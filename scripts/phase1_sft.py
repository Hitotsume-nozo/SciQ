"""
Forge — Phase 1: Reasoning SFT (Supervised Fine-Tuning)

Teaches the base model to produce structured reasoning before answering,
using distilled Chain-of-Thought traces. This establishes the foundation
for self-play in Phase 2.

Usage:
    python -m scripts.phase1_sft --config config.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import torch
from datasets import Dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.config import ForgeConfig
from forge.lora import load_base_model_with_stratified_lora
from forge.data.datasets import load_sft_dataset

logger = logging.getLogger(__name__)


def run_phase1_sft(cfg: ForgeConfig) -> str:
    """
    Execute Phase 1: Reasoning SFT.

    Returns:
        Path to the saved SFT checkpoint.
    """
    logger.info("=" * 60)
    logger.info("FORGE — Phase 1: Reasoning SFT")
    logger.info("=" * 60)

    # --- Load model with Stratified LoRA ---
    model, tokenizer = load_base_model_with_stratified_lora(cfg)

    # --- Load and prepare dataset ---
    logger.info("Loading SFT dataset...")
    dataset = load_sft_dataset(cfg)
    logger.info(f"Dataset size: {len(dataset)} examples")

    # Pre-format: apply chat template to convert messages → text strings
    # This makes the dataset compatible with all TRL versions
    def apply_chat_template(example):
        messages = example.get("messages", [])
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    logger.info("Applying chat template to dataset...")
    dataset = dataset.map(apply_chat_template, num_proc=4, desc="Formatting")

    # Split into train/eval
    split = dataset.train_test_split(test_size=0.05, seed=cfg.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # --- Initialize SFT Trainer (compatible with TRL v0.12+) ---
    from trl import SFTTrainer, SFTConfig

    sft_config = SFTConfig(
        output_dir=cfg.sft_output_dir,
        num_train_epochs=cfg.sft_epochs,
        per_device_train_batch_size=max(1, cfg.sft_batch_size // 2),
        per_device_eval_batch_size=max(1, cfg.sft_batch_size // 2),
        gradient_accumulation_steps=cfg.sft_grad_accum * 2,
        learning_rate=cfg.sft_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_8bit",
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"{cfg.wandb_run_name}-sft",
        seed=cfg.seed,
        dataloader_num_workers=0,
        # SFT-specific settings
        max_length=cfg.max_seq_length,
        packing=False,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # --- Train ---
    logger.info("Starting SFT training...")
    trainer.train()

    # --- Save ---
    final_path = os.path.join(cfg.sft_output_dir, "final")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Phase 1 SFT model saved to: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Forge Phase 1: Reasoning SFT")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ForgeConfig.from_yaml(args.config)
    run_phase1_sft(cfg)


if __name__ == "__main__":
    main()
