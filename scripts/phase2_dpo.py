"""
Forge — Phase 2B: DPO Training on Self-Play Preference Pairs

Takes the verified solutions from Phase 2A, constructs preference pairs
(correct > incorrect, concise > verbose), and trains via Direct Preference
Optimization (DPO).

Usage:
    python -m scripts.phase2_dpo --config config.yaml \
        --adapter-path ./checkpoints/phase1_sft/final \
        --solutions-cache ./cache/selfplay_solutions
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.config import ForgeConfig
from forge.lora import load_base_model_with_stratified_lora
from forge.data.pair_builder import (
    SolutionWithVerdict,
    build_all_preference_pairs,
    pairs_to_dpo_dataset,
)
from forge.data.datasets import load_problem_pool

logger = logging.getLogger(__name__)


def load_cached_solutions(cache_dir: str) -> dict[int, list[SolutionWithVerdict]]:
    """Load cached solutions from Phase 2A and convert to SolutionWithVerdict."""
    cache_path = Path(cache_dir) / "solutions.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Solutions cache not found: {cache_path}")

    with open(cache_path) as f:
        raw = json.load(f)

    result: dict[int, list[SolutionWithVerdict]] = {}
    for idx_str, solutions in raw.items():
        idx = int(idx_str)
        result[idx] = [
            SolutionWithVerdict(
                text=s["text"],
                correct=s["correct"],
                quality_score=s.get("quality_score", 0.0),
                quality_tier=s.get("quality_tier", "D"),
                word_count=s.get("word_count", len(s["text"].split())),
            )
            for s in solutions
        ]

    logger.info(f"Loaded {sum(len(v) for v in result.values())} solutions for {len(result)} problems")
    return result


def run_phase2_dpo(
    cfg: ForgeConfig,
    adapter_path: str,
    solutions_cache: str,
) -> str:
    """
    Execute Phase 2B: DPO training on self-play preference pairs.

    Args:
        cfg: Forge configuration.
        adapter_path: Path to Phase 1 SFT adapter (used as reference model).
        solutions_cache: Path to cached solutions from Phase 2A.

    Returns:
        Path to the saved DPO checkpoint.
    """
    logger.info("=" * 60)
    logger.info("FORGE — Phase 2B: DPO Training")
    logger.info("=" * 60)

    # --- Load cached solutions ---
    all_solutions = load_cached_solutions(solutions_cache)

    # --- Load problem pool (same order as generation) ---
    problems = load_problem_pool(cfg, max_problems=cfg.max_selfplay_problems)
    problems_list = [problems[i] for i in range(len(problems))]

    # --- Build preference pairs ---
    logger.info("Building preference pairs...")
    pairs = build_all_preference_pairs(problems_list, all_solutions, cfg)

    if not pairs:
        logger.error("No preference pairs generated! Check solution quality.")
        return ""

    logger.info(f"Total preference pairs: {len(pairs)}")

    # Convert to DPO dataset format
    dpo_data = pairs_to_dpo_dataset(pairs)
    dpo_dataset = Dataset.from_dict(dpo_data)

    # Split
    split = dpo_dataset.train_test_split(test_size=0.05, seed=cfg.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    logger.info(f"DPO Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # --- Load model (fresh, with LoRA initialized from SFT) ---
    logger.info("Loading model with Stratified LoRA for DPO...")

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization,
        bnb_4bit_compute_dtype=getattr(torch, cfg.torch_dtype),
        bnb_4bit_use_double_quant=cfg.double_quant,
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load SFT adapter as the starting point
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
    logger.info(f"Loaded SFT adapter from: {adapter_path}")

    # --- Initialize DPO Trainer (TRL v0.12+ API) ---
    from trl import DPOTrainer, DPOConfig

    dpo_config = DPOConfig(
        output_dir=cfg.dpo_output_dir,
        num_train_epochs=cfg.dpo_epochs,
        per_device_train_batch_size=max(1, cfg.dpo_batch_size),
        per_device_eval_batch_size=max(1, cfg.dpo_batch_size),
        gradient_accumulation_steps=cfg.dpo_grad_accum,
        learning_rate=cfg.dpo_lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
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
        run_name=f"{cfg.wandb_run_name}-dpo",
        seed=cfg.seed,
        dataloader_num_workers=0,
        # DPO-specific settings
        beta=cfg.dpo_beta,
        loss_type=cfg.dpo_loss_type,
        max_length=cfg.max_seq_length,
        max_prompt_length=cfg.max_seq_length // 2,
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # PEFT handles reference model automatically
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=dpo_config,
    )

    # --- Train ---
    logger.info("Starting DPO training...")
    dpo_trainer.train()

    # --- Save ---
    final_path = os.path.join(cfg.dpo_output_dir, "final")
    dpo_trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Phase 2 DPO model saved to: {final_path}")

    return final_path


def main():
    parser = argparse.ArgumentParser(description="Forge Phase 2B: DPO Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--solutions-cache", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ForgeConfig.from_yaml(args.config)
    run_phase2_dpo(cfg, args.adapter_path, args.solutions_cache)


if __name__ == "__main__":
    main()
