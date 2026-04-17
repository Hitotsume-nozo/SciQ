"""
Stratified LoRA: Depth-Aware Adapter Architecture for Forge.

Instead of applying a uniform LoRA rank across all transformer layers,
Stratified LoRA allocates parameter budget based on layer depth:

  - Bottom layers (Foundation): Low rank — these encode general language
    understanding that needs minimal task-specific adaptation.
  - Middle layers (Reasoning Core/Deep Reasoning): High rank — this is
    where algorithmic planning and multi-step computation happen.
  - Top layers (Generation): Moderate rank — output formatting and code
    synthesis need some adaptation but less than reasoning.

This concentrates trainable parameters where they matter most for
reasoning-intensive tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

if TYPE_CHECKING:
    from transformers import PreTrainedModel
    from .config import ForgeConfig

logger = logging.getLogger(__name__)


def build_stratified_lora_config(cfg: ForgeConfig) -> LoraConfig:
    """
    Build a PEFT LoraConfig with per-layer rank and alpha overrides
    derived from the Stratified LoRA specification in ForgeConfig.

    Uses PEFT's `rank_pattern` and `alpha_pattern` to assign different
    ranks to different layer groups based on their depth in the model.

    Args:
        cfg: Forge configuration with lora_layer_groups defined.

    Returns:
        A LoraConfig ready to be applied to a model.
    """
    if not cfg.lora_layer_groups:
        logger.warning(
            "No LoRA layer groups defined — falling back to uniform rank=32"
        )
        return LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=cfg.lora_target_modules,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            task_type="CAUSAL_LM",
        )

    # Build per-module rank/alpha patterns using PEFT regex matching.
    # Pattern format: "model.layers.{idx}.self_attn.{module}" or
    #                 "model.layers.{idx}.mlp.{module}"
    rank_pattern: dict[str, int] = {}
    alpha_pattern: dict[str, int] = {}

    # Determine the default rank (use the most common group's rank)
    default_group = max(
        cfg.lora_layer_groups,
        key=lambda g: g.end - g.start + 1,
    )
    default_rank = default_group.rank
    default_alpha = default_group.alpha

    total_trainable = 0

    for group in cfg.lora_layer_groups:
        for layer_idx in range(group.start, group.end + 1):
            for module in cfg.lora_target_modules:
                # PEFT uses regex-like keys for rank_pattern
                # The key should match the full module name in the model
                key = f"model.layers.{layer_idx}.*.{module}"
                rank_pattern[key] = group.rank
                alpha_pattern[key] = group.alpha

        num_layers = group.end - group.start + 1
        num_modules = len(cfg.lora_target_modules)
        # Rough estimate: each LoRA adapter = 2 * hidden_dim * rank params
        # For Qwen2.5-7B: hidden_dim=3584, intermediate=18944
        # Attention projections: 3584 * rank * 2 each
        # MLP projections: varies (gate/up: 18944*rank*2, down: 18944*rank*2)
        estimated_params = num_layers * num_modules * group.rank * 3584 * 2
        total_trainable += estimated_params

        logger.info(
            f"  {group.name}: layers {group.start}-{group.end}, "
            f"rank={group.rank}, α={group.alpha}, "
            f"~{estimated_params / 1e6:.1f}M params"
        )

    logger.info(f"  Total estimated trainable: ~{total_trainable / 1e6:.1f}M params")

    config = LoraConfig(
        r=default_rank,
        lora_alpha=default_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias=cfg.lora_bias,
        task_type="CAUSAL_LM",
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    return config


def load_base_model_with_stratified_lora(
    cfg: ForgeConfig,
) -> tuple[PreTrainedModel, any]:
    """
    Load the quantized base model and apply Stratified LoRA adapters.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    logger.info(f"Loading base model: {cfg.model_name}")

    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization,
        bnb_4bit_compute_dtype=getattr(torch, cfg.torch_dtype),
        bnb_4bit_use_double_quant=cfg.double_quant,
    )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=cfg.trust_remote_code,
        attn_implementation=cfg.attn_implementation,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=cfg.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    # Apply Stratified LoRA
    logger.info("Applying Stratified LoRA configuration:")
    lora_config = build_stratified_lora_config(cfg)
    model = get_peft_model(model, lora_config)

    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    return model, tokenizer


def load_model_for_generation(
    cfg: ForgeConfig,
    adapter_path: str | None = None,
) -> tuple[PreTrainedModel, any]:
    """
    Load model for inference/generation (self-play phase).
    Optionally loads a trained LoRA adapter.

    Args:
        cfg: Forge configuration.
        adapter_path: Path to saved LoRA adapter weights. If None,
                      loads base model only.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.quantization,
        bnb_4bit_compute_dtype=getattr(torch, cfg.torch_dtype),
        bnb_4bit_use_double_quant=cfg.double_quant,
    )

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

    if adapter_path:
        logger.info(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, tokenizer
