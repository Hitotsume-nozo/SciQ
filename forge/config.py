"""
Configuration loader for Forge.
Reads config.yaml and provides typed access to all settings.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class LoraLayerGroup:
    """Configuration for one group of layers in Stratified LoRA."""
    name: str
    start: int
    end: int
    rank: int
    alpha: int


@dataclass
class ForgeConfig:
    """
    Central configuration object parsed from config.yaml.
    All pipeline stages read from this single source of truth.
    """

    # --- Raw YAML dict (for passthrough to libraries) ---
    _raw: dict = field(default_factory=dict, repr=False)

    # --- Model ---
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    quantization: str = "nf4"
    double_quant: bool = True
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    max_seq_length: int = 4096
    trust_remote_code: bool = True

    # --- LoRA ---
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    lora_layer_groups: list[LoraLayerGroup] = field(default_factory=list)

    # --- SFT ---
    sft_lr: float = 2e-4
    sft_epochs: int = 3
    sft_batch_size: int = 4
    sft_grad_accum: int = 4
    sft_output_dir: str = "./checkpoints/phase1_sft"

    # --- Self-Play Generation ---
    selfplay_k: int = 8
    selfplay_temperature: float = 0.7
    selfplay_top_p: float = 0.95
    selfplay_max_new_tokens: int = 2048
    selfplay_cache_dir: str = "./cache/selfplay_solutions"

    # --- Verification ---
    code_timeout: int = 5
    code_memory_limit: str = "256m"
    math_tolerance: float = 1e-6
    max_docker_workers: int = 10
    docker_image: str = "python:3.11-slim"

    # --- DPO ---
    dpo_beta: float = 0.1
    dpo_lr: float = 5e-6
    dpo_epochs: int = 2
    dpo_batch_size: int = 2
    dpo_grad_accum: int = 8
    dpo_loss_type: str = "sigmoid"
    dpo_output_dir: str = "./checkpoints/phase2_dpo"

    # --- Refinement ---
    refine_dpo_epochs: int = 1
    refine_output_dir: str = "./checkpoints/phase3_final"

    # --- Data ---
    max_sft_examples: int = 20000
    max_selfplay_problems: int = 5000
    max_refine_problems: int = 2000

    # --- Curriculum ---
    easy_threshold: float = 0.7
    hard_threshold: float = 0.3

    # --- Infrastructure ---
    seed: int = 42
    output_base: str = "./outputs"
    cache_dir: str = "./cache"
    wandb_project: str = "forge"
    wandb_run_name: str = "forge-v1"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ForgeConfig":
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f)

        cfg = cls(_raw=raw)

        # --- Model ---
        model = raw.get("model", {})
        cfg.model_name = model.get("name", cfg.model_name)
        cfg.quantization = model.get("quantization", cfg.quantization)
        cfg.double_quant = model.get("double_quant", cfg.double_quant)
        cfg.torch_dtype = model.get("torch_dtype", cfg.torch_dtype)
        cfg.attn_implementation = model.get("attn_implementation", cfg.attn_implementation)
        cfg.max_seq_length = model.get("max_seq_length", cfg.max_seq_length)
        cfg.trust_remote_code = model.get("trust_remote_code", cfg.trust_remote_code)

        # --- LoRA ---
        lora = raw.get("lora", {})
        cfg.lora_target_modules = lora.get("target_modules", cfg.lora_target_modules)
        cfg.lora_dropout = lora.get("dropout", cfg.lora_dropout)
        cfg.lora_bias = lora.get("bias", cfg.lora_bias)

        cfg.lora_layer_groups = []
        for group_name, group_cfg in lora.get("layers", {}).items():
            cfg.lora_layer_groups.append(LoraLayerGroup(
                name=group_name,
                start=group_cfg["start"],
                end=group_cfg["end"],
                rank=group_cfg["rank"],
                alpha=group_cfg["alpha"],
            ))

        # --- SFT ---
        sft = raw.get("sft", {})
        cfg.sft_lr = sft.get("learning_rate", cfg.sft_lr)
        cfg.sft_epochs = sft.get("num_epochs", cfg.sft_epochs)
        cfg.sft_batch_size = sft.get("per_device_batch_size", cfg.sft_batch_size)
        cfg.sft_grad_accum = sft.get("gradient_accumulation_steps", cfg.sft_grad_accum)
        cfg.sft_output_dir = sft.get("output_dir", cfg.sft_output_dir)

        # --- Self-Play ---
        sp = raw.get("selfplay", {})
        gen = sp.get("generation", {})
        cfg.selfplay_k = gen.get("num_candidates", cfg.selfplay_k)
        cfg.selfplay_temperature = gen.get("temperature", cfg.selfplay_temperature)
        cfg.selfplay_top_p = gen.get("top_p", cfg.selfplay_top_p)
        cfg.selfplay_max_new_tokens = gen.get("max_new_tokens", cfg.selfplay_max_new_tokens)
        cfg.selfplay_cache_dir = gen.get("output_dir", cfg.selfplay_cache_dir)

        ver = sp.get("verification", {})
        cfg.code_timeout = ver.get("code_timeout", cfg.code_timeout)
        cfg.code_memory_limit = ver.get("code_memory_limit", cfg.code_memory_limit)
        cfg.math_tolerance = ver.get("math_tolerance", cfg.math_tolerance)
        cfg.max_docker_workers = ver.get("max_docker_workers", cfg.max_docker_workers)
        cfg.docker_image = ver.get("docker_image", cfg.docker_image)

        # --- DPO ---
        dpo = raw.get("dpo", {})
        cfg.dpo_beta = dpo.get("beta", cfg.dpo_beta)
        cfg.dpo_lr = dpo.get("learning_rate", cfg.dpo_lr)
        cfg.dpo_epochs = dpo.get("num_epochs", cfg.dpo_epochs)
        cfg.dpo_batch_size = dpo.get("per_device_batch_size", cfg.dpo_batch_size)
        cfg.dpo_grad_accum = dpo.get("gradient_accumulation_steps", cfg.dpo_grad_accum)
        cfg.dpo_loss_type = dpo.get("loss_type", cfg.dpo_loss_type)
        cfg.dpo_output_dir = dpo.get("output_dir", cfg.dpo_output_dir)

        # --- Refinement ---
        ref = raw.get("refinement", {})
        cfg.refine_dpo_epochs = ref.get("dpo_epochs", cfg.refine_dpo_epochs)
        cfg.refine_output_dir = ref.get("output_dir", cfg.refine_output_dir)

        # --- Data ---
        data = raw.get("data", {})
        cfg.max_sft_examples = data.get("max_sft_examples", cfg.max_sft_examples)
        cfg.max_selfplay_problems = data.get("max_selfplay_problems", cfg.max_selfplay_problems)
        cfg.max_refine_problems = data.get("max_refine_problems", cfg.max_refine_problems)

        # --- Curriculum ---
        cur = raw.get("curriculum", {})
        cfg.easy_threshold = cur.get("easy_threshold", cfg.easy_threshold)
        cfg.hard_threshold = cur.get("hard_threshold", cfg.hard_threshold)

        # --- Infrastructure ---
        infra = raw.get("infrastructure", {})
        cfg.seed = infra.get("seed", cfg.seed)
        cfg.output_base = infra.get("output_base", cfg.output_base)
        cfg.cache_dir = infra.get("cache_dir", cfg.cache_dir)
        cfg.wandb_project = infra.get("wandb_project", cfg.wandb_project)
        cfg.wandb_run_name = infra.get("wandb_run_name", cfg.wandb_run_name)

        return cfg

    def get_raw(self, *keys: str, default: Any = None) -> Any:
        """Access nested raw config values by dot-separated keys."""
        obj = self._raw
        for key in keys:
            if isinstance(obj, dict):
                obj = obj.get(key, default)
            else:
                return default
        return obj
