"""
Forge — Phase 3: Iterative Refinement

Takes the DPO-trained model and runs one more self-play iteration
focused on hard problems. This exploits the improved model's ability
to solve problems it previously couldn't, generating higher-quality
preference pairs for a final DPO round.

Usage:
    python -m scripts.phase3_refine --config config.yaml \
        --dpo-adapter-path ./checkpoints/phase2_dpo/final \
        --sft-adapter-path ./checkpoints/phase1_sft/final
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.config import ForgeConfig

logger = logging.getLogger(__name__)


def run_phase3_refine(
    cfg: ForgeConfig,
    dpo_adapter_path: str,
    sft_adapter_path: str,
) -> str:
    """
    Execute Phase 3: Iterative Refinement.

    1. Re-generate solutions on hard problems using the DPO model
    2. Verify and build refined preference pairs
    3. Run a quick 1-epoch DPO round

    Returns:
        Path to the final model checkpoint.
    """
    logger.info("=" * 60)
    logger.info("FORGE — Phase 3: Iterative Refinement")
    logger.info("=" * 60)

    # Step 1: Override config for refinement focus
    refinement_cache = os.path.join(cfg.cache_dir, "refinement_solutions")

    # Reduce problem count and focus on hard problems
    original_max = cfg.max_selfplay_problems
    cfg.max_selfplay_problems = cfg.max_refine_problems

    # Step 2: Re-generate solutions using the DPO model
    logger.info("Step 1/3: Re-generating solutions with DPO model on hard problems...")
    from scripts.phase2_generate import run_phase2_generate

    # Temporarily redirect cache
    original_cache = cfg.selfplay_cache_dir
    cfg.selfplay_cache_dir = refinement_cache

    run_phase2_generate(cfg, adapter_path=dpo_adapter_path)

    # Restore config
    cfg.selfplay_cache_dir = original_cache
    cfg.max_selfplay_problems = original_max

    # Step 3: Run DPO with refined pairs
    logger.info("Step 2/3: Building refined preference pairs and training DPO...")
    from scripts.phase2_dpo import run_phase2_dpo

    # Override DPO config for quick refinement
    original_epochs = cfg.dpo_epochs
    original_output = cfg.dpo_output_dir
    cfg.dpo_epochs = cfg.refine_dpo_epochs
    cfg.dpo_output_dir = cfg.refine_output_dir

    final_path = run_phase2_dpo(
        cfg,
        adapter_path=dpo_adapter_path,  # Start from DPO model
        solutions_cache=refinement_cache,
    )

    # Restore config
    cfg.dpo_epochs = original_epochs
    cfg.dpo_output_dir = original_output

    logger.info(f"Phase 3 refinement complete. Final model: {final_path}")
    return final_path


def main():
    parser = argparse.ArgumentParser(description="Forge Phase 3: Iterative Refinement")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dpo-adapter-path", required=True)
    parser.add_argument("--sft-adapter-path", required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ForgeConfig.from_yaml(args.config)
    run_phase3_refine(cfg, args.dpo_adapter_path, args.sft_adapter_path)


if __name__ == "__main__":
    main()
