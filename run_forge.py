#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════╗
║  FORGE — Self-Improving Code & Math Reasoning            ║
║  via Verified Self-Play Fine-Tuning                      ║
║                                                          ║
║  One A100 40GB. Twenty-four hours.                       ║
║  No reward model. No human annotators.                   ║
╚═══════════════════════════════════════════════════════════╝

Main orchestration script that runs the full 3-phase pipeline:
  Phase 1: Reasoning SFT    (Hours  0–6)
  Phase 2: Self-Play DPO    (Hours  6–20)
  Phase 3: Refinement       (Hours 20–24)

Usage:
    python run_forge.py --config config.yaml [--phase all|1|2|3]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forge.config import ForgeConfig

logger = logging.getLogger(__name__)
console = Console()


BANNER = """
╔═══════════════════════════════════════════════════════════╗
║                                                          ║
║   🔨 FORGE v0.1.0                                       ║
║                                                          ║
║   Self-Improving Code & Math Reasoning                   ║
║   via Verified Self-Play Fine-Tuning                     ║
║                                                          ║
║   Target: 1× A100 40GB, ~24 hours                       ║
║                                                          ║
╚═══════════════════════════════════════════════════════════╝
"""


def display_config_summary(cfg: ForgeConfig):
    """Display a summary table of the configuration."""
    table = Table(title="Forge Configuration Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Base Model", cfg.model_name)
    table.add_row("Quantization", f"{cfg.quantization} (double={cfg.double_quant})")
    table.add_row("Max Seq Length", str(cfg.max_seq_length))
    table.add_row("", "")
    table.add_row("LoRA Type", "Stratified (depth-aware)")
    for group in cfg.lora_layer_groups:
        table.add_row(
            f"  └ {group.name}",
            f"layers {group.start}-{group.end}, r={group.rank}, α={group.alpha}",
        )
    table.add_row("", "")
    table.add_row("SFT LR", str(cfg.sft_lr))
    table.add_row("SFT Epochs", str(cfg.sft_epochs))
    table.add_row("SFT Effective Batch", str(cfg.sft_batch_size * cfg.sft_grad_accum))
    table.add_row("", "")
    table.add_row("Self-Play K", str(cfg.selfplay_k))
    table.add_row("Self-Play Temperature", str(cfg.selfplay_temperature))
    table.add_row("Max Self-Play Problems", str(cfg.max_selfplay_problems))
    table.add_row("", "")
    table.add_row("DPO β", str(cfg.dpo_beta))
    table.add_row("DPO LR", str(cfg.dpo_lr))
    table.add_row("DPO Epochs", str(cfg.dpo_epochs))
    table.add_row("", "")
    table.add_row("Refinement DPO Epochs", str(cfg.refine_dpo_epochs))
    table.add_row("Max Refine Problems", str(cfg.max_refine_problems))

    console.print(table)


def run_full_pipeline(cfg: ForgeConfig, start_phase: int = 1):
    """Run the complete Forge training pipeline."""
    pipeline_start = time.time()
    checkpoints = {}

    # ===== PHASE 1: Reasoning SFT =====
    if start_phase <= 1:
        console.print(Panel(
            "[bold blue]Phase 1: Reasoning SFT[/bold blue]\n"
            "Teaching structured reasoning via supervised fine-tuning",
            title="🔨 Forge",
            border_style="blue",
        ))

        from scripts.phase1_sft import run_phase1_sft
        sft_path = run_phase1_sft(cfg)
        checkpoints["sft"] = sft_path

        # Mini evaluation
        console.print("\n[yellow]Running mini-evaluation on SFT checkpoint...[/yellow]")
        from scripts.evaluate import run_evaluation
        run_evaluation(cfg, sft_path, mode="mini", output_dir="./eval_results/phase1")

        elapsed = (time.time() - pipeline_start) / 3600
        console.print(f"[green]Phase 1 complete.[/green] Elapsed: {elapsed:.1f}h")
    else:
        # Load existing SFT checkpoint
        sft_path = os.path.join(cfg.sft_output_dir, "final")
        checkpoints["sft"] = sft_path
        console.print(f"[yellow]Skipping Phase 1, using SFT adapter: {sft_path}[/yellow]")

    # ===== PHASE 2: Self-Play DPO =====
    if start_phase <= 2:
        console.print(Panel(
            "[bold yellow]Phase 2: Verified Self-Play DPO[/bold yellow]\n"
            "Generate → Verify → Build preference pairs → DPO train",
            title="🔨 Forge",
            border_style="yellow",
        ))

        # Phase 2A: Generation
        from scripts.phase2_generate import run_phase2_generate
        cache_path = run_phase2_generate(cfg, adapter_path=checkpoints["sft"])
        checkpoints["solutions_cache"] = cache_path

        # Phase 2B: DPO
        from scripts.phase2_dpo import run_phase2_dpo
        dpo_path = run_phase2_dpo(
            cfg,
            adapter_path=checkpoints["sft"],
            solutions_cache=cache_path,
        )
        checkpoints["dpo"] = dpo_path

        # Mini evaluation
        console.print("\n[yellow]Running mini-evaluation on DPO checkpoint...[/yellow]")
        from scripts.evaluate import run_evaluation
        run_evaluation(cfg, dpo_path, mode="mini", output_dir="./eval_results/phase2")

        elapsed = (time.time() - pipeline_start) / 3600
        console.print(f"[green]Phase 2 complete.[/green] Elapsed: {elapsed:.1f}h")
    else:
        dpo_path = os.path.join(cfg.dpo_output_dir, "final")
        checkpoints["dpo"] = dpo_path
        console.print(f"[yellow]Skipping Phase 2, using DPO adapter: {dpo_path}[/yellow]")

    # ===== PHASE 3: Iterative Refinement =====
    if start_phase <= 3:
        console.print(Panel(
            "[bold red]Phase 3: Iterative Refinement[/bold red]\n"
            "Re-generate on hard problems → Final DPO round",
            title="🔨 Forge",
            border_style="red",
        ))

        from scripts.phase3_refine import run_phase3_refine
        final_path = run_phase3_refine(
            cfg,
            dpo_adapter_path=checkpoints["dpo"],
            sft_adapter_path=checkpoints["sft"],
        )
        checkpoints["final"] = final_path

        # Full evaluation
        console.print("\n[yellow]Running final evaluation...[/yellow]")
        from scripts.evaluate import run_evaluation
        run_evaluation(cfg, final_path, mode="mini", output_dir="./eval_results/final")

        elapsed = (time.time() - pipeline_start) / 3600
        console.print(f"[green]Phase 3 complete.[/green] Elapsed: {elapsed:.1f}h")

    # ===== SUMMARY =====
    total_time = (time.time() - pipeline_start) / 3600
    summary = Table(title="🔨 Forge Training Complete")
    summary.add_column("Phase", style="cyan")
    summary.add_column("Checkpoint", style="green")

    for name, path in checkpoints.items():
        summary.add_row(name, str(path))
    summary.add_row("", "")
    summary.add_row("Total Time", f"{total_time:.1f} hours")

    console.print(summary)
    console.print(Panel(
        f"[bold green]✅ Forge training complete![/bold green]\n\n"
        f"Final model: {checkpoints.get('final', checkpoints.get('dpo', 'N/A'))}\n"
        f"Total time: {total_time:.1f} hours\n\n"
        f"Run full evaluation:\n"
        f"  python -m scripts.evaluate --adapter-path {checkpoints.get('final', '')} --mode full",
        title="🔨 Forge",
        border_style="green",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Forge: Self-Improving Code & Math Reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_forge.py --config config.yaml              # Run full pipeline
  python run_forge.py --config config.yaml --phase 2    # Start from Phase 2
  python run_forge.py --config config.yaml --phase 3    # Start from Phase 3 (refinement only)
        """,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Phase to start from (1=SFT, 2=Self-Play, 3=Refine)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("forge_training.log"),
        ],
    )

    console.print(BANNER)

    # Load config
    cfg = ForgeConfig.from_yaml(args.config)
    display_config_summary(cfg)

    # Run
    run_full_pipeline(cfg, start_phase=args.phase)


if __name__ == "__main__":
    main()
