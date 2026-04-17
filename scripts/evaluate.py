"""
Forge — Evaluation Script

Runs mini and full evaluations on Forge checkpoints using both
custom verification and lm-eval-harness benchmarks.

Usage:
    python -m scripts.evaluate --config config.yaml \
        --adapter-path ./checkpoints/phase3_final/final \
        --mode mini
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forge.config import ForgeConfig
from forge.lora import load_model_for_generation
from forge.data.datasets import FORGE_SYSTEM_PROMPT
from forge.verify.math_verifier import verify_math_solution
from forge.verify.code_sandbox import DockerSandbox

logger = logging.getLogger(__name__)


def evaluate_gsm8k(
    model,
    tokenizer,
    num_samples: int = 200,
    cfg: ForgeConfig | None = None,
) -> dict:
    """Evaluate on GSM8K test set."""
    from datasets import load_dataset

    logger.info(f"Evaluating GSM8K ({num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    if num_samples and num_samples < len(ds):
        ds = ds.select(range(num_samples))

    correct = 0
    total = 0

    for example in ds:
        question = example["question"]
        answer_text = example["answer"]
        ground_truth = answer_text.rsplit("####", 1)[-1].strip() if "####" in answer_text else ""

        # Generate solution
        messages = [
            {"role": "system", "content": FORGE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.0,  # Greedy for eval
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        solution = tokenizer.decode(generated, skip_special_tokens=True)

        # Verify
        verdict = verify_math_solution(solution, ground_truth)
        if verdict.correct:
            correct += 1
        total += 1

        if total % 50 == 0:
            logger.info(f"  GSM8K progress: {total}/{num_samples}, accuracy: {correct/total:.3f}")

    accuracy = correct / max(total, 1)
    logger.info(f"GSM8K: {correct}/{total} = {accuracy:.4f}")
    return {"gsm8k_accuracy": accuracy, "gsm8k_correct": correct, "gsm8k_total": total}


def evaluate_humaneval(
    model,
    tokenizer,
    cfg: ForgeConfig | None = None,
) -> dict:
    """
    Evaluate on HumanEval.
    Uses a simplified subset evaluation since full HumanEval
    requires the official evaluation harness.
    """
    logger.info("HumanEval evaluation — use lm-eval-harness for full results")
    return {"humaneval_note": "Run via lm-eval-harness for official scores"}


def run_evaluation(
    cfg: ForgeConfig,
    adapter_path: str,
    mode: str = "mini",
    output_dir: str = "./eval_results",
) -> dict:
    """
    Run model evaluation.

    Args:
        cfg: Forge configuration.
        adapter_path: Path to the adapter to evaluate.
        mode: "mini" for quick eval, "full" for comprehensive.
        output_dir: Where to save results.

    Returns:
        Dict of evaluation metrics.
    """
    logger.info("=" * 60)
    logger.info(f"FORGE — Evaluation ({mode} mode)")
    logger.info(f"Adapter: {adapter_path}")
    logger.info("=" * 60)

    # Load model
    model, tokenizer = load_model_for_generation(cfg, adapter_path=adapter_path)

    results = {"adapter_path": adapter_path, "mode": mode}
    start = time.time()

    if mode == "mini":
        # Quick evaluation
        mini_cfg = cfg._raw.get("evaluation", {}).get("mini_eval", {})
        gsm8k_results = evaluate_gsm8k(
            model, tokenizer,
            num_samples=mini_cfg.get("gsm8k_samples", 200),
            cfg=cfg,
        )
        results.update(gsm8k_results)

    elif mode == "full":
        # Full evaluation
        gsm8k_results = evaluate_gsm8k(model, tokenizer, num_samples=None, cfg=cfg)
        results.update(gsm8k_results)

        # Suggest lm-eval-harness for standardized benchmarks
        logger.info(
            "\nFor full standardized evaluation, run:\n"
            f"  lm_eval --model hf "
            f"--model_args pretrained={cfg.model_name},peft={adapter_path} "
            f"--tasks gsm8k,minerva_math,humaneval,mbpp "
            f"--batch_size 8 --output_path {output_dir}/\n"
        )

    elapsed = time.time() - start
    results["eval_time_seconds"] = elapsed

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, f"eval_{mode}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Forge Evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--mode", choices=["mini", "full"], default="mini")
    parser.add_argument("--output-dir", default="./eval_results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ForgeConfig.from_yaml(args.config)
    run_evaluation(cfg, args.adapter_path, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
