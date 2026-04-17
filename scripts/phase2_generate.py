"""
Forge — Phase 2A: Self-Play Solution Generation

Generates K candidate solutions per problem using the SFT-trained model.
Solutions are cached to disk for subsequent verification and pair building.

This is the core of the self-play loop: the model generates diverse
solutions via temperature sampling, which are then verified by the
code sandbox and math verifier to produce preference pairs.

Usage:
    python -m scripts.phase2_generate --config config.yaml --adapter-path ./checkpoints/phase1_sft/final
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
from forge.data.datasets import load_problem_pool, FORGE_SYSTEM_PROMPT
from forge.verify.code_sandbox import DockerSandbox, verify_code_batch
from forge.verify.math_verifier import verify_math_solution
from forge.verify.process_verifier import analyze_reasoning
from forge.data.pair_builder import SolutionWithVerdict

logger = logging.getLogger(__name__)


def generate_solutions_for_problem(
    model,
    tokenizer,
    problem: dict,
    num_candidates: int = 8,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 1024,
) -> list[str]:
    """
    Generate K candidate solutions for a single problem.

    Batches all K candidates into a single model.generate() call
    for much higher GPU utilization and throughput.
    """
    # Build the prompt
    messages = [
        {"role": "system", "content": FORGE_SYSTEM_PROMPT},
        {"role": "user", "content": problem["prompt"]},
    ]

    # Apply chat template
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    # Repeat input K times to batch-generate all candidates at once
    input_ids = inputs["input_ids"].repeat(num_candidates, 1)
    attention_mask = inputs["attention_mask"].repeat(num_candidates, 1)
    prompt_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode each candidate (skip the prompt tokens)
    solutions = []
    for i in range(num_candidates):
        generated_ids = outputs[i][prompt_len:]
        solution_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        solutions.append(solution_text)

    return solutions


def verify_solution(
    problem: dict,
    solution_text: str,
    cfg: ForgeConfig,
    code_sandbox: DockerSandbox | None = None,
) -> SolutionWithVerdict:
    """
    Verify a single solution and produce a SolutionWithVerdict.
    """
    domain = problem.get("domain", "math")
    correct = False
    quality_score = 0.0
    quality_tier = "D"

    if domain == "math":
        # Math verification
        ground_truth = problem.get("ground_truth", "")
        if ground_truth:
            verdict = verify_math_solution(
                solution_text,
                ground_truth,
                tolerance=cfg.math_tolerance,
            )
            correct = verdict.correct
        else:
            correct = False  # Can't verify without ground truth

    elif domain == "code":
        # Code verification via sandbox
        import re
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", solution_text, re.DOTALL)
        if code_match and code_sandbox:
            code = code_match.group(1)
            test_code = problem.get("test_code", "")
            result = code_sandbox.execute(
                code=code,
                test_code=test_code,
                language="python",
            )
            correct = result.passed
        else:
            correct = False

    # Process-level analysis
    process_verdict = analyze_reasoning(solution_text, correct)
    quality_score = process_verdict.quality_score
    quality_tier = process_verdict.tier

    return SolutionWithVerdict(
        text=solution_text,
        correct=correct,
        quality_score=quality_score,
        quality_tier=quality_tier,
        word_count=len(solution_text.split()),
    )


def run_phase2_generate(cfg: ForgeConfig, adapter_path: str) -> str:
    """
    Execute Phase 2A: Generate and verify solutions.

    Returns:
        Path to the cached solutions directory.
    """
    logger.info("=" * 60)
    logger.info("FORGE — Phase 2A: Self-Play Solution Generation")
    logger.info("=" * 60)

    # --- Load model ---
    logger.info(f"Loading model with adapter from: {adapter_path}")
    model, tokenizer = load_model_for_generation(cfg, adapter_path=adapter_path)

    # --- Load problem pool ---
    logger.info("Loading problem pool...")
    problems = load_problem_pool(cfg, max_problems=cfg.max_selfplay_problems)
    logger.info(f"Problem pool: {len(problems)} problems")

    # --- Initialize code sandbox ---
    code_sandbox = DockerSandbox(
        timeout=cfg.code_timeout,
        memory_limit=cfg.code_memory_limit,
        docker_image=cfg.docker_image,
    )

    # --- Generate and verify ---
    cache_dir = Path(cfg.selfplay_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[int, list[dict]] = {}
    stats = {
        "total_problems": len(problems),
        "total_solutions": 0,
        "total_correct": 0,
        "total_incorrect": 0,
    }

    start_time = time.time()

    from tqdm import tqdm
    pbar = tqdm(range(len(problems)), desc="Generating solutions", unit="problem")

    for idx in pbar:
        problem = problems[idx]

        # Generate K solutions (batched — single model.generate call)
        try:
            solutions_text = generate_solutions_for_problem(
                model=model,
                tokenizer=tokenizer,
                problem=problem,
                num_candidates=cfg.selfplay_k,
                temperature=cfg.selfplay_temperature,
                top_p=cfg.selfplay_top_p,
                max_new_tokens=min(cfg.selfplay_max_new_tokens, 1024),
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            logger.warning(f"OOM on problem {idx}, falling back to sequential generation")
            # Fallback: generate one at a time if batch OOMs
            solutions_text = []
            for _ in range(cfg.selfplay_k):
                sol = generate_solutions_for_problem(
                    model=model, tokenizer=tokenizer, problem=problem,
                    num_candidates=1, temperature=cfg.selfplay_temperature,
                    top_p=cfg.selfplay_top_p, max_new_tokens=min(cfg.selfplay_max_new_tokens, 1024),
                )
                solutions_text.extend(sol)
                torch.cuda.empty_cache()

        # Verify each solution
        verified_solutions = []
        for sol_text in solutions_text:
            verdict = verify_solution(
                problem=problem,
                solution_text=sol_text,
                cfg=cfg,
                code_sandbox=code_sandbox,
            )
            verified_solutions.append({
                "text": verdict.text,
                "correct": verdict.correct,
                "quality_score": verdict.quality_score,
                "quality_tier": verdict.quality_tier,
                "word_count": verdict.word_count,
            })
            stats["total_solutions"] += 1
            if verdict.correct:
                stats["total_correct"] += 1
            else:
                stats["total_incorrect"] += 1

        all_results[idx] = verified_solutions

        # Update progress bar
        correct_rate = stats["total_correct"] / max(stats["total_solutions"], 1)
        pbar.set_postfix({
            "correct": f"{correct_rate:.0%}",
            "sols": stats["total_solutions"],
        })

        # Cache periodically
        if (idx + 1) % 100 == 0:
            _save_cache(cache_dir, all_results, problems, stats)

    # Final save
    _save_cache(cache_dir, all_results, problems, stats)

    elapsed = time.time() - start_time
    logger.info(f"Generation complete in {elapsed:.0f}s")
    logger.info(f"Stats: {stats}")
    logger.info(
        f"Pass rate: {stats['total_correct']}/{stats['total_solutions']} "
        f"({100 * stats['total_correct'] / max(stats['total_solutions'], 1):.1f}%)"
    )

    return str(cache_dir)


def _save_cache(
    cache_dir: Path,
    results: dict,
    problems,
    stats: dict,
):
    """Save results to disk."""
    # Save solutions
    with open(cache_dir / "solutions.json", "w") as f:
        json.dump(
            {str(k): v for k, v in results.items()},
            f,
            indent=2,
            default=str,
        )

    # Save stats
    with open(cache_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Cache saved to {cache_dir} ({len(results)} problems)")


def main():
    parser = argparse.ArgumentParser(
        description="Forge Phase 2A: Self-Play Solution Generation"
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Path to Phase 1 SFT adapter",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = ForgeConfig.from_yaml(args.config)
    run_phase2_generate(cfg, args.adapter_path)


if __name__ == "__main__":
    main()
