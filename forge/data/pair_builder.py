"""
Preference Pair Builder for Forge.

Constructs DPO-ready (chosen, rejected) preference pairs from
solutions that have been verified by the code sandbox and math verifier.

Key design decisions:
  - Primary signal: correct > incorrect (highest confidence pairs)
  - Secondary signal: concise correct > verbose correct (efficiency reward)
  - Tertiary signal: good reasoning + correct > no reasoning + correct
  - Process-level quality tiers weight the DPO loss
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..verify.process_verifier import analyze_reasoning

if TYPE_CHECKING:
    from ..config import ForgeConfig

logger = logging.getLogger(__name__)


@dataclass
class PreferencePair:
    """A single (chosen, rejected) preference pair for DPO training."""
    prompt: str
    chosen: str
    rejected: str
    margin: float = 1.0       # Confidence in this pair (1.0 = very confident)
    pair_type: str = "correct_vs_incorrect"  # Type of signal
    domain: str = "math"
    metadata: dict = field(default_factory=dict)


@dataclass
class SolutionWithVerdict:
    """A generated solution paired with its verification result."""
    text: str
    correct: bool
    quality_score: float = 0.0
    quality_tier: str = "D"
    word_count: int = 0


def build_preference_pairs(
    problem: dict,
    solutions: list[SolutionWithVerdict],
    max_pairs_per_problem: int = 4,
    conciseness_bonus: bool = True,
    seed: int = 42,
) -> list[PreferencePair]:
    """
    Build DPO preference pairs from a set of verified solutions.

    Generates pairs using multiple signals:
      1. Correct vs. Incorrect (primary, margin=1.0)
      2. High-quality correct vs. Low-quality correct (secondary, margin=0.5)
      3. Concise correct vs. Verbose correct (tertiary, margin=0.3)

    Args:
        problem: The problem dict with "prompt", "domain", etc.
        solutions: List of solutions with verification verdicts.
        max_pairs_per_problem: Maximum pairs to generate per problem.
        conciseness_bonus: Whether to generate conciseness pairs.
        seed: Random seed for pair selection.

    Returns:
        List of PreferencePair objects.
    """
    rng = random.Random(seed)

    correct_solutions = [s for s in solutions if s.correct]
    incorrect_solutions = [s for s in solutions if not s.correct]

    pairs: list[PreferencePair] = []
    prompt = problem.get("prompt", "")
    domain = problem.get("domain", "math")

    # -----------------------------------------------------------------------
    # Signal 1: Correct vs. Incorrect (strongest signal)
    # -----------------------------------------------------------------------
    if correct_solutions and incorrect_solutions:
        # Pick the best correct solution (highest quality score)
        best_correct = max(correct_solutions, key=lambda s: s.quality_score)

        # Pair with each incorrect solution (up to limit)
        shuffled_incorrect = incorrect_solutions[:]
        rng.shuffle(shuffled_incorrect)

        for inc_sol in shuffled_incorrect[:max_pairs_per_problem]:
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=best_correct.text,
                rejected=inc_sol.text,
                margin=1.0,
                pair_type="correct_vs_incorrect",
                domain=domain,
                metadata={
                    "chosen_tier": best_correct.quality_tier,
                    "rejected_tier": inc_sol.quality_tier,
                },
            ))

    # -----------------------------------------------------------------------
    # Signal 2: High-quality reasoning vs. Low-quality reasoning
    # (both correct, but different reasoning quality)
    # -----------------------------------------------------------------------
    if len(correct_solutions) >= 2:
        sorted_by_quality = sorted(
            correct_solutions,
            key=lambda s: s.quality_score,
            reverse=True,
        )
        best = sorted_by_quality[0]
        worst = sorted_by_quality[-1]

        # Only create this pair if there's a meaningful quality difference
        if best.quality_score - worst.quality_score >= 0.15:
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=best.text,
                rejected=worst.text,
                margin=0.5,
                pair_type="quality_reasoning",
                domain=domain,
                metadata={
                    "chosen_tier": best.quality_tier,
                    "rejected_tier": worst.quality_tier,
                    "quality_gap": best.quality_score - worst.quality_score,
                },
            ))

    # -----------------------------------------------------------------------
    # Signal 3: Concise correct vs. Verbose correct
    # (both correct, prefer efficiency)
    # -----------------------------------------------------------------------
    if conciseness_bonus and len(correct_solutions) >= 2:
        sorted_by_length = sorted(
            correct_solutions,
            key=lambda s: s.word_count,
        )
        shortest = sorted_by_length[0]
        longest = sorted_by_length[-1]

        # Only if there's a meaningful length difference (>50% longer)
        if longest.word_count > shortest.word_count * 1.5 and shortest.word_count > 10:
            pairs.append(PreferencePair(
                prompt=prompt,
                chosen=shortest.text,
                rejected=longest.text,
                margin=0.3,
                pair_type="conciseness",
                domain=domain,
                metadata={
                    "chosen_words": shortest.word_count,
                    "rejected_words": longest.word_count,
                },
            ))

    # Cap total pairs
    if len(pairs) > max_pairs_per_problem:
        # Prioritize by margin (higher margin = more informative)
        pairs.sort(key=lambda p: p.margin, reverse=True)
        pairs = pairs[:max_pairs_per_problem]

    return pairs


def build_all_preference_pairs(
    problems: list[dict],
    all_solutions: dict[int, list[SolutionWithVerdict]],
    cfg: ForgeConfig,
) -> list[PreferencePair]:
    """
    Build preference pairs for all problems in the self-play pool.

    Args:
        problems: List of problem dicts.
        all_solutions: Dict mapping problem index → list of verified solutions.
        cfg: Forge configuration.

    Returns:
        Flat list of all PreferencePair objects, ready for DPO.
    """
    pair_cfg = cfg._raw.get("selfplay", {}).get("pair_building", {})
    max_pairs = pair_cfg.get("max_pairs_per_problem", 4)
    conciseness = pair_cfg.get("conciseness_bonus", True)
    min_correct = pair_cfg.get("min_correct", 1)
    min_incorrect = pair_cfg.get("min_incorrect", 1)

    all_pairs: list[PreferencePair] = []
    stats = {"total_problems": 0, "yielded_pairs": 0, "skipped_all_correct": 0,
             "skipped_all_incorrect": 0, "skipped_no_solutions": 0}

    for idx, problem in enumerate(problems):
        stats["total_problems"] += 1
        solutions = all_solutions.get(idx, [])

        if not solutions:
            stats["skipped_no_solutions"] += 1
            continue

        correct = sum(1 for s in solutions if s.correct)
        incorrect = len(solutions) - correct

        if correct < min_correct and incorrect < min_incorrect:
            stats["skipped_no_solutions"] += 1
            continue
        if correct >= len(solutions):
            stats["skipped_all_correct"] += 1
            # Still useful for conciseness pairs
        if incorrect >= len(solutions):
            stats["skipped_all_incorrect"] += 1
            continue

        pairs = build_preference_pairs(
            problem=problem,
            solutions=solutions,
            max_pairs_per_problem=max_pairs,
            conciseness_bonus=conciseness,
            seed=cfg.seed + idx,
        )
        all_pairs.extend(pairs)
        if pairs:
            stats["yielded_pairs"] += 1

    logger.info(f"Preference pair building stats: {stats}")
    logger.info(f"Total pairs generated: {len(all_pairs)}")

    # Log pair type distribution
    type_counts = {}
    for p in all_pairs:
        type_counts[p.pair_type] = type_counts.get(p.pair_type, 0) + 1
    for ptype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {ptype}: {count} pairs")

    return all_pairs


def pairs_to_dpo_dataset(pairs: list[PreferencePair]) -> dict[str, list]:
    """
    Convert PreferencePair objects into a format suitable for
    TRL's DPOTrainer.

    Returns a dict with keys: "prompt", "chosen", "rejected".
    """
    return {
        "prompt": [p.prompt for p in pairs],
        "chosen": [p.chosen for p in pairs],
        "rejected": [p.rejected for p in pairs],
    }
