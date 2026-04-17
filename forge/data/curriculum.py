"""
Curriculum-Aware Difficulty Scheduler for Forge.

Classifies problems by difficulty based on the base model's pass rate
(pass@K), then controls the difficulty mix at each training phase.

Phase 1 (SFT):     40% easy, 40% medium, 20% hard — build foundation
Phase 2 (Self-Play): 20% easy, 50% medium, 30% hard — maximize signal
Phase 3 (Refine):   5% easy, 35% medium, 60% hard — push boundaries
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import ForgeConfig

logger = logging.getLogger(__name__)


@dataclass
class DifficultyProfile:
    """Difficulty classification for a problem."""
    problem_index: int
    pass_rate: float        # Fraction of K solutions that were correct
    difficulty: str         # "easy", "medium", or "hard"
    num_correct: int
    num_total: int


def classify_difficulty(
    pass_rate: float,
    easy_threshold: float = 0.7,
    hard_threshold: float = 0.3,
) -> str:
    """Classify a problem's difficulty based on pass rate."""
    if pass_rate >= easy_threshold:
        return "easy"
    elif pass_rate <= hard_threshold:
        return "hard"
    else:
        return "medium"


def classify_problems(
    pass_rates: list[tuple[int, float, int, int]],
    cfg: ForgeConfig,
) -> list[DifficultyProfile]:
    """
    Classify a list of problems by difficulty.

    Args:
        pass_rates: List of (problem_index, pass_rate, num_correct, num_total).
        cfg: Forge configuration.

    Returns:
        List of DifficultyProfile objects.
    """
    profiles = []
    for idx, rate, correct, total in pass_rates:
        profiles.append(DifficultyProfile(
            problem_index=idx,
            pass_rate=rate,
            difficulty=classify_difficulty(rate, cfg.easy_threshold, cfg.hard_threshold),
            num_correct=correct,
            num_total=total,
        ))

    # Log distribution
    counts = {"easy": 0, "medium": 0, "hard": 0}
    for p in profiles:
        counts[p.difficulty] += 1
    logger.info(f"Difficulty distribution: {counts}")

    return profiles


def sample_by_curriculum(
    profiles: list[DifficultyProfile],
    phase: str,
    target_count: int,
    cfg: ForgeConfig,
) -> list[int]:
    """
    Sample problem indices according to the curriculum mix for a given phase.

    Args:
        profiles: Difficulty profiles for all problems.
        phase: Training phase ("phase1", "phase2", "phase3").
        target_count: How many problems to sample.
        cfg: Forge configuration.

    Returns:
        List of selected problem indices.
    """
    # Get the mix for this phase
    curriculum = cfg._raw.get("curriculum", {})
    mix_key = f"{phase}_mix"
    mix = curriculum.get(mix_key, {"easy": 0.33, "medium": 0.34, "hard": 0.33})

    # Group problems by difficulty
    by_difficulty: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    for profile in profiles:
        by_difficulty[profile.difficulty].append(profile.problem_index)

    # Shuffle each group
    rng = random.Random(cfg.seed)
    for group in by_difficulty.values():
        rng.shuffle(group)

    # Sample according to mix
    selected: list[int] = []
    for difficulty in ["easy", "medium", "hard"]:
        fraction = mix.get(difficulty, 0.33)
        n_want = int(target_count * fraction)
        available = by_difficulty[difficulty]
        n_take = min(n_want, len(available))
        selected.extend(available[:n_take])

    # If we're short, fill from medium (most informative)
    deficit = target_count - len(selected)
    if deficit > 0:
        remaining = [
            p.problem_index for p in profiles
            if p.problem_index not in set(selected)
        ]
        rng.shuffle(remaining)
        selected.extend(remaining[:deficit])

    rng.shuffle(selected)
    logger.info(
        f"Sampled {len(selected)} problems for {phase} "
        f"(target: {target_count}, mix: {mix})"
    )

    return selected[:target_count]
