"""
Process-Level Reasoning Verifier for Forge.

Goes beyond binary correct/incorrect by analyzing the quality of
individual reasoning steps in a solution. Used to build richer
preference pairs for DPO training.

Capabilities:
  - Step segmentation: split reasoning into logical steps
  - Coherence checking: detect contradictions and logical leaps
  - Quality scoring: assign step-level quality metrics
  - Tier classification: rank solutions by reasoning quality
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    index: int
    text: str
    step_type: str = "reasoning"  # reasoning, computation, conclusion, setup
    has_equation: bool = False
    has_code: bool = False
    word_count: int = 0


@dataclass
class ProcessVerdict:
    """
    Process-level analysis of a solution's reasoning chain.
    """
    steps: list[ReasoningStep] = field(default_factory=list)
    num_steps: int = 0
    total_word_count: int = 0
    has_structured_reasoning: bool = False
    reasoning_depth: str = "none"   # none, shallow, moderate, deep
    quality_score: float = 0.0      # 0.0 to 1.0
    tier: str = "D"                 # A (best), B, C, D (worst)
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Step segmentation
# ---------------------------------------------------------------------------

def segment_reasoning(text: str) -> list[ReasoningStep]:
    """
    Split solution text into discrete reasoning steps.

    Handles multiple formats:
      - Think tags: <think>...</think>
      - Numbered steps: "Step 1:", "1."
      - Bullet points: "- ", "* "
      - Paragraph breaks
    """
    steps: list[ReasoningStep] = []

    # Extract reasoning section (if using <think> tags)
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    reasoning_text = think_match.group(1) if think_match else text

    # Try numbered / bulleted segmentation first
    numbered_pattern = r"(?:^|\n)(?:(?:step\s*\d+|#{1,3}\s+|\d+[\.\)]\s*|\-\s+|\*\s+))(.*?)(?=\n(?:step\s*\d+|#{1,3}\s+|\d+[\.\)]\s*|\-\s+|\*\s+)|$)"
    numbered_matches = re.findall(numbered_pattern, reasoning_text, re.IGNORECASE | re.DOTALL)

    if len(numbered_matches) >= 2:
        # Use numbered/structured segmentation
        raw_steps = numbered_matches
    else:
        # Fallback: split by double newlines (paragraphs)
        raw_steps = [s.strip() for s in re.split(r"\n\n+", reasoning_text) if s.strip()]

    for idx, step_text in enumerate(raw_steps):
        step_text = step_text.strip()
        if not step_text or len(step_text) < 5:
            continue

        # Classify step type
        step_type = _classify_step(step_text)

        steps.append(ReasoningStep(
            index=idx,
            text=step_text,
            step_type=step_type,
            has_equation=bool(re.search(r"[=<>≤≥±]", step_text)),
            has_code=bool(re.search(r"```|def |import |return |print\(", step_text)),
            word_count=len(step_text.split()),
        ))

    return steps


def _classify_step(text: str) -> str:
    """Classify a reasoning step by its type."""
    text_lower = text.lower()

    if any(kw in text_lower for kw in ["therefore", "thus", "answer is", "conclude", "result"]):
        return "conclusion"
    elif any(kw in text_lower for kw in ["let ", "given", "assume", "define", "set up"]):
        return "setup"
    elif re.search(r"\d+\s*[+\-*/=]\s*\d+", text):
        return "computation"
    else:
        return "reasoning"


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------

def analyze_reasoning(text: str, is_correct: bool) -> ProcessVerdict:
    """
    Perform process-level analysis of a solution's reasoning.

    Considers:
      - Whether structured reasoning is present
      - Number and depth of reasoning steps
      - Presence of setup, computation, and conclusion stages
      - Coherence indicators
      - Correctness of the final answer

    Args:
        text: The full solution text.
        is_correct: Whether the final answer is correct.

    Returns:
        ProcessVerdict with quality metrics and tier classification.
    """
    steps = segment_reasoning(text)
    verdict = ProcessVerdict(steps=steps, num_steps=len(steps))

    if not steps:
        verdict.reasoning_depth = "none"
        verdict.quality_score = 0.1 if is_correct else 0.0
        verdict.tier = "D"
        verdict.issues.append("No identifiable reasoning steps")
        return verdict

    # --- Metrics ---
    verdict.total_word_count = sum(s.word_count for s in steps)
    step_types = {s.step_type for s in steps}
    verdict.has_structured_reasoning = (
        "setup" in step_types and "conclusion" in step_types
    )

    # Reasoning depth
    if len(steps) >= 5:
        verdict.reasoning_depth = "deep"
    elif len(steps) >= 3:
        verdict.reasoning_depth = "moderate"
    elif len(steps) >= 1:
        verdict.reasoning_depth = "shallow"
    else:
        verdict.reasoning_depth = "none"

    # Quality score (0-1)
    score = 0.0
    issues = []

    # Correctness is the foundation
    if is_correct:
        score += 0.5
    else:
        issues.append("Final answer is incorrect")

    # Structured reasoning bonus
    if verdict.has_structured_reasoning:
        score += 0.15
    elif len(steps) >= 2:
        score += 0.05
        issues.append("Missing structured setup/conclusion")

    # Depth bonus (but penalize excessive verbosity)
    if verdict.reasoning_depth == "deep":
        score += 0.1
        if verdict.total_word_count > 800:
            score -= 0.05
            issues.append("Overly verbose reasoning")
    elif verdict.reasoning_depth == "moderate":
        score += 0.1
    elif verdict.reasoning_depth == "shallow":
        score += 0.05

    # Computation steps present (shows work)
    if "computation" in step_types:
        score += 0.1

    # Has equations (mathematical rigor)
    if any(s.has_equation for s in steps):
        score += 0.05

    # Consistency check: correct answer but no reasoning = suspicious
    if is_correct and verdict.reasoning_depth == "none":
        score -= 0.1
        issues.append("Correct answer without reasoning (potential memorization)")

    # Incorrect answer with good reasoning = dangerous hallucination
    if not is_correct and verdict.reasoning_depth in ("moderate", "deep"):
        issues.append("Plausible reasoning leading to wrong answer (dangerous)")

    verdict.quality_score = max(0.0, min(1.0, score))
    verdict.issues = issues

    # --- Tier classification ---
    if is_correct and verdict.quality_score >= 0.7:
        verdict.tier = "A"  # Correct + good reasoning
    elif is_correct and verdict.quality_score >= 0.5:
        verdict.tier = "B"  # Correct + some reasoning
    elif is_correct:
        verdict.tier = "C"  # Correct but weak reasoning
    elif verdict.quality_score >= 0.3:
        verdict.tier = "C"  # Wrong but with effort
    else:
        verdict.tier = "D"  # Wrong with no/bad reasoning

    return verdict
