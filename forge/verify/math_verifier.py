"""
Math Answer Verification for Forge.

Multi-strategy extraction and comparison of mathematical answers:
  1. LaTeX boxed answers (\\boxed{...})
  2. "Answer: X" patterns
  3. Terminal numeric values
  4. Symbolic comparison via SymPy (handles equivalent expressions)

Designed for robustness: the model may format answers in many ways,
so we try multiple extraction strategies and multiple comparison methods.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MathVerdict:
    """Result of verifying a math solution."""
    correct: bool
    predicted_answer: str | None = None
    expected_answer: str | None = None
    reason: str = ""
    confidence: float = 1.0  # 1.0 = exact match, <1.0 = approximate


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

# Ordered by specificity (most reliable first)
ANSWER_PATTERNS = [
    # LaTeX \boxed{...} — most reliable, explicitly marked
    (r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", "boxed"),
    # "The answer is X" / "Answer: X"
    (r"[Tt]he\s+(?:final\s+)?answer\s+is\s*[:=]?\s*(.+?)(?:\.|$)", "explicit"),
    (r"[Aa]nswer\s*[:=]\s*(.+?)(?:\.|$)", "explicit"),
    # "Therefore, X" or "Thus, X" at end of reasoning
    (r"[Tt]herefore,?\s+.*?(\-?\d+[\d,]*\.?\d*)", "therefore"),
    (r"[Tt]hus,?\s+.*?(\-?\d+[\d,]*\.?\d*)", "therefore"),
    # "= X" at end of a line (likely final computation)
    (r"=\s*(\-?\d+[\d,]*\.?\d*)\s*$", "equation_terminal"),
    # Last standalone number on its own line
    (r"^(\-?\d+[\d,]*\.?\d*)\s*$", "standalone_number"),
]


def extract_answer(text: str) -> tuple[str | None, str]:
    """
    Extract the final mathematical answer from solution text.

    Uses multiple extraction strategies in order of reliability.
    Returns (answer_string, extraction_method) or (None, "none").
    """
    if not text:
        return None, "none"

    for pattern, method in ANSWER_PATTERNS:
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            # Take the last match (most likely to be the final answer)
            answer = matches[-1].strip()
            # Clean up common formatting
            answer = answer.replace(",", "")  # Remove thousands separators
            answer = answer.strip("$").strip()  # Remove dollar signs
            if answer:
                return answer, method

    return None, "none"


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def _try_numeric_compare(predicted: str, expected: str, tol: float = 1e-6) -> bool | None:
    """
    Try to compare as floating-point numbers.
    Returns True/False if both parse as numbers, None if they don't.
    """
    try:
        pred_num = float(predicted.replace(",", "").strip())
        exp_num = float(expected.replace(",", "").strip())

        if exp_num == 0:
            return abs(pred_num) < tol
        return abs(pred_num - exp_num) / max(abs(exp_num), 1e-10) < tol
    except (ValueError, TypeError):
        return None


def _try_fraction_compare(predicted: str, expected: str) -> bool | None:
    """
    Try to compare as fractions (e.g., "3/4" vs "0.75").
    """
    from fractions import Fraction

    try:
        # Handle LaTeX fractions: \frac{a}{b} → a/b
        pred_clean = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", predicted)
        exp_clean = re.sub(r"\\frac\{(\d+)\}\{(\d+)\}", r"\1/\2", expected)

        pred_frac = Fraction(pred_clean).limit_denominator(10000)
        exp_frac = Fraction(exp_clean).limit_denominator(10000)
        return pred_frac == exp_frac
    except (ValueError, ZeroDivisionError):
        return None


def _try_symbolic_compare(predicted: str, expected: str) -> bool | None:
    """
    Try symbolic comparison via SymPy.
    Handles algebraic expressions, radicals, etc.
    """
    try:
        from sympy import simplify, sympify, oo, zoo, nan

        # Clean up LaTeX-isms
        for old, new in [("\\pi", "pi"), ("\\sqrt", "sqrt"), ("\\cdot", "*"),
                         ("\\times", "*"), ("\\div", "/"), ("\\infty", "oo")]:
            predicted = predicted.replace(old, new)
            expected = expected.replace(old, new)

        pred_expr = sympify(predicted)
        exp_expr = sympify(expected)

        # Avoid comparing infinities/NaN
        if pred_expr in (oo, -oo, zoo, nan) or exp_expr in (oo, -oo, zoo, nan):
            return str(pred_expr) == str(exp_expr)

        diff = simplify(pred_expr - exp_expr)
        return diff == 0
    except Exception:
        return None


def compare_answers(
    predicted: str,
    expected: str,
    tolerance: float = 1e-6,
) -> tuple[bool, float]:
    """
    Compare predicted and expected answers using multiple strategies.

    Returns (is_correct, confidence) where confidence indicates
    how certain we are about the comparison.
    """
    if not predicted or not expected:
        return False, 1.0

    # Normalize whitespace
    predicted = predicted.strip()
    expected = expected.strip()

    # Strategy 1: Exact string match (highest confidence)
    if predicted.lower() == expected.lower():
        return True, 1.0

    # Strategy 2: Numeric comparison
    numeric_result = _try_numeric_compare(predicted, expected, tolerance)
    if numeric_result is not None:
        return numeric_result, 1.0

    # Strategy 3: Fraction comparison
    fraction_result = _try_fraction_compare(predicted, expected)
    if fraction_result is not None:
        return fraction_result, 0.95

    # Strategy 4: Symbolic comparison (lowest confidence due to parsing)
    symbolic_result = _try_symbolic_compare(predicted, expected)
    if symbolic_result is not None:
        return symbolic_result, 0.9

    # All strategies failed — answers are different
    return False, 0.8


# ---------------------------------------------------------------------------
# High-level verification function
# ---------------------------------------------------------------------------

def verify_math_solution(
    solution_text: str,
    ground_truth: str,
    tolerance: float = 1e-6,
) -> MathVerdict:
    """
    Verify a math solution against a ground truth answer.

    Extracts the answer from the solution text, then compares
    it against the ground truth using multiple strategies.

    Args:
        solution_text: The full solution text (may include reasoning).
        ground_truth: The expected correct answer.
        tolerance: Numeric comparison tolerance.

    Returns:
        MathVerdict with correctness and diagnostics.
    """
    predicted, method = extract_answer(solution_text)

    if predicted is None:
        return MathVerdict(
            correct=False,
            predicted_answer=None,
            expected_answer=ground_truth,
            reason=f"Could not extract answer from solution",
            confidence=1.0,
        )

    is_correct, confidence = compare_answers(predicted, ground_truth, tolerance)

    return MathVerdict(
        correct=is_correct,
        predicted_answer=predicted,
        expected_answer=ground_truth,
        reason=f"Extracted via '{method}', compared with confidence {confidence:.2f}",
        confidence=confidence,
    )


def verify_math_batch(
    solutions: list[dict],
    tolerance: float = 1e-6,
) -> list[MathVerdict]:
    """
    Verify a batch of math solutions.

    Each solution dict should have:
      - "text": str — the full solution text
      - "ground_truth": str — the expected answer

    Args:
        solutions: List of solution dicts.
        tolerance: Numeric comparison tolerance.

    Returns:
        List of MathVerdict objects.
    """
    results = []
    for sol in solutions:
        verdict = verify_math_solution(
            solution_text=sol.get("text", ""),
            ground_truth=sol.get("ground_truth", ""),
            tolerance=tolerance,
        )
        results.append(verdict)
    return results
