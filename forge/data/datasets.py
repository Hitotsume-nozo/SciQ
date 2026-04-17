"""
Dataset Loading and Formatting for Forge.

Handles:
  - Loading datasets from HuggingFace Hub
  - Formatting into structured chat format with <think> reasoning tags
  - Creating SFT-ready datasets
  - Creating problem pools for self-play generation
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from datasets import Dataset, load_dataset, concatenate_datasets

if TYPE_CHECKING:
    from ..config import ForgeConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt for all Forge models
# ---------------------------------------------------------------------------

FORGE_SYSTEM_PROMPT = (
    "You are a precise problem solver. For every problem, think step-by-step "
    "inside <think> tags before providing your solution. Show your reasoning "
    "clearly, then give your final answer.\n\n"
    "For math problems, put your final answer in \\boxed{}.\n"
    "For code problems, provide a complete, working solution in a code block."
)


# ---------------------------------------------------------------------------
# Dataset-specific formatters
# ---------------------------------------------------------------------------

def _format_gsm8k(example: dict) -> dict:
    """Format a GSM8K example into chat format with reasoning."""
    question = example["question"]
    answer_text = example["answer"]

    # Extract the numeric answer (after "####")
    if "####" in answer_text:
        reasoning, final_answer = answer_text.rsplit("####", 1)
        reasoning = reasoning.strip()
        final_answer = final_answer.strip()
    else:
        reasoning = answer_text
        final_answer = ""

    # Build the assistant response with <think> tags
    assistant_response = f"<think>\n{reasoning}\n</think>\n\nThe answer is \\boxed{{{final_answer}}}."

    return {
        "messages": [
            {"role": "system", "content": FORGE_SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_response},
        ],
        "domain": "math",
        "ground_truth": final_answer,
        "difficulty": "easy",
    }


def _format_math(example: dict) -> dict:
    """Format a MATH dataset example into chat format."""
    problem = example.get("problem", "")
    solution = example.get("solution", "")
    level = example.get("level", "Level 1")

    # Extract boxed answer from solution
    import re
    boxed_match = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", solution)
    ground_truth = boxed_match.group(1) if boxed_match else ""

    # Map level to difficulty
    level_num = int(re.search(r"\d+", level).group()) if re.search(r"\d+", level) else 1
    if level_num <= 2:
        difficulty = "easy"
    elif level_num <= 3:
        difficulty = "medium"
    else:
        difficulty = "hard"

    assistant_response = f"<think>\n{solution}\n</think>\n\nThe answer is \\boxed{{{ground_truth}}}."

    return {
        "messages": [
            {"role": "system", "content": FORGE_SYSTEM_PROMPT},
            {"role": "user", "content": problem},
            {"role": "assistant", "content": assistant_response},
        ],
        "domain": "math",
        "ground_truth": ground_truth,
        "difficulty": difficulty,
    }


def _format_code_contests(example: dict) -> dict:
    """Format a Code Contests example into chat format."""
    description = example.get("description", "")

    # Get Python solutions if available
    solutions = example.get("solutions", {})
    python_solutions = []
    if isinstance(solutions, dict):
        langs = solutions.get("language", [])
        codes = solutions.get("solution", [])
        for lang, code in zip(langs, codes):
            # Language 3 = Python3 in the Code Contests dataset
            if lang == 3:
                python_solutions.append(code)

    if not python_solutions:
        return None  # Skip if no Python solution

    solution_code = python_solutions[0]

    # Get test cases
    public_tests = example.get("public_tests", {})
    test_inputs = public_tests.get("input", [])
    test_outputs = public_tests.get("output", [])

    # Build test code
    test_code = ""
    if test_inputs and test_outputs:
        test_code = f"# Test input: {test_inputs[0][:100]}\n# Expected output: {test_outputs[0][:100]}"

    assistant_response = (
        f"<think>\nLet me analyze this problem and develop a solution.\n\n"
        f"The problem asks us to process the input and produce the correct output.\n"
        f"</think>\n\n```python\n{solution_code}\n```"
    )

    return {
        "messages": [
            {"role": "system", "content": FORGE_SYSTEM_PROMPT},
            {"role": "user", "content": description[:4000]},  # Truncate long descriptions
            {"role": "assistant", "content": assistant_response},
        ],
        "domain": "code",
        "ground_truth": "",
        "test_code": test_code,
        "difficulty": "hard",
    }


def _format_mbpp(example: dict) -> dict:
    """Format an MBPP example into chat format."""
    prompt = example.get("text", example.get("prompt", ""))
    code = example.get("code", "")
    test_list = example.get("test_list", [])

    test_code = "\n".join(test_list) if test_list else ""

    assistant_response = (
        f"<think>\nLet me break down what's needed:\n"
        f"- {prompt}\n\n"
        f"I'll implement this step by step.\n</think>\n\n"
        f"```python\n{code}\n```"
    )

    return {
        "messages": [
            {"role": "system", "content": FORGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": assistant_response},
        ],
        "domain": "code",
        "ground_truth": "",
        "test_code": test_code,
        "difficulty": "easy",
    }


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

FORMATTERS = {
    "gsm8k": _format_gsm8k,
    "math": _format_math,
    "code_contests": _format_code_contests,
    "mbpp": _format_mbpp,
}


def load_sft_dataset(cfg: ForgeConfig) -> Dataset:
    """
    Load and format all training datasets for Phase 1 SFT.

    Combines GSM8K, MATH, Code Contests, and MBPP into a single
    shuffled dataset formatted for chat-based SFT training.
    """
    all_examples = []
    data_sources = cfg._raw.get("data", {}).get("sources", {})

    for source_name, source_cfg in data_sources.items():
        formatter = FORMATTERS.get(source_name)
        if formatter is None:
            logger.warning(f"No formatter for dataset: {source_name}, skipping")
            continue

        logger.info(f"Loading dataset: {source_name} ({source_cfg.get('path', '')})")
        try:
            ds = load_dataset(
                source_cfg["path"],
                name=source_cfg.get("subset"),
                split=source_cfg.get("split", "train"),
            )
        except Exception as e:
            logger.error(f"Failed to load {source_name}: {e}")
            continue

        # Format each example
        count = 0
        for example in ds:
            formatted = formatter(example)
            if formatted is not None:
                all_examples.append(formatted)
                count += 1
        logger.info(f"  → {count} examples from {source_name}")

    # Shuffle and cap at max_sft_examples
    random.seed(cfg.seed)
    random.shuffle(all_examples)

    if len(all_examples) > cfg.max_sft_examples:
        all_examples = all_examples[:cfg.max_sft_examples]
        logger.info(f"Capped to {cfg.max_sft_examples} examples")

    logger.info(f"Total SFT dataset size: {len(all_examples)}")

    # Convert to HuggingFace Dataset
    return Dataset.from_list(all_examples)


def load_problem_pool(
    cfg: ForgeConfig,
    max_problems: int | None = None,
    difficulty_filter: str | None = None,
) -> Dataset:
    """
    Load a pool of problems for self-play generation (no solutions).

    Each problem has:
      - "prompt": the problem statement
      - "domain": "math" or "code"
      - "ground_truth": the expected answer (for verification)
      - "test_code": test cases (for code problems)
      - "difficulty": "easy", "medium", or "hard"

    Args:
        cfg: Forge configuration.
        max_problems: Maximum number of problems to return.
        difficulty_filter: If set, only return problems of this difficulty.
    """
    problems = []
    data_sources = cfg._raw.get("data", {}).get("sources", {})

    for source_name, source_cfg in data_sources.items():
        logger.info(f"Loading problems from: {source_name}")
        try:
            ds = load_dataset(
                source_cfg["path"],
                name=source_cfg.get("subset"),
                split=source_cfg.get("split", "train"),
            )
        except Exception as e:
            logger.error(f"Failed to load {source_name}: {e}")
            continue

        domain = source_cfg.get("domain", "math")

        for example in ds:
            problem = _extract_problem(example, source_name, domain)
            if problem is not None:
                if difficulty_filter and problem.get("difficulty") != difficulty_filter:
                    continue
                problems.append(problem)

    random.seed(cfg.seed)
    random.shuffle(problems)

    max_n = max_problems or cfg.max_selfplay_problems
    if len(problems) > max_n:
        problems = problems[:max_n]

    logger.info(f"Problem pool size: {len(problems)}")
    return Dataset.from_list(problems)


def _extract_problem(example: dict, source_name: str, domain: str) -> dict | None:
    """Extract a problem (without solution) from a dataset example."""
    import re

    if source_name == "gsm8k":
        answer_text = example.get("answer", "")
        ground_truth = ""
        if "####" in answer_text:
            ground_truth = answer_text.rsplit("####", 1)[-1].strip()
        return {
            "prompt": example["question"],
            "domain": "math",
            "ground_truth": ground_truth,
            "test_code": "",
            "difficulty": "easy",
            "source": "gsm8k",
        }

    elif source_name == "math":
        solution = example.get("solution", "")
        boxed = re.search(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}", solution)
        ground_truth = boxed.group(1) if boxed else ""
        level = example.get("level", "Level 1")
        level_num = int(re.search(r"\d+", level).group()) if re.search(r"\d+", level) else 1
        difficulty = "easy" if level_num <= 2 else ("medium" if level_num <= 3 else "hard")
        return {
            "prompt": example.get("problem", ""),
            "domain": "math",
            "ground_truth": ground_truth,
            "test_code": "",
            "difficulty": difficulty,
            "source": "math",
        }

    elif source_name == "mbpp":
        test_list = example.get("test_list", [])
        return {
            "prompt": example.get("text", example.get("prompt", "")),
            "domain": "code",
            "ground_truth": "",
            "test_code": "\n".join(test_list),
            "difficulty": "easy",
            "source": "mbpp",
        }

    elif source_name == "code_contests":
        description = example.get("description", "")
        if not description or len(description) < 20:
            return None
        public_tests = example.get("public_tests", {})
        test_inputs = public_tests.get("input", [])
        test_outputs = public_tests.get("output", [])
        test_code = ""
        for inp, out in zip(test_inputs[:3], test_outputs[:3]):
            test_code += f"# Input: {inp[:200]}\n# Expected: {out[:200]}\n"
        return {
            "prompt": description[:4000],
            "domain": "code",
            "ground_truth": "",
            "test_code": test_code,
            "difficulty": "hard",
            "source": "code_contests",
        }

    return None
