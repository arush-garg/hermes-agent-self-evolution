"""Evaluation dataset generation for hermes-agent-self-evolution.

Sources:
A) Synthetic generation — LLM reads a skill/tool/prompt and generates test cases
B) SessionDB mining — extract real usage patterns and score with LLM-as-judge
C) Golden sets — hand-curated JSONL files
"""

import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import dspy

from evolution.core.config import EvolutionConfig
from evolution.core.llm_api import get_lm_for_role


@dataclass
class EvalExample:
    """A single evaluation example."""
    task_input: str  # What the user asks
    expected_behavior: str  # Rubric — what a good response looks like
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # Category for stratified eval
    source: str = "synthetic"  # synthetic, sessiondb, golden

    def to_dict(self) -> dict:
        return {
            "task_input": self.task_input,
            "expected_behavior": self.expected_behavior,
            "difficulty": self.difficulty,
            "category": self.category,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalExample":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvalDataset:
    """Train/val/holdout split of evaluation examples."""
    train: list[EvalExample] = field(default_factory=list)
    val: list[EvalExample] = field(default_factory=list)
    holdout: list[EvalExample] = field(default_factory=list)

    @property
    def all_examples(self) -> list[EvalExample]:
        return self.train + self.val + self.holdout

    def save(self, path: Path):
        """Save dataset splits to JSONL files."""
        path.mkdir(parents=True, exist_ok=True)
        for split_name, split_data in [("train", self.train), ("val", self.val), ("holdout", self.holdout)]:
            with open(path / f"{split_name}.jsonl", "w") as f:
                for ex in split_data:
                    f.write(json.dumps(ex.to_dict()) + "\n")

    @classmethod
    def load(cls, path: Path) -> "EvalDataset":
        """Load dataset splits from JSONL files."""
        dataset = cls()
        for split_name in ["train", "val", "holdout"]:
            split_file = path / f"{split_name}.jsonl"
            if split_file.exists():
                examples = []
                with open(split_file) as f:
                    for line in f:
                        if line.strip():
                            examples.append(EvalExample.from_dict(json.loads(line)))
                setattr(dataset, split_name, examples)
        return dataset

    def to_dspy_examples(self, split: str = "train") -> list[dspy.Example]:
        """Convert a split to DSPy Example objects."""
        data = getattr(self, split)
        return [
            dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
            ).with_inputs("task_input")
            for ex in data
        ]


# ── Fallback case generation ───────────────────────────────────────────────

def _fallback_cases(artifact_text: str, n: int = 5) -> list[dict]:
    """Generate minimal test cases from artifact text using heuristics.

    Used when LLM output is completely unparseable. Extracts key phrases
    from the text and wraps them in simple (task_input, expected_behavior)
    pairs so that optimisation can still proceed (even if with low-quality
    evaluation).
    """
    import sys

    # Strip frontmatter / markdown fences
    text = artifact_text
    if text.startswith("---"):
        _, _, text = text.partition("---")
        _, _, text = text.partition("---")
    text = "\n".join(
        line for line in text.splitlines()
        if not line.startswith("```")
    )

    # Pick topic sentences (first ~20 non-empty lines)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    topics = lines[: min(20, len(lines))]

    # Build pairs from every other sentence
    cases: list[dict] = []
    difficulties = ["easy", "medium", "hard"]
    categories = ["general", "edge_case", "integration"]

    for i in range(0, min(n * 2, len(topics)), 2):
        topic_a = topics[i] if i < len(topics) else ""
        topic_b = topics[i + 1] if i + 1 < len(topics) else "follow instructions"
        if not topic_a:
            continue
        cases.append({
            "task_input": f"Apply {topic_a[:120]}",
            "expected_behavior": topic_b[:200] if topic_b else f"Respond correctly about {topic_a[:80]}",
            "difficulty": difficulties[i % len(difficulties)],
            "category": categories[i % len(categories)],
        })
        if len(cases) >= n:
            break

    if not cases:
        # absolute last resort
        cases.append({
            "task_input": "Perform the described task",
            "expected_behavior": "Follow the instructions correctly and completely",
            "difficulty": "medium",
            "category": "general",
        })

    print(
        f"  [dataset_fallback] Generated {len(cases)} heuristic cases from artifact text.",
        file=sys.stderr,
    )
    return cases


class SyntheticDatasetBuilder:
    """Generate evaluation datasets using a strong LLM.

    Reads the target artifact (skill file, tool description, etc.)
    and generates realistic (task_input, expected_behavior) pairs.
    """

    class GenerateTestCases(dspy.Signature):
        """Generate realistic evaluation test cases for an agent skill or tool.

        Given the full text of a skill/tool description, generate diverse test cases
        that would exercise different aspects of the skill. Each test case should include:
        - A realistic task_input (what a user would actually ask)
        - An expected_behavior rubric (what a good response should contain/do, NOT exact text)
        - A difficulty level (easy, medium, hard)
        - A category (what aspect of the skill this tests)
        """
        artifact_text: str = dspy.InputField(desc="The full text of the skill/tool/prompt being tested")
        artifact_type: str = dspy.InputField(desc="Type: 'skill', 'tool_description', or 'prompt_section'")
        num_cases: int = dspy.InputField(desc="Number of test cases to generate")
        test_cases: str = dspy.OutputField(desc="JSON array of test cases, each with: task_input, expected_behavior, difficulty, category")

    def __init__(self, config: EvolutionConfig):
        self.config = config
        self.generator = dspy.ChainOfThought(self.GenerateTestCases)

    def generate(
        self,
        artifact_text: str,
        artifact_type: str = "skill",
        num_cases: Optional[int] = None,
    ) -> EvalDataset:
        """Generate a full eval dataset with train/val/holdout splits."""

        n = num_cases or self.config.eval_dataset_size

        lm = get_lm_for_role("judge", override=self.config.judge_model)

        with dspy.context(lm=lm):
            result = self.generator(
                artifact_text=artifact_text,
                artifact_type=artifact_type,
                num_cases=n,
            )

        # Parse the generated test cases
        # The LLM often wraps the JSON array in markdown fences or adds surrounding text.
        # We attempt a series of increasingly tolerant parses:
        # 1. Direct JSON load – works when the model returns clean JSON.
        # 2. Strip common markdown fences (e.g. ```json ... ```) and retry.
        # 3. Fallback to a balanced‑bracket extractor that finds the first top‑level JSON
        #    array without being greedy (handles nested brackets correctly).
        def _extract_balanced_array(text: str) -> Optional[str]:
            """Return the first balanced ``[ ... ]`` substring or ``None``.

            This scans the string character‑by‑character, counting brackets, and stops
            when the opening bracket that started at position ``i`` is closed. It
            therefore avoids the greedy ``r'\\[.*\\]'`` pattern that would swallow the
            last ``]`` in the entire response.
            """

            start = text.find('[')
            if start == -1:
                return None
            depth = 0
            for idx in range(start, len(text)):
                if text[idx] == '[':
                    depth += 1
                elif text[idx] == ']':
                    depth -= 1
                    if depth == 0:
                        return text[start : idx + 1]
            return None

        try:
            cases_raw = json.loads(result.test_cases)
        except json.JSONDecodeError:
            # Strip markdown code fences if present.
            import re

            # Typical pattern: ```json\n[ ... ]\n``` (case‑insensitive ``json`` label optional)
            fence_match = re.search(r'```\s*json?\s*(\[.*?\])\s*```', result.test_cases, re.DOTALL | re.IGNORECASE)
            if fence_match:
                try:
                    cases_raw = json.loads(fence_match.group(1))
                except json.JSONDecodeError:
                    # Fallback to balanced‑bracket extraction on the captured group.
                    raw = _extract_balanced_array(fence_match.group(1))
                    if raw:
                        cases_raw = json.loads(raw)
                    else:
                        raise
            else:
                # Try balanced bracket extraction on the whole output.
                raw = _extract_balanced_array(result.test_cases)
                if raw:
                    cases_raw = json.loads(raw)
                else:
                    # Last resort: fallback to minimal heuristic-based cases
                    cases_raw = _fallback_cases(artifact_text, n)
                    if cases_raw:
                        print(f"  [warn] LLM output unparseable, using {len(cases_raw)} heuristic fallback cases.", file=sys.stderr)
                    else:
                        preview = result.test_cases[:300]
                        raise ValueError(f"Could not parse test cases from LLM output: {preview}")

        examples = [
            EvalExample(
                task_input=c.get("task_input", ""),
                expected_behavior=c.get("expected_behavior", ""),
                difficulty=c.get("difficulty", "medium"),
                category=c.get("category", "general"),
                source="synthetic",
            )
            for c in cases_raw
            if c.get("task_input") and c.get("expected_behavior")
        ]

        # Shuffle and split
        random.shuffle(examples)
        n_total = len(examples)
        n_train = max(1, int(n_total * self.config.train_ratio))
        n_val = max(1, int(n_total * self.config.val_ratio))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )


class GoldenDatasetLoader:
    """Load hand-curated evaluation datasets from JSONL files."""

    @staticmethod
    def load(path: Path) -> EvalDataset:
        """Load a golden dataset. If no splits exist, auto-split the single file."""
        if (path / "train.jsonl").exists():
            return EvalDataset.load(path)

        # Single file — auto-split
        golden_file = path if path.suffix == ".jsonl" else path / "golden.jsonl"
        if not golden_file.exists():
            raise FileNotFoundError(f"No golden dataset found at {golden_file}")

        examples = []
        with open(golden_file) as f:
            for line in f:
                if line.strip():
                    examples.append(EvalExample.from_dict(json.loads(line)))

        random.shuffle(examples)
        n = len(examples)
        n_train = max(1, int(n * 0.5))
        n_val = max(1, int(n * 0.25))

        return EvalDataset(
            train=examples[:n_train],
            val=examples[n_train:n_train + n_val],
            holdout=examples[n_train + n_val:],
        )
