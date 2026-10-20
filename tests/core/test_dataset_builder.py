"""Tests for evaluation dataset building and loading."""

import json
import pytest
from pathlib import Path

from evolution.core.dataset_builder import EvalExample, EvalDataset


class TestEvalExample:
    def test_to_dict_roundtrip(self):
        ex = EvalExample(
            task_input="Find all Python files",
            expected_behavior="Use search_files tool",
            difficulty="easy",
            category="tool_selection",
            source="synthetic",
        )
        d = ex.to_dict()
        restored = EvalExample.from_dict(d)

        assert restored.task_input == ex.task_input
        assert restored.expected_behavior == ex.expected_behavior
        assert restored.difficulty == ex.difficulty
        assert restored.category == ex.category
        assert restored.source == ex.source

    def test_from_dict_ignores_unknown_fields(self):
        d = {
            "task_input": "hello",
            "expected_behavior": "world",
            "difficulty": "hard",
            "category": "general",
            "source": "golden",
            "unknown_field": "ignored",
        }
        ex = EvalExample.from_dict(d)
        assert ex.task_input == "hello"

    def test_default_values(self):
        ex = EvalExample(task_input="x", expected_behavior="y")
        assert ex.difficulty == "medium"
        assert ex.category == "general"
        assert ex.source == "synthetic"


class TestEvalDataset:
    def test_all_examples_combines_splits(self):
        train = [EvalExample("a", "b")]
        val = [EvalExample("c", "d")]
        holdout = [EvalExample("e", "f")]
        dataset = EvalDataset(train=train, val=val, holdout=holdout)

        assert len(dataset.all_examples) == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        dataset = EvalDataset(
            train=[EvalExample("t1", "b1", source="synthetic")],
            val=[EvalExample("v1", "b2", difficulty="hard")],
            holdout=[EvalExample("h1", "b3", category="cat")],
        )
        dataset.save(tmp_path)

        loaded = EvalDataset.load(tmp_path)
        assert len(loaded.train) == 1
        assert len(loaded.val) == 1
        assert len(loaded.holdout) == 1
        assert loaded.train[0].task_input == "t1"
        assert loaded.val[0].difficulty == "hard"
        assert loaded.holdout[0].category == "cat"

    def test_save_creates_jsonl_files(self, tmp_path):
        dataset = EvalDataset(
            train=[EvalExample("t", "b")],
            val=[],
            holdout=[],
        )
        dataset.save(tmp_path)

        assert (tmp_path / "train.jsonl").exists()
        assert (tmp_path / "val.jsonl").exists()
        assert (tmp_path / "holdout.jsonl").exists()

    def test_load_missing_splits_are_empty(self, tmp_path):
        # Only write train
        with open(tmp_path / "train.jsonl", "w") as f:
            f.write(json.dumps({"task_input": "x", "expected_behavior": "y",
                                "difficulty": "easy", "category": "g", "source": "s"}) + "\n")

        dataset = EvalDataset.load(tmp_path)
        assert len(dataset.train) == 1
        assert len(dataset.val) == 0
        assert len(dataset.holdout) == 0

    def test_to_dspy_examples_train(self):
        import dspy
        dataset = EvalDataset(
            train=[
                EvalExample("task1", "behavior1"),
                EvalExample("task2", "behavior2"),
            ],
        )
        dspy_examples = dataset.to_dspy_examples("train")

        assert len(dspy_examples) == 2
        assert dspy_examples[0].task_input == "task1"
        assert dspy_examples[0].expected_behavior == "behavior1"

    def test_empty_dataset_save_load(self, tmp_path):
        dataset = EvalDataset()
        dataset.save(tmp_path)
        loaded = EvalDataset.load(tmp_path)
        assert len(loaded.all_examples) == 0
