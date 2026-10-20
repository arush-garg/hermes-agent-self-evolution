"""Tests for fitness functions."""

import pytest
import dspy

from evolution.core.fitness import skill_fitness_metric, _parse_score, FitnessScore


class TestParseScore:
    def test_float_passthrough(self):
        assert _parse_score(0.75) == 0.75

    def test_int_converts_to_float(self):
        assert _parse_score(1) == 1.0

    def test_string_float(self):
        assert _parse_score("0.5") == 0.5

    def test_clamps_above_one(self):
        assert _parse_score(1.5) == 1.0

    def test_clamps_below_zero(self):
        assert _parse_score(-0.1) == 0.0

    def test_invalid_string_returns_neutral(self):
        assert _parse_score("invalid") == 0.5

    def test_none_returns_neutral(self):
        assert _parse_score(None) == 0.5

    def test_string_with_whitespace(self):
        assert _parse_score("  0.8  ") == 0.8


class TestFitnessScore:
    def test_composite_weighted_sum(self):
        score = FitnessScore(
            correctness=1.0,
            procedure_following=1.0,
            conciseness=1.0,
        )
        assert score.composite == pytest.approx(1.0)

    def test_composite_with_length_penalty(self):
        score = FitnessScore(
            correctness=1.0,
            procedure_following=1.0,
            conciseness=1.0,
            length_penalty=0.3,
        )
        assert score.composite == pytest.approx(0.7)

    def test_composite_never_negative(self):
        score = FitnessScore(
            correctness=0.0,
            procedure_following=0.0,
            conciseness=0.0,
            length_penalty=1.0,
        )
        assert score.composite == 0.0


class TestSkillFitnessMetric:
    def _make_example(self, task: str, expected: str) -> dspy.Example:
        return dspy.Example(
            task_input=task,
            expected_behavior=expected,
        ).with_inputs("task_input")

    def _make_prediction(self, output: str) -> dspy.Prediction:
        return dspy.Prediction(output=output)

    def test_empty_output_returns_zero(self):
        ex = self._make_example("find files", "use search_files")
        pred = self._make_prediction("")
        assert skill_fitness_metric(ex, pred) == 0.0

    def test_whitespace_only_returns_zero(self):
        ex = self._make_example("find files", "use search_files")
        pred = self._make_prediction("   ")
        assert skill_fitness_metric(ex, pred) == 0.0

    def test_exact_match_keywords_scores_high(self):
        ex = self._make_example(
            "search for files",
            "should search files using search_files tool",
        )
        pred = self._make_prediction(
            "I will search files using the search_files tool for you."
        )
        score = skill_fitness_metric(ex, pred)
        assert score > 0.5

    def test_completely_unrelated_output_scores_low(self):
        ex = self._make_example(
            "find python files",
            "search for files with .py extension",
        )
        pred = self._make_prediction("The weather is nice today.")
        score = skill_fitness_metric(ex, pred)
        assert score < 0.5

    def test_score_bounded_zero_to_one(self):
        ex = self._make_example("task", "expected")
        pred = self._make_prediction("some output " * 100)
        score = skill_fitness_metric(ex, pred)
        assert 0.0 <= score <= 1.0

    def test_missing_prediction_field_returns_zero(self):
        ex = self._make_example("task", "expected")
        pred = dspy.Prediction()  # No 'output' field
        assert skill_fitness_metric(ex, pred) == 0.0
