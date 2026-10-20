"""Tests for benchmark gating logic."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult, _parse_benchmark_score
from evolution.core.config import EvolutionConfig


@pytest.fixture
def config():
    return EvolutionConfig(run_tblite=True, tblite_regression_threshold=0.02)


@pytest.fixture
def gate(config):
    return BenchmarkGate(config)


class TestCheckRegression:
    def test_no_regression_returns_true(self, gate):
        assert gate.check_regression(0.80, 0.81) is True

    def test_acceptable_regression_returns_true(self, gate):
        # 2% threshold: 0.80 - 0.02 = 0.78 is the floor
        assert gate.check_regression(0.80, 0.78) is True

    def test_unacceptable_regression_returns_false(self, gate):
        assert gate.check_regression(0.80, 0.77) is False

    def test_exact_threshold_boundary(self, gate):
        # Exactly at the threshold is acceptable
        assert gate.check_regression(0.80, 0.78) is True

    def test_custom_threshold(self, gate):
        assert gate.check_regression(0.90, 0.85, threshold=0.05) is True
        assert gate.check_regression(0.90, 0.84, threshold=0.05) is False

    def test_improvement_always_passes(self, gate):
        assert gate.check_regression(0.75, 0.95) is True


class TestRunTbliteFastSkip:
    def test_skips_when_benchmark_not_found(self, gate, tmp_path):
        # tmp_path has no benchmark scripts
        result = gate.run_tblite_fast(tmp_path)

        assert result.skipped is True
        assert result.passed is True
        assert "not found" in result.details.lower()

    def test_skips_when_yc_bench_not_found(self, gate, tmp_path):
        result = gate.run_yc_bench_fast(tmp_path)

        assert result.skipped is True
        assert result.passed is True


class TestParseBenchmarkScore:
    def test_parses_score_colon_format(self):
        assert _parse_benchmark_score("Score: 0.742") == pytest.approx(0.742)

    def test_parses_score_equals_format(self):
        assert _parse_benchmark_score("score=0.5") == pytest.approx(0.5)

    def test_parses_percentage_format(self):
        assert _parse_benchmark_score("Pass rate: 74.2%") == pytest.approx(0.742)

    def test_parses_fraction_format(self):
        assert _parse_benchmark_score("74/100 tasks passed") == pytest.approx(0.74)

    def test_returns_zero_for_no_match(self):
        assert _parse_benchmark_score("No score here") == 0.0

    def test_returns_zero_for_empty_string(self):
        assert _parse_benchmark_score("") == 0.0

    def test_normalizes_percentage_above_one(self):
        # "74.2%" → 0.742, not 74.2
        result = _parse_benchmark_score("Pass rate: 74.2%")
        assert result < 1.0


class TestBenchmarkResult:
    def test_dataclass_fields(self):
        result = BenchmarkResult(
            benchmark_name="tblite-fast",
            score=0.85,
            passed=True,
            details="All good",
        )
        assert result.score == 0.85
        assert result.passed is True
        assert result.skipped is False
