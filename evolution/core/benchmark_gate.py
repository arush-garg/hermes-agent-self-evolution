"""Benchmark gating for evolved artifacts.

Runs TBLite (fast subset) and YC-Bench against the hermes-agent repo and
checks whether an evolved variant regresses below the baseline threshold.

Benchmarks are GATES, not fitness functions. The fitness function is
task-specific (did the skill/tool/prompt do its job better?). Benchmarks
ensure the improvement didn't break something else.
"""

import subprocess
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolution.core.config import EvolutionConfig


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    benchmark_name: str
    score: float        # 0.0–1.0 pass rate
    passed: bool        # True if no regression vs baseline
    details: str        # Human-readable summary
    skipped: bool = False  # True if benchmark not available


class BenchmarkGate:
    """Run benchmarks and check for regressions against a baseline score.

    Benchmarks are run as subprocesses in the hermes-agent repo. If the
    benchmark scripts don't exist (e.g., hermes-agent is not installed),
    the gate gracefully skips rather than blocking evolution.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def run_tblite_fast(
        self,
        hermes_repo: Path,
        baseline_score: Optional[float] = None,
    ) -> BenchmarkResult:
        """Run TBLite fast subset (~20 tasks, ~20 min) as a quick regression check.

        Args:
            hermes_repo: Path to the hermes-agent repository.
            baseline_score: If provided, check this score against the threshold.

        Returns:
            BenchmarkResult with score and pass/fail status.
        """
        benchmark_script = hermes_repo / "environments" / "benchmarks" / "tblite" / "run.py"

        if not benchmark_script.exists():
            return BenchmarkResult(
                benchmark_name="tblite-fast",
                score=0.0,
                passed=True,  # Skip is not a failure
                details="TBLite benchmark not found — skipping gate",
                skipped=True,
            )

        try:
            result = subprocess.run(
                ["python", "-m", "environments.benchmarks.tblite.run", "--fast"],
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
                cwd=str(hermes_repo),
            )

            score = _parse_benchmark_score(result.stdout + result.stderr)

            passed = True
            if baseline_score is not None:
                passed = self.check_regression(
                    baseline=baseline_score,
                    evolved=score,
                    threshold=self.config.tblite_regression_threshold,
                )

            status = "PASS" if passed else "FAIL"
            details = f"TBLite fast: {score:.3f} [{status}]"
            if baseline_score is not None:
                details += f" (baseline: {baseline_score:.3f}, delta: {score - baseline_score:+.3f})"

            return BenchmarkResult(
                benchmark_name="tblite-fast",
                score=score,
                passed=passed,
                details=details,
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                benchmark_name="tblite-fast",
                score=0.0,
                passed=False,
                details="TBLite timed out (3600s)",
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark_name="tblite-fast",
                score=0.0,
                passed=False,
                details=f"TBLite failed to run: {e}",
            )

    def run_yc_bench_fast(
        self,
        hermes_repo: Path,
        baseline_score: Optional[float] = None,
    ) -> BenchmarkResult:
        """Run YC-Bench fast_test preset as a coherence check.

        Args:
            hermes_repo: Path to the hermes-agent repository.
            baseline_score: If provided, check this score against the threshold.

        Returns:
            BenchmarkResult with score and pass/fail status.
        """
        benchmark_script = hermes_repo / "environments" / "benchmarks" / "yc_bench" / "run.py"

        if not benchmark_script.exists():
            return BenchmarkResult(
                benchmark_name="yc-bench-fast",
                score=0.0,
                passed=True,
                details="YC-Bench not found — skipping gate",
                skipped=True,
            )

        try:
            result = subprocess.run(
                ["python", "-m", "environments.benchmarks.yc_bench.run", "--preset", "fast_test"],
                capture_output=True,
                text=True,
                timeout=7200,
                cwd=str(hermes_repo),
            )

            score = _parse_benchmark_score(result.stdout + result.stderr)

            passed = True
            if baseline_score is not None:
                passed = self.check_regression(
                    baseline=baseline_score,
                    evolved=score,
                    threshold=self.config.tblite_regression_threshold,
                )

            status = "PASS" if passed else "FAIL"
            details = f"YC-Bench fast: {score:.3f} [{status}]"
            if baseline_score is not None:
                details += f" (baseline: {baseline_score:.3f}, delta: {score - baseline_score:+.3f})"

            return BenchmarkResult(
                benchmark_name="yc-bench-fast",
                score=score,
                passed=passed,
                details=details,
            )

        except subprocess.TimeoutExpired:
            return BenchmarkResult(
                benchmark_name="yc-bench-fast",
                score=0.0,
                passed=False,
                details="YC-Bench timed out (7200s)",
            )
        except Exception as e:
            return BenchmarkResult(
                benchmark_name="yc-bench-fast",
                score=0.0,
                passed=False,
                details=f"YC-Bench failed to run: {e}",
            )

    def check_regression(
        self,
        baseline: float,
        evolved: float,
        threshold: float = 0.02,
    ) -> bool:
        """Return True if the evolved score is within the acceptable regression threshold.

        Args:
            baseline: The baseline benchmark score (0.0–1.0).
            evolved: The evolved variant's benchmark score.
            threshold: Max allowed regression (default 2%).

        Returns:
            True if evolved >= baseline - threshold (no unacceptable regression).
        """
        return evolved >= (baseline - threshold)


def _parse_benchmark_score(output: str) -> float:
    """Parse a benchmark pass rate from stdout/stderr.

    Looks for patterns like:
      "Score: 0.742"
      "Pass rate: 74.2%"
      "74/100 tasks passed"
      "score=0.742"
    """
    # Pattern: "Score: 0.742" or "score=0.742" or "score: 0.742"
    for pattern in [
        r'[Ss]core[:\s=]+([0-9]+(?:\.[0-9]+)?)',
        r'[Pp]ass\s+rate[:\s=]+([0-9]+(?:\.[0-9]+)?)%',
        r'([0-9]+)/([0-9]+)\s+tasks?\s+passed',
    ]:
        match = re.search(pattern, output)
        if match:
            if len(match.groups()) == 2:
                # "74/100 tasks passed" form
                passed, total = int(match.group(1)), int(match.group(2))
                return passed / total if total > 0 else 0.0
            value = float(match.group(1))
            # Percentages > 1.0 need normalizing
            return value / 100.0 if value > 1.0 else value

    return 0.0
