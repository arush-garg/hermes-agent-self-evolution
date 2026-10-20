"""Core infrastructure shared across all evolution phases."""

from evolution.core.config import EvolutionConfig, get_hermes_agent_path
from evolution.core.benchmark_gate import BenchmarkGate, BenchmarkResult
from evolution.core.pr_builder import PRBuilder, PRMetrics
from evolution.core.regression_gate import check_regression, RegressionError
from evolution.core.llm_api import (
    get_lm,
    get_lm_for_role,
    configure_default_lm,
    call,
    resolve_model,
    PROVIDER_DEFAULTS,
)
