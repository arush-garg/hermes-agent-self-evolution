"""Statistical promotion gate over paired pairwise scores.

Per-example candidate scores come from :mod:`evolution.core.pairwise`
(win=1, tie=0.5, loss=0; judge failures excluded).  A seeded paired bootstrap
(>= 2000 resamples) gives a CI for the mean score.

Decision:
  promote  iff  mean - 0.5 >= min_effect  AND  CI_low > 0.5  AND  n_pairs >= min_pairs
  stage    iff  mean > 0.5 (positive but not significant / too small)
  reject   otherwise

``noise_check`` runs the same statistic on a baseline-vs-baseline comparison
(two independent baseline generations judged against each other) so reports
show how much "improvement" pure noise produces.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

MIN_RESAMPLES = 2000


@dataclass
class GateResult:
    decision: str  # 'promote' | 'stage' | 'reject'
    mean: float
    ci_low: float
    ci_high: float
    p_value: float
    n_pairs: int
    min_effect: float
    reasons: list = field(default_factory=list)
    noise_check: Optional[dict] = None

    @property
    def improvement(self) -> float:
        return self.mean - 0.5

    # backwards-friendly alias
    @property
    def mean_win_rate(self) -> float:
        return self.mean

    def to_dict(self) -> dict:
        d = asdict(self)
        d["improvement"] = self.improvement
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def scores_from(results: Sequence) -> np.ndarray:
    """Accept PairResult-like objects (``.score``) or raw floats; drop None."""
    vals = []
    for r in results:
        s = getattr(r, "score", r)
        if s is None:
            continue
        vals.append(float(s))
    return np.asarray(vals, dtype=float)


def bootstrap(scores: np.ndarray, n_resamples: int = MIN_RESAMPLES, seed: int = 0,
              alpha: float = 0.05) -> tuple[float, float, float, float]:
    """Return (mean, ci_low, ci_high, one-sided p for H0: mean <= 0.5)."""
    n = len(scores)
    if n == 0:
        return 0.5, 0.0, 1.0, 1.0
    n_resamples = max(MIN_RESAMPLES, int(n_resamples))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_resamples, n))
    means = scores[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    # Shift-to-null bootstrap p-value: P(mean* - mean + 0.5 >= observed)
    obs = float(scores.mean())
    null_means = means - obs + 0.5
    p = float((np.sum(null_means >= obs) + 1) / (n_resamples + 1))
    return obs, float(lo), float(hi), p


def decide(results: Sequence, min_effect: float = 0.05, min_pairs: int = 8,
           n_resamples: int = MIN_RESAMPLES, seed: int = 0) -> GateResult:
    scores = scores_from(results)
    mean, lo, hi, p = bootstrap(scores, n_resamples, seed)
    n = len(scores)
    reasons = []
    if mean - 0.5 < min_effect:
        reasons.append(f"effect {mean - 0.5:+.3f} < min_effect {min_effect}")
    if not lo > 0.5:
        reasons.append(f"CI low {lo:.3f} <= 0.5")
    if n < min_pairs:
        reasons.append(f"n_pairs {n} < {min_pairs}")
    if not reasons:
        decision = "promote"
    elif mean > 0.5:
        decision = "stage"
    else:
        decision = "reject"
    return GateResult(decision, mean, lo, hi, p, n, min_effect, reasons)


def noise_check(baseline_vs_baseline: Sequence, n_resamples: int = MIN_RESAMPLES, seed: int = 0) -> dict:
    """Summarize a baseline-vs-baseline comparison (expected mean ~0.5)."""
    s = scores_from(baseline_vs_baseline)
    mean, lo, hi, p = bootstrap(s, n_resamples, seed)
    return {"n": int(len(s)), "mean": mean, "ci_low": lo, "ci_high": hi,
            "abs_effect": abs(mean - 0.5),
            "would_promote": bool(len(s) >= 8 and lo > 0.5 and mean - 0.5 >= 0.05)}
