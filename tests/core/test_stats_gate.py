"""Tests for the paired-bootstrap promotion gate."""

from types import SimpleNamespace

import numpy as np

from evolution.core import stats_gate


def test_strong_win_promotes():
    g = stats_gate.decide([1.0] * 10 + [0.5] * 2, min_effect=0.05)
    assert g.decision == "promote" and g.ci_low > 0.5 and g.n_pairs == 12
    assert g.p_value < 0.05


def test_too_few_pairs_stages():
    g = stats_gate.decide([1.0] * 7)
    assert g.decision == "stage" and any("n_pairs" in r for r in g.reasons)


def test_positive_not_significant_stages():
    g = stats_gate.decide([1, 0, 1, 0, 1, 0.5, 1, 0, 0.5, 1])
    assert g.mean > 0.5 and g.decision == "stage"


def test_effect_below_min_effect_stages():
    scores = [0.5] * 40 + [1.0] * 4  # mean 0.545, tight CI
    g = stats_gate.decide(scores, min_effect=0.05)
    assert g.decision == "stage" and any("min_effect" in r for r in g.reasons)


def test_loss_or_parity_rejects():
    assert stats_gate.decide([0.0] * 10).decision == "reject"
    assert stats_gate.decide([0.5] * 10).decision == "reject"
    assert stats_gate.decide([]).decision == "reject"


def test_none_scores_excluded_and_objects_accepted():
    rs = [SimpleNamespace(score=1.0)] * 9 + [SimpleNamespace(score=None)] * 5
    g = stats_gate.decide(rs)
    assert g.n_pairs == 9 and g.decision == "promote"


def test_seeded_and_min_resamples():
    s = np.array([1, 0.5, 0, 1, 1, 0.5, 1, 0])
    assert stats_gate.bootstrap(s, seed=7) == stats_gate.bootstrap(s, seed=7)
    # n_resamples below 2000 is raised to 2000 → identical to explicit 2000
    assert stats_gate.bootstrap(s, n_resamples=10, seed=7) == stats_gate.bootstrap(s, n_resamples=2000, seed=7)


def test_to_dict_has_required_keys():
    d = stats_gate.decide([1.0] * 8).to_dict()
    for k in ("decision", "improvement", "ci_low", "ci_high", "p_value", "n_pairs"):
        assert k in d


def test_noise_check():
    n = stats_gate.noise_check([0.5, 1, 0, 0.5, 1, 0, 0.5, 0.5])
    assert n["n"] == 8 and abs(n["mean"] - 0.5) < 1e-9 and not n["would_promote"]
