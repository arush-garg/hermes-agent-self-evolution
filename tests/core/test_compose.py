"""Tests for Propose → Probe → Compose with mocked LLMs."""

import json

from evolution.core import compose
from evolution.core.pairwise import CallBudget, PairwiseJudge

BASE = "# Skill\n" + "Step one. Step two. Verify output.\n" * 10


def _examples(n, prefix):
    return [{"task_input": f"{prefix} task {i}", "expected_behavior": "do it", "task_class": "git"}
            for i in range(n)]


def make_env(rank, budget=None, merge_ok=True):
    """rank: dict marker->quality. Agent echoes the marker found in its system prompt."""
    budget = budget or CallBudget(max_calls=10_000)
    proposals = []

    def optimizer(prompt):
        budget.spend("optimizer")
        if prompt.startswith("Two improved versions"):
            return "<document>\n" + BASE + "MARK_COMPOSITE\n</document>" if merge_ok else "nope"
        name = next(n for n, i in compose.STRATEGIES if i in prompt) if any(
            i in prompt for _, i in compose.STRATEGIES) else "x"
        proposals.append((name, prompt))
        return f"<document>\n{BASE}MARK_{name.upper()}\n</document>\nCHANGES: x"

    def agent(user, system=None):
        budget.spend("eval")
        for m in rank:
            if f"MARK_{m}" in system:
                return f"answer {m} for {user}"
        return f"answer BASE for {user}"

    def judge_chat(prompt):
        budget.spend("judge")
        a = prompt.split("=== Response A ===")[1].split("=== End A ===")[0]
        b = prompt.split("=== Response B ===")[1].split("=== End B ===")[0]

        def q(t):
            return next((rank[m] for m in rank if f"answer {m} " in t), 0)
        if q(a) == q(b):
            return '{"winner": "tie"}'
        return '{"winner": "%s"}' % ("A" if q(a) > q(b) else "B")

    return budget, optimizer, agent, PairwiseJudge(judge_chat, repeats=2, seed=0), proposals


def run(rank, k=3, holdout_n=10, **kw):
    budget, opt, agent, judge, props = make_env(rank, kw.pop("budget", None), kw.pop("merge_ok", True))
    res = compose.propose_probe_compose(
        baseline_body=BASE, train=_examples(4, "tr"), val=_examples(4, "val"),
        holdout=_examples(holdout_n, "ho"), agent_chat=agent, optimizer_chat=opt, judge=judge,
        budget=budget, k=k, log=lambda s: None, **kw)
    return res, props, budget


def test_composite_wins_and_is_promoted():
    res, props, _ = run({"GENERAL": 1, "DOMAIN": 1, "COMPOSITE": 2, "CONCISE": -1})
    assert [c.name for c in res.candidates] == ["general", "domain", "concise"]
    assert res.composite is not None and res.composite.probe_score == 1.0
    assert res.winner is res.composite
    assert res.gate.decision == "promote" and res.gate.n_pairs == 10


def test_baseline_kept_when_nothing_beats_it():
    res, _, budget = run({"GENERAL": -1, "DOMAIN": -1, "CONCISE": 0})
    assert res.winner is None and res.gate.decision == "reject"
    assert res.composite is None
    assert res.holdout_results == []  # no holdout spending on losers
    assert res.final_text == BASE


def test_top_single_beats_composite():
    res, _, _ = run({"GENERAL": 3, "DOMAIN": 1, "COMPOSITE": -1})
    assert res.winner.name == "general"


def test_lessons_strategy_only_with_seed_material(monkeypatch, tmp_path):
    res, props, _ = run({"GENERAL": 1}, k=3)
    assert "lessons" not in [c.name for c in res.candidates]
    ctx = tmp_path / "fc.json"
    ctx.write_text(json.dumps([{"pattern": "forgot to push branch" * 200}]))
    monkeypatch.setenv("EVOLUTION_FAILURE_CONTEXT", str(ctx))
    fc = compose.load_failure_context()
    assert 0 < len(fc) <= 2000
    lessons = [{"cue": "git push", "lesson": "always set upstream", "evidence": "x"}]
    res, props, _ = run({"GENERAL": 1}, k=3, lessons=lessons, failure_context=fc)
    assert [c.name for c in res.candidates] == ["general", "domain", "lessons"]
    lp = dict(props)["lessons"]
    assert "always set upstream" in lp and "forgot to push branch" in lp


def test_growth_and_validation_reject():
    res, _, _ = run({"GENERAL": 1}, max_growth=0.0, validate=lambda t: "nope")
    assert all(c.error for c in res.candidates)
    assert res.gate.decision == "reject"


def test_budget_cap_is_respected():
    budget = CallBudget(max_calls=30)
    res, _, b = run({"GENERAL": 1, "DOMAIN": 1}, budget=budget)
    assert b.used <= 30
    assert any("cap" in n for n in res.notes)
    assert res.gate.decision in ("reject", "stage")


def test_noise_check_runs():
    budget, opt, agent, judge, _ = make_env({"GENERAL": 2, "DOMAIN": 1})
    res = compose.propose_probe_compose(
        baseline_body=BASE, train=_examples(4, "tr"), val=_examples(4, "val"),
        holdout=_examples(10, "ho"), agent_chat=agent, optimizer_chat=opt, judge=judge,
        budget=budget, k=2, noise_agent_chat=agent, log=lambda s: None)
    assert res.noise is not None and res.noise["n"] == 8 and res.noise["mean"] == 0.5


def test_extract_document():
    assert compose.extract_document("x <document>\nhello\n</document> y") == "hello"
    assert compose.extract_document("<think><document>no</document></think>") is None
    assert compose.extract_document("```markdown\n" + "a" * 60 + "\n```") == "a" * 60
    assert compose.extract_document("") is None


def test_lessons_import_tolerant(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_LESSONS_DIR", str(tmp_path))
    assert compose.load_lessons(["git"]) == []
    row = compose.record_lesson("git", "github-workflow", "set upstream", "mean=0.8")
    assert row is not None
    assert compose.load_lessons(["git"])[0]["lesson"] == "set upstream"
