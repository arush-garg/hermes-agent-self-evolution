"""Tests for grader-free pairwise judging (mocked LLM)."""

import pytest

from evolution.core.pairwise import (BudgetExceeded, CallBudget, PairwiseJudge, calibrate_agreement,
                                     parse_verdict, summarize)


class TestParseVerdict:
    @pytest.mark.parametrize("text,exp", [
        ('{"winner": "A", "reason": "x"}', "A"),
        ('{"winner": "b"}', "B"),
        ('{"winner": "tie"}', "TIE"),
        ('```json\n{"winner": "B", "reason": "better"}\n```', "B"),
        ('<think>maybe {"winner": "A"}</think>\n{"winner": "B"}', "B"),
        ('Some prose first.\n{"verdict": "Response A"}', "A"),
        ('winner: tie', "TIE"),
        ('```choice\nA\n```', "A"),
        ('After thinking, the answer is\nB', "B"),
        ('', None),
        ('I cannot decide', None),
        ('{"winner": "C"}', None),
    ])
    def test_parse(self, text, exp):
        assert parse_verdict(text) == exp

    def test_truncated_json_falls_back_to_kv(self):
        assert parse_verdict('{"winner": "A", "reason": "long reason cut off') == "A"


def _judge_preferring(content_marker, log=None):
    """Mock judge that always prefers the response containing marker."""
    def chat(prompt):
        if log is not None:
            log.append(prompt)
        a = prompt.split("=== Response A ===")[1].split("=== End A ===")[0]
        return '{"winner": "%s"}' % ("A" if content_marker in a else "B")
    return chat


class TestPairwiseJudge:
    def test_consistent_preference_wins(self):
        j = PairwiseJudge(_judge_preferring("GOOD"), repeats=2, seed=1)
        r = j.compare("1", "task", "", "bad answer", "GOOD answer")
        assert r.score == 1.0 and not r.position_inconsistent
        r = j.compare("2", "task", "", "GOOD answer", "bad answer")
        assert r.score == 0.0

    def test_both_orders_run_and_anonymized(self):
        log = []
        j = PairwiseJudge(_judge_preferring("GOOD", log), repeats=2, seed=3)
        j.compare("1", "task", "", "baseline text", "GOOD candidate")
        assert len(log) == 2
        slots = []
        for p in log:
            a = p.split("=== Response A ===")[1].split("=== End A ===")[0]
            slots.append("cand" if "GOOD" in a else "base")
            assert "baseline" not in p.lower().split("=== response a ===")[0]  # no labels leak
            assert "candidate" not in p.split("TASK:")[0].lower()
        assert sorted(slots) == ["base", "cand"]

    def test_position_bias_becomes_tie(self):
        j = PairwiseJudge(lambda p: '{"winner": "A"}', repeats=2, seed=0)
        r = j.compare("1", "t", "", "x", "y")
        assert r.position_inconsistent and r.score == 0.5

    def test_order_randomized_with_single_repeat(self):
        first_slots = set()
        for seed in range(20):
            log = []
            j = PairwiseJudge(_judge_preferring("CAND", log), repeats=1, seed=seed)
            j.compare("1", "t", "", "base", "CAND")
            a = log[0].split("=== Response A ===")[1].split("=== End A ===")[0]
            first_slots.add("CAND" in a)
        assert first_slots == {True, False}

    def test_retry_once_then_exclude(self):
        calls = []

        def chat(p):
            calls.append(p)
            return "no idea"
        j = PairwiseJudge(chat, repeats=2, seed=0)
        r = j.compare("1", "t", "", "x", "y")
        assert r.score is None and len(calls) == 4 and r.judge_failures == 2
        st = summarize([r])
        assert st.judge_failures == 1 and st.n_examples == 0

    def test_retry_recovers(self):
        replies = iter(["garbage", '{"winner": "tie"}', '{"winner": "tie"}'])
        j = PairwiseJudge(lambda p: next(replies), repeats=2, seed=0)
        assert j.compare("1", "t", "", "x", "y").score == 0.5

    def test_identical_outputs_skip_judge(self):
        def boom(p):
            raise AssertionError("should not be called")
        assert PairwiseJudge(boom).compare("1", "t", "", "same", "same ").score == 0.5

    def test_prefilter_tie_short_circuits(self):
        def boom(p):
            raise AssertionError("should not be called")
        j = PairwiseJudge(boom, prefilter=lambda t, a, b: "TIE")
        assert j.compare("1", "t", "", "x", "y").score == 0.5

    def test_evaluate_stops_on_budget(self):
        budget = CallBudget(max_calls=3)

        def chat(p):
            budget.spend("judge")
            return '{"winner": "tie"}'
        j = PairwiseJudge(chat, repeats=2)
        exs = [{"task_input": f"t{i}"} for i in range(5)]
        res = j.evaluate(exs, ["a"] * 5, ["b"] * 5)
        assert len(res) == 1  # second example's 2nd call exceeds cap

    def test_tolerates_extra_dataset_fields(self):
        j = PairwiseJudge(_judge_preferring("GOOD"))
        ex = {"task_input": "t", "expected_behavior": "e", "task_class": "git", "session_id": "s",
              "tool_trace_summary": "ran ls", "outcome": "failed", "failure_signals": ["x"]}
        assert j.evaluate([ex], ["bad"], ["GOOD"])[0].score == 1.0


def test_budget():
    b = CallBudget(max_calls=2)
    b.spend("judge")
    b.spend("eval")
    with pytest.raises(BudgetExceeded):
        b.spend("judge")
    assert b.summary() == {"used": 2, "max": 2, "by_role": {"judge": 1, "eval": 1}}


def test_calibrate_agreement():
    pairs = [("t", "a", "b")] * 12
    judge = ["A"] * 9 + ["B"] * 3
    res = calibrate_agreement(pairs, judge, cheap=lambda t, a, b: "A")
    assert res["n"] == 12 and res["agreement"] == 0.75 and res["enable"]
    res = calibrate_agreement(pairs[:10], judge[:10], cheap=lambda t, a, b: "A")
    assert not res["enable"]  # too few pairs
