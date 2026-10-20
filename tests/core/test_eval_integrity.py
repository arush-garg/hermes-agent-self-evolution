"""Tests for evaluation-integrity guards: correction-aware rubric, provenance gate, split leakage."""

from types import SimpleNamespace

from evolution.core import stats_gate
from evolution.core.pairwise import PairwiseJudge, rubric_for
from evolution.skills.evolve_skill import _resplit

WRONG = "WRONG_ANSWER: run git push --force on main"


class TestRubric:
    def test_corrected_drops_superseded_answer(self):
        ex = {"outcome": "corrected", "source": "sessiondb", "session_id": "s1",
              "expected_behavior": "Correction takes priority: no, never force push to main\n" + WRONG}
        r = rubric_for(ex)
        assert "never force push" in r and "WRONG_ANSWER" not in r
        assert "CORRECTED" in r

    def test_failed_drops_answer(self):
        r = rubric_for({"outcome": "failed", "expected_behavior": WRONG, "session_id": "s"})
        assert "WRONG_ANSWER" not in r and "FAILED" in r

    def test_real_success_is_hint_only(self):
        r = rubric_for({"outcome": "success", "source": "sessiondb", "session_id": "s",
                        "expected_behavior": "use gh pr create"})
        assert "use gh pr create" in r and "do NOT reward verbatim" in r

    def test_synthetic_passthrough(self):
        assert rubric_for({"expected_behavior": "rubric", "source": "synthetic"}) == "rubric"

    def test_judge_prompt_never_contains_superseded_answer(self):
        seen = []

        def chat(p):
            seen.append(p)
            return '{"winner": "tie"}'
        ex = {"task_input": "push my branch", "outcome": "corrected", "session_id": "s",
              "expected_behavior": "Correction takes priority: use a feature branch\n" + WRONG}
        PairwiseJudge(chat).evaluate([ex], ["a"], ["b"])
        assert seen and all("WRONG_ANSWER" not in p for p in seen)


class TestProvenanceGuard:
    def _gate(self):
        return stats_gate.decide([1.0] * 10)

    def test_synthetic_holdout_cannot_promote(self):
        hold = [{"source": "sessiondb"}] * 9 + [{"source": "synthetic"}]
        g = stats_gate.apply_provenance_guard(self._gate(), hold)
        assert g.decision == "stage" and any("synthetic" in r for r in g.reasons)

    def test_all_real_holdout_can_promote(self):
        hold = [{"source": "sessiondb"}] * 5 + [SimpleNamespace(source="golden")] * 5
        assert stats_gate.apply_provenance_guard(self._gate(), hold).decision == "promote"

    def test_reject_untouched(self):
        g = stats_gate.apply_provenance_guard(stats_gate.decide([0.0] * 10), [{"source": "synthetic"}])
        assert g.decision == "reject"


class TestResplit:
    def test_real_rows_not_moved_and_no_leakage(self):
        train = [{"task_input": f"r{i}", "session_id": f"s{i}", "source": "sessiondb"} for i in range(6)]
        train += [{"task_input": f"syn{i}", "source": "synthetic"} for i in range(6)]
        train.append({"task_input": "dup", "session_id": "h1", "source": "sessiondb"})  # same session as holdout
        val = [{"task_input": "v", "session_id": "s99", "source": "sessiondb"},
               {"task_input": "held task", "source": "synthetic"}]  # same task text as holdout
        hold = [{"task_input": "held task", "session_id": "h1", "source": "sessiondb"}]
        ds = SimpleNamespace(train=train, val=val, holdout=hold)
        tr, va, ho = _resplit(ds, n_val=6, n_holdout_min=5, seed=0)
        moved = [e for e in ho if e is not hold[0]]
        assert moved and all(not e.get("session_id") for e in moved)  # only synthetic moved
        ho_sids = {e.get("session_id") for e in ho if e.get("session_id")}
        ho_tasks = {e["task_input"] for e in ho}
        for e in tr + va:
            assert e.get("session_id") not in ho_sids
            assert e["task_input"] not in ho_tasks
