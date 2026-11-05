"""GEPA's evolved instructions (JSON or prose) become sanitised per-tool descriptions, and
evolve() scores and stages exactly those descriptions."""
import json
from pathlib import Path

import dspy
import pytest

from evolution.core.dataset_builder import EvalDataset, EvalExample
from evolution.tools import evolve_tool_descriptions as etd
from evolution.tools import evolved_tools as et
from evolution.tools.tool_module import ToolDescription, format_tools_for_prompt

ALWAYS_VALID = lambda new, orig: True  # noqa: E731

# Shape of what GEPA actually produced on 2026-11-04: prose rules, not the tools JSON.
GEPA_PROSE = """You are a tool router. Rules:
1. File reading – use `read_file` for viewing file contents, never `terminal` with cat.
2. Searching – use `search_files` to locate files or text."""


def _tools():
    return [ToolDescription("read_file", "Read a file", {"path": "Path"}, Path("/x/file_tools.py")),
            ToolDescription("search_files", "Search files", {"pattern": "Pattern"}),
            ToolDescription("terminal", "Run a shell command", {"command": "Cmd"})]


class FakeDistill:
    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def __call__(self, tools_json, guidance):
        self.calls.append(json.loads(tools_json))
        answer = self.answers.pop(0) if self.answers else "[]"
        if isinstance(answer, Exception):
            raise answer
        return dspy.Prediction(revised=answer)


# ── parse_tools_json ────────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    '[{"name": "a", "description": "x"}]',
    'Here you go:\n```json\n[{"name": "a", "description": "x"}]\n```\nDone.',
    'Updated tools: [{"name": "a", "description": "x"}] as requested',
])
def test_parse_tools_json_accepts_plain_fenced_and_wrapped(text):
    assert et.parse_tools_json(text) == {"a": "x"}


@pytest.mark.parametrize("text", [GEPA_PROSE, "", '{"name": "a"}', "[1, 2]", "[]", None])
def test_parse_tools_json_rejects_prose_and_non_tool_json(text):
    assert et.parse_tools_json(text) is None


# ── sanitize ────────────────────────────────────────────────────────────────

def test_sanitize_freezes_structure_and_reverts_bad_text():
    base = _tools()
    out = et.sanitize(base, {"read_file": "  Read a text file  ", "search_files": "   ",
                             "terminal": "TOO LONG", "rm_rf": "new tool"},
                      is_valid=lambda new, orig: new != "TOO LONG")
    assert [t.name for t in out] == ["read_file", "search_files", "terminal"]  # no new tools
    assert out[0].description == "Read a text file"
    assert out[0].parameters == {"path": "Path"} and out[0].source_file == Path("/x/file_tools.py")
    assert out[1].description == "Search files" and out[2].description == "Run a shell command"
    assert base[0].description == "Read a file"  # baseline not mutated


# ── evolved_tools_from_instructions ─────────────────────────────────────────

def test_unchanged_instructions_cost_no_lm_call():
    base, fake = _tools(), FakeDistill([])
    tools, how = et.evolved_tools_from_instructions(base, format_tools_for_prompt(base), ALWAYS_VALID, fake)
    assert how == "unchanged" and tools == base and fake.calls == []


def test_json_instructions_are_taken_directly():
    base, fake = _tools(), FakeDistill([])
    text = json.dumps([{"name": "terminal", "description": "Run shell; prefer read_file for files",
                        "parameters": {"command": "HACKED"}}])
    tools, how = et.evolved_tools_from_instructions(base, text, ALWAYS_VALID, fake)
    assert how == "json" and fake.calls == []
    assert tools[2].description == "Run shell; prefer read_file for files"
    assert tools[2].parameters == {"command": "Cmd"}  # parameters never evolve
    assert et.changed_names(base, tools) == ["terminal"]


def test_prose_instructions_are_distilled_into_descriptions():
    """Regression: prose from GEPA used to be dropped ('No description changes detected')."""
    base = _tools()
    fake = FakeDistill(['[{"name": "read_file", "description": "Read a file. Use instead of cat in terminal."}]'])
    tools, how = et.evolved_tools_from_instructions(base, GEPA_PROSE, ALWAYS_VALID, fake)
    assert how == "distilled"
    assert et.changed_names(base, tools) == ["read_file"]
    assert tools[0].description == "Read a file. Use instead of cat in terminal."


def test_distill_batches_and_survives_bad_answers(monkeypatch):
    monkeypatch.setattr(et, "DISTILL_BATCH", 2)
    base = [ToolDescription(f"t{i}", f"d{i}", {}) for i in range(5)]
    fake = FakeDistill([RuntimeError("lm down"), "not json at all",
                        '[{"name": "t4", "description": "new4"}, {"name": "t0", "description": "sneaky"}]'])
    revised = et.distill_descriptions(base, GEPA_PROSE, fake)
    assert [len(c) for c in fake.calls] == [2, 2, 1]
    assert revised == {"t4": "new4"}  # t0 was not in that batch, so it is ignored


# ── evolve() end to end, LMs and GEPA faked ─────────────────────────────────

class _FakeLM:
    pass


def _dataset(tmp_path):
    ex = lambda t, i: EvalExample(task_input=f"{t} task {i}", expected_behavior=f"Select tool: {t}")  # noqa: E731
    ds = EvalDataset(train=[ex("read_file", i) for i in range(6)], val=[ex("terminal", i) for i in range(3)],
                     holdout=[ex(t, i) for t in ("read_file", "terminal") for i in range(6)])
    ds.save(tmp_path / "ds")
    return tmp_path / "ds"


@pytest.fixture
def harness(tmp_path, monkeypatch):
    src = tmp_path / "repo" / "tools"
    src.mkdir(parents=True)
    (src / "file_tools.py").write_text('registry.register(name="read_file", description="Read a file")\n'
                                       'registry.register(name="terminal", description="Run a shell command")\n')
    staged, state = [], {"instructions": None}

    class FakeGEPA:
        def __init__(self, **kwargs):
            state["gepa_kwargs"] = kwargs

        def compile(self, student, trainset, valset):
            return etd.ToolSelectionModule(state["instructions"])

    def fake_scores(module, holdout, lm):
        # Evolved descriptions that mention "instead of cat" fix every terminal-vs-read_file miss.
        good = "instead of cat" in module.tools_text
        return [1.0 if good or ex.correct_tool == "read_file" else 0.0 for ex in holdout]

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(etd, "get_lm_for_role", lambda *a, **k: _FakeLM())
    monkeypatch.setattr(etd, "configure_default_lm", lambda *a, **k: None)
    monkeypatch.setattr(etd, "tool_rating_signals", lambda: {})
    monkeypatch.setattr(etd.dspy, "GEPA", FakeGEPA)
    monkeypatch.setattr(etd, "_holdout_scores", fake_scores)
    monkeypatch.setattr("evolution.core.staging.finalize",
                        lambda art, kind, live, text, stats, **kw: staged.append((art, live, text, stats, kw))
                        or {"staged_path": str(tmp_path / art), "status": "staged"})
    return {"repo": tmp_path / "repo", "dataset": _dataset(tmp_path), "staged": staged, "state": state,
            "file": src / "file_tools.py"}


def _run(h):
    etd.evolve(iterations=1, dataset_path=str(h["dataset"]), hermes_repo=str(h["repo"]))


def test_evolve_stages_distilled_descriptions_into_source(harness, monkeypatch, capsys):
    harness["state"]["instructions"] = GEPA_PROSE
    monkeypatch.setattr(et, "distill_descriptions", lambda tools, guidance, predictor=None: {
        "terminal": "Shell; use read_file instead of cat"})
    _run(harness)

    [(art, live, text, stats, kw)] = harness["staged"]
    assert art == "tool_tools_file_tools.py" and Path(live) == harness["file"]
    assert 'description="Shell; use read_file instead of cat"' in text
    assert 'registry.register(name="read_file", description="Read a file")' in text
    assert stats["description_source"] == "distilled" and stats["tools_changed"] == ["terminal"]
    assert stats["evolved_score"] > stats["baseline_score"]
    assert kw["extra_files"]["evolved_tools.json"][0]["name"] == "terminal"
    assert kw["apply"] is False  # stage-only by default
    assert harness["file"].read_text().count("instead of cat") == 0  # live file untouched
    result = json.loads(capsys.readouterr().out.split("EVOLUTION_RESULT ")[1].splitlines()[0])
    assert result["staged_paths"] and result["improvement"] > 0


def test_evolve_with_no_real_change_stages_nothing(harness, monkeypatch, capsys):
    harness["state"]["instructions"] = GEPA_PROSE
    monkeypatch.setattr(et, "distill_descriptions", lambda tools, guidance, predictor=None: {})
    _run(harness)
    assert harness["staged"] == []
    out = capsys.readouterr().out
    assert "No description changes detected" in out and '"improvement": 0.0' in out


def test_evolve_budget_is_still_capped(harness, monkeypatch):
    harness["state"]["instructions"] = GEPA_PROSE
    monkeypatch.setattr(et, "distill_descriptions", lambda *a, **k: {})
    _run(harness)
    assert harness["state"]["gepa_kwargs"]["max_metric_calls"] == etd._gepa_budget(1, 6, 3)


def test_evolve_never_stages_descriptions_that_break_constraints(harness, monkeypatch):
    harness["state"]["instructions"] = GEPA_PROSE
    monkeypatch.setattr(et, "distill_descriptions", lambda *a, **k: {"terminal": "instead of cat " * 50})
    _run(harness)
    assert harness["staged"] == []


def test_unpatchable_changes_are_not_scored_or_staged(harness, monkeypatch, capsys):
    """A registry-only tool (no source location) can't ship, so its rewrite must not count
    towards the holdout score: the run must report no improvement and stage nothing."""
    real_load = etd.load_tool_descriptions
    monkeypatch.setattr(etd, "load_tool_descriptions",
                        lambda repo: real_load(repo) + [ToolDescription("web_search", "Search the web", {})])
    harness["state"]["instructions"] = GEPA_PROSE
    monkeypatch.setattr(et, "distill_descriptions", lambda *a, **k: {"web_search": "Web; not instead of cat"})
    _run(harness)
    assert harness["staged"] == []
    assert '"improvement": 0.0' in capsys.readouterr().out


def test_only_applicable_reads_descriptions_back_from_patched_source(tmp_path):
    path = tmp_path / "tools" / "f.py"
    path.parent.mkdir()
    path.write_text('registry.register(name="a", description="A")\n')
    tools = [ToolDescription("a", "A", {}, path), ToolDescription("u", "U", {})]
    patched = [(Path("tools/f.py"), 'registry.register(name="a", description="A2")\n')]
    out = etd._only_applicable(tools, patched, tmp_path)
    assert [(t.name, t.description) for t in out] == [("a", "A2"), ("u", "U")]
    assert etd._only_applicable(tools, [], tmp_path) == tools
