"""Tests for tool description module."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock
import dspy

from evolution.tools.tool_module import (
    ToolDescription,
    tool_selection_metric,
    format_tools_for_prompt,
    apply_evolved_descriptions,
    _extract_from_source,
    _load_via_regex,
)


@pytest.fixture
def sample_tools():
    return [
        ToolDescription(
            name="search_files",
            description="Search for files by pattern or content",
            parameters={"pattern": "The search pattern"},
        ),
        ToolDescription(
            name="read_file",
            description="Read the contents of a file",
            parameters={"path": "Path to the file to read"},
        ),
    ]


class TestToolDescription:
    def test_dataclass_fields(self, sample_tools):
        tool = sample_tools[0]
        assert tool.name == "search_files"
        assert "Search" in tool.description
        assert "pattern" in tool.parameters


class TestToolSelectionMetric:
    def _make_example(self, correct_tool: str) -> dspy.Example:
        ex = dspy.Example(
            task_input="find python files",
            expected_behavior=f"Select tool: {correct_tool}",
            correct_tool=correct_tool,
        ).with_inputs("task_input")
        return ex

    def _make_prediction(self, selected: str) -> dspy.Prediction:
        return dspy.Prediction(selected_tool=selected)

    def test_exact_match_scores_one(self):
        ex = self._make_example("search_files")
        pred = self._make_prediction("search_files")
        assert tool_selection_metric(ex, pred) == 1.0

    def test_case_insensitive_match(self):
        ex = self._make_example("search_files")
        pred = self._make_prediction("SEARCH_FILES")
        assert tool_selection_metric(ex, pred) == 1.0

    def test_partial_match_scores_half(self):
        ex = self._make_example("search_files")
        pred = self._make_prediction("use search_files to find it")
        assert tool_selection_metric(ex, pred) == 0.5

    def test_wrong_tool_scores_zero(self):
        ex = self._make_example("search_files")
        pred = self._make_prediction("read_file")
        assert tool_selection_metric(ex, pred) == 0.0

    def test_empty_correct_tool_scores_zero(self):
        ex = dspy.Example(task_input="t", expected_behavior="b", correct_tool="").with_inputs("task_input")
        pred = self._make_prediction("search_files")
        assert tool_selection_metric(ex, pred) == 0.0

    def test_empty_prediction_scores_zero(self):
        ex = self._make_example("search_files")
        pred = dspy.Prediction(selected_tool="")
        assert tool_selection_metric(ex, pred) == 0.0


class TestFormatToolsForPrompt:
    def test_returns_valid_json(self, sample_tools):
        result = format_tools_for_prompt(sample_tools)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_includes_all_fields(self, sample_tools):
        result = format_tools_for_prompt(sample_tools)
        parsed = json.loads(result)
        assert parsed[0]["name"] == "search_files"
        assert "description" in parsed[0]
        assert "parameters" in parsed[0]


class TestExtractFromSource:
    def test_extracts_registry_calls(self, tmp_path):
        source = '''
from tools.registry import registry

registry.register(
    name="my_tool",
    description="Does something useful",
    parameters={"path": "the path"},
)
'''
        tmp_file = tmp_path / "my_tool.py"
        results = _extract_from_source(source, tmp_file)

        assert len(results) == 1
        assert results[0].name == "my_tool"
        assert results[0].description == "Does something useful"

    def test_multiple_tools_in_one_file(self, tmp_path):
        source = '''
registry.register(name="tool_a", description="Tool A description")
registry.register(name="tool_b", description="Tool B description")
'''
        results = _extract_from_source(source, tmp_path / "tools.py")
        assert len(results) == 2
        names = {r.name for r in results}
        assert "tool_a" in names
        assert "tool_b" in names

    def test_empty_source_returns_empty(self):
        results = _extract_from_source("", Path("empty.py"))
        assert results == []


class TestLoadViaRegex:
    def test_returns_empty_when_tools_dir_missing(self, tmp_path):
        # tmp_path has no tools/ subdirectory
        result = _load_via_regex(tmp_path)
        assert result == []

    def test_discovers_tools_from_py_files(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "my_tool.py").write_text(
            'registry.register(name="my_tool", description="A test tool")\n'
        )

        result = _load_via_regex(tmp_path)
        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_skips_registry_py(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "registry.py").write_text(
            'registry.register(name="hidden", description="Should be skipped")\n'
        )

        result = _load_via_regex(tmp_path)
        assert result == []

    def test_skips_private_files(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        (tools_dir / "_helpers.py").write_text(
            'registry.register(name="private", description="Skip me")\n'
        )

        result = _load_via_regex(tmp_path)
        assert result == []


class TestApplyEvolvedDescriptions:
    def test_applies_changed_description(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()

        source = 'registry.register(name="my_tool", description="Old description")\n'
        tool_file = tools_dir / "my_tool.py"
        tool_file.write_text(source)

        original = [ToolDescription("my_tool", "Old description", {}, tool_file)]
        evolved = [ToolDescription("my_tool", "Improved description", {})]

        changes = apply_evolved_descriptions(original, evolved, tmp_path)
        assert len(changes) == 1

    def test_no_change_when_descriptions_identical(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        tool_file = tools_dir / "my_tool.py"
        tool_file.write_text('registry.register(name="t", description="Same")\n')

        original = [ToolDescription("t", "Same", {}, tool_file)]
        evolved = [ToolDescription("t", "Same", {})]  # identical

        changes = apply_evolved_descriptions(original, evolved, tmp_path)
        assert len(changes) == 0
