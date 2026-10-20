"""Tests for system prompt section loading and saving."""

import ast
import pytest
from pathlib import Path

from evolution.prompts.prompt_module import (
    EVOLVABLE_SECTIONS,
    load_prompt_section,
    save_prompt_section,
    prompt_fitness_metric,
    _extract_constant,
    _replace_constant,
)
import dspy


SAMPLE_PROMPT_BUILDER = '''"""Prompt builder for Hermes agent."""

DEFAULT_AGENT_IDENTITY = "You are Hermes, a helpful AI assistant."

MEMORY_GUIDANCE = """When users share important preferences or context,
save them to memory for future sessions."""

SESSION_SEARCH_GUIDANCE = "Search past sessions when the user references prior work."

SKILLS_GUIDANCE = "Load relevant skills before starting complex tasks."

PLATFORM_HINTS = "Use plain text on CLI, markdown on web interfaces."
'''


@pytest.fixture
def hermes_repo(tmp_path):
    """Create a minimal hermes-agent repo structure."""
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    (agent_dir / "prompt_builder.py").write_text(SAMPLE_PROMPT_BUILDER)
    return tmp_path


class TestEvolvableSections:
    def test_expected_sections_present(self):
        for section in ["DEFAULT_AGENT_IDENTITY", "MEMORY_GUIDANCE", "SESSION_SEARCH_GUIDANCE",
                        "SKILLS_GUIDANCE"]:
            assert section in EVOLVABLE_SECTIONS

    def test_no_user_data_sections(self):
        # These should never be in the evolvable list
        forbidden = ["memory_block", "skills_index", "context_files"]
        for f in forbidden:
            assert f.upper() not in [s.upper() for s in EVOLVABLE_SECTIONS]


class TestLoadPromptSection:
    def test_loads_simple_string(self, hermes_repo):
        text = load_prompt_section("DEFAULT_AGENT_IDENTITY", hermes_repo)
        assert "Hermes" in text
        assert "helpful AI assistant" in text

    def test_loads_multiline_string(self, hermes_repo):
        text = load_prompt_section("MEMORY_GUIDANCE", hermes_repo)
        assert "memory" in text.lower()

    def test_raises_for_unknown_section(self, hermes_repo):
        with pytest.raises(KeyError):
            load_prompt_section("NONEXISTENT_SECTION", hermes_repo)

    def test_raises_when_prompt_builder_missing(self, tmp_path):
        # No agent/prompt_builder.py
        with pytest.raises(FileNotFoundError):
            load_prompt_section("MEMORY_GUIDANCE", tmp_path)


class TestSavePromptSection:
    def test_save_changes_constant_value(self, hermes_repo):
        evolved = "You are Hermes, an improved AI assistant with better capabilities."
        save_prompt_section("DEFAULT_AGENT_IDENTITY", evolved, hermes_repo)

        loaded = load_prompt_section("DEFAULT_AGENT_IDENTITY", hermes_repo)
        assert "improved AI assistant" in loaded

    def test_save_preserves_other_sections(self, hermes_repo):
        save_prompt_section(
            "DEFAULT_AGENT_IDENTITY",
            "New identity text.",
            hermes_repo,
        )

        # Other sections should be unchanged
        memory = load_prompt_section("MEMORY_GUIDANCE", hermes_repo)
        assert "memory" in memory.lower()

    def test_save_produces_valid_python(self, hermes_repo):
        save_prompt_section(
            "SESSION_SEARCH_GUIDANCE",
            'Search past sessions when the user says "like last time" or similar.',
            hermes_repo,
        )

        pb_path = hermes_repo / "agent" / "prompt_builder.py"
        ast.parse(pb_path.read_text())  # Should not raise SyntaxError

    def test_roundtrip(self, hermes_repo):
        original = load_prompt_section("SKILLS_GUIDANCE", hermes_repo)
        evolved = original + " Also load debugging skills for error-related tasks."

        save_prompt_section("SKILLS_GUIDANCE", evolved, hermes_repo)
        loaded = load_prompt_section("SKILLS_GUIDANCE", hermes_repo)

        assert "Also load debugging" in loaded


class TestExtractConstant:
    def test_extracts_simple_string(self):
        source = 'MY_CONST = "hello world"'
        result = _extract_constant(source, "MY_CONST", Path("fake.py"))
        assert result == "hello world"

    def test_extracts_triple_quoted(self):
        source = 'MY_CONST = """multi\nline"""'
        result = _extract_constant(source, "MY_CONST", Path("fake.py"))
        assert "multi" in result

    def test_raises_key_error_for_missing(self):
        source = 'OTHER_CONST = "value"'
        with pytest.raises(KeyError):
            _extract_constant(source, "MISSING_CONST", Path("fake.py"))


class TestReplaceConstant:
    def test_replaces_simple_string(self):
        source = 'MY_CONST = "old value"\n'
        result = _replace_constant(source, "MY_CONST", "old value", "new value")
        assert "new value" in result
        assert "old value" not in result

    def test_preserves_other_content(self):
        source = 'A = "keep this"\nB = "change this"\n'
        result = _replace_constant(source, "B", "change this", "new value")
        assert 'A = "keep this"' in result

    def test_raises_when_section_not_found(self):
        source = 'DIFFERENT = "value"\n'
        with pytest.raises(KeyError):
            _replace_constant(source, "MISSING", "value", "new")


class TestPromptFitnessMetric:
    def _make_example(self, expected: str) -> dspy.Example:
        return dspy.Example(
            task_input="task",
            expected_behavior=expected,
        ).with_inputs("task_input")

    def _make_prediction(self, output: str) -> dspy.Prediction:
        return dspy.Prediction(output=output)

    def test_empty_output_returns_zero(self):
        ex = self._make_example("save memory when user shares preferences")
        pred = self._make_prediction("")
        assert prompt_fitness_metric(ex, pred) == 0.0

    def test_relevant_output_scores_above_zero(self):
        ex = self._make_example("save memory when user shares preferences")
        pred = self._make_prediction("I will save your preferences to memory.")
        score = prompt_fitness_metric(ex, pred)
        assert score > 0.0

    def test_score_bounded_zero_to_one(self):
        ex = self._make_example("some expected behavior")
        pred = self._make_prediction("some output text here")
        score = prompt_fitness_metric(ex, pred)
        assert 0.0 <= score <= 1.0

    def test_empty_expected_returns_neutral(self):
        ex = self._make_example("")
        pred = self._make_prediction("some output")
        score = prompt_fitness_metric(ex, pred)
        assert score == 0.5
