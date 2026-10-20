"""Tests for code evolution organism and fitness evaluation."""

import ast
import subprocess
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from evolution.code.code_organism import (
    CodeTarget,
    CodeFitnessEvaluator,
    CodeFitnessResult,
    check_signatures_unchanged,
    _extract_function_signatures,
    _extract_registry_calls,
)


SIMPLE_TOOL = '''"""A simple tool for testing."""

def read_file(path: str, offset: int = 0) -> str:
    """Read a file."""
    with open(path) as f:
        content = f.read()
    return content[offset:]


def write_file(path: str, content: str) -> None:
    """Write a file."""
    with open(path, "w") as f:
        f.write(content)
'''

TOOL_WITH_REGISTRY = '''"""Tool with registry registration."""
from tools.registry import registry

def search_files(pattern: str, directory: str = ".") -> list:
    """Search files."""
    import glob
    return glob.glob(f"{directory}/**/{pattern}", recursive=True)


registry.register(
    name="search_files",
    description="Search for files by pattern",
    parameters={"pattern": "The search pattern", "directory": "Search root"},
)
'''

EVOLVED_SAME_SIGS = SIMPLE_TOOL.replace(
    "content = f.read()\n    return content[offset:]",
    "content = f.read()\n    # optimized: strip trailing whitespace\n    return content[offset:].rstrip()",
)

EVOLVED_BROKEN_SIG = SIMPLE_TOOL.replace(
    "def read_file(path: str, offset: int = 0) -> str:",
    "def read_file(path: str, offset: int = 0, encoding: str = 'utf-8') -> str:",
)

EVOLVED_REMOVED_FUNC = SIMPLE_TOOL.replace(
    "def write_file(path: str, content: str) -> None:",
    "def _write_file_internal(path: str, content: str) -> None:",
)

EVOLVED_CHANGED_REGISTRY = TOOL_WITH_REGISTRY.replace(
    'name="search_files"',
    'name="find_files"',
)


class TestCheckSignaturesUnchanged:
    def test_identical_code_passes(self):
        assert check_signatures_unchanged(SIMPLE_TOOL, SIMPLE_TOOL) is True

    def test_implementation_change_passes(self):
        assert check_signatures_unchanged(SIMPLE_TOOL, EVOLVED_SAME_SIGS) is True

    def test_added_parameter_fails(self):
        assert check_signatures_unchanged(SIMPLE_TOOL, EVOLVED_BROKEN_SIG) is False

    def test_renamed_function_fails(self):
        assert check_signatures_unchanged(SIMPLE_TOOL, EVOLVED_REMOVED_FUNC) is False

    def test_registry_name_change_fails(self):
        assert check_signatures_unchanged(TOOL_WITH_REGISTRY, EVOLVED_CHANGED_REGISTRY) is False

    def test_registry_unchanged_passes(self):
        evolved = TOOL_WITH_REGISTRY.replace(
            '"Search for files by pattern"',
            '"Find files matching a glob pattern"',
        )
        assert check_signatures_unchanged(TOOL_WITH_REGISTRY, evolved) is True

    def test_syntax_error_in_evolved_fails(self):
        assert check_signatures_unchanged(SIMPLE_TOOL, "def broken(") is False

    def test_syntax_error_in_original_fails(self):
        assert check_signatures_unchanged("def broken(", SIMPLE_TOOL) is False


class TestExtractFunctionSignatures:
    def test_extracts_simple_function(self):
        tree = ast.parse("def foo(a, b): pass")
        sigs = _extract_function_signatures(tree)
        assert "foo" in sigs
        assert "foo(a, b)" == sigs["foo"]

    def test_extracts_function_with_defaults(self):
        tree = ast.parse("def bar(x, y=10): pass")
        sigs = _extract_function_signatures(tree)
        assert "bar" in sigs
        assert "y=10" in sigs["bar"]

    def test_extracts_kwargs(self):
        tree = ast.parse("def baz(*args, **kwargs): pass")
        sigs = _extract_function_signatures(tree)
        assert "*args" in sigs["baz"]
        assert "**kwargs" in sigs["baz"]

    def test_empty_source(self):
        tree = ast.parse("")
        sigs = _extract_function_signatures(tree)
        assert sigs == {}


class TestExtractRegistryCalls:
    def test_finds_registry_register(self):
        tree = ast.parse(TOOL_WITH_REGISTRY)
        calls = _extract_registry_calls(tree)
        assert "search_files" in calls

    def test_captures_kwarg_names(self):
        tree = ast.parse(TOOL_WITH_REGISTRY)
        calls = _extract_registry_calls(tree)
        assert "name" in calls["search_files"]
        assert "description" in calls["search_files"]

    def test_no_registry_calls(self):
        tree = ast.parse(SIMPLE_TOOL)
        calls = _extract_registry_calls(tree)
        assert calls == {}


class TestCodeFitnessEvaluator:
    def test_returns_zero_when_original_not_found(self, tmp_path):
        evaluator = CodeFitnessEvaluator(tmp_path)
        target = CodeTarget(source_file=Path("nonexistent.py"))

        result = evaluator.score("def foo(): pass", target)
        assert result.score == 0.0
        assert not result.tests_passed

    def test_returns_zero_when_signatures_change(self, tmp_path):
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        original = tools_dir / "test_tool.py"
        original.write_text(SIMPLE_TOOL)

        evaluator = CodeFitnessEvaluator(tmp_path)
        target = CodeTarget(source_file=Path("tools/test_tool.py"))

        result = evaluator.score(EVOLVED_BROKEN_SIG, target)
        assert result.score == 0.0
        assert not result.signatures_unchanged

    def test_passes_gate_with_valid_evolution(self, tmp_path):
        """Test that identical code (trivial evolution) passes the gate."""
        tools_dir = tmp_path / "tools"
        tools_dir.mkdir()
        original = tools_dir / "test_tool.py"
        original.write_text(SIMPLE_TOOL)

        evaluator = CodeFitnessEvaluator(tmp_path)
        target = CodeTarget(
            source_file=Path("tools/test_tool.py"),
            test_command="python -c 'print(\"ok\")'",  # Always passes
        )

        result = evaluator.score(EVOLVED_SAME_SIGS, target)
        # Signature check passes; test result depends on subprocess
        assert result.signatures_unchanged is True


class TestCodeTarget:
    def test_default_test_command(self):
        target = CodeTarget(source_file=Path("tools/file_tools.py"))
        assert "pytest" in target.test_command

    def test_custom_test_command(self):
        target = CodeTarget(
            source_file=Path("tools/file_tools.py"),
            test_command="python -m pytest tests/tools/ -q",
        )
        assert "tests/tools/" in target.test_command
