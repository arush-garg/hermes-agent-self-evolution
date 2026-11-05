"""Evolved tool descriptions are written back into source exactly, by AST span, or not at all."""
import ast
import os
from pathlib import Path

import pytest

from evolution.tools.source_patch import _chunks, apply_to_sources, python_literal
from evolution.tools.tool_module import ToolDescription, _extract_from_source, _load_via_regex


def _tool_file(tmp_path, source, name="t.py"):
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir(exist_ok=True)
    path = tools_dir / name
    path.write_text(source, encoding="utf-8")
    return path


def _apply(tmp_path, path, new_by_name):
    originals = _extract_from_source(path.read_text(encoding="utf-8"), path)
    evolved = [ToolDescription(n, d, {}) for n, d in new_by_name.items()]
    return apply_to_sources(originals, evolved, tmp_path)


def _descs(source, path):
    return {t.name: t.description for t in _extract_from_source(source, path)}


KWARG = 'registry.register(name="a", description="Old a", handler=h)\n'

SCHEMA_CONCAT = '''from tools.registry import registry

_SCHEMA = {
    "name": "b",
    "description": (
        "Reads a file. "
        "Use for \\"text\\" files.\\nNot for binaries."
    ),
    "parameters": {"type": "object"},
}

registry.register(name="b", schema=_SCHEMA, handler=h)
'''

CONSTANT = '''DESC = "Old " + "c"

def helper():
    return 1

registry.register(name="c", description=DESC)
'''


def test_kwarg_literal_replaced_and_rest_of_file_untouched(tmp_path):
    path = _tool_file(tmp_path, "x = 1  # keep\n" + KWARG + "y = 2\n")
    [(rel, new)], notes = _apply(tmp_path, path, {"a": "New a"})
    assert rel == Path("tools/t.py") and notes == []
    assert new == 'x = 1  # keep\nregistry.register(name="a", description="New a", handler=h)\ny = 2\n'


def test_multiline_concatenation_in_schema_dict(tmp_path):
    path = _tool_file(tmp_path, SCHEMA_CONCAT)
    assert _descs(SCHEMA_CONCAT, path)["b"] == 'Reads a file. Use for "text" files.\nNot for binaries.'
    [(_, new)], _ = _apply(tmp_path, path, {"b": "Reads text files; use search_files to find them."})
    assert _descs(new, path) == {"b": "Reads text files; use search_files to find them."}
    assert new.startswith('from tools.registry import registry\n\n_SCHEMA = {\n    "name": "b",\n')
    assert new.endswith('    "parameters": {"type": "object"},\n}\n\nregistry.register(name="b", schema=_SCHEMA, handler=h)\n')


def test_named_constant_is_rewritten_where_defined(tmp_path):
    path = _tool_file(tmp_path, CONSTANT)
    [(_, new)], _ = _apply(tmp_path, path, {"c": "New c"})
    assert new.splitlines()[0] == 'DESC = "New c"'
    assert 'registry.register(name="c", description=DESC)' in new


@pytest.mark.parametrize("text", [
    'Quotes "double" and \'single\', a backslash \\ and a tab\t.',
    "Line one.\nLine two.",
    "Unicode — café ✓ 日本語",
    "  leading and  double  spaces  ",
    "word " * 60,
])
def test_awkward_text_round_trips_exactly(tmp_path, text):
    path = _tool_file(tmp_path, KWARG)
    [(_, new)], _ = _apply(tmp_path, path, {"a": text})
    assert _descs(new, path) == {"a": text}


def test_two_tools_on_one_line_with_non_ascii_prefix(tmp_path):
    # ast column offsets are UTF-8 byte offsets; the second replacement must still land right.
    src = 'X = "é"; registry.register(name="a", description="A"); registry.register(name="b", description="B")\n'
    path = _tool_file(tmp_path, src)
    [(_, new)], _ = _apply(tmp_path, path, {"a": "Alpha ✓", "b": "Beta"})
    assert _descs(new, path) == {"a": "Alpha ✓", "b": "Beta"}
    assert new.startswith('X = "é"; ')


def test_long_text_is_wrapped_inside_parentheses_and_parses(tmp_path):
    src = 'def register_all():\n    registry.register(\n        name="a",\n        description="Old",\n    )\n'
    path = _tool_file(tmp_path, src)
    long_text = "Searches file contents with ripgrep and returns matching lines with paths. " * 3
    [(_, new)], _ = _apply(tmp_path, path, {"a": long_text})
    ast.parse(new)
    assert _descs(new, path) == {"a": long_text}
    assert max(len(line) for line in new.splitlines()) < 100


def test_shared_constant_with_conflicting_rewrites_is_skipped(tmp_path):
    src = 'D = "Shared"\nregistry.register(name="a", description=D)\nregistry.register(name="b", description=D)\n'
    path = _tool_file(tmp_path, src)
    changes, notes = _apply(tmp_path, path, {"a": "For a"})  # b keeps "Shared": conflict
    assert changes == [] and "evolved differently" in notes[0]
    [(_, new)], _ = _apply(tmp_path, path, {"a": "Both", "b": "Both"})
    assert _descs(new, path) == {"a": "Both", "b": "Both"}


def test_source_edited_since_load_is_skipped(tmp_path):
    path = _tool_file(tmp_path, KWARG)
    originals = _extract_from_source(KWARG, path)
    path.write_text(KWARG.replace("Old a", "Hand-edited"), encoding="utf-8")
    changes, notes = apply_to_sources(originals, [ToolDescription("a", "New", {})], tmp_path)
    assert changes == [] and "changed since load" in notes[0]


def test_unchanged_unknown_and_unsourced_tools_produce_no_changes(tmp_path):
    path = _tool_file(tmp_path, KWARG)
    assert _apply(tmp_path, path, {"a": "Old a", "zzz": "ignored"}) == ([], [])
    unsourced = [ToolDescription("a", "Old a", {})]
    assert apply_to_sources(unsourced, [ToolDescription("a", "New", {})], tmp_path) == ([], [])


def test_files_on_disk_are_never_written(tmp_path):
    path = _tool_file(tmp_path, KWARG)
    _apply(tmp_path, path, {"a": "New a"})
    assert path.read_text(encoding="utf-8") == KWARG


def test_f_string_descriptions_are_not_extracted_so_never_patched(tmp_path):
    path = _tool_file(tmp_path, 'N = 3\nregistry.register(name="a", description=f"Top {N}")\n')
    assert _extract_from_source(path.read_text(), path) == []


@pytest.mark.parametrize("text", ["", "a", " a", "a ", "a  b", "x" * 200, "one two " * 40])
def test_chunks_are_lossless(text):
    assert "".join(_chunks(text, 20)) == text


def test_python_literal_evaluates_to_text():
    for text in ["short", "y " * 100, 'q"uote\n' * 20]:
        assert ast.literal_eval(python_literal(text, indent=8)) == text


HERMES_REPO = Path(os.environ.get("HERMES_AGENT_REPO", Path.home() / ".hermes" / "hermes-agent"))


@pytest.mark.skipif(not (HERMES_REPO / "tools").is_dir(), reason="hermes-agent checkout not present")
def test_every_real_hermes_tool_description_can_be_patched():
    """Regression for the real repo: before the AST patch only 8 of 40 descriptions matched
    the literal text that string replacement searched for."""
    tools = _load_via_regex(HERMES_REPO)
    assert tools, "no tools extracted from hermes-agent"
    evolved = [ToolDescription(t.name, "EVOLVED. " + t.description + " ✓", {}) for t in tools]
    changes, notes = apply_to_sources(tools, evolved, HERMES_REPO)
    assert notes == []
    patched = {}
    for rel, content in changes:
        ast.parse(content)
        patched.update(_descs(content, HERMES_REPO / rel))
    assert patched == {t.name: "EVOLVED. " + t.description + " ✓" for t in tools}


def test_already_parenthesised_concatenation_gets_no_second_pair(tmp_path):
    src = ('S = {\n    "name": "a",\n    "description": (\n        "Old one. "\n        "Old two."\n    ),\n}\n'
           'registry.register(name="a", schema=S)\n')
    path = _tool_file(tmp_path, src)
    text = "A rewritten description that is long enough to need wrapping over several lines. " * 2
    [(_, new)], _ = _apply(tmp_path, path, {"a": text})
    assert _descs(new, path) == {"a": text}
    assert "((" not in new.replace(" ", "").replace("\n", "")
    assert new.startswith('S = {\n    "name": "a",\n    "description": (\n        "A rewritten')
    assert new.endswith('\n    ),\n}\nregistry.register(name="a", schema=S)\n')


def test_bracket_detection_ignores_call_arguments_with_neighbours(tmp_path):
    src = 'registry.register(name="a", description="Old")\n'
    path = _tool_file(tmp_path, src)
    [(_, new)], _ = _apply(tmp_path, path, {"a": "word " * 40})
    ast.parse(new)
    assert 'description=(\n' in new  # not alone in the call's brackets, so it gets its own
