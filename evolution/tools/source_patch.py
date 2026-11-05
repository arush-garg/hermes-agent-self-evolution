"""Rewrite tool description string literals in hermes-agent source, located by AST span.

The extractor records the exact source span of the string expression each description came
from (a literal, an implicit/explicit concatenation, or a module-level constant it names).
Patching replaces that span and nothing else, so escapes, quoting style, multi-line
concatenations and surrounding code never have to match textually. Every patched file is
re-parsed and re-extracted: the patch is kept only if the target tools now carry exactly the
evolved text and every other tool in the file is unchanged.
"""
import json
from pathlib import Path

WRAP_WIDTH = 72  # chars of description text per line when a long literal is wrapped

Span = tuple[int, int, int, int]  # (lineno, col, end_lineno, end_col); cols are UTF-8 byte offsets


def _byte_offset(line_starts: list[int], lineno: int, col: int) -> int:
    return line_starts[lineno - 1] + col


def _line_starts(data: bytes) -> list[int]:
    starts, pos = [0], 0
    for line in data.splitlines(keepends=True):
        pos += len(line)
        starts.append(pos)
    return starts


def _chunks(text: str, width: int) -> list[str]:
    """Split on spaces into pieces of about ``width`` chars; ``"".join`` gives ``text`` back."""
    words = text.split(" ")
    pieces, current = [], words[0]
    for word in words[1:]:
        if len(current) + 1 + len(word) > width:
            pieces.append(current + " ")
            current = word
        else:
            current += " " + word
    pieces.append(current)
    return pieces


def python_literal(text: str, indent: int, bracketed: bool = False) -> str:
    """A Python expression evaluating to ``text``: one literal, or an implicit concatenation
    when it is long. The concatenation gets its own parentheses (which make the newlines legal)
    unless ``bracketed`` says the span already sits alone inside a pair of brackets."""
    if len(text) <= WRAP_WIDTH:
        return json.dumps(text, ensure_ascii=False)
    pieces = [json.dumps(piece, ensure_ascii=False) for piece in _chunks(text, WRAP_WIDTH)]
    if bracketed:
        return ("\n" + " " * indent).join(pieces)
    pad = " " * (indent + 4)
    return "(\n" + "\n".join(pad + piece for piece in pieces) + "\n" + " " * indent + ")"


def _is_bracketed(data: bytes, begin: int, end: int) -> bool:
    """The span is the only thing inside ``( ... )``: e.g. ``"description": ("a" "b")``."""
    return data[:begin].rstrip()[-1:] == b"(" and data[end:].lstrip()[:1] == b")"


def _indent_of(data: bytes, line_starts: list[int], lineno: int) -> int:
    line = data[line_starts[lineno - 1]:line_starts[lineno]]
    return len(line) - len(line.lstrip(b" "))


def patch_source(source: str, replacements: dict[Span, str]) -> str:
    """Replace each span with a literal for its new text; spans are applied bottom-up."""
    data = source.encode("utf-8")
    starts = _line_starts(data)
    for span in sorted(replacements, key=lambda s: (s[0], s[1]), reverse=True):
        lineno, col, end_lineno, end_col = span
        begin = _byte_offset(starts, lineno, col)
        end = _byte_offset(starts, end_lineno, end_col)
        bracketed = _is_bracketed(data, begin, end)
        literal = python_literal(replacements[span], _indent_of(data, starts, lineno), bracketed)
        data = data[:begin] + literal.encode("utf-8") + data[end:]
    return data.decode("utf-8")


def _spans_for_file(originals, evolved_map) -> tuple[dict[Span, str], list[str]]:
    """Span -> new text for one file. A span shared by several tools (a module constant) is
    rewritten only when every tool using it gets the same new text."""
    texts: dict[Span, set[str]] = {}
    for orig in originals:
        if orig.description_span is not None:
            texts.setdefault(orig.description_span, set()).add(evolved_map.get(orig.name, orig.description))
    current = {t.description_span: t.description for t in originals}
    spans, notes = {}, []
    for span, candidates in texts.items():
        if len(candidates) > 1:
            notes.append(f"{originals[0].source_file.name}: tools sharing one description were "
                         f"evolved differently, skipped")
        elif (new := next(iter(candidates))) != current[span]:
            spans[span] = new
    return spans, notes


def _verify(new_source: str, path: Path, before, spans: dict[Span, str]) -> bool:
    """New source parses and each tool carries its span's new text (others unchanged)."""
    from evolution.tools.tool_module import _extract_from_source
    after = {t.name: t.description for t in _extract_from_source(new_source, path)}
    want = {t.name: spans.get(t.description_span, t.description) for t in before}
    return after == want


def apply_to_sources(original_tools, evolved_tools, hermes_repo: Path) -> tuple[list[tuple[Path, str]], list[str]]:
    """``([(relative_path, new_content)], notes)`` for every file whose patch verified."""
    from evolution.tools.tool_module import _extract_from_source
    evolved_map = {t.name: t.description for t in evolved_tools}
    by_file: dict[Path, list] = {}
    for orig in original_tools:
        if orig.source_file is not None:
            by_file.setdefault(orig.source_file, []).append(orig)

    changes, notes = [], []
    for path, originals in by_file.items():
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        before = _extract_from_source(source, path)  # fresh spans: the file is the source of truth
        current = {(t.name, t.description) for t in before}
        if not all((t.name, t.description) in current for t in originals):
            notes.append(f"{path.name}: source changed since load, skipped")
            continue
        spans, conflict_notes = _spans_for_file(before, evolved_map)
        notes.extend(conflict_notes)
        if not spans:
            continue
        new_source = patch_source(source, spans)
        if not _verify(new_source, path, before, spans):
            notes.append(f"{path.name}: patched source failed verification, skipped")
            continue
        try:
            rel = path.relative_to(hermes_repo)
        except ValueError:
            rel = path
        changes.append((rel, new_source))
    return changes, notes
