"""Wraps hermes-agent tool descriptions as DSPy modules for optimization.

Tool descriptions are the natural language text in tool schemas — what the
agent sees when deciding which tool to use. Optimizing these descriptions
improves tool selection accuracy without changing any tool behavior.

FROZEN: tool schema structure (parameter names, types, required fields).
EVOLVABLE: description text strings only.
"""

import ast
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import dspy


@dataclass
class ToolDescription:
    """A single tool's description data."""
    name: str                           # e.g. "search_files"
    description: str                    # Top-level tool description text
    parameters: dict[str, str]          # param_name -> description text
    source_file: Optional[Path] = None  # tools/*.py it came from
    # Source span (lineno, col, end_lineno, end_col; UTF-8 byte cols) of the string expression
    # the description was read from, so it can be rewritten in place. None if not from source.
    description_span: Optional[tuple[int, int, int, int]] = None


def load_tool_descriptions(hermes_repo: Path) -> list[ToolDescription]:
    """Load tool descriptions from the hermes-agent repo.

    Strategy:
    1. Regex scan of tools/*.py — covers every file and carries source_file,
       which is required to apply evolved descriptions back to source.
    2. Tool registry import as a supplement (covers tools registered but not
       matching the regex patterns). Dedupe by name, preferring regex entries.

    Args:
        hermes_repo: Path to the hermes-agent repository root.

    Returns:
        List of ToolDescription objects, one per tool.
    """
    descriptions: list[ToolDescription] = []
    # Regex scan first: it finds every tools/*.py file and sets source_file.
    try:
        descriptions = _load_via_regex(hermes_repo)
    except Exception:
        pass
    # Registry as a supplement; dedupe by name, keep regex entries (source_file).
    try:
        reg = _load_via_registry(hermes_repo)
        by_name = {d.name: d for d in descriptions}
        for d in reg:
            if d.name not in by_name:
                descriptions.append(d)
    except Exception:
        pass
    return descriptions


def _load_via_registry(hermes_repo: Path) -> list[ToolDescription]:
    """Load tool descriptions by importing the hermes-agent tool registry."""
    if str(hermes_repo) not in sys.path:
        sys.path.insert(0, str(hermes_repo))

    try:
        import importlib
        registry_mod = importlib.import_module("tools.registry")

        # First, discover and import all built-in tool modules so they call
        # registry.register() and populate the singleton registry.
        try:
            registry_mod.discover_builtin_tools(hermes_repo / "tools")
        except Exception:
            # Individual module import failures are logged by discover_builtin_tools;
            # proceed with whatever entries are already registered.
            pass

        reg = registry_mod.registry
        entries = reg.get_all_entries()

        descriptions = []
        for entry in entries:
            name = entry.name
            if not name:
                continue

            schema = entry.schema if isinstance(entry.schema, dict) else {}
            desc = entry.description or schema.get("description", "")
            params = {}
            for param_name, param_info in schema.get("parameters", {}).get("properties", {}).items():
                params[param_name] = param_info.get("description", "")

            descriptions.append(ToolDescription(
                name=name,
                description=desc,
                parameters=params,
            ))

        return descriptions
    finally:
        if str(hermes_repo) in sys.path:
            sys.path.remove(str(hermes_repo))


def _load_via_regex(hermes_repo: Path) -> list[ToolDescription]:
    """Parse tool descriptions from tools/*.py using AST + regex.

    Looks for patterns like:
      TOOL_DESCRIPTION = "..."
      registry.register(name="...", description="...", ...)
    """
    tools_dir = hermes_repo / "tools"
    if not tools_dir.exists():
        return []

    descriptions = []

    for py_file in sorted(tools_dir.glob("*.py")):
        if py_file.name.startswith("_") or py_file.name == "registry.py":
            continue

        try:
            source = py_file.read_text()
            found = _extract_from_source(source, py_file)
            descriptions.extend(found)
        except Exception:
            continue

    return descriptions


def _eval_str(node: ast.AST, constants: dict[str, ast.AST]) -> Optional[str]:
    """Best-effort static evaluation of an AST node to a string.

    Handles string literals, name references to module-level string constants,
    and adjacent/concatenated string literals. Returns None for anything that
    can't be resolved without executing code (f-strings, function calls, etc.).
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        const = constants.get(node.id)
        if const is not None:
            return _eval_str(const, constants)
    # Adjacent string literals ("a" "b") are folded by the parser into one Constant.
    # Explicit concatenation ("a" + "b") — best effort: resolve each side.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _eval_str(node.left, constants)
        right = _eval_str(node.right, constants)
        if left is not None and right is not None:
            return left + right
    return None


def _string_node(node: ast.AST, constants: dict[str, ast.AST]) -> Optional[ast.AST]:
    """The expression node that holds the string text: follows a name to its module constant."""
    if isinstance(node, ast.Name) and isinstance(constants.get(node.id), ast.AST):
        return _string_node(constants[node.id], constants)
    return node


def _desc_value_from_dict_node(dict_node: ast.Dict) -> Optional[ast.AST]:
    """The value node of the ``"description": ...`` entry of a schema dict literal."""
    for k, v in zip(dict_node.keys, dict_node.values):
        if isinstance(k, ast.Constant) and k.value == "description":
            return v
    return None


def _desc_from_dict_node(dict_node: ast.Dict) -> Optional[str]:
    """Pull the ``"description": "..."`` entry out of a schema dict literal."""
    value = _desc_value_from_dict_node(dict_node)
    return _eval_str(value, {}) if value is not None else None


def _schema_desc_node(schema_node: ast.AST, constants: dict[str, ast.AST]) -> Optional[ast.AST]:
    """Description value node of an inline schema dict or a named module-level schema dict."""
    if isinstance(schema_node, ast.Name):
        schema_node = constants.get(schema_node.id)
    if isinstance(schema_node, ast.Dict):
        return _desc_value_from_dict_node(schema_node)
    return None


def _description_from_schema(schema_node: ast.AST, constants: dict[str, ast.AST]) -> Optional[str]:
    """Resolve a schema argument to a description string.

    Supports an inline dict literal or a name reference to a module-level
    schema constant. Function-call schemas (e.g. build_xxx_schema()) can't be
    resolved statically and return None.
    """
    value = _schema_desc_node(schema_node, constants)
    return _eval_str(value, {}) if value is not None else None


def _span(node: Optional[ast.AST]) -> Optional[tuple[int, int, int, int]]:
    if node is None or getattr(node, "end_lineno", None) is None:
        return None
    return node.lineno, node.col_offset, node.end_lineno, node.end_col_offset


def _extract_from_source(source: str, source_file: Path) -> list[ToolDescription]:
    """Extract tool registrations from a single Python source file.

    Walks the AST looking for ``registry.register(...)`` calls and pulls the
    tool ``name`` plus its ``description``. The description may come from:

    * a ``description="..."`` keyword on the register call itself, or
    * the ``"description": "..."`` entry of a schema dict passed via
      ``schema=_CONSTANT`` (module-level constant) or an inline dict.

    Schemas built by function calls (``build_xxx_schema()``) can't be resolved
    statically and are skipped.
    """
    results = []
    try:
        tree = ast.parse(source, filename=str(source_file))
    except SyntaxError:
        return results

    # Collect module-level constant assignments (name -> AST value).
    constants: dict[str, ast.AST] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name):
                    constants[target.id] = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            if stmt.value is not None:
                constants[stmt.target.id] = stmt.value

    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "register"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "registry"
        ):
            continue

        call = node.value
        name: Optional[str] = None
        description: Optional[str] = None
        desc_node: Optional[ast.AST] = None
        for kw in call.keywords:
            if kw.arg == "name":
                name = _eval_str(kw.value, constants)
            elif kw.arg == "description":
                description = _eval_str(kw.value, constants)
                desc_node = _string_node(kw.value, constants)
            elif kw.arg == "schema" and not description:
                description = _description_from_schema(kw.value, constants)
                desc_node = _schema_desc_node(kw.value, constants)

        if name and description:
            results.append(ToolDescription(
                name=name,
                description=description,
                parameters={},
                source_file=source_file,
                description_span=_span(desc_node),
            ))

    return results


def apply_evolved_descriptions(
    original_tools: list[ToolDescription],
    evolved_tools: list[ToolDescription],
    hermes_repo: Path,
) -> list[tuple[Path, str]]:
    """Apply evolved descriptions back to source files.

    Only modifies description text — never touches parameter names,
    types, required fields, or any other schema structure. Each description is
    rewritten at its recorded AST span and the result is re-parsed and verified
    (see ``evolution.tools.source_patch``).

    Returns:
        List of (relative_path, new_content) tuples for pr_builder.
    """
    from evolution.tools.source_patch import apply_to_sources
    changes, _notes = apply_to_sources(original_tools, evolved_tools, hermes_repo)
    return changes


class ToolSelectionModule(dspy.Module):
    """DSPy module that selects the right tool for a task.

    Tool descriptions are embedded as the DSPy signature instructions — this is
    what GEPA mutates during optimization. After evolution, tools_text
    returns the GEPA-optimized version.
    """

    def __init__(self, tool_descriptions_text: str):
        super().__init__()
        self._tools_text = tool_descriptions_text
        sig = _BaseToolSelectionSig.with_instructions(tool_descriptions_text)
        self.selector = dspy.ChainOfThought(sig)

    @property
    def tools_text(self) -> str:
        """Return the current (possibly GEPA-evolved) tool descriptions."""
        return self.selector.predict.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.selector(task_input=task_input)
        return dspy.Prediction(
            selected_tool=result.selected_tool,
            rationale=result.rationale,
        )


class _BaseToolSelectionSig(dspy.Signature):
    """Placeholder — replaced with tool descriptions via with_instructions."""
    task_input: str = dspy.InputField(desc="The task that needs to be completed")
    selected_tool: str = dspy.OutputField(
        desc="The exact name of the tool to use (must match one of the available tools)"
    )
    rationale: str = dspy.OutputField(
        desc="Brief explanation of why this tool is the right choice"
    )


def tool_selection_metric(gold, pred, trace=None, pred_name=None, pred_trace=None) -> float:
    """DSPy-compatible metric for tool selection accuracy (GEPA v2).

    Scores whether the agent selected the correct tool. Uses exact match
    as primary signal with partial credit for close matches.

    Returns:
        1.0 for exact correct tool, 0.5 for partial match, 0.0 for wrong tool.
    """
    correct_tool = getattr(gold, "correct_tool", "") or ""
    selected = getattr(pred, "selected_tool", "") or ""

    if not correct_tool or not selected:
        return 0.0

    correct_lower = correct_tool.strip().lower()
    selected_lower = selected.strip().lower()

    # Exact match
    if correct_lower == selected_lower:
        return 1.0

    # Partial match: one contains the other
    if correct_lower in selected_lower or selected_lower in correct_lower:
        return 0.5

    return 0.0


def format_tools_for_prompt(tools: list[ToolDescription]) -> str:
    """Format tool descriptions as a JSON-like string for the DSPy module."""
    import json
    return json.dumps([
        {
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ], indent=2)
