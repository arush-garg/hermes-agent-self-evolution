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


def load_tool_descriptions(hermes_repo: Path) -> list[ToolDescription]:
    """Load tool descriptions from the hermes-agent repo.

    Strategy:
    1. Try importing the tool registry directly (most accurate)
    2. Fall back to regex parsing of tools/*.py files

    Args:
        hermes_repo: Path to the hermes-agent repository root.

    Returns:
        List of ToolDescription objects, one per registered tool.
    """
    try:
        return _load_via_registry(hermes_repo)
    except Exception:
        return _load_via_regex(hermes_repo)


def _load_via_registry(hermes_repo: Path) -> list[ToolDescription]:
    """Load tool descriptions by importing the hermes-agent tool registry."""
    if str(hermes_repo) not in sys.path:
        sys.path.insert(0, str(hermes_repo))

    try:
        import importlib
        registry_mod = importlib.import_module("tools.registry")
        tools = registry_mod.get_all_tools()

        descriptions = []
        for tool in tools:
            schema = tool.get("schema", {}) if isinstance(tool, dict) else {}
            name = schema.get("name", "") or tool.get("name", "")
            if not name:
                continue

            desc = schema.get("description", "")
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


def _extract_from_source(source: str, source_file: Path) -> list[ToolDescription]:
    """Extract tool registrations from a single Python source file."""
    results = []

    # Look for registry.register(...) calls
    # Pattern: registry.register(name="...", description="...", parameters={...})
    register_pattern = re.compile(
        r'registry\.register\s*\(',
        re.MULTILINE,
    )

    for match in register_pattern.finditer(source):
        start = match.start()
        # Find the matching closing paren
        depth = 0
        end = start
        for i in range(start, len(source)):
            if source[i] == '(':
                depth += 1
            elif source[i] == ')':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break

        call_text = source[start:end]

        # Extract name and description using simple regex
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', call_text)
        desc_match = re.search(r'description\s*=\s*["\']([^"\']*)["\']', call_text)

        if name_match and desc_match:
            results.append(ToolDescription(
                name=name_match.group(1),
                description=desc_match.group(1),
                parameters={},
                source_file=source_file,
            ))

    return results


def apply_evolved_descriptions(
    original_tools: list[ToolDescription],
    evolved_tools: list[ToolDescription],
    hermes_repo: Path,
) -> list[tuple[Path, str]]:
    """Apply evolved descriptions back to source files.

    Only modifies description text — never touches parameter names,
    types, required fields, or any other schema structure.

    Returns:
        List of (relative_path, new_content) tuples for pr_builder.
    """
    # Build lookup: name -> evolved description
    evolved_map = {t.name: t for t in evolved_tools}
    changes: dict[Path, str] = {}

    for orig in original_tools:
        evolved = evolved_map.get(orig.name)
        if not evolved or evolved.description == orig.description:
            continue
        if not orig.source_file or not orig.source_file.exists():
            continue

        source = orig.source_file.read_text()
        # Replace the specific description string for this tool
        new_source = source.replace(
            f'description="{orig.description}"',
            f'description="{evolved.description}"',
            1,
        )
        if new_source != source:
            changes[orig.source_file] = new_source

    # Return as (relative_path, content) tuples
    result = []
    for abs_path, content in changes.items():
        try:
            rel = abs_path.relative_to(hermes_repo)
        except ValueError:
            rel = abs_path
        result.append((rel, content))

    return result


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
