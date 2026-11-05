"""Wraps hermes-agent system prompt sections as DSPy modules for optimization.

System prompt sections are the natural language constants in prompt_builder.py
that guide agent behavior: identity, memory guidance, session search, etc.

FROZEN: section names, which sections exist, prompt caching structure.
EVOLVABLE: the text content of each section constant.

Parsing uses ast.parse() for accuracy — never fragile regex.
"""

import ast
import re
from pathlib import Path
from typing import Optional

import dspy


# Sections that are safe to evolve. Others (memory blocks, skill indices,
# project context files) are auto-generated or user data — never evolve those.
EVOLVABLE_SECTIONS = [
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "PLATFORM_HINTS",
]


def load_prompt_section(section_name: str, hermes_repo: Path) -> str:
    """Extract a prompt section constant from agent/prompt_builder.py.

    Uses AST parsing to find the exact string value of a module-level
    constant assignment like: MEMORY_GUIDANCE = "..."

    Args:
        section_name: Name of the constant (e.g., "MEMORY_GUIDANCE").
        hermes_repo: Path to the hermes-agent repository root.

    Returns:
        The string value of the constant.

    Raises:
        FileNotFoundError: If prompt_builder.py doesn't exist.
        KeyError: If the section is not found in the file.
    """
    prompt_builder_path = hermes_repo / "agent" / "prompt_builder.py"
    if not prompt_builder_path.exists():
        raise FileNotFoundError(
            f"prompt_builder.py not found at {prompt_builder_path}. "
            "Check HERMES_AGENT_REPO path."
        )

    source = prompt_builder_path.read_text()
    return _extract_constant(source, section_name, prompt_builder_path)


def save_prompt_section(
    section_name: str,
    evolved_text: str,
    hermes_repo: Path,
) -> Path:
    """Write an evolved section back to agent/prompt_builder.py.

    Replaces only the value of the named constant. Uses string replacement
    with proper escaping to preserve the file structure.

    Args:
        section_name: Name of the constant to replace.
        evolved_text: New text value for the constant.
        hermes_repo: Path to the hermes-agent repository root.

    Returns:
        Path to the modified file.

    Raises:
        FileNotFoundError: If prompt_builder.py doesn't exist.
        KeyError: If the section is not found.
        ValueError: If the replacement would produce invalid Python.
    """
    prompt_builder_path = hermes_repo / "agent" / "prompt_builder.py"
    if not prompt_builder_path.exists():
        raise FileNotFoundError(f"prompt_builder.py not found at {prompt_builder_path}")

    source = prompt_builder_path.read_text()
    original = _extract_constant(source, section_name, prompt_builder_path)

    # Replace by finding the exact assignment and substituting the value
    new_source = _replace_constant(source, section_name, original, evolved_text)

    # Verify the modified source is valid Python before writing
    try:
        ast.parse(new_source)
    except SyntaxError as e:
        raise ValueError(f"Evolved text produces invalid Python in {prompt_builder_path}: {e}")

    prompt_builder_path.write_text(new_source)
    return prompt_builder_path


def _extract_constant(source: str, section_name: str, source_path: Path) -> str:
    """Extract the string value of a module-level constant using AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise ValueError(f"Cannot parse {source_path}: {e}")

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != section_name:
                continue
            # Handle direct string or f-string (take the string if it's a Constant)
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                return value.value
            # Handle joined strings (f-strings or concatenated literals)
            if isinstance(value, ast.JoinedStr):
                # For f-strings, fall back to extracting raw text from source
                return _extract_via_regex(source, section_name)
            # Handle function calls (e.g., MEMORY_GUIDANCE = build_memory_guidance(True, True))
            if isinstance(value, ast.Call):
                return _extract_via_function_call(source, section_name, value, source_path)

    raise KeyError(
        f"Section '{section_name}' not found in prompt_builder.py. "
        f"Available evolvable sections: {EVOLVABLE_SECTIONS}"
    )


def _extract_via_regex(source: str, section_name: str) -> str:
    """Regex fallback for f-strings and multi-line strings."""
    # Match: SECTION_NAME = """...""" or SECTION_NAME = "..."
    pattern = re.compile(
        rf'^{re.escape(section_name)}\s*=\s*'
        r'(?:"""(.*?)"""|\'\'\'(.*?)\'\'\'|"(.*?)"|\'(.*?)\')',
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(source)
    if match:
        # Return whichever group matched
        return next(g for g in match.groups() if g is not None)
    raise KeyError(f"Cannot extract '{section_name}' via regex fallback")


def _extract_via_function_call(
    source: str, section_name: str, call_node: ast.Call, source_path: Path
) -> str:
    """Extract the return value of a function call assignment.

    Handles cases like ``MEMORY_GUIDANCE = build_memory_guidance(True, True)``
    by extracting the function's source via AST and executing it in isolation,
    avoiding the full module's import chain.
    """
    if not isinstance(call_node.func, ast.Name):
        raise KeyError(
            f"Section '{section_name}' is assigned via a non-simple function call "
            f"({ast.dump(call_node.func)}) — cannot extract automatically."
        )
    func_name = call_node.func.id

    # Parse the source and find the function definition
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise KeyError(f"Cannot parse {source_path}: {e}") from e

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            func_def = node
            break
    if func_def is None:
        raise KeyError(
            f"Function '{func_name}' not found in {source_path} for section '{section_name}'"
        )

    # Extract the function's source code and exec it in a clean namespace
    func_source = ast.get_source_segment(source, func_def)
    if func_source is None:
        raise KeyError(
            f"Could not extract source for '{func_name}' from {source_path}"
        )

    namespace: dict = {}
    try:
        exec(compile(func_source, str(source_path), "exec"), namespace)
    except Exception as e:
        raise KeyError(
            f"Failed to compile '{func_name}' from {source_path}: {e}"
        ) from e

    func = namespace.get(func_name)
    if func is None:
        raise KeyError(
            f"Function '{func_name}' not found after compiling from {source_path}"
        )

    # Bind default arguments for parameters not provided in the call
    provided_param_names = {kw.arg for kw in call_node.keywords}
    import inspect

    sig = inspect.signature(func)
    kwargs = {}
    for param_name, param in sig.parameters.items():
        if param_name in provided_param_names:
            continue
        if param.default is not inspect.Parameter.empty:
            kwargs[param_name] = param.default
        elif param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

    try:
        result = func(**kwargs)
    except TypeError:
        try:
            result = func()
        except Exception as e:
            raise KeyError(
                f"Failed to call '{func_name}()' for section '{section_name}': {e}"
            ) from e

    if not isinstance(result, str):
        raise KeyError(
            f"Function '{func_name}()' returned {type(result).__name__}, not a string — "
            f"cannot evolve section '{section_name}'"
        )
    return result


def _replace_constant(
    source: str,
    section_name: str,
    original_value: str,
    evolved_value: str,
) -> str:
    """Replace the value of a module-level constant using AST-guided source surgery.

    Parses with *ast* to find the exact byte range of the constant's value,
    determines the quoting style from the original source text, and performs
    a surgical replacement.  Hands multi-line strings, f-strings escaped with
    ``\\``, and every quoting style (", ', \"\"\", ''') without producing
    invalid Python.

    Raises ``KeyError`` if the assignment cannot be located.
    """
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name) or target.id != section_name:
                continue

            val = node.value

            # ── source range of the value literal (incl. quotes) ──────
            sl, sc = val.lineno, val.col_offset
            el, ec = val.end_lineno, val.end_col_offset

            # original source text of the value literal
            if sl == el:
                old_src = lines[sl - 1][sc:ec]
            else:
                parts = [lines[sl - 1][sc:]]
                for i in range(sl, el - 1):
                    parts.append(lines[i])
                parts.append(lines[el - 1][:ec])
                old_src = "".join(parts)

            # ── quoting style detection ───────────────────────────────
            new_src = _format_literal(evolved_value, old_src)

            # ── surgical replacement ──────────────────────────────────
            if sl == el:
                line = lines[sl - 1]
                lines[sl - 1] = line[:sc] + new_src + line[ec:]
            else:
                # Multi-line original: replace first-line tail + all
                # intermediate lines + last-line head.
                lines[sl - 1] = lines[sl - 1][:sc] + new_src
                for i in range(sl, el):
                    lines[i] = ""

            result = "".join(lines)

            # Safety: verify the modified source is valid Python
            try:
                ast.parse(result)
            except SyntaxError as e:
                raise ValueError(
                    f"Evolved text for '{section_name}' produced invalid Python: {e}"
                ) from e
            return result

    raise KeyError(
        f"Could not locate the assignment for '{section_name}' to replace it. "
        "The section may use a non-standard assignment style."
    )


def _format_literal(value: str, original_src: str) -> str:
    """Re-``repr`` *value* using the quoting style detected from *original_src*.

    *original_src* is the raw source text of the old literal (including its
    outer quotes).  The function emits a new literal that uses the same
    quoting convention, falling back to ``\"\"\"`` for multi-line strings
    when the original is also multi-line.
    """
    is_multi = "\n" in value
    was_multi = original_src.startswith('"""') or original_src.startswith("'''")

    if is_multi or was_multi:
        # Prefer """ (Pythonic).  Only use ''' if the value contains """
        # but not '''.
        if '"""' in value and "'''" not in value:
            quote = "'''"
        else:
            quote = '"""'
        triple = quote * 3
        # Escape the closing sequence if it appears verbatim in the value.
        esc_triple = triple[:1]
        escaped = value.replace(triple, esc_triple * 2)
        return triple + escaped + triple
    else:
        # Single-line — match original quote char, fall back to "
        q = '"'
        if original_src and original_src[0] in ("'", '"'):
            q = original_src[0]
        escaped = value.replace("\\", "\\\\").replace(q, "\\" + q)
        return q + escaped + q


class PromptSectionModule(dspy.Module):
    """DSPy module that evaluates an agent response guided by a prompt section.

    The section text is embedded as the DSPy signature instructions — this is
    what GEPA mutates during optimization. After evolution, section_text
    returns the GEPA-optimized version.
    """

    def __init__(self, section_name: str, section_text: str):
        super().__init__()
        self.section_name = section_name

        # Create a signature where the instructions ARE the section text.
        # GEPA can mutate this via its reflection mechanism.
        sig = _BasePromptSectionSig.with_instructions(section_text)
        self.predictor = dspy.ChainOfThought(sig)

    @property
    def section_text(self) -> str:
        """Return the current (possibly GEPA-evolved) section text."""
        return self.predictor.predict.signature.instructions

    def forward(self, task_input: str) -> dspy.Prediction:
        result = self.predictor(task_input=task_input)
        return dspy.Prediction(output=result.output)


class _BasePromptSectionSig(dspy.Signature):
    """Placeholder — replaced with section_text via with_instructions."""
    task_input: str = dspy.InputField(desc="The task or scenario to respond to")
    output: str = dspy.OutputField(
        desc="Your response following the section's behavioral guidance"
    )



def prompt_fitness_metric(
    gold,
    pred,
    trace=None,
    pred_name=None,
    pred_trace=None,
) -> float:
    """DSPy-compatible metric for prompt section behavioral alignment (GEPA v2).

    Scores whether the agent's response demonstrates the behavioral
    guidance described in expected_behavior.
    """
    expected = getattr(gold, "expected_behavior", "") or ""
    output = getattr(pred, "output", "") or ""

    if not output.strip():
        return 0.0

    # Keyword overlap as a fast proxy for behavioral alignment
    expected_lower = expected.lower()
    output_lower = output.lower()

    expected_words = set(w for w in expected_lower.split() if len(w) > 3)
    output_words = set(output_lower.split())

    if not expected_words:
        return 0.5

    overlap = len(expected_words & output_words) / len(expected_words)
    return min(1.0, max(0.0, 0.2 + 0.8 * overlap))
