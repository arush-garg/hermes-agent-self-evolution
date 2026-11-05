"""Turn GEPA's evolved selector instructions back into per-tool descriptions.

GEPA mutates the selector's instruction text, which starts as the tools JSON. Its reflection LM
usually rewrites that into prose routing rules, so the output is often no longer JSON. The rules
are only useful if they end up in the tool descriptions Hermes actually ships, so:

1. if the evolved text still parses as the tools JSON, take its descriptions directly;
2. otherwise a distill LM folds the rules into the descriptions of the tools they concern.

Either way the result is sanitised against the baseline: only known tool names, parameters
frozen, empty or constraint-failing descriptions revert to the original. The holdout gate then
scores *these* descriptions, so what is measured is exactly what would be applied.
"""
import json
import re
from dataclasses import replace
from typing import Callable, Optional

import dspy

from evolution.tools.tool_module import ToolDescription, format_tools_for_prompt

DISTILL_BATCH = 10  # tools per distill call keeps each JSON answer short


def parse_tools_json(text: str) -> Optional[dict[str, str]]:
    """``{name: description}`` if ``text`` (optionally fenced / wrapped in prose) holds a JSON
    array of tool objects; ``None`` when it does not."""
    if not isinstance(text, str):
        return None
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))
    bracketed = re.search(r"\[.*\]", text, re.DOTALL)
    if bracketed:
        candidates.append(bracketed.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(data, list):
            found = {d["name"]: d.get("description", "") for d in data
                     if isinstance(d, dict) and isinstance(d.get("name"), str)}
            if found:
                return found
    return None


class DistillToolDescriptions(dspy.Signature):
    """Rewrite tool descriptions so they carry the routing guidance an optimizer learned.

    `guidance` was optimized to make an agent pick the right tool. Fold whatever it says about
    a tool (when to use it, when another tool is better) into that tool's description. Keep
    each description factual, self-contained, and about as long as the original (at most
    roughly double). Return only tools whose description should change; return [] if none.
    """
    tools_json: str = dspy.InputField(desc="JSON array of {name, description} for the tools to consider")
    guidance: str = dspy.InputField(desc="Optimized tool-selection instructions")
    revised: str = dspy.OutputField(desc='JSON array of {"name": str, "description": str}; only changed tools')


def distill_descriptions(tools: list[ToolDescription], guidance: str,
                         predictor: Optional[Callable] = None) -> dict[str, str]:
    """``{name: new_description}`` folded from prose ``guidance``. A batch whose answer does not
    parse contributes nothing (its tools keep their baseline descriptions)."""
    predictor = predictor or dspy.Predict(DistillToolDescriptions)
    revised: dict[str, str] = {}
    for i in range(0, len(tools), DISTILL_BATCH):
        batch = tools[i:i + DISTILL_BATCH]
        tools_json = json.dumps([{"name": t.name, "description": t.description} for t in batch], indent=2)
        try:
            answer = predictor(tools_json=tools_json, guidance=guidance).revised
        except Exception:  # one bad LM call must not sink the whole run
            continue
        names = {t.name for t in batch}
        revised.update({k: v for k, v in (parse_tools_json(answer) or {}).items() if k in names})
    return revised


def sanitize(baseline: list[ToolDescription], proposed: dict[str, str],
             is_valid: Callable[[str, str], bool]) -> list[ToolDescription]:
    """Baseline tools with valid proposed descriptions swapped in. Unknown names are ignored,
    parameters and source locations are kept, invalid or blank text keeps the original."""
    result = []
    for tool in baseline:
        new = proposed.get(tool.name)
        if isinstance(new, str) and new.strip() and new.strip() != tool.description \
                and is_valid(new.strip(), tool.description):
            result.append(replace(tool, description=new.strip()))
        else:
            result.append(tool)
    return result


def evolved_tools_from_instructions(baseline: list[ToolDescription], evolved_text: str,
                                    is_valid: Callable[[str, str], bool],
                                    predictor: Optional[Callable] = None) -> tuple[list[ToolDescription], str]:
    """``(tools, how)`` where ``how`` is ``"unchanged"``, ``"json"`` or ``"distilled"``."""
    if not evolved_text or evolved_text.strip() == format_tools_for_prompt(baseline).strip():
        return list(baseline), "unchanged"
    parsed = parse_tools_json(evolved_text)
    if parsed is not None:
        return sanitize(baseline, parsed, is_valid), "json"
    return sanitize(baseline, distill_descriptions(baseline, evolved_text, predictor), is_valid), "distilled"


def changed_names(baseline: list[ToolDescription], evolved: list[ToolDescription]) -> list[str]:
    before = {t.name: t.description for t in baseline}
    return [t.name for t in evolved if before.get(t.name) != t.description]
