"""Propose → Probe → Compose (Harness²-style) candidate search.

1. PROPOSE  K candidate edits of an artifact body, each with a different
            instruction ("general/invariant", "domain-specific",
            "seeded from lessons/failure context", ...).
2. PROBE    each candidate pairwise vs the baseline on a small VAL slice.
3. COMPOSE  the top-2 probed candidates into one composite via an LLM merge
            prompt; probe the composite too.
4. FINAL    best of {composite, top single} — only if it beat the baseline on
            val (baseline always stays a contender) — is evaluated on HOLDOUT
            with the pairwise judge, then gated by :mod:`stats_gate`.

The keyword-overlap metric is not used for the keep/reject decision anywhere.
All calls go through a shared :class:`CallBudget`.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from evolution.core.pairwise import (BudgetExceeded, CallBudget, PairResult, PairwiseJudge,
                                     rubric_for, summarize)
from evolution.core import stats_gate

logger = logging.getLogger(__name__)

ChatFn = Callable[..., str]

STRATEGIES = [
    ("general", "Make GENERAL, task-invariant improvements: clearer step ordering, explicit "
                "verification steps, sharper decision rules, removing ambiguity. Do not add "
                "content specific to individual example tasks."),
    ("domain", "Make DOMAIN-SPECIFIC improvements: add the concrete commands, flags, pitfalls and "
               "edge cases that a practitioner of this skill's domain would need for tasks like "
               "the examples."),
    ("lessons", "Fix the FAILURE MODES and apply the LESSONS listed below. Each change should "
                "address a specific observed failure or lesson; keep everything else intact."),
    ("concise", "Make the instructions more CONCISE and scannable without losing any procedure: "
                "merge redundant sections, convert prose to checklists, keep all commands."),
]


@dataclass
class Candidate:
    name: str
    strategy: str
    body: str
    probe_results: list = field(default_factory=list)
    probe_score: Optional[float] = None
    error: Optional[str] = None


@dataclass
class ComposeResult:
    baseline_body: str
    candidates: list
    composite: Optional[Candidate]
    winner: Optional[Candidate]  # None => baseline kept
    holdout_results: list
    gate: stats_gate.GateResult
    noise: Optional[dict]
    budget: dict
    notes: list = field(default_factory=list)

    @property
    def final_text(self) -> str:
        return self.winner.body if self.winner else self.baseline_body


# ── Context helpers (lessons / failure context) ────────────────────────────


def load_failure_context(max_chars: int = 2000) -> str:
    p = os.getenv("EVOLUTION_FAILURE_CONTEXT", "").strip()
    if not p:
        return ""
    try:
        with open(p) as f:
            raw = f.read()
    except OSError:
        return ""
    try:
        raw = json.dumps(json.loads(raw), indent=1)
    except ValueError:
        pass
    return raw[:max_chars]


def load_lessons(task_classes: Sequence[str], k: int = 3) -> list[dict]:
    try:
        from evolution.core import lesson_library
    except ImportError:
        return []
    seen, out = set(), []
    for tc in task_classes:
        if not tc:
            continue
        try:
            rows = lesson_library.retrieve(tc, k=k)
        except Exception as e:
            logger.debug("lesson retrieve failed for %s: %s", tc, e)
            continue
        for r in rows:
            key = (r.get("cue"), r.get("lesson"))
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out[: k * 2]


def record_lesson(task_class: str, cue: str, lesson: str, evidence: str) -> Optional[dict]:
    try:
        from evolution.core import lesson_library
        return lesson_library.append(task_class, cue, lesson, evidence)
    except Exception as e:  # ImportError or storage problem; never fatal
        logger.warning("could not record lesson: %s", e)
        return None


# ── Prompts ─────────────────────────────────────────────────────────────────

PROPOSE_PROMPT = """You are improving an instruction document ("{kind}") that an AI agent reads before doing tasks.

STRATEGY: {instruction}

CURRENT DOCUMENT:
<document>
{body}
</document>

EXAMPLE TASKS the agent will face (with notes on what a good response covers):
{examples}
{lessons}{failures}
Rules:
- Keep the document's purpose and scope; preserve every correct command/path/fact.
- Do not invent tools, flags or facts you are unsure of.
- Stay within ~{max_growth}% of the current length.
- Output the COMPLETE revised document between <document> and </document> tags, then one line starting with CHANGES: summarizing what you changed."""

MERGE_PROMPT = """Two improved versions of the same instruction document ("{kind}") were produced independently. Both beat the original on some tasks. Merge them into ONE document that keeps the best, non-conflicting improvements of each. Where they conflict, prefer the more precise/verifiable instruction. Do not add new content beyond what either version contains.

ORIGINAL:
<document>
{base}
</document>

VERSION 1:
<document>
{a}
</document>

VERSION 2:
<document>
{b}
</document>

Stay within ~{max_growth}% of the ORIGINAL length. Output the COMPLETE merged document between <document> and </document> tags."""

_DOC = re.compile(r"<document>\s*(.*?)\s*</document>", re.S)
_FENCED = re.compile(r"```(?:markdown|md)?\s*\n(.*?)\n```", re.S)


def extract_document(text: str) -> Optional[str]:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    m = _DOC.findall(text)
    if m:
        doc = max(m, key=len).strip()
        return doc or None
    m = _FENCED.findall(text)
    if m:
        doc = max(m, key=len).strip()
        return doc if len(doc) > 50 else None
    return None


def _fmt_examples(examples: Sequence[dict], n: int = 4) -> str:
    lines = []
    for ex in list(examples)[:n]:
        lines.append(f"- TASK: {str(ex.get('task_input', ''))[:400]}\n  GOOD RESPONSE: "
                     f"{rubric_for(ex)[:300]}")
    return "\n".join(lines) or "(none)"


def _fmt_lessons(lessons: Sequence[dict]) -> str:
    if not lessons:
        return ""
    rows = [f"- When {l.get('cue', '?')}: {l.get('lesson', '')}" for l in lessons]
    return "\nLESSONS FROM PAST RUNS:\n" + "\n".join(rows) + "\n"


def _fmt_failures(ex_failures: Sequence[dict], failure_ctx: str) -> str:
    parts = []
    for ex in ex_failures[:4]:
        sig = ", ".join(ex.get("failure_signals") or [])
        parts.append(f"- {str(ex.get('task_input', ''))[:200]} (outcome={ex.get('outcome')}; {sig})")
    s = ""
    if parts:
        s += "\nOBSERVED FAILURES ON REAL TASKS:\n" + "\n".join(parts) + "\n"
    if failure_ctx:
        s += "\nRECENT FAILURE CONTEXT (mined from sessions):\n" + failure_ctx[:2000] + "\n"
    return s


# ── Agent output generation ─────────────────────────────────────────────────

AGENT_SYSTEM = """You are Hermes, an AI agent with shell, file and web tools. The following {kind} is loaded into your context; follow it.

<{tag}>
{body}
</{tag}>

You cannot execute tools in this evaluation; instead describe exactly what you would do (commands, checks, and the final answer) as you would in a real session. Be concrete."""


def generate_outputs(chat: ChatFn, body: str, examples: Sequence[dict], kind: str = "skill",
                     cache: Optional[dict] = None) -> list[str]:
    """One agent response per example with ``body`` loaded. Budget errors propagate."""
    cache = cache if cache is not None else {}
    system = AGENT_SYSTEM.format(kind=kind, tag=kind.replace(" ", "_"), body=body)
    outs = []
    for ex in examples:
        user = str(ex.get("task_input", ""))
        if ex.get("tool_trace_summary"):
            user += f"\n\n(Context from the original session: {str(ex['tool_trace_summary'])[:500]})"
        key = (hash(body), user)
        if key not in cache:
            try:
                cache[key] = chat(user, system=system)
            except BudgetExceeded:
                raise
            except Exception as e:
                logger.warning("agent generation failed: %s", e)
                cache[key] = ""
        outs.append(cache[key])
    return outs


# ── Pipeline ────────────────────────────────────────────────────────────────


def _within_growth(new: str, base: str, max_growth: float) -> bool:
    return len(new) <= len(base) * (1 + max_growth) + 200 and len(new) >= 0.3 * len(base)


def _pick_strategies(k: int, have_seed_material: bool) -> list:
    pool = list(STRATEGIES) if have_seed_material else [s for s in STRATEGIES if s[0] != "lessons"]
    out = pool[:k]
    i = 0
    while len(out) < k:
        out.append((f"general-{i}", STRATEGIES[0][1]))
        i += 1
    return out


def propose_probe_compose(
    *,
    baseline_body: str,
    train: Sequence[dict],
    val: Sequence[dict],
    holdout: Sequence[dict],
    agent_chat: ChatFn,
    optimizer_chat: ChatFn,
    judge: PairwiseJudge,
    budget: CallBudget,
    k: int = 3,
    kind: str = "skill",
    lessons: Sequence[dict] = (),
    failure_context: str = "",
    max_growth: float = 0.2,
    min_effect: float = 0.05,
    min_pairs: int = 8,
    seed: int = 0,
    noise_agent_chat: Optional[ChatFn] = None,
    validate: Optional[Callable[[str], Optional[str]]] = None,
    log: Callable[[str], None] = print,
) -> ComposeResult:
    notes: list = []
    gen_cache: dict = {}
    failures = [e for e in list(train) + list(val)
                if e.get("outcome") in ("failed", "corrected") or e.get("failure_signals")]

    def _gen(body, exs):
        return generate_outputs(agent_chat, body, exs, kind, gen_cache)

    def _probe(c: Candidate, base_val_outs):
        outs = _gen(c.body, val)
        c.probe_results = judge.evaluate(val, base_val_outs, outs)
        st = summarize(c.probe_results)
        c.probe_score = st.win_rate if st.n_examples else None

    def _accept(text: Optional[str], label: str) -> Optional[str]:
        if not text:
            return f"{label}: no <document> in reply"
        if text.strip() == baseline_body.strip():
            return f"{label}: identical to baseline"
        if not _within_growth(text, baseline_body, max_growth):
            return f"{label}: size {len(text)} vs baseline {len(baseline_body)} violates growth bounds"
        if validate:
            return validate(text)
        return None

    candidates: list[Candidate] = []
    composite = None
    winner = None
    holdout_results: list[PairResult] = []
    noise = None
    try:
        base_val = _gen(baseline_body, val)
        # ── PROPOSE ──
        for name, instr in _pick_strategies(k, bool(lessons or failures or failure_context)):
            prompt = PROPOSE_PROMPT.format(
                kind=kind, instruction=instr, body=baseline_body,
                examples=_fmt_examples(train or val),
                lessons=_fmt_lessons(lessons),
                failures=_fmt_failures(failures, failure_context) if name == "lessons" else "",
                max_growth=int(max_growth * 100))
            try:
                raw = optimizer_chat(prompt)
            except BudgetExceeded:
                raise
            except Exception as e:
                candidates.append(Candidate(name, instr, "", error=f"proposal failed: {e}"))
                log(f"  proposed [{name}]: FAILED {e}")
                continue
            doc = extract_document(raw)
            err = _accept(doc, name)
            candidates.append(Candidate(name, instr, doc or "", error=err))
            log(f"  proposed [{name}]: {len(doc or '')} chars" + (f" — rejected: {err}" if err else ""))

        # ── PROBE ──
        live = [c for c in candidates if not c.error]
        for c in live:
            _probe(c, base_val)
            st = summarize(c.probe_results)
            if c.probe_score is None:
                log(f"  probe [{c.name}] no valid judgments")
            else:
                log(f"  probe [{c.name}] val win-rate={c.probe_score:.3f} "
                    f"(W{st.wins}/T{st.ties}/L{st.losses}, order-inconsistent={st.order_inconsistent})")
        ranked = sorted([c for c in live if c.probe_score is not None],
                        key=lambda c: c.probe_score, reverse=True)

        # ── COMPOSE ──
        if len(ranked) >= 2 and ranked[1].probe_score > 0.5:
            a, b = ranked[0], ranked[1]
            try:
                raw = optimizer_chat(MERGE_PROMPT.format(kind=kind, base=baseline_body, a=a.body,
                                                         b=b.body, max_growth=int(max_growth * 100)))
                doc = extract_document(raw)
                err = _accept(doc, "composite")
                composite = Candidate(f"composite({a.name}+{b.name})", "merge", doc or "", error=err)
                if not err:
                    _probe(composite, base_val)
                    log(f"  probe [{composite.name}] val win-rate={composite.probe_score}")
                else:
                    log(f"  composite rejected: {err}")
            except BudgetExceeded:
                raise
            except Exception as e:
                notes.append(f"compose failed: {e}")
        else:
            notes.append("compose skipped: fewer than 2 candidates beat baseline on val")
            log("  compose skipped: fewer than 2 candidates beat baseline on val")

        contenders = [c for c in ([composite] if composite else []) + ranked[:1]
                      if not c.error and c.probe_score is not None]
        best = max(contenders, key=lambda c: c.probe_score, default=None)
        if best is None or best.probe_score <= 0.5:
            notes.append("baseline kept: no candidate beat baseline on val probe")
            log("  baseline remains best on val — skipping holdout candidate eval")
        else:
            winner = best
            # ── FINAL: holdout ──
            log(f"  holdout eval of [{best.name}] on {len(holdout)} examples")
            base_h = _gen(baseline_body, holdout)
            cand_h = _gen(best.body, holdout)
            holdout_results = judge.evaluate(holdout, base_h, cand_h)
            if noise_agent_chat is not None:
                n_noise = min(len(holdout), max(min_pairs, 4))
                try:
                    alt = generate_outputs(noise_agent_chat, baseline_body, holdout[:n_noise], kind, {})
                    noise = stats_gate.noise_check(judge.evaluate(holdout[:n_noise], base_h[:n_noise], alt))
                except BudgetExceeded:
                    notes.append("noise check skipped: budget")
    except BudgetExceeded as e:
        notes.append(str(e))
        log(f"  [budget] {e}")

    gate = stats_gate.decide(holdout_results, min_effect=min_effect, min_pairs=min_pairs, seed=seed)
    if winner is None:
        gate.decision = "reject"
        gate.reasons.append("no candidate beat baseline on val")
    gate.noise_check = noise
    # The baseline-vs-baseline noise check must GATE promotion, not just be reported:
    # if the judge "finds" an effect between two samples of the same artifact, a candidate
    # effect of similar size is indistinguishable from judge/sampling noise.
    if gate.decision == "promote":
        effect = (gate.mean if getattr(gate, "mean", None) is not None else 0.5) - 0.5
        if noise is None:
            gate.decision = "stage"
            gate.reasons.append("noise check unavailable: cannot rule out judge noise -> stage")
        elif noise.get("would_promote") or effect < noise.get("abs_effect", 0.0) + min_effect:
            gate.decision = "stage"
            gate.reasons.append(
                f"effect {effect:+.3f} not above noise {noise.get('abs_effect', 0.0):.3f} + "
                f"min_effect {min_effect:.2f} (baseline-vs-baseline mean {noise.get('mean', 0.5):.3f}) -> stage")
    return ComposeResult(baseline_body, candidates, composite, winner, holdout_results, gate,
                         noise, budget.summary(), notes)
