"""Grader-free pairwise comparison (Harness²-style) for self-evolution.

For each eval example we hold a BASELINE output and a CANDIDATE output and ask
a judge which of two *anonymized* responses ("Response A" / "Response B") better
accomplishes the task.  Which artifact sits in slot A is randomized (seeded);
with ``repeats >= 2`` both orders are always run, and a position-inconsistent
pair (candidate wins in one order, loses in the other) is scored as a tie.

Per-example candidate score: win=1, tie=0.5, loss=0, averaged over orders.
Judge failures (unparsable twice) are *excluded* (``score=None``), never zero.

All LLM traffic goes through :class:`LLMClient`, which counts calls against a
shared :class:`CallBudget` so a whole run can be hard-capped (``--max-calls``).
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)


# ── Call budget ─────────────────────────────────────────────────────────────


class BudgetExceeded(RuntimeError):
    """Raised when the run-wide LLM call cap would be exceeded."""


@dataclass
class CallBudget:
    max_calls: int = 400
    used: int = 0
    by_role: dict = field(default_factory=dict)

    @property
    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def spend(self, role: str = "other", n: int = 1) -> None:
        if self.used + n > self.max_calls:
            raise BudgetExceeded(f"LLM call cap reached ({self.used}/{self.max_calls})")
        self.used += n
        self.by_role[role] = self.by_role.get(role, 0) + n

    def summary(self) -> dict:
        return {"used": self.used, "max": self.max_calls, "by_role": dict(self.by_role)}


# ── LLM client ──────────────────────────────────────────────────────────────


def _resolve_role_model(role: str) -> str:
    """Model string for a role. Judge prefers OmniRoute combo `judge`."""
    from evolution.core.llm_api import resolve_model, _to_omniroute_model  # lazy

    provider = os.getenv("HERMES_LLM_PROVIDER", "omniroute").strip().lower()
    env_override = os.getenv({"judge": "HERMES_JUDGE_MODEL", "eval": "HERMES_EVAL_MODEL",
                              "optimizer": "HERMES_OPTIMIZER_MODEL"}.get(role, ""), "").strip()
    model = resolve_model(role if role in ("judge", "eval", "optimizer") else "eval",
                          override=env_override or None)
    if provider == "omniroute":
        if role == "judge" and not env_override:
            model = "openai/judge"  # dedicated combo; llm_api maps judge->eval by default
        model = _to_omniroute_model(model)
    return model


def _extract_text(out) -> str:
    """dspy.LM returns list[str] or list[dict(text=..., reasoning_content=...)]."""
    if isinstance(out, list):
        out = out[0] if out else ""
    if isinstance(out, dict):
        return str(out.get("text") or out.get("content") or "")
    return str(out or "")


class LLMClient:
    """Thin chat wrapper with role-based model resolution and call budgeting."""

    def __init__(
        self,
        budget: CallBudget,
        role: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        model: Optional[str] = None,
        fallback_role: Optional[str] = None,
        cache: bool = True,
    ):
        self.budget = budget
        self.role = role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model or _resolve_role_model(role)
        self.fallback_role = fallback_role
        self.cache = cache
        self._lm = None
        self._fallback_lm = None

    def _make(self, model: str):
        import dspy

        return dspy.LM(model, temperature=self.temperature, max_tokens=self.max_tokens,
                       cache=self.cache, num_retries=2)

    def chat(self, user: str, system: Optional[str] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        self.budget.spend(self.role)
        if self._lm is None:
            self._lm = self._make(self.model)
        try:
            return _extract_text(self._lm(messages=messages))
        except BudgetExceeded:
            raise
        except Exception as e:
            if not self.fallback_role:
                raise
            logger.warning("%s model %s failed (%s); falling back to %s",
                           self.role, self.model, e, self.fallback_role)
            if self._fallback_lm is None:
                self._fallback_lm = self._make(_resolve_role_model(self.fallback_role))
            self.budget.spend(self.fallback_role)
            return _extract_text(self._fallback_lm(messages=messages))

    __call__ = chat


ChatFn = Callable[[str], str]


# ── Verdict parsing ─────────────────────────────────────────────────────────

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S | re.I)
_OBJ = re.compile(r"\{[^{}]*\}", re.S)
_CHOICE = re.compile(r"```choice\s*\n?\s*([A-Za-z]+)\s*\n?\s*```", re.S | re.I)
_KV = re.compile(r"[\"']?(?:winner|verdict|choice|better)[\"']?\s*[:=]\s*[\"']?\s*(A|B|TIE|EQUAL)\b",
                 re.I)


def _norm(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().strip("\"'.").upper()
    s = s.replace("RESPONSE ", "")
    if s in ("A", "B"):
        return s
    if s in ("TIE", "EQUAL", "DRAW", "SAME", "NEITHER", "BOTH"):
        return "TIE"
    return None


def parse_verdict(text: str) -> Optional[str]:
    """Return 'A', 'B', 'TIE' or None. Tolerates fences, think blocks, prose."""
    if not text:
        return None
    t = _THINK.sub("", text)
    candidates = [m.group(1) for m in _FENCE.finditer(t)] + _OBJ.findall(t)
    for blob in reversed(candidates):  # last JSON object wins
        try:
            obj = json.loads(blob)
        except ValueError:
            continue
        if isinstance(obj, dict):
            for k in ("winner", "verdict", "choice", "better"):
                if k in obj and _norm(obj[k]):
                    return _norm(obj[k])
    m = _CHOICE.search(t)
    if m and _norm(m.group(1)):
        return _norm(m.group(1))
    kvs = _KV.findall(t)
    if kvs:
        return _norm(kvs[-1])
    last = t.strip().splitlines()[-1] if t.strip() else ""
    return _norm(last)


# ── Judge ───────────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial evaluator. Two anonymous assistant responses were produced for the same task. Decide which one better accomplishes the task for the user: correct, actionable, follows sound procedure, no fabrication, appropriately concise. Ignore length and style unless they affect usefulness. Do not favour a response because of its position.

TASK:
{task}
{rubric}
=== Response A ===
{a}
=== End A ===

=== Response B ===
{b}
=== End B ===

Reply with ONLY a JSON object, no other text:
{{"winner": "A" | "B" | "tie", "reason": "<one short sentence>"}}"""

_MAX_RESP_CHARS = 6000


def _clip(s: str, n: int = _MAX_RESP_CHARS) -> str:
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + "\n…[truncated]…\n" + s[-n // 2:]


@dataclass
class PairResult:
    example_id: str
    score: Optional[float]  # candidate score in [0,1]; None = judge failed
    order_verdicts: list = field(default_factory=list)  # [(cand_slot, verdict), ...]
    position_inconsistent: bool = False
    judge_failures: int = 0

    @property
    def verdict(self) -> int:
        """Legacy: 1 = candidate better, 0 = tie, -1 = baseline better."""
        if self.score is None or self.score == 0.5:
            return 0
        return 1 if self.score > 0.5 else -1


@dataclass
class PairwiseStats:
    n_examples: int
    wins: int
    ties: int
    losses: int
    order_inconsistent: int
    judge_failures: int = 0

    @property
    def win_rate(self) -> float:
        n = self.wins + self.ties + self.losses
        return 0.5 if n == 0 else (self.wins + 0.5 * self.ties) / n


def summarize(results: Sequence[PairResult]) -> PairwiseStats:
    valid = [r for r in results if r.score is not None]
    return PairwiseStats(
        n_examples=len(valid),
        wins=sum(1 for r in valid if r.score > 0.5),
        ties=sum(1 for r in valid if r.score == 0.5),
        losses=sum(1 for r in valid if r.score < 0.5),
        order_inconsistent=sum(1 for r in results if r.position_inconsistent),
        judge_failures=sum(1 for r in results if r.score is None),
    )


class PairwiseJudge:
    """Anonymized, order-randomized pairwise judge."""

    def __init__(self, chat: ChatFn, repeats: int = 2, seed: int = 0,
                 prefilter: Optional[Callable[[str, str, str], Optional[str]]] = None):
        self.chat = chat
        self.repeats = max(1, int(repeats))
        self.rng = random.Random(seed)
        self.prefilter = prefilter  # optional cheap tie filter: returns 'TIE' to short-circuit

    def _ask(self, task: str, rubric: str, a: str, b: str) -> Optional[str]:
        prompt = JUDGE_PROMPT.format(
            task=_clip(task, 3000),
            rubric=(f"\nREFERENCE NOTES (what a good answer covers; not a strict checklist):\n{_clip(rubric, 1500)}\n"
                    if rubric else ""),
            a=_clip(a), b=_clip(b),
        )
        for _attempt in range(2):  # retry once on unparsable
            try:
                raw = self.chat(prompt)
            except BudgetExceeded:
                raise
            except Exception as e:  # network/provider failure → treat as unparsable
                logger.warning("judge call failed: %s", e)
                raw = ""
            v = parse_verdict(raw)
            if v is not None:
                return v
            logger.info("unparsable judge reply (attempt %d): %r", _attempt + 1, (raw or "")[:200])
        return None

    def compare(self, example_id: str, task: str, rubric: str,
                baseline: str, candidate: str) -> PairResult:
        if (baseline or "").strip() == (candidate or "").strip():
            return PairResult(example_id, 0.5, [("identical", "TIE")])
        if self.prefilter is not None:
            try:
                if self.prefilter(task, baseline, candidate) == "TIE":
                    return PairResult(example_id, 0.5, [("prefilter", "TIE")])
            except Exception as e:  # prefilter is best-effort
                logger.debug("prefilter failed: %s", e)

        first = self.rng.choice(["A", "B"])  # candidate's slot in first order
        slots = [first if i % 2 == 0 else ("B" if first == "A" else "A") for i in range(self.repeats)]
        order_scores, verdicts, failures = [], [], 0
        for cand_slot in slots:
            a, b = (candidate, baseline) if cand_slot == "A" else (baseline, candidate)
            v = self._ask(task, rubric, a, b)
            verdicts.append((cand_slot, v))
            if v is None:
                failures += 1
                continue
            order_scores.append(0.5 if v == "TIE" else (1.0 if v == cand_slot else 0.0))
        if not order_scores:
            return PairResult(example_id, None, verdicts, False, failures)
        inconsistent = 1.0 in order_scores and 0.0 in order_scores
        score = 0.5 if inconsistent else sum(order_scores) / len(order_scores)
        return PairResult(example_id, score, verdicts, inconsistent, failures)

    def evaluate(self, examples: Sequence[dict], baseline_outputs: Sequence[str],
                 candidate_outputs: Sequence[str]) -> list[PairResult]:
        out = []
        for i, (ex, b, c) in enumerate(zip(examples, baseline_outputs, candidate_outputs)):
            eid = str(ex.get("id") or ex.get("session_id") or i)
            try:
                out.append(self.compare(eid, ex.get("task_input", ""), ex.get("expected_behavior", ""), b, c))
            except BudgetExceeded:
                logger.warning("budget exhausted during pairwise eval after %d/%d", i, len(examples))
                break
        return out


# ── Optional System One tie prefilter (off by default) ─────────────────────

SYSTEMONE_URL = os.getenv("SYSTEMONE_URL", "http://127.0.0.1:11436/v1/systemone")


def systemone_verdict(task: str, a: str, b: str, timeout: float = 20.0) -> Optional[str]:
    """Ask the cheap System One router for A/B/tie. Returns None on failure."""
    import httpx

    state = f"TASK:\n{_clip(task, 1500)}\n\nRESPONSE A:\n{_clip(a, 2500)}\n\nRESPONSE B:\n{_clip(b, 2500)}"
    body = {"state": state, "questions": {"better": {
        "type": "choice",
        "instructions": "Which response better accomplishes the task? Answer tie if they are about equally good.",
        "criteria": {"A": None, "B": None, "tie": None}}}}
    try:
        r = httpx.post(SYSTEMONE_URL, json=body, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.debug("systemone failed: %s", e)
        return None
    ans = data.get("answers", data).get("better") if isinstance(data, dict) else None
    if isinstance(ans, dict):
        ans = ans.get("choice") or ans.get("answer") or ans.get("value")
    return _norm(ans)


def calibrate_agreement(pairs: Sequence[tuple[str, str, str]], judge_verdicts: Sequence[Optional[str]],
                        cheap: Callable[[str, str, str], Optional[str]] = systemone_verdict) -> dict:
    """Agreement of a cheap verdict fn with the judge on the same (task, a, b) pairs."""
    agree = n = 0
    for (task, a, b), jv in zip(pairs, judge_verdicts):
        if jv is None:
            continue
        cv = cheap(task, a, b)
        if cv is None:
            continue
        n += 1
        agree += int(cv == jv)
    return {"n": n, "agree": agree, "agreement": (agree / n) if n else 0.0,
            "enable": n >= 12 and n > 0 and agree / n >= 0.75}
