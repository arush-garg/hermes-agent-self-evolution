"""Central LLM routing layer for hermes-agent-self-evolution.

All DSPy LM calls go through get_lm() so you can swap providers,
models, and credentials in one place — without touching call sites.

──────────────────────────────────────────────────────────────────
QUICK PROVIDER SWITCH
──────────────────────────────────────────────────────────────────
Set env vars before running any evolution command:

  # Switch everything to Omniroute (local proxy — handles auth + fallback)
  export HERMES_LLM_PROVIDER=omniroute

  # Switch everything to OpenRouter
  export HERMES_LLM_PROVIDER=openrouter
  export OPENROUTER_API_KEY=sk-or-v1-...

  # Use a specific model for one role
  export HERMES_OPTIMIZER_MODEL=openrouter/anthropic/claude-opus-4-5
  export HERMES_EVAL_MODEL=openrouter/google/gemini-2.5-flash
  export HERMES_JUDGE_MODEL=openrouter/meta-llama/llama-3.3-70b-instruct

  # Or override a single role without changing provider
  export HERMES_EVAL_MODEL=openai/gpt-4o-mini

Env vars take priority over EvolutionConfig fields, which take
priority over the provider defaults defined in PROVIDER_DEFAULTS.
──────────────────────────────────────────────────────────────────
"""

import os
import functools
from typing import Literal, Optional

import httpx
import dspy


# ── Provider defaults ─────────────────────────────────────────────────────────
# Edit this dict to change what "anthropic", "openrouter", etc. resolve to.
# Keys are provider names (HERMES_LLM_PROVIDER values).
# Each entry maps role → LiteLLM model string.

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {
        "optimizer": "anthropic/claude-opus-4-8",
        "eval":      "anthropic/claude-haiku-4-5-20251001",
        "judge":     "anthropic/claude-sonnet-4-6",
    },
    "omniroute": {
        # Model IDs understood by the local omniroute proxy.
        # Sent via LiteLLM's openai/ provider using OPENAI_API_BASE=http://localhost:20128/v1.
        # Using combo names: omniroute routes through the fallback chain automatically.
        "optimizer": "openai/optimizer",
        "eval":      "openai/eval",
        "judge":     "openai/eval",  # judge and eval both generate/score data — use same combo
    },
    "openrouter": {
        "optimizer": "openrouter/anthropic/claude-opus-4-5",
        "eval":      "openrouter/google/gemini-2.5-flash",
        "judge":     "openrouter/anthropic/claude-sonnet-4-5",
    },
    "openai": {
        "optimizer": "openai/o3",
        "eval":      "openai/gpt-4o-mini",
        "judge":     "openai/gpt-4o",
    },
    "groq": {
        "optimizer": "groq/llama-3.3-70b-versatile",
        "eval":      "groq/llama-3.1-8b-instant",
        "judge":     "groq/llama-3.3-70b-versatile",
    },
    "ollama": {
        "optimizer": "ollama/llama3.3:70b",
        "eval":      "ollama/llama3.2:3b",
        "judge":     "ollama/llama3.3:70b",
    },
}

# Role → env var override name
_ROLE_ENV: dict[str, str] = {
    "optimizer": "HERMES_OPTIMIZER_MODEL",
    "eval":      "HERMES_EVAL_MODEL",
    "judge":     "HERMES_JUDGE_MODEL",
}

Role = Literal["optimizer", "eval", "judge"]


def resolve_model(
    role: Role,
    override: Optional[str] = None,
) -> str:
    """Return the LiteLLM model string for a given role.

    Resolution order (first non-empty wins):
    1. override argument — explicit caller override
    2. Role-specific env var — e.g. HERMES_EVAL_MODEL
    3. Provider defaults — from HERMES_LLM_PROVIDER (default: omniroute)

    When HERMES_LLM_PROVIDER='omniroute' (the default), all LiteLLM calls
    are routed through the local omniroute proxy. This happens regardless
    of which level provided the model string — override, env var, or defaults
    — so even CLI-hardcoded model strings go through the proxy.

    Args:
        role: One of "optimizer", "eval", "judge".
        override: Explicit model string from EvolutionConfig or CLI flag.

    Returns:
        A LiteLLM-compatible model string.
    """
    provider = os.getenv("HERMES_LLM_PROVIDER", "omniroute").strip().lower()

    # Route ALL providers through omniroute — do this FIRST regardless of
    # which resolution path is taken below.
    if provider == "omniroute":
        _enable_omniroute_proxy()

    if override:
        return override

    env_val = os.getenv(_ROLE_ENV[role], "").strip()
    if env_val:
        return env_val

    defaults = PROVIDER_DEFAULTS.get(provider)
    if defaults is None:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. "
            f"Set HERMES_LLM_PROVIDER to one of: {list(PROVIDER_DEFAULTS)}"
        )

    return defaults[role]


# ── Omniroute proxy integration ──────────────────────────────────────────────

_OMNIROUTE_CONFIGURED = False


def _enable_omniroute_proxy() -> None:
    """Set LiteLLM env vars so all providers route through omniroute.

    Omniroute is a local OpenAI-compatible proxy at localhost:20128.
    By setting OPENROUTER_API_KEY and OPENROUTER_API_BASE, LiteLLM's
    openrouter backend forwards all requests to omniroute, which handles
    auth, rate limiting, and provider fallback transparently.

    Also sets ANTHROPIC_* and OPENAI_* to dummy values so that
    non-openrouter provider strings also hit omniroute.

    Idempotent — only runs once per process.
    """
    global _OMNIROUTE_CONFIGURED
    if _OMNIROUTE_CONFIGURED:
        return

    omniroute_url = os.getenv("OMNIROUTE_BASE_URL", "http://localhost:20128/v1")
    dummy_key = os.getenv("OMNIROUTE_API_KEY", "omniroute-no-auth-needed")

    # OpenRouter backend: LiteLLM sends to OPENROUTER_API_BASE
    os.environ.setdefault("OPENROUTER_API_KEY", dummy_key)
    os.environ.setdefault("OPENROUTER_API_BASE", omniroute_url)

    # Anthropic backend: redirect to omniroute
    os.environ.setdefault("ANTHROPIC_API_KEY", dummy_key)
    os.environ.setdefault("ANTHROPIC_BASE_URL", omniroute_url)

    # OpenAI backend: redirect to omniroute
    os.environ.setdefault("OPENAI_API_KEY", dummy_key)
    os.environ.setdefault("OPENAI_API_BASE", omniroute_url)

    # Force HTTP/1.1 — omniroute doesn't support HTTP/2 ALPN negotiation,
    # causing Python httpx clients to hang waiting for response headers.
    import litellm as _litellm
    _litellm.client_session = httpx.Client(
        transport=httpx.HTTPTransport(http2=False),
    )

    _OMNIROUTE_CONFIGURED = True


_DEFAULT_MAX_TOKENS = 4096


@functools.lru_cache(maxsize=64)
def get_lm(
    model: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> dspy.LM:
    """Return a cached dspy.LM instance for the given model string.

    Cached so repeated calls with the same (model, temperature) don't
    re-instantiate. The cache is per-process; clear it with
    get_lm.cache_clear() if needed.

    If HERMES_LLM_PROVIDER=omniroute, model strings are rewritten to use
    LiteLM's openai/ provider with omniroute-compatible model IDs so that
    all requests hit the local OpenAI-compatible proxy at localhost:20128.

    Pass temperature=0 for deterministic evaluation output. The default
    (None) uses the provider default, which is typically ~0.7 for most
    chat models.

    max_tokens defaults to _DEFAULT_MAX_TOKENS (4096) when the env var
    HERMES_MAX_TOKENS is not set. Set HERMES_MAX_TOKENTS=0 (or pass
    max_tokens=0) to disable; some providers pick their own default then.
    """
    if max_tokens is None:
        max_tokens = int(os.getenv("HERMES_MAX_TOKENS", str(_DEFAULT_MAX_TOKENS)))
    if os.getenv("HERMES_LLM_PROVIDER", "").strip().lower() == "omniroute":
        model = _to_omniroute_model(model)
    return dspy.LM(
        model,
        temperature=temperature,
        max_tokens=max_tokens if max_tokens else None,
    )


# ── Model-name translation for omniroute ─────────────────────────────────────

# Mapping from "legacy" model strings to omniroute-native IDs.
# Any model not in this map is passed through as-is (openai/ prefixed).
_OMNIROUTE_MODEL_MAP: dict[str, str] = {
    # Anthropic
    "anthropic/claude-haiku-4-5-20251001":  "openai/cc/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-6":          "openai/cc/claude-sonnet-4-6",
    "anthropic/claude-opus-4-8":            "openai/cc/claude-opus-4-6",
    "anthropic/claude-opus-4-5-20251101":   "openai/cc/claude-opus-4-6",
    # OpenRouter (when someone accidentally uses these with omniroute)
    "openrouter/anthropic/claude-opus-4-5": "openai/cc/claude-opus-4-5-20251101",
    "openrouter/anthropic/claude-sonnet-4-5": "openai/cc/claude-sonnet-4-6",
    "openrouter/anthropic/claude-haiku-4-5-20251001": "openai/cc/claude-haiku-4-5-20251001",
}


def _to_omniroute_model(model: str) -> str:
    """Rewrite a model string for omniroute.

    Already-prefixed openai/ models pass through unchanged.
    Known anthropic/ and openrouter/ models are mapped to cc/* equivalents.
    Unknown models get openai/ prepended so they go through the proxy.
    """
    if model in _OMNIROUTE_MODEL_MAP:
        return _OMNIROUTE_MODEL_MAP[model]
    if model.startswith("openai/"):
        return model
    # Unknown model — wrap in openai/ and hope omniroute knows it
    return f"openai/{model}"


def get_lm_for_role(
    role: Role,
    override: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dspy.LM:
    """Convenience wrapper: resolve model for role then return its LM.

    Pass temperature=0 for deterministic evaluation output.

    Usage:
        lm = get_lm_for_role("eval", temperature=0)
        with dspy.context(lm=lm):
            result = my_predictor(...)
    """
    model = resolve_model(role, override)
    return get_lm(model, temperature=temperature)


def configure_default_lm(
    role: Role = "eval",
    override: Optional[str] = None,
    temperature: Optional[float] = None,
) -> dspy.LM:
    """Call dspy.configure(lm=...) with the resolved model and return the LM.

    Use this at the top of an evolve_* function to set the global DSPy default.
    Prefer get_lm_for_role + dspy.context for scoped calls.

    Pass temperature=0 for deterministic evaluation output.
    """
    lm = get_lm_for_role(role, override, temperature=temperature)
    dspy.configure(lm=lm)
    return lm


def call(
    prompt: str,
    role: Role = "judge",
    override: Optional[str] = None,
    **kwargs,
) -> str:
    """Send a plain text prompt to an LLM and return the response string.

    This is for one-off calls that don't need a DSPy module.
    """
    lm = get_lm_for_role(role, override)
    response = lm(prompt, **kwargs)
    if isinstance(response, list):
        return response[0] if response else ""
    return str(response)