"""Regression-rollback gate for evolved artifacts.

Every evolve pipeline calls ``check_regression()`` before applying changes.
If the evolved artifact fails to improve (or regresses), the gate:

1. Logs the failure with before/after scores.
2. Refuses to deploy — the apply flag is ignored.
3. When the artifact type supports it, restores the known-good baseline file
   so a partial or concurrent write cant leave a half-applied state.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RegressionError(Exception):
    """Raised when an evolved artifact regresses and is denied deployment."""


_LOG_FILE: Path | None = None


def _ensure_log() -> Path:
    global _LOG_FILE
    if _LOG_FILE is None:
        p = Path("logs") / "regression_gate.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = p
    return _LOG_FILE


def check_regression(
    improvement: float,
    baseline_score: float,
    evolved_score: float,
    artifact_type: str,
    artifact_name: str,
    *,
    deployed_path: Optional[Path] = None,
    baseline_content: Optional[str] = None,
) -> bool:
    """Return ``True`` if the artifact passed (no regression).

    When ``improvement <= 0`` and a ``deployed_path`` + ``baseline_content``
    are provided the gate *restores* the baseline file so that a partially
    applied evolution is rolled back automatically.

    Every check (pass or fail) is appended as a JSON line to
    ``logs/regression_gate.jsonl`` for auditability.
    """
    passed = improvement > 0
    log_dir = _ensure_log()

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "artifact_type": artifact_type,
        "artifact_name": artifact_name,
        "baseline_score": round(baseline_score, 4),
        "evolved_score": round(evolved_score, 4),
        "improvement": round(improvement, 4),
        "passed": passed,
        "rolled_back": False,
    }

    if passed:
        log_dir.parent.mkdir(parents=True, exist_ok=True)
        with log_dir.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        return True

    # ── Regression: roll back if we have the content ────────────────
    if deployed_path is not None and baseline_content is not None:
        try:
            deployed_path.write_text(baseline_content)
            record["rolled_back"] = True
            record["restored_path"] = str(deployed_path)
            logger.warning(
                "Regression rollback restored %s to baseline",
                deployed_path,
            )
        except OSError as exc:
            logger.error("Rollback failed for %s: %s", deployed_path, exc)

    log_dir.parent.mkdir(parents=True, exist_ok=True)
    with log_dir.open("a") as fh:
        fh.write(json.dumps(record) + "\n")

    return False
