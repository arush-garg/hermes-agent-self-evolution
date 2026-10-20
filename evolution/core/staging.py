"""Staged deployment with backup + rollback for evolved artifacts.

Default is STAGE, never overwrite:
  * candidate  -> output/<artifact>/<ts>/<evolved_file>
  * manifest   -> staging/<artifact>.json  (paths, sha256 of live file at
                  staging time, stats, decision)

A live write happens only via :func:`promote` — callers must invoke it only
when ``--promote`` was given, or ``decision == 'promote' and --apply``.
Promotion:
  1. aborts if the live file's sha256 changed since staging (concurrent edit),
  2. validates the payload (skills: YAML frontmatter parses and name/description
     are preserved; JSON payloads must parse),
  3. backs the live file up to staging/backups/<artifact>/<ts>.<ext>
     (last 5 per artifact kept),
  4. writes atomically (tmp + os.replace).

CLI::

    python -m evolution.core.staging list
    python -m evolution.core.staging rollback --artifact X
    python -m evolution.core.staging promote --artifact X

Total size of staging/ + output/ is capped at 50 MB (oldest files pruned first;
manifests and the newest backup of each artifact are never pruned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BACKUPS = 5
MAX_TOTAL_BYTES = 50 * 1024 * 1024

_FILENAMES = {"skill": "evolved_skill.md", "prompt_section": "evolved_section.txt",
              "tool_descriptions": "evolved_tools.json"}


def _root(root: Optional[Path] = None) -> Path:
    return Path(root or os.getenv("EVOLUTION_STAGING_ROOT") or REPO_ROOT)


def staging_dir(root=None) -> Path:
    return _root(root) / "staging"


def output_dir(root=None) -> Path:
    return _root(root) / "output"


def backups_dir(artifact: str, root=None) -> Path:
    return staging_dir(root) / "backups" / _safe(artifact)


def manifest_path(artifact: str, root=None) -> Path:
    return staging_dir(root) / f"{_safe(artifact)}.json"


def _safe(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    if not s:
        raise ValueError(f"bad artifact name: {name!r}")
    return s


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> Optional[str]:
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


# ── Validation ──────────────────────────────────────────────────────────────


def parse_frontmatter(text: str) -> Optional[dict]:
    if not text.lstrip().startswith("---"):
        return None
    parts = text.lstrip().split("---", 2)
    if len(parts) < 3:
        return None
    try:
        fm = yaml.safe_load(parts[1])
    except yaml.YAMLError:
        return None
    return fm if isinstance(fm, dict) else None


def validate_frontmatter(candidate: str, live: Optional[str] = None) -> tuple[bool, str]:
    """YAML parses, has name+description; if ``live`` given they must match it."""
    fm = parse_frontmatter(candidate)
    if fm is None:
        return False, "frontmatter missing or YAML does not parse"
    for k in ("name", "description"):
        if not fm.get(k):
            return False, f"frontmatter missing '{k}'"
    if live is not None:
        live_fm = parse_frontmatter(live)
        if live_fm:
            for k in ("name", "description"):
                if live_fm.get(k) and live_fm.get(k) != fm.get(k):
                    return False, f"frontmatter '{k}' changed ({live_fm.get(k)!r} -> {fm.get(k)!r})"
    body = candidate.lstrip().split("---", 2)[2].strip()
    if not body:
        return False, "empty skill body"
    return True, "ok"


def validate_payload(artifact_type: str, candidate: str, live: Optional[str]) -> tuple[bool, str]:
    if artifact_type == "skill":
        return validate_frontmatter(candidate, live)
    if artifact_type == "tool_descriptions":
        try:
            json.loads(candidate)
        except ValueError as e:
            return False, f"invalid JSON: {e}"
        return True, "ok"
    return (bool(candidate.strip()), "ok" if candidate.strip() else "empty payload")


# ── Caps ────────────────────────────────────────────────────────────────────


def prune_backups(artifact: str, keep: int = MAX_BACKUPS, root=None) -> list[Path]:
    d = backups_dir(artifact, root)
    if not d.exists():
        return []
    files = sorted((p for p in d.iterdir() if p.is_file()), key=lambda p: p.name, reverse=True)
    removed = []
    for p in files[keep:]:
        p.unlink()
        removed.append(p)
    return removed


def enforce_size_cap(max_bytes: int = MAX_TOTAL_BYTES, root=None) -> list[Path]:
    """Prune oldest files under staging/ + output/ until total <= max_bytes."""
    sd, od = staging_dir(root), output_dir(root)
    files = [p for d in (sd, od) if d.exists() for p in d.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    if total <= max_bytes:
        return []
    protected = {p for p in sd.glob("*.json")}  # manifests
    bdir = sd / "backups"
    if bdir.exists():  # newest backup per artifact
        for a in bdir.iterdir():
            if a.is_dir():
                bs = sorted(a.iterdir(), key=lambda p: p.name)
                if bs:
                    protected.add(bs[-1])
    # staged candidates referenced by manifests are also protected
    for m in sd.glob("*.json"):
        try:
            protected.add(Path(json.loads(m.read_text()).get("staged_path", "")))
        except Exception:
            pass
    removed = []
    for p in sorted(files, key=lambda p: p.stat().st_mtime):
        if total <= max_bytes:
            break
        if p in protected:
            continue
        total -= p.stat().st_size
        p.unlink()
        removed.append(p)
    for d in (od,):  # drop empty run dirs
        if d.exists():
            for sub in sorted(d.rglob("*"), key=lambda x: len(x.parts), reverse=True):
                if sub.is_dir() and not any(sub.iterdir()):
                    sub.rmdir()
    return removed


# ── Stage / promote / rollback ──────────────────────────────────────────────


def stage(artifact: str, artifact_type: str, live_path: Path, candidate_text: str,
          stats: Optional[dict] = None, root=None, extra_files: Optional[dict] = None) -> dict:
    """Write candidate + manifest. Never touches ``live_path``."""
    live_path = Path(live_path)
    ts = _ts()
    run_dir = output_dir(root) / _safe(artifact) / ts
    staged = run_dir / _FILENAMES.get(artifact_type, f"evolved_{artifact_type}.txt")
    _atomic_write(staged, candidate_text)
    for name, content in (extra_files or {}).items():
        _atomic_write(run_dir / name, content if isinstance(content, str) else json.dumps(content, indent=2))
    manifest = {
        "artifact": artifact,
        "artifact_type": artifact_type,
        "timestamp": ts,
        "live_path": str(live_path),
        "staged_path": str(staged),
        "live_sha256_at_staging": sha256_file(live_path),
        "candidate_sha256": sha256_text(candidate_text),
        "stats": stats or {},
        "decision": (stats or {}).get("decision", "stage"),
        "status": "staged",
        "history": [],
    }
    _atomic_write(manifest_path(artifact, root), json.dumps(manifest, indent=2, default=str))
    _atomic_write(run_dir / "manifest.json", json.dumps(manifest, indent=2, default=str))
    enforce_size_cap(root=root)
    return manifest


def load_manifest(artifact: str, root=None) -> Optional[dict]:
    p = manifest_path(artifact, root)
    return json.loads(p.read_text()) if p.exists() else None


class PromotionError(RuntimeError):
    pass


def promote(artifact: str, root=None) -> dict:
    """Make the staged candidate live (with backup). Raises PromotionError."""
    m = load_manifest(artifact, root)
    if not m:
        raise PromotionError(f"no staged manifest for {artifact}")
    if m.get("status") == "promoted":
        raise PromotionError(f"{artifact} already promoted at {m.get('promoted_at')}")
    live, staged = Path(m["live_path"]), Path(m["staged_path"])
    if not staged.exists():
        raise PromotionError(f"staged file missing: {staged}")
    current = sha256_file(live)
    if current != m.get("live_sha256_at_staging"):
        raise PromotionError("live file changed since staging (concurrent edit) — aborting")
    candidate = staged.read_text()
    live_text = live.read_text() if live.exists() else None
    ok, why = validate_payload(m["artifact_type"], candidate, live_text)
    if not ok:
        raise PromotionError(f"validation failed: {why}")
    backup = None
    if live_text is not None:
        bd = backups_dir(artifact, root)
        bd.mkdir(parents=True, exist_ok=True)
        backup = bd / f"{_ts()}{live.suffix or '.bak'}"
        shutil.copy2(live, backup)
        prune_backups(artifact, root=root)
    _atomic_write(live, candidate)
    m.update(status="promoted", promoted_at=_ts(), backup_path=str(backup) if backup else None,
             live_sha256_after_promote=sha256_text(candidate))
    m["history"].append({"event": "promote", "at": m["promoted_at"], "backup": m["backup_path"]})
    _atomic_write(manifest_path(artifact, root), json.dumps(m, indent=2, default=str))
    return m


def rollback(artifact: str, root=None) -> dict:
    """Restore the newest backup to the live path."""
    m = load_manifest(artifact, root)
    if not m:
        raise PromotionError(f"no manifest for {artifact}")
    bd = backups_dir(artifact, root)
    backups = sorted((p for p in bd.iterdir() if p.is_file()), key=lambda p: p.name) if bd.exists() else []
    if not backups:
        raise PromotionError(f"no backups for {artifact}")
    src = backups[-1]
    live = Path(m["live_path"])
    _atomic_write(live, src.read_text())
    src.unlink()  # consumed; next rollback goes one further back
    m.update(status="rolled_back", rolled_back_at=_ts())
    m["history"].append({"event": "rollback", "at": m["rolled_back_at"], "from": str(src)})
    _atomic_write(manifest_path(artifact, root), json.dumps(m, indent=2, default=str))
    return m


def list_staged(root=None) -> list[dict]:
    sd = staging_dir(root)
    out = []
    for p in sorted(sd.glob("*.json")) if sd.exists() else []:
        try:
            m = json.loads(p.read_text())
        except ValueError:
            continue
        s = m.get("stats", {})
        out.append({k: m.get(k) for k in ("artifact", "artifact_type", "timestamp", "status",
                                          "decision", "staged_path", "live_path")}
                   | {k: s.get(k) for k in ("improvement", "ci_low", "ci_high", "p_value", "n_pairs")})
    return out


def finalize(artifact: str, artifact_type: str, live_path: Path, candidate_text: str,
             stats: dict, *, apply: bool = False, force_promote: bool = False, root=None,
             extra_files: Optional[dict] = None) -> dict:
    """Stage, then promote iff force_promote or (decision=='promote' and apply)."""
    m = stage(artifact, artifact_type, live_path, candidate_text, stats, root, extra_files)
    want = force_promote or (stats.get("decision") == "promote" and apply)
    if want:
        try:
            m = promote(artifact, root)
        except PromotionError as e:
            m["promotion_error"] = str(e)
            logger.error("promotion aborted: %s", e)
    return m


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evolution.core.staging")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("rollback", "promote", "show"):
        sp = sub.add_parser(name)
        sp.add_argument("--artifact", required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "list":
            rows = list_staged()
            if not rows:
                print("No staged artifacts.")
            for r in rows:
                print(json.dumps(r, default=str))
        elif args.cmd == "show":
            print(json.dumps(load_manifest(args.artifact), indent=2))
        elif args.cmd == "promote":
            m = promote(args.artifact)
            print(f"Promoted {args.artifact} -> {m['live_path']} (backup {m.get('backup_path')})")
        elif args.cmd == "rollback":
            m = rollback(args.artifact)
            print(f"Rolled back {args.artifact} -> {m['live_path']}")
    except PromotionError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
