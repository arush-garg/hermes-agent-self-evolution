"""Gauntlet runner: evolves skills, tool descriptions, prompt sections, and tool code in rotation.

Each tick picks one pipeline type (weighted: skills most often),
selects the best candidate for that pipeline, runs evolution,
persists state, and exits.  LaunchAgent fires every 2 hours via StartInterval.

Usage:
    python -m evolution.monitor.daily_runner
    python -m evolution.monitor.daily_runner --dry-run
    python -m evolution.monitor.daily_runner --force-pipeline skills --force-artifact linear
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).parent.parent.parent
STATE_FILE = REPO_ROOT / "logs" / "evolution_schedule.json"

# Augment PATH so hermes CLI is found even under launchd's minimal environment
_EXTRA_PATH = str(Path.home() / ".local" / "bin")
if _EXTRA_PATH not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _EXTRA_PATH + ":" + os.environ.get("PATH", "")

# 6-tick cycle: skills × 3, tools × 1, prompts × 1, code × 1
# tick 0 → skills, 1 → tools, 2 → skills, 3 → prompts, 4 → skills, 5 → code
TICK_SCHEDULE = ["skills", "tools", "skills", "prompts", "skills", "code"]

EVOLVABLE_SECTIONS = [
    "DEFAULT_AGENT_IDENTITY",
    "MEMORY_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "SKILLS_GUIDANCE",
    "PLATFORM_HINTS",
]

STALE_THRESHOLD = 3


# ── State ──────────────────────────────────────────────────────────────────


def read_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {"tick": 0, "pipelines": {}, "stale_skills": {}}


def write_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Skill discovery ────────────────────────────────────────────────────────


def discover_skills() -> list[str]:
    """Return skill directory names from ~/.hermes/skills/ and the hermes-agent repo."""
    skill_names: list[str] = []

    user_skills_dir = Path.home() / ".hermes" / "skills"
    if user_skills_dir.exists():
        for skill_md in user_skills_dir.rglob("SKILL.md"):
            skill_names.append(skill_md.parent.name)

    try:
        from evolution.core.config import get_hermes_agent_path
        hermes_path = get_hermes_agent_path()
        hermes_skills = hermes_path / "skills"
        if hermes_skills.exists():
            for skill_md in hermes_skills.rglob("SKILL.md"):
                name = skill_md.parent.name
                if name not in skill_names:
                    skill_names.append(name)
    except (FileNotFoundError, ImportError):
        pass

    return sorted(skill_names)


def load_past_metrics() -> dict:
    """Read the most recent metrics.json for each skill in output/."""
    output_dir = REPO_ROOT / "output"
    metrics: dict = {}

    if not output_dir.exists():
        return metrics

    for skill_dir in output_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        run_dirs = sorted(
            (d for d in skill_dir.iterdir() if d.is_dir()),
            reverse=True,
        )
        for run_dir in run_dirs:
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text())
                    metrics[skill_dir.name] = {
                        "baseline_score": data.get("baseline_score"),
                        "evolved_score": data.get("evolved_score"),
                        "improvement": data.get("improvement"),
                        "timestamp": data.get("timestamp"),
                    }
                except (json.JSONDecodeError, OSError):
                    pass
                break

    return metrics


# ── Candidate selection per pipeline ───────────────────────────────────────


def select_skill_via_hermes(
    skills: list[str],
    past_metrics: dict,
    recently_evolved: Optional[set] = None,
    stale_skills: Optional[dict] = None,
    dry_run: bool = False,
) -> Optional[str]:
    """Deterministically rank failure evidence, staleness, and past metrics.

    Seven-day cooldown applies unless new failure evidence exists.
    """
    if not skills:
        raise ValueError('No skills available')
    from evolution.core.failure_mining import mine
    failures = {row['artifact']: row for row in mine(days=30)}
    recently_evolved = recently_evolved or set()
    now = datetime.now()
    last_runs = {}
    log = REPO_ROOT / 'logs/daily_evolution.jsonl'
    if log.exists():
        for line in log.read_text(errors='replace').splitlines()[-2000:]:
            try:
                row = json.loads(line)
                if row.get('pipeline') == 'skills' and row.get('artifact'):
                    last_runs[row['artifact']] = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None)
            except (ValueError, KeyError, TypeError):
                continue
    def rank(name):
        metric = past_metrics.get(name, {})
        failure = failures.get(name, {})
        return (failure.get('score', 0) * 10 + (1 - (metric.get('evolved_score') or 0.5)) * 3
                + (stale_skills or {}).get(f'skills/{name}', 0), name)
    eligible = [s for s in skills if (s not in recently_evolved and
                (s not in last_runs or (now - last_runs[s]).days >= 7)) or
                failures.get(s, {}).get('score', 0) > 0]
    return sorted(eligible or skills, key=lambda name: (-rank(name)[0], rank(name)[1]))[0]

def select_prompt_section(state: dict) -> str:
    """Rotate through evolvable prompt sections."""
    pipe_state = state["pipelines"].get("prompts", {})
    last = pipe_state.get("last_artifact")
    if last and last in EVOLVABLE_SECTIONS:
        idx = (EVOLVABLE_SECTIONS.index(last) + 1) % len(EVOLVABLE_SECTIONS)
        return EVOLVABLE_SECTIONS[idx]
    return EVOLVABLE_SECTIONS[0]


def select_code_tool(state: dict) -> Optional[str]:
    """Return the next tool file to evolve, or None if darwinian-evolver not installed."""
    if not shutil.which("darwinian-evolver"):
        venv_path = REPO_ROOT / ".venv" / "bin" / "darwinian-evolver"
        if not venv_path.exists():
            print("[warn] darwinian-evolver not found — skipping code pipeline", file=sys.stderr)
            return None

    try:
        from evolution.core.config import get_hermes_agent_path
        agent_path = get_hermes_agent_path()
        tools_dir = agent_path / "tools"
        if not tools_dir.exists():
            print("[warn] hermes-agent tools directory not found", file=sys.stderr)
            return None

        tools = sorted(
            f.stem for f in tools_dir.glob("*.py")
            if f.is_file() and not f.name.startswith("_")
        )
        if not tools:
            print("[warn] no tool files found in hermes-agent/agent/tools/", file=sys.stderr)
            return None

        pipe_state = state["pipelines"].get("code", {})
        last = pipe_state.get("last_artifact")
        if last and last in tools:
            idx = (tools.index(last) + 1) % len(tools)
            return tools[idx]
        return tools[0]

    except (FileNotFoundError, ImportError) as e:
        print(f"[warn] could not discover code tools: {e}", file=sys.stderr)
        return None


# ── Subprocess dispatch ────────────────────────────────────────────────────


def build_command(pipeline: str, artifact: str) -> list[str]:
    """Return the subprocess argv for a given pipeline + artifact."""
    python = sys.executable
    if pipeline == "skills":
        return [python, "-m", "evolution.skills.evolve_skill",
                "--skill", artifact, "--iterations", "10", "--eval-source", "mixed",
                "--candidates", "3", "--repeats", "2", "--apply"]
    elif pipeline == "tools":
        return [python, "-m", "evolution.tools.evolve_tool_descriptions",
                "--iterations", "5"]
    elif pipeline == "prompts":
        return [python, "-m", "evolution.prompts.evolve_prompt_section",
                "--section", artifact, "--iterations", "5", "--apply"]
    elif pipeline == "code":
        # No --apply: the code pipeline still uses the legacy sign-only gate and would write into
        # ~/.hermes/hermes-agent (which holds the user's uncommitted work). Report-only until it is
        # migrated to pairwise/stats gating + staging.
        return [python, "-m", "evolution.code.evolve_tool_code",
                "--tool", artifact, "--iterations", "10"]
    raise ValueError(f"Unknown pipeline: {pipeline}")


def run_evolution(pipeline: str, artifact: str) -> tuple[bool, Optional[float]]:
    """Run the evolution subprocess. Returns (success, improvement)."""
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = logs_dir / f"evolve_{pipeline}_{artifact}_{ts}.log"

    cmd = build_command(pipeline, artifact)

    with log_path.open("w") as log_f:
        log_f.write(f"Command: {' '.join(cmd)}\nTimestamp: {ts}\n\n")

        env = os.environ.copy()
        env.setdefault("HERMES_LLM_PROVIDER", "omniroute")
        from evolution.core.failure_mining import mine
        context_path = logs_dir / 'failure_context.json'
        context_path.write_text(json.dumps(mine(days=30)[:30], indent=2))
        env['EVOLUTION_FAILURE_CONTEXT'] = str(context_path)

        try:
            result = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=log_f,
                                    stderr=subprocess.STDOUT, env=env, timeout=3 * 3600)
        except subprocess.TimeoutExpired:
            log_f.write('\nEvolution wall-clock limit (3h) exceeded.\n')
            print(f'  Evolution timed out — full log: {log_path}', file=sys.stderr)
            return False, None

    # Latest-log symlink
    latest_link = logs_dir / f"evolve_{pipeline}_latest.log"
    if latest_link.exists() or latest_link.is_symlink():
        latest_link.unlink()
    latest_link.symlink_to(log_path.name)

    # Structured result is authoritative when available; older pipelines use metrics fallback.
    for line in reversed(log_path.read_text(errors='replace').splitlines()):
        if line.startswith('EVOLUTION_RESULT '):
            try:
                data = json.loads(line.removeprefix('EVOLUTION_RESULT '))
                globals()['_LAST_EVOLUTION_RESULT'] = data
                return bool(data.get('success')), data.get('improvement')
            except (ValueError, TypeError):
                break
    if result.returncode != 0:
        print(f"  Evolution failed — full log: {log_path}", file=sys.stderr)
        return False, None

    # Read improvement from the freshest metrics.json
    if pipeline == "skills":
        output_dir = REPO_ROOT / "output" / artifact
    elif pipeline == "tools":
        output_dir = REPO_ROOT / "output" / "tool_descriptions"
    elif pipeline == "prompts":
        output_dir = REPO_ROOT / "output" / "prompts" / artifact
    elif pipeline == "code":
        output_dir = REPO_ROOT / "output" / "code" / artifact
    else:
        output_dir = None

    if output_dir and output_dir.exists():
        run_dirs = sorted(
            (d for d in output_dir.iterdir() if d.is_dir()),
            reverse=True,
        )
        for run_dir in run_dirs:
            metrics_file = run_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    data = json.loads(metrics_file.read_text())
                    return True, data.get("improvement")
                except (json.JSONDecodeError, OSError):
                    pass
                break

    return True, None


# ── Pipeline selection logic ────────────────────────────────────────────────


def select_pipeline(state: dict, dry_run: bool) -> Optional[tuple[str, str]]:
    """Determine (pipeline_type, artifact_name) for this tick."""
    tick = state.get("tick", 0)
    pipeline_type = TICK_SCHEDULE[tick % len(TICK_SCHEDULE)]

    # If code is not available, fall back to skills
    if pipeline_type == "code":
        tool = select_code_tool(state)
        if tool is None:
            print("  Darwinian-evolver not available — falling back to skills")
            pipeline_type = "skills"

    pipe_state = state["pipelines"].get(pipeline_type, {})
    recently_evolved = set(pipe_state.get("evolved_this_cycle", []))

    if pipeline_type == "skills":
        skills = discover_skills()
        if not skills:
            print("  No skills found.", file=sys.stderr)
            return None
        past_metrics = load_past_metrics()
        print(f"  Skills: {len(skills)} candidates, {len(recently_evolved)} evolved this cycle")
        skill_name = select_skill_via_hermes(skills, past_metrics, recently_evolved, stale_skills=state.get("stale_skills", {}), dry_run=dry_run)
        if skill_name is None:
            return None
        return pipeline_type, skill_name

    elif pipeline_type == "tools":
        return pipeline_type, "all"

    elif pipeline_type == "prompts":
        section = select_prompt_section(state)
        return pipeline_type, section

    elif pipeline_type == "code":
        tool = select_code_tool(state)
        if tool is None:
            # Use skills pipeline's recently-evolved set, not code's
            skills_recent = set(state["pipelines"].get("skills", {}).get("evolved_this_cycle", []))
            skills = discover_skills()
            if skills:
                past_metrics = load_past_metrics()
                sn = select_skill_via_hermes(skills, past_metrics, skills_recent, stale_skills=state.get("stale_skills", {}), dry_run=dry_run)
                if sn:
                    return "skills", sn
            return None
        return pipeline_type, tool

    return None


# ── Notifications & logging ────────────────────────────────────────────────


def send_notification(pipeline: str, artifact: str, success: bool, improvement: Optional[float]) -> None:
    label = f"{pipeline}/{artifact}"
    if not success:
        body = f"Evolution FAILED for '{label}'"
    elif improvement is not None and improvement > 0:
        body = f"Evolved '{label}' — improvement: {improvement:+.3f}"
    else:
        body = f"No improvement for '{label}' (try more iterations)"

    subprocess.run(
        ["osascript", "-e",
         f'display notification "{body}" with title "Hermes Self-Evolution"'],
        check=False,
    )


def write_log_entry(pipeline: str, artifact: str, success: bool, improvement: Optional[float]) -> None:
    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline": pipeline,
        "artifact": artifact,
        "success": success,
        "improvement": improvement,
        "decision": globals().get('_LAST_EVOLUTION_RESULT', {}).get('decision'),
        "ci_low": globals().get('_LAST_EVOLUTION_RESULT', {}).get('ci_low'),
        "ci_high": globals().get('_LAST_EVOLUTION_RESULT', {}).get('ci_high'),
        "n_pairs": globals().get('_LAST_EVOLUTION_RESULT', {}).get('n_pairs'),
    }
    log_file = logs_dir / "daily_evolution.jsonl"
    with log_file.open("a") as f:
        f.write(json.dumps(entry) + "\n")


# ── Entry point ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hermes Self-Evolution Gauntlet: rotate through skills, tools, prompts, and code."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running evolution.",
    )
    parser.add_argument(
        "--force-pipeline",
        choices=list(set(TICK_SCHEDULE)),  # unique pipeline names
        default=None,
        help="Override and run a specific pipeline type.",
    )
    parser.add_argument(
        "--force-artifact", default=None,
        help="Override and evolve a specific artifact (skill name, prompt section, or code tool).",
    )
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    from evolution.core.housekeeping import housekeeping
    if args.dry_run:
        # Dry run is strictly read-only: no runner log, no symlink, no housekeeping writes,
        # no state/log updates. Print the plan and proposed selection only.
        hk = housekeeping(REPO_ROOT, dry_run=True)
        print(f"[dry-run] Housekeeping plan: {json.dumps(hk['plan'])}")
        print(f"[dry-run] Housekeeping usage (bytes): {hk['before']} -> projected {hk['projected']}")
        acts = hk['actions']
        print(f"[dry-run] Housekeeping would perform {len(acts)} action(s)"
              + (": " + "; ".join(acts[:10]) + (" ..." if len(acts) > 10 else "") if acts else ""))
        _main(args)
        return

    logs_dir = REPO_ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    runner_log = logs_dir / f"daily_runner_{ts}.log"
    runner_latest = logs_dir / "daily_runner_latest.log"
    # Enforce disk caps on every real run, before creating new artifacts; never touch
    # the current run's log.
    usage = housekeeping(REPO_ROOT, dry_run=False, protect=[runner_log])
    print(f"Housekeeping (bytes): {usage['before']} -> {usage['after']}")
    if runner_latest.exists() or runner_latest.is_symlink():
        runner_latest.unlink()
    runner_latest.symlink_to(runner_log.name)

    # Tee stdout/stderr to both console and the runner log
    runner_fh = runner_log.open("w")
    _orig_stdout = sys.stdout
    _orig_stderr = sys.stderr

    class _Tee:
        def __init__(self, *files):
            self._files = files
        def write(self, s):
            for f in self._files:
                f.write(s)
        def flush(self):
            for f in self._files:
                f.flush()

    sys.stdout = _Tee(_orig_stdout, runner_fh)
    sys.stderr = _Tee(_orig_stderr, runner_fh)

    try:
        _main(args, protect=[runner_log])
    except BaseException as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout = _orig_stdout
        sys.stderr = _orig_stderr
        runner_fh.close()


def _main(args, protect=()) -> None:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] Starting self-evolution gauntlet")

    state = read_state()
    state.setdefault("stale_skills", {})
    dry_run = args.dry_run

    # ── Determine pipeline + artifact ──────────────────────────────────
    if args.force_pipeline:
        pipeline_type = args.force_pipeline
        if args.force_artifact:
            artifact = args.force_artifact
        else:
            # Auto-select artifact for the forced pipeline
            if pipeline_type == "skills":
                skills = discover_skills()
                past_metrics = load_past_metrics()
                sn = select_skill_via_hermes(skills, past_metrics, set(), stale_skills=state.get("stale_skills", {}), dry_run=dry_run)
                artifact = sn or "unknown"
            elif pipeline_type == "tools":
                artifact = "all"
            elif pipeline_type == "prompts":
                artifact = select_prompt_section(state)
            elif pipeline_type == "code":
                tool = select_code_tool(state)
                artifact = tool or "unknown"
            else:
                artifact = "unknown"
    else:
        result = select_pipeline(state, dry_run=dry_run)
        if result is None:
            print("  Nothing to evolve this tick. Exiting.")
            return
        pipeline_type, artifact = result

    if artifact is None or artifact == "unknown":
        print("  No suitable artifact found. Exiting.")
        return

    tick = state.get("tick", 0)
    print(f"  Pipeline: {pipeline_type}")
    print(f"  Artifact: {artifact}")
    print(f"  Tick:     {tick}")

    if dry_run:
        cmd = build_command(pipeline_type, artifact)
        print(f"\n[dry-run] Would run:\n  {' '.join(cmd)}")
        from evolution.core.failure_mining import mine
        row = next((r for r in mine(days=30) if r['artifact'] == artifact), None)
        if row:
            print(f"[dry-run] Failure evidence for {artifact}: score={row['score']} "
                  f"failure_sessions={row['failure_sessions']} correction_sessions={row['correction_sessions']} "
                  f"low_rated={row.get('low_rated_sessions', 0)} high_rated={row.get('high_rated_sessions', 0)}")
            for fb in row.get('user_feedback', [])[:3]:
                print(f"    /rate {fb['rating']}/5: {fb['comment'][:120]}")
            for f in row['failures'][:3]:
                print(f"    {f['tool']} x{f['sessions']}: {f['error_signature'][:120]}")
        else:
            print(f"[dry-run] Failure evidence for {artifact}: none")
        return

    # ── Run ────────────────────────────────────────────────────────────
    globals()['_LAST_EVOLUTION_RESULT'] = {}
    if pipeline_type == 'code' and select_code_tool(state) is None:
        print('  Code prerequisites missing; skipping cleanly.')
        return
    print("  Running evolution...")
    success, improvement = run_evolution(pipeline_type, artifact)

    # ── Update stale-skills tracker ────────────────────────────────────
    stale_key = f"{pipeline_type}/{artifact}"
    if improvement is not None and improvement > 0:
        state["stale_skills"][stale_key] = 0
    elif improvement is not None and improvement <= 0:
        state["stale_skills"][stale_key] = state["stale_skills"].get(stale_key, 0) + 1

    # ── Notify ─────────────────────────────────────────────────────────
    send_notification(pipeline_type, artifact, success, improvement)

    # ── Log ────────────────────────────────────────────────────────────
    write_log_entry(pipeline_type, artifact, success, improvement)

    # ── Persist state ──────────────────────────────────────────────────
    pipe_key = pipeline_type
    if pipe_key not in state["pipelines"]:
        state["pipelines"][pipe_key] = {}
    pipe_state = state["pipelines"][pipe_key]

    pipe_state["last_artifact"] = artifact
    pipe_state["last_run"] = datetime.now(timezone.utc).isoformat()

    # Track evolved-this-cycle for skills (to avoid repeating)
    if pipeline_type == "skills":
        if "evolved_this_cycle" not in pipe_state:
            pipe_state["evolved_this_cycle"] = []
        if artifact not in pipe_state["evolved_this_cycle"]:
            pipe_state["evolved_this_cycle"].append(artifact)

    state["tick"] = tick + 1

    # Reset evolved_this_cycle every 12 ticks (2 full rotations of the schedule)
    if state["tick"] % 12 == 0:
        for p in state["pipelines"]:
            state["pipelines"][p]["evolved_this_cycle"] = []
        print("  Reset evolved_this_cycle (12-tick milestone)")

    write_state(state)

    if success:
        msg = f"  Done. Improvement: {improvement:+.3f}" if improvement is not None else "  Done."
        print(msg)
    else:
        print("  Evolution failed.", file=sys.stderr)
        sys.exit(1)

    # ── Housekeeping (after evolution, not before; dry-run handled earlier) ─────
    if not dry_run:
        from evolution.core.housekeeping import housekeeping
        usage = housekeeping(REPO_ROOT, dry_run=False, protect=protect)
        print(f"Housekeeping (bytes): {usage['before']} -> {usage['after']}")


if __name__ == "__main__":
    main()
