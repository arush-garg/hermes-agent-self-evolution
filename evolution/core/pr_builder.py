"""Auto-generate pull requests with evolution metrics and diffs.

Creates a branch in the hermes-agent repo, commits evolved changes,
and opens a PR via `gh pr create`. All evolved changes deploy through
PRs — never direct commits.
"""

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class PRMetrics:
    """Metrics to include in the PR body."""
    target_name: str
    artifact_type: str        # skill, tool_description, prompt_section, code
    baseline_score: float
    evolved_score: float
    improvement: float
    optimizer: str
    iterations: int
    elapsed_seconds: float
    eval_dataset_size: int = 0
    holdout_score: Optional[float] = None
    constraint_violations_rejected: int = 0


class PRBuilder:
    """Create PRs in the hermes-agent repo for evolved artifacts.

    Uses raw subprocess calls to git and gh — no extra dependencies.
    """

    def create_pr(
        self,
        hermes_repo: Path,
        changed_files: list[tuple[Path, str]],
        metrics: PRMetrics,
    ) -> str:
        """Create a branch, commit changes, and open a PR.

        Args:
            hermes_repo: Path to the hermes-agent repository.
            changed_files: List of (relative_path, new_content) tuples.
            metrics: Evolution metrics to include in the PR body.

        Returns:
            PR URL from `gh pr create`.

        Raises:
            RuntimeError: If git or gh operations fail.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        branch_name = f"evolve/{metrics.target_name}-{timestamp}"

        # 1. Create and switch to a new branch
        self._run_git(hermes_repo, ["checkout", "-b", branch_name])

        try:
            # 2. Write evolved files
            for rel_path, content in changed_files:
                full_path = hermes_repo / rel_path
                full_path.write_text(content)
                self._run_git(hermes_repo, ["add", str(rel_path)])

            # 3. Commit
            commit_msg = _build_commit_message(metrics)
            self._run_git(hermes_repo, ["commit", "-m", commit_msg])

            # 4. Push
            self._run_git(hermes_repo, ["push", "-u", "origin", branch_name])

            # 5. Create PR
            pr_body = _build_pr_body(metrics)
            pr_title = f"evolve: {metrics.target_name} ({metrics.improvement:+.3f} score improvement)"

            result = subprocess.run(
                ["gh", "pr", "create", "--title", pr_title, "--body", pr_body],
                capture_output=True,
                text=True,
                cwd=str(hermes_repo),
            )

            if result.returncode != 0:
                raise RuntimeError(f"gh pr create failed: {result.stderr.strip()}")

            pr_url = result.stdout.strip()
            return pr_url

        except Exception:
            # Clean up: switch back to main if anything fails
            try:
                self._run_git(hermes_repo, ["checkout", "main"])
                self._run_git(hermes_repo, ["branch", "-D", branch_name])
            except Exception:
                pass
            raise

    def _run_git(self, repo: Path, args: list[str]) -> str:
        """Run a git command in the given repo. Raises RuntimeError on failure."""
        result = subprocess.run(
            ["git"] + args,
            capture_output=True,
            text=True,
            cwd=str(repo),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"git {' '.join(args)} failed:\n{result.stderr.strip()}"
            )
        return result.stdout.strip()


def _build_commit_message(metrics: PRMetrics) -> str:
    improvement_pct = (
        metrics.improvement / max(0.001, abs(metrics.baseline_score)) * 100
        if metrics.baseline_score
        else 0.0
    )
    lines = [
        f"evolve: {metrics.target_name} — score {metrics.baseline_score:.3f} → {metrics.evolved_score:.3f} ({improvement_pct:+.1f}%)",
        "",
        f"Optimizer: {metrics.optimizer} ({metrics.iterations} iterations)",
        f"Eval dataset: {metrics.eval_dataset_size} examples",
        f"Before: {metrics.baseline_score:.3f}",
        f"After: {metrics.evolved_score:.3f}",
    ]
    if metrics.holdout_score is not None:
        lines.append(f"Holdout: {metrics.holdout_score:.3f}")
    if metrics.constraint_violations_rejected > 0:
        lines.append(f"Rejected variants (constraint violations): {metrics.constraint_violations_rejected}")
    return "\n".join(lines)


def _build_pr_body(metrics: PRMetrics) -> str:
    improvement_pct = (
        metrics.improvement / max(0.001, abs(metrics.baseline_score)) * 100
        if metrics.baseline_score
        else 0.0
    )
    elapsed_min = metrics.elapsed_seconds / 60.0

    rows = [
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Target | `{metrics.target_name}` |",
        f"| Type | {metrics.artifact_type} |",
        f"| Baseline score | {metrics.baseline_score:.3f} |",
        f"| Evolved score | {metrics.evolved_score:.3f} |",
        f"| Improvement | {metrics.improvement:+.3f} ({improvement_pct:+.1f}%) |",
    ]
    if metrics.holdout_score is not None:
        rows.append(f"| Holdout score | {metrics.holdout_score:.3f} |")
    rows += [
        f"| Optimizer | {metrics.optimizer} |",
        f"| Iterations | {metrics.iterations} |",
        f"| Eval dataset | {metrics.eval_dataset_size} examples |",
        f"| Runtime | {elapsed_min:.1f} min |",
    ]
    if metrics.constraint_violations_rejected > 0:
        rows.append(f"| Rejected variants | {metrics.constraint_violations_rejected} |")

    sections = [
        "## Evolution Summary",
        "",
        "\n".join(rows),
        "",
        "## Review Checklist",
        "",
        "- [ ] Changes are semantically correct (the artifact still does what it's supposed to)",
        "- [ ] No unintended behavioral drift",
        "- [ ] Size/token budget is acceptable",
        "- [ ] All benchmarks passed (see CI)",
        "",
        "---",
        "_Generated by [hermes-agent-self-evolution](https://github.com/NousResearch/hermes-agent-self-evolution)_",
    ]

    return "\n".join(sections)
