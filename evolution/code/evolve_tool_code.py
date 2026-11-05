"""Evolve hermes-agent tool implementation code using Darwinian Evolver.

Darwinian Evolver is called as an EXTERNAL subprocess (AGPL v3) — never
imported as a Python module. This module manages the full pipeline:
load → gate check → evolve → validate → save/PR.

Usage:
    python -m evolution.code.evolve_tool_code --tool file_tools --iterations 10
    python -m evolution.code.evolve_tool_code --tool file_tools --dry-run
    python -m evolution.code.evolve_tool_code \\
        --tool search_files \\
        --bug-description "Regex edge cases with unicode filenames" \\
        --iterations 15
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.constraints import ConstraintValidator
from evolution.core.regression_gate import check_regression
from evolution.code.code_organism import (
    CodeTarget,
    CodeFitnessEvaluator,
    check_signatures_unchanged,
    _write_fitness_script,
)

console = Console()


def _check_darwinian_evolver() -> Optional[str]:
    """Return path to darwinian-evolver CLI, or None if not installed."""
    path = shutil.which("darwinian-evolver")
    if path:
        return path
    # Check in .venv
    venv_path = Path(__file__).parent.parent.parent / ".venv" / "bin" / "darwinian-evolver"
    if venv_path.exists():
        return str(venv_path)
    return None


def evolve(
    tool: str,
    iterations: int = 10,
    bug_description: str = "",
    optimizer_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    run_full_tests: bool = True,
    dry_run: bool = False,
    apply: bool = False,
    create_pr: bool = False,
) -> None:
    """Evolve a hermes-agent tool implementation using Darwinian Evolver."""

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        run_pytest=run_full_tests,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print(f"\n[bold cyan]Code Evolution (Darwinian Evolver)[/bold cyan] — Tool: [bold]{tool}[/bold]\n")

    # ── 1. Find the tool source file ─────────────────────────────────────
    tools_dir = config.hermes_agent_path / "tools"
    source_path = tools_dir / f"{tool}.py"

    if not source_path.exists():
        # Try without .py extension already
        candidates = list(tools_dir.glob(f"{tool}*.py"))
        if not candidates:
            console.print(f"[red]✗ Tool file not found: {source_path}[/red]")
            console.print(f"  Searched in: {tools_dir}")
            sys.exit(1)
        source_path = candidates[0]

    rel_path = source_path.relative_to(config.hermes_agent_path)
    original_code = source_path.read_text()

    console.print(f"  Source: {rel_path}")
    console.print(f"  Size: {len(original_code):,} chars")
    if bug_description:
        console.print(f"  Target bug: {bug_description}")

    # ── 2. Check Darwinian Evolver is available ──────────────────────────
    de_path = _check_darwinian_evolver()
    if not de_path:
        console.print("[red]✗ darwinian-evolver CLI not found.[/red]")
        console.print("  Install with: pip install darwinian-evolver")
        console.print("  Or: pip install -e '.[darwinian]'")
        if not dry_run:
            sys.exit(1)

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated.[/bold green]")
        console.print(f"  Source file: {source_path}")
        console.print(f"  darwinian-evolver: {'found' if de_path else 'NOT FOUND'}")
        console.print(f"  Iterations: {iterations}")
        console.print(f"  Apply if improved: {apply}")
        return

    target = CodeTarget(
        source_file=rel_path,
        test_command=f"{sys.executable} -m pytest tests/ -q --tb=short --ignore=tests/integration --ignore=tests/e2e --ignore=tests/stress --ignore=tests/manual --ignore=tests/gateway --ignore=tests/ci -n=0",
        bug_description=bug_description,
    )

    evaluator = CodeFitnessEvaluator(config.hermes_agent_path)
    validator = ConstraintValidator(config)

    # ── 3. Validate original code passes tests ───────────────────────────
    console.print(f"\n[bold]Validating baseline: running test suite[/bold]")
    baseline_result = evaluator.score(original_code, target)

    if not baseline_result.tests_passed:
        console.print(f"[red]✗ Baseline tests FAILED — cannot evolve broken code[/red]")
        console.print(f"  {baseline_result.details}")
        sys.exit(1)

    console.print(f"  [green]✓ Baseline tests pass[/green]")

    # ── 4. Run Darwinian Evolver ─────────────────────────────────────────
    with tempfile.TemporaryDirectory(prefix="hermes-evolve-code-") as tmpdir:
        tmp = Path(tmpdir)
        fitness_script = tmp / "fitness.py"
        _write_fitness_script(config.hermes_agent_path, target, fitness_script)

        console.print(f"\n[bold cyan]Running Darwinian Evolver ({iterations} iterations)...[/bold cyan]\n")
        start_time = time.time()

        de_cmd = [
            de_path,
            "evolve",
            "--file", str(source_path),
            "--fitness", str(fitness_script),
            "--iterations", str(iterations),
            "--output-dir", str(tmp / "evolved"),
        ]
        if bug_description:
            de_cmd += ["--goal", bug_description]
        if optimizer_model:
            de_cmd += ["--model", optimizer_model]

        import subprocess
        de_result = subprocess.run(
            de_cmd,
            capture_output=False,  # Stream output to terminal
            cwd=str(config.hermes_agent_path),
        )

        elapsed = time.time() - start_time
        console.print(f"\n  Darwinian Evolver completed in {elapsed:.1f}s")

        # ── 5. Read evolved output ───────────────────────────────────────
        evolved_dir = tmp / "evolved"
        evolved_candidates = list(evolved_dir.glob("*.py")) if evolved_dir.exists() else []

        if not evolved_candidates:
            console.print("[yellow]⚠ Darwinian Evolver produced no output files[/yellow]")
            return

        # Pick the best candidate (Darwinian Evolver names them by fitness score)
        evolved_path = sorted(evolved_candidates)[-1]
        evolved_code = evolved_path.read_text()

    # ── 6. Validate evolved code ─────────────────────────────────────────
    console.print(f"\n[bold]Validating evolved code[/bold]")

    if not check_signatures_unchanged(original_code, evolved_code):
        console.print("[red]✗ REJECTED: Function signatures or registry calls changed[/red]")
        return

    console.print("  [green]✓ Signatures unchanged[/green]")

    evolved_result = evaluator.score(evolved_code, target)
    if not evolved_result.tests_passed:
        console.print(f"[red]✗ REJECTED: Tests failed after evolution[/red]")
        console.print(f"  {evolved_result.details}")
        return

    console.print("  [green]✓ Full test suite passes[/green]")

    # ── 6a. Regression gate ────────────────────────────────────────────────
    # Code fitness is binary: 1.0 = tests pass AND sigs match; 0.0 otherwise.
    code_fitness = 1.0
    passed_gate = check_regression(
        improvement=code_fitness,
        baseline_score=0.0,
        evolved_score=code_fitness,
        artifact_type="code",
        artifact_name=tool,
        deployed_path=source_path if apply else None,
        baseline_content=original_code if apply else None,
    )

    # ── 7. Report and save ───────────────────────────────────────────────
    table = Table(title="Code Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("Tool", tool)
    table.add_row("Lines (original)", str(len(original_code.splitlines())))
    table.add_row("Lines (evolved)", str(len(evolved_code.splitlines())))
    table.add_row("Signatures unchanged", "[green]✓[/green]")
    table.add_row("Tests pass", "[green]✓[/green]")
    table.add_row("Time", f"{elapsed:.1f}s")
    console.print()
    console.print(table)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "code" / tool / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_code.py").write_text(evolved_code)
    (output_dir / "baseline_code.py").write_text(original_code)
    metrics = {
        "tool": tool,
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "bug_description": bug_description,
        "original_lines": len(original_code.splitlines()),
        "evolved_lines": len(evolved_code.splitlines()),
        "tests_passed": True,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if apply and passed_gate:
        source_path.write_text(evolved_code)
        console.print(f"\n[bold green]✓ Applied evolved code → {source_path}[/bold green]")
    elif apply and not passed_gate:
        console.print(f"\n[yellow]⚠ Regression - restoring baseline → {source_path}[/yellow]")
        source_path.write_text(original_code)
    else:
        console.print(f"  Review: diff {output_dir}/baseline_code.py {output_dir}/evolved_code.py")
        console.print(f"  Apply: cp {output_dir}/evolved_code.py {source_path}")


@click.command()
@click.option("--tool", required=True, help="Name of the tool file to evolve (e.g., file_tools)")
@click.option("--iterations", default=10, help="Number of Darwinian Evolver iterations")
@click.option("--bug-description", default="", help="Description of the bug/edge case to target")
@click.option("--optimizer-model", default=None,
              help="LLM model for Darwinian Evolver mutations (default: Darwian Evolver's own default)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--no-tests", is_flag=True, help="Skip full test suite validation (not recommended)")
@click.option("--dry-run", is_flag=True, help="Validate setup without running evolution")
@click.option("--apply", is_flag=True, help="Write evolved code back to source if tests pass")
@click.option("--create-pr", is_flag=True, help="Create PR in hermes-agent repo")
def main(tool, iterations, bug_description, optimizer_model, hermes_repo,
         no_tests, dry_run, apply, create_pr):
    """Evolve hermes-agent tool code using Darwinian Evolver (external CLI)."""
    evolve(
        tool=tool,
        iterations=iterations,
        bug_description=bug_description,
        optimizer_model=optimizer_model,
        hermes_repo=hermes_repo,
        run_full_tests=not no_tests,
        dry_run=dry_run,
        apply=apply,
        create_pr=create_pr,
    )


if __name__ == "__main__":
    main()
