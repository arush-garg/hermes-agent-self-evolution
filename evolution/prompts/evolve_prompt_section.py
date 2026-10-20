"""Evolve hermes-agent system prompt sections using DSPy + GEPA.

Usage:
    python -m evolution.prompts.evolve_prompt_section --section MEMORY_GUIDANCE --iterations 5
    python -m evolution.prompts.evolve_prompt_section --section DEFAULT_AGENT_IDENTITY --dry-run
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import click
import dspy
from rich.console import Console
from rich.table import Table

from evolution.core.config import EvolutionConfig
from evolution.core.llm_api import configure_default_lm, get_lm_for_role
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset
from evolution.core.constraints import ConstraintValidator
from evolution.core.regression_gate import check_regression
from evolution.prompts.prompt_module import (
    EVOLVABLE_SECTIONS,
    PromptSectionModule,
    load_prompt_section,
    save_prompt_section,
    prompt_fitness_metric,
)

console = Console()


def evolve(
    section: str,
    iterations: int = 5,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    benchmark_gate: bool = False,
    dry_run: bool = False,
    apply: bool = False,
    create_pr: bool = False,
) -> None:
    """Evolve a single system prompt section."""

    if section not in EVOLVABLE_SECTIONS:
        console.print(f"[red]✗ '{section}' is not an evolvable section.[/red]")
        console.print(f"  Evolvable sections: {', '.join(EVOLVABLE_SECTIONS)}")
        sys.exit(1)

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=None,
        run_tblite=benchmark_gate,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print(f"\n[bold cyan]Prompt Section Evolution[/bold cyan] — Section: [bold]{section}[/bold]\n")

    # ── 1. Load the section ──────────────────────────────────────────────
    try:
        section_text = load_prompt_section(section, config.hermes_agent_path)
    except (FileNotFoundError, KeyError) as e:
        console.print(f"[red]✗ {e}[/red]")
        sys.exit(1)

    console.print(f"  Loaded: {len(section_text):,} chars")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated.[/bold green]")
        console.print(f"  Would generate behavioral test suite for {section}")
        console.print(f"  Would run GEPA optimization ({iterations} iterations)")
        console.print(f"  Benchmark gate: {'enabled (zero tolerance)' if benchmark_gate else 'disabled'}")
        console.print(f"  Apply if improved: {apply}")
        return

    # ── 2. Build behavioral evaluation dataset ───────────────────────────
    console.print(f"\n[bold]Building behavioral test suite[/bold]")

    # Global default LM — GEPA uses dspy.settings.lm for student evaluation
    configure_default_lm("eval", override=eval_model, temperature=0.0)
    lm = get_lm_for_role("eval", override=eval_model, temperature=0.0)

    builder = SyntheticDatasetBuilder(config)
    dataset = builder.generate(
        artifact_text=section_text,
        artifact_type="prompt_section",
        num_cases=40,
    )
    save_path = Path("datasets") / "prompts" / section.lower()
    dataset.save(save_path)

    console.print(f"  Generated {len(dataset.all_examples)} behavioral test cases")
    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate baseline constraints ────────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    baseline_constraints = validator.validate_all(section_text, "prompt_section")
    for c in baseline_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "yellow"
        console.print(f"  [{color}]{icon}[/{color}] {c.constraint_name}: {c.message}")

    # ── 4. Run GEPA ──────────────────────────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")
    console.print(f"  Optimizer: GEPA ({iterations} iterations)")
    console.print(f"  Optimizer model: {optimizer_model}")

    baseline_module = PromptSectionModule(section, section_text)
    trainset = dataset.to_dspy_examples("train")
    valset = dataset.to_dspy_examples("val")

    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    try:
        reflection_lm = get_lm_for_role("optimizer", override=optimizer_model)
        optimizer = dspy.GEPA(
            metric=prompt_fitness_metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
        )
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as e:
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(metric=prompt_fitness_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset)

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 5. Extract evolved section text ──────────────────────────────────
    # The optimized module's section_text contains the evolved text
    evolved_text = optimized_module.section_text

    # ── 6. Validate evolved text ─────────────────────────────────────────
    console.print(f"\n[bold]Validating evolved section[/bold]")
    evolved_constraints = validator.validate_all(evolved_text, "prompt_section", baseline_text=section_text)
    all_pass = True
    for c in evolved_constraints:
        icon = "✓" if c.passed else "✗"
        color = "green" if c.passed else "red"
        console.print(f"  [{color}]{icon}[/{color}] {c.constraint_name}: {c.message}")
        if not c.passed:
            all_pass = False

    if not all_pass:
        console.print("[red]✗ Evolved section FAILED constraints — not deploying[/red]")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path("output") / "prompts" / section / f"evolved_FAILED_{timestamp}.txt"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(evolved_text)
        console.print(f"  Saved failed variant to {output_path}")
        return

    # ── 7. Evaluate on holdout ───────────────────────────────────────────
    holdout = dataset.to_dspy_examples("holdout")
    baseline_scores = []
    evolved_scores = []

    with dspy.context(lm=lm):
        for ex in holdout:
            b_pred = baseline_module(task_input=ex.task_input)
            baseline_scores.append(prompt_fitness_metric(ex, b_pred))
            e_pred = optimized_module(task_input=ex.task_input)
            evolved_scores.append(prompt_fitness_metric(ex, e_pred))

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 8a. Regression gate ────────────────────────────────────────────
    passed_gate = check_regression(
        improvement=improvement,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        artifact_type="prompt_section",
        artifact_name=section,
    )

    # ── 8. Report results ────────────────────────────────────────────────
    table = Table(title=f"Prompt Section Evolution — {section}")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row("Behavioral Score", f"{avg_baseline:.3f}", f"{avg_evolved:.3f}",
                  f"[{change_color}]{improvement:+.3f}[/{change_color}]")
    table.add_row("Size", f"{len(section_text):,}", f"{len(evolved_text):,}",
                  f"{len(evolved_text) - len(section_text):+,} chars")
    console.print()
    console.print(table)

    # ── 9. Save output ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "prompts" / section / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "evolved_section.txt").write_text(evolved_text)
    (output_dir / "baseline_section.txt").write_text(section_text)
    metrics = {
        "section": section,
        "timestamp": timestamp,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "baseline_size": len(section_text),
        "evolved_size": len(evolved_text),
        "elapsed_seconds": elapsed,
        "constraints_passed": all_pass,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    console.print(f"\n  Output saved to {output_dir}/")

    if passed_gate:
        console.print(f"\n[bold green]✓ Section improved by {improvement:+.3f}[/bold green]")
        if apply:
            try:
                save_prompt_section(section, evolved_text, config.hermes_agent_path)
                console.print(f"  [green]✓ Applied to {config.hermes_agent_path}/agent/prompt_builder.py[/green]")
            except (ValueError, KeyError) as e:
                console.print(f"  [red]✗ Could not apply: {e}[/red]")
        else:
            console.print(f"  Review diff: diff {output_dir}/baseline_section.txt {output_dir}/evolved_section.txt")
    else:
        console.print(f"\n[yellow]⚠ No improvement (change: {improvement:+.3f})[/yellow]")


@click.command()
@click.option("--section", required=True,
              type=click.Choice(EVOLVABLE_SECTIONS),
              help="System prompt section to evolve")
@click.option("--iterations", default=5, help="Number of GEPA iterations")
@click.option("--optimizer-model", default=None, help="Model for GEPA reflections (default: provider default)")
@click.option("--eval-model", default=None, help="Model for evaluations (default: provider default)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--benchmark-gate", is_flag=True, help="Run TBLite (zero tolerance for regression)")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--apply", is_flag=True, help="Write evolved section back to prompt_builder.py")
@click.option("--create-pr", is_flag=True, help="Create PR in hermes-agent repo")
def main(section, iterations, optimizer_model, eval_model, hermes_repo,
         benchmark_gate, dry_run, apply, create_pr):
    """Evolve a hermes-agent system prompt section using DSPy + GEPA."""
    evolve(
        section=section,
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        benchmark_gate=benchmark_gate,
        dry_run=dry_run,
        apply=apply,
        create_pr=create_pr,
    )


if __name__ == "__main__":
    main()
