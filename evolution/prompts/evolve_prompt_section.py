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
    promote: bool = False,
    min_effect: float = 0.05,
    repeats: int = 2,
    max_calls: int = 200,
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

    # ── 7. Pairwise holdout evaluation (keyword metric only steered GEPA) ──
    from evolution.core import staging, stats_gate
    from evolution.core.compose import generate_outputs
    from evolution.core.pairwise import CallBudget, LLMClient, PairwiseJudge, summarize
    from evolution.prompts.prompt_module import _extract_constant, _replace_constant

    budget = CallBudget(max_calls=max_calls)
    agent = LLMClient(budget, "eval", temperature=0.0, max_tokens=2048)
    judge = PairwiseJudge(LLMClient(budget, "judge", temperature=0.0, max_tokens=2048,
                                    fallback_role="eval"), repeats=repeats, seed=0)
    holdout = [ex.to_dict() for ex in dataset.holdout]
    try:
        base_out = generate_outputs(agent, section_text, holdout, kind="system prompt section")
        evo_out = generate_outputs(agent, evolved_text, holdout, kind="system prompt section")
        pairs = judge.evaluate(holdout, base_out, evo_out)
    except Exception as e:  # BudgetExceeded or provider failure
        console.print(f"[yellow]pairwise eval incomplete: {e}[/yellow]")
        pairs = []
    gate = stats_gate.decide(pairs, min_effect=min_effect, min_pairs=8)
    st = summarize(pairs)
    improvement = gate.improvement

    table = Table(title=f"Prompt Section Evolution — {section}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Holdout pairs", f"{gate.n_pairs} (W{st.wins}/T{st.ties}/L{st.losses})")
    table.add_row("Win score (0.5 = parity)", f"{gate.mean:.3f}")
    table.add_row("95% CI", f"[{gate.ci_low:.3f}, {gate.ci_high:.3f}]  p={gate.p_value:.4f}")
    table.add_row("Decision", gate.decision)
    table.add_row("Size", f"{len(section_text):,} -> {len(evolved_text):,}")
    table.add_row("LLM calls (eval)", f"{budget.used}/{budget.max_calls}")
    console.print()
    console.print(table)

    # ── 8. Stage (default) / promote ────────────────────────────────────
    live_path = config.hermes_agent_path / "agent" / "prompt_builder.py"
    staged_path = None
    if evolved_text.strip() != section_text.strip():
        source = live_path.read_text()
        original = _extract_constant(source, section, live_path)
        new_source = _replace_constant(source, section, original, evolved_text)
        stats = gate.to_dict() | {"wins": st.wins, "ties": st.ties, "losses": st.losses,
                                  "order_inconsistent": st.order_inconsistent,
                                  "elapsed_seconds": elapsed}
        m = staging.finalize(f"prompt_{section}", "prompt_section", live_path, new_source, stats,
                             apply=apply, force_promote=promote,
                             extra_files={"evolved_section.txt": evolved_text,
                                          "baseline_section.txt": section_text,
                                          "metrics.json": {"section": section, "baseline_score": 0.5,
                                                           "evolved_score": gate.mean,
                                                           "improvement": improvement, **stats}})
        staged_path = m["staged_path"]
        console.print(f"  staged → {staged_path}")
        if m.get("status") == "promoted":
            console.print(f"  [green]✓ promoted to {live_path} (backup {m.get('backup_path')})[/green]")
        elif m.get("promotion_error"):
            console.print(f"  [red]✗ promotion aborted: {m['promotion_error']}[/red]")

    if improvement > 0 and gate.decision != "reject":
        console.print(f"\n[bold green]✓ Section improved by {improvement:+.3f} (decision={gate.decision})[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement (change: {improvement:+.3f})[/yellow]")
    print("EVOLUTION_RESULT " + json.dumps(dict(
        success=True, improvement=improvement, ci_low=gate.ci_low, ci_high=gate.ci_high,
        p_value=gate.p_value, decision=gate.decision, staged_path=staged_path,
        n_pairs=gate.n_pairs)), flush=True)


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
@click.option("--apply", is_flag=True, help="Write to prompt_builder.py ONLY if the stats gate says promote")
@click.option("--create-pr", is_flag=True, help="Create PR in hermes-agent repo")
@click.option("--stage-only", is_flag=True, help="Default: stage only, never write live")
@click.option("--promote", "force_promote", is_flag=True, help="Explicitly promote the staged candidate")
@click.option("--min-effect", default=0.05, show_default=True)
@click.option("--repeats", default=2, show_default=True)
@click.option("--max-calls", default=200, show_default=True, help="Cap on pairwise-eval LLM calls")
def main(section, iterations, optimizer_model, eval_model, hermes_repo,
         benchmark_gate, dry_run, apply, create_pr, stage_only, force_promote, min_effect,
         repeats, max_calls):
    """Evolve a hermes-agent system prompt section using DSPy + GEPA."""
    if stage_only:
        apply, force_promote = False, False
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
        promote=force_promote,
        min_effect=min_effect,
        repeats=repeats,
        max_calls=max_calls,
    )


if __name__ == "__main__":
    main()
