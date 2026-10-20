"""Evolve hermes-agent tool descriptions using DSPy + GEPA.

Usage:
    python -m evolution.tools.evolve_tool_descriptions --iterations 5
    python -m evolution.tools.evolve_tool_descriptions --dry-run
    python -m evolution.tools.evolve_tool_descriptions --benchmark-gate --iterations 5
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
from evolution.core.llm_api import configure_default_lm, get_lm_for_role

from evolution.core.config import EvolutionConfig
from evolution.core.dataset_builder import SyntheticDatasetBuilder, EvalDataset, EvalExample
from evolution.core.constraints import ConstraintValidator
from evolution.core.regression_gate import check_regression
from evolution.tools.tool_module import (
    ToolDescription,
    ToolSelectionModule,
    load_tool_descriptions,
    tool_selection_metric,
    format_tools_for_prompt,
    apply_evolved_descriptions,
)

console = Console()


def _build_tool_selection_dataset(
    tools: list[ToolDescription],
    config: EvolutionConfig,
) -> EvalDataset:
    """Generate a synthetic tool-selection evaluation dataset.

    Creates (task_input, correct_tool, correct_params) triples:
    - 10–20 clear-cut tasks per tool
    - 10–20 "confusing" tasks where two tools could work but one is better
    """

    class GenerateToolSelectionCases(dspy.Signature):
        """Generate tool selection evaluation cases for AI agent tool descriptions.

        For each tool, generate realistic tasks where that tool is clearly the right
        choice. Also generate "confusing" cases where two tools could work but one is
        clearly better. Each case must include the task, the correct tool name, and
        why the other tools are not the best choice.
        """
        tools_json: str = dspy.InputField(
            desc="JSON array of tool objects with name, description, and parameters"
        )
        num_cases_per_tool: int = dspy.InputField(desc="Number of clear cases per tool")
        num_confusion_cases: int = dspy.InputField(desc="Number of cross-tool confusion cases")
        cases: str = dspy.OutputField(
            desc='JSON array of cases, each with: task_input (str), correct_tool (str), rationale (str), difficulty (easy/medium/hard)'
        )

    lm = get_lm_for_role("judge", override=config.judge_model)
    generator = dspy.ChainOfThought(GenerateToolSelectionCases)

    tools_json = format_tools_for_prompt(tools)
    n_per_tool = max(3, 15 // max(1, len(tools)))

    with dspy.context(lm=lm):
        result = generator(
            tools_json=tools_json,
            num_cases_per_tool=n_per_tool,
            num_confusion_cases=10,
        )

    try:
        cases_raw = json.loads(result.cases)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\[.*\]', result.cases, re.DOTALL)
        if match:
            cases_raw = json.loads(match.group())
        else:
            raise ValueError(f"Could not parse tool selection cases: {result.cases[:200]}")

    examples = []
    for c in cases_raw:
        task = c.get("task_input", "")
        correct = c.get("correct_tool", "")
        difficulty = c.get("difficulty", "medium")
        if not task or not correct:
            continue
        examples.append(EvalExample(
            task_input=task,
            expected_behavior=f"Select tool: {correct}",
            difficulty=difficulty,
            category="tool_selection",
            source="synthetic",
        ))
        # Store correct_tool as an attribute for the metric
        examples[-1].__dict__["correct_tool"] = correct

    import random
    random.shuffle(examples)
    n = len(examples)
    n_train = max(1, int(n * 0.6))
    n_val = max(1, int(n * 0.2))

    return EvalDataset(
        train=examples[:n_train],
        val=examples[n_train:n_train + n_val],
        holdout=examples[n_train + n_val:],
    )


def _validate_tool_constraints(
    tools: list[ToolDescription],
    validator: ConstraintValidator,
    baseline_tools: Optional[list[ToolDescription]] = None,
) -> tuple[bool, list[str]]:
    """Validate all tool descriptions against size constraints."""
    baseline_map = {t.name: t for t in (baseline_tools or [])}
    all_pass = True
    messages = []

    for tool in tools:
        # Top-level description
        result = validator.validate_all(
            tool.description,
            "tool_description",
            baseline_text=baseline_map.get(tool.name, tool).description if baseline_tools else None,
        )
        for r in result:
            if not r.passed:
                all_pass = False
                messages.append(f"  {tool.name}: {r.constraint_name}: {r.message}")

        # Parameter descriptions
        for param_name, param_desc in tool.parameters.items():
            param_result = validator.validate_all(
                param_desc,
                "param_description",
            )
            for r in param_result:
                if not r.passed:
                    all_pass = False
                    messages.append(f"  {tool.name}.{param_name}: {r.constraint_name}: {r.message}")

    return all_pass, messages


def evolve(
    iterations: int = 5,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    benchmark_gate: bool = False,
    dry_run: bool = False,
    create_pr: bool = False,
    apply: bool = False,
) -> None:
    """Evolve all tool descriptions using DSPy + GEPA.

    Parameters
    ----------
    apply: bool
        When True, write any improved tool descriptions back to the hermes-agent
        source files using ``apply_evolved_descriptions``.
    """

    config = EvolutionConfig(
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=None,
        run_pytest=run_tests,
        run_tblite=benchmark_gate,
    )
    if hermes_repo:
        config.hermes_agent_path = Path(hermes_repo)

    console.print("\n[bold cyan]Tool Description Evolution[/bold cyan]\n")

    # ── 1. Load tool descriptions ────────────────────────────────────────
    console.print(f"[bold]Loading tool descriptions[/bold] from {config.hermes_agent_path}")
    tools = load_tool_descriptions(config.hermes_agent_path)

    if not tools:
        console.print("[red]✗ No tool descriptions found. Check HERMES_AGENT_REPO path.[/red]")
        sys.exit(1)

    console.print(f"  Found {len(tools)} tools: {', '.join(t.name for t in tools[:5])}{'...' if len(tools) > 5 else ''}")

    if dry_run:
        console.print(f"\n[bold green]DRY RUN — setup validated.[/bold green]")
        console.print(f"  Would generate tool selection dataset (source: {eval_source})")
        console.print(f"  Would run GEPA optimization ({iterations} iterations) on {len(tools)} tools")
        console.print(f"  Benchmark gate: {'enabled' if benchmark_gate else 'disabled'}")
        return

    # ── 2. Build evaluation dataset ──────────────────────────────────────
    console.print(f"\n[bold]Building tool selection dataset[/bold]")

    # Global default LM — GEPA uses dspy.settings.lm for student evaluation
    configure_default_lm("eval", override=eval_model, temperature=0.0)
    lm = get_lm_for_role("eval", override=eval_model, temperature=0.0)

    if dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        dataset = _build_tool_selection_dataset(tools, config)
        save_path = Path("datasets") / "tools"
        dataset.save(save_path)
        console.print(f"  Generated {len(dataset.all_examples)} tool selection examples")

    console.print(f"  Split: {len(dataset.train)} train / {len(dataset.val)} val / {len(dataset.holdout)} holdout")

    # ── 3. Validate baseline constraints ────────────────────────────────
    console.print(f"\n[bold]Validating baseline constraints[/bold]")
    validator = ConstraintValidator(config)
    all_pass, messages = _validate_tool_constraints(tools, validator)
    if messages:
        for msg in messages:
            console.print(f"  [yellow]{msg}[/yellow]")
    if all_pass:
        console.print("  [green]✓ All tool descriptions pass constraints[/green]")

    # ── 4. Set up GEPA optimizer ─────────────────────────────────────────
    console.print(f"\n[bold]Configuring optimizer[/bold]")

    tools_text = format_tools_for_prompt(tools)
    baseline_module = ToolSelectionModule(tools_text)

    # Convert dataset to DSPy examples with correct_tool attribute
    def _to_dspy_with_tool(examples: list[EvalExample]) -> list[dspy.Example]:
        result = []
        for ex in examples:
            dspy_ex = dspy.Example(
                task_input=ex.task_input,
                expected_behavior=ex.expected_behavior,
                correct_tool=ex.__dict__.get("correct_tool", ""),
            ).with_inputs("task_input")
            result.append(dspy_ex)
        return result

    trainset = _to_dspy_with_tool(dataset.train)
    valset = _to_dspy_with_tool(dataset.val)

    # ── 5. Run GEPA ─────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    try:
        reflection_lm = get_lm_for_role("optimizer", override=optimizer_model)
        optimizer = dspy.GEPA(
            metric=tool_selection_metric,
            max_full_evals=iterations,
            reflection_lm=reflection_lm,
        )
        optimized_module = optimizer.compile(baseline_module, trainset=trainset, valset=valset)
    except Exception as e:
        console.print(f"[yellow]GEPA not available ({e}), falling back to MIPROv2[/yellow]")
        optimizer = dspy.MIPROv2(metric=tool_selection_metric, auto="light")
        optimized_module = optimizer.compile(baseline_module, trainset=trainset)

    elapsed = time.time() - start_time
    console.print(f"\n  Optimization completed in {elapsed:.1f}s")

    # ── 6. Evaluate on holdout ───────────────────────────────────────────
    holdout = _to_dspy_with_tool(dataset.holdout)
    baseline_scores = []
    evolved_scores = []

    with dspy.context(lm=lm):
        for ex in holdout:
            b_pred = baseline_module(task_input=ex.task_input)
            baseline_scores.append(tool_selection_metric(ex, b_pred))

            e_pred = optimized_module(task_input=ex.task_input)
            evolved_scores.append(tool_selection_metric(ex, e_pred))

    avg_baseline = sum(baseline_scores) / max(1, len(baseline_scores))
    avg_evolved = sum(evolved_scores) / max(1, len(evolved_scores))
    improvement = avg_evolved - avg_baseline

    # ── 7a. Regression gate ──────────────────────────────────────────────
    passed_gate = check_regression(
        improvement=improvement,
        baseline_score=avg_baseline,
        evolved_score=avg_evolved,
        artifact_type="tool_descriptions",
        artifact_name="all",
    )

    # ── 7. Report results ────────────────────────────────────────────────
    table = Table(title="Tool Description Evolution Results")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Evolved", justify="right")
    table.add_column("Change", justify="right")

    change_color = "green" if improvement > 0 else "red"
    table.add_row(
        "Selection Accuracy (holdout)",
        f"{avg_baseline:.3f}",
        f"{avg_evolved:.3f}",
        f"[{change_color}]{improvement:+.3f}[/{change_color}]",
    )
    console.print()
    console.print(table)

    # ── 8. Save output ───────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("output") / "tool_descriptions" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "timestamp": timestamp,
        "iterations": iterations,
        "optimizer_model": optimizer_model,
        "eval_model": eval_model,
        "baseline_score": avg_baseline,
        "evolved_score": avg_evolved,
        "improvement": improvement,
        "tools_evolved": len(tools),
        "elapsed_seconds": elapsed,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    console.print(f"\n  Output saved to {output_dir}/")

    if passed_gate:
        console.print(f"\n[bold green]✓ Tool descriptions improved by {improvement:+.3f}[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement (change: {improvement:+.3f})[/yellow]")
        console.print("  Try: more iterations, larger dataset, or different optimizer model")

    # ── 9. Optionally apply evolved descriptions ──────────────────────────────────────
    if apply and passed_gate:
        # ``optimized_module.tools_text`` contains the (potentially) evolved JSON string.
        try:
            evolved_raw = json.loads(optimized_module.tools_text)
        except json.JSONDecodeError as e:
            console.print(f"[red]Failed to parse evolved tool descriptions: {e}[/red]")
            evolved_raw = []

        evolved_tools: list[ToolDescription] = []
        for td in evolved_raw:
            name = td.get("name", "")
            description = td.get("description", "")
            parameters = td.get("parameters", {})
            evolved_tools.append(ToolDescription(name=name, description=description, parameters=parameters))

        changes = apply_evolved_descriptions(tools, evolved_tools, config.hermes_agent_path)
        if changes:
            for rel_path, new_content in changes:
                abs_path = config.hermes_agent_path / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                abs_path.write_text(new_content)
            console.print(f"[green]Applied evolved descriptions to {len(changes)} file(s).[/green]")
        else:
            console.print("[yellow]No description changes detected to apply.[/yellow]")


@click.command()
@click.option("--iterations", default=5, help="Number of GEPA iterations")
@click.option("--eval-source", default="synthetic", type=click.Choice(["synthetic"]),
              help="Source for evaluation dataset")
@click.option("--dataset-path", default=None, help="Path to existing eval dataset")
@click.option("--optimizer-model", default=None, help="Model for GEPA reflections (default: provider default)")
@click.option("--eval-model", default=None, help="Model for evaluations (default: provider default)")
@click.option("--hermes-repo", default=None, help="Path to hermes-agent repo")
@click.option("--run-tests", is_flag=True, help="Run pytest as constraint gate")
@click.option("--benchmark-gate", is_flag=True, help="Run TBLite as regression gate")
@click.option("--dry-run", is_flag=True, help="Validate setup without running optimization")
@click.option("--create-pr", is_flag=True, help="Create PR in hermes-agent repo")
@click.option("--apply", is_flag=True, help="Apply evolved descriptions back to source files if improved")
def main(iterations, eval_source, dataset_path, optimizer_model, eval_model,
         hermes_repo, run_tests, benchmark_gate, dry_run, create_pr, apply):
    """Evolve hermes-agent tool descriptions using DSPy + GEPA."""
    evolve(
        iterations=iterations,
        eval_source=eval_source,
        dataset_path=dataset_path,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        hermes_repo=hermes_repo,
        run_tests=run_tests,
        benchmark_gate=benchmark_gate,
        dry_run=dry_run,
        create_pr=create_pr,
        apply=apply,
    )


if __name__ == "__main__":
    main()
