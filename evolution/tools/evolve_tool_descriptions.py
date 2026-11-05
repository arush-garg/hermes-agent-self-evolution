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
from evolution.tools.tool_module import (
    ToolDescription,
    ToolSelectionModule,
    load_tool_descriptions,
    tool_selection_metric,
    format_tools_for_prompt,
    apply_evolved_descriptions,
)
from evolution.tools.tool_ratings import feedback_for_tools, flagged_tools, tool_rating_signals

console = Console()

# GEPA budget. ``max_full_evals`` scales with the dataset: 5 evals of a ~290-example set is
# ~1160 metric calls, which overran the runner's 3 h wall-clock limit (2026-11-04). Cap both
# the splits and the total metric calls so a run finishes in well under an hour.
MAX_TRAIN_EXAMPLES = 60
MAX_VAL_EXAMPLES = 30
MAX_METRIC_CALLS = 300
GEPA_THREADS = 4


_SELECT_PREFIX = "Select tool:"


def _correct_tool(ex: EvalExample) -> str:
    """The gold tool: the in-memory attribute, else parsed from ``expected_behavior`` (a dataset
    reloaded with ``--dataset-path`` only round-trips the EvalExample fields)."""
    tool = ex.__dict__.get("correct_tool", "")
    if not tool and (ex.expected_behavior or "").startswith(_SELECT_PREFIX):
        tool = ex.expected_behavior[len(_SELECT_PREFIX):].strip()
    return tool


def _to_dspy_with_tool(examples: list[EvalExample]) -> list[dspy.Example]:
    return [dspy.Example(task_input=ex.task_input, expected_behavior=ex.expected_behavior,
                         correct_tool=_correct_tool(ex)).with_inputs("task_input")
            for ex in examples]


def _gepa_budget(iterations: int, n_train: int, n_val: int) -> int:
    """Metric-call budget: ``iterations`` full passes, never more than ``MAX_METRIC_CALLS``."""
    return max(1, min(iterations * (n_train + n_val), MAX_METRIC_CALLS))


def _bounded_splits(trainset: list, valset: list, flagged: set[str]) -> tuple[list, list]:
    """Cap train/val; tools the user's low ratings point at are kept in val first."""
    def prioritized(examples):
        return sorted(examples, key=lambda e: getattr(e, "correct_tool", "") not in flagged)
    return prioritized(trainset)[:MAX_TRAIN_EXAMPLES], prioritized(valset)[:MAX_VAL_EXAMPLES]


def make_gepa_metric(signals: dict):
    """GEPA metric with textual feedback: the reflection LM sees what was expected and, for the
    tools involved, what the user said about them via /rate."""
    def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        score = tool_selection_metric(gold, pred)
        expected = getattr(gold, "correct_tool", "") or ""
        selected = (getattr(pred, "selected_tool", "") or "").strip()
        feedback = ("Correct tool." if score == 1.0 else
                    f"Expected {expected}, but selected {selected or 'nothing'}.")
        user = feedback_for_tools(signals, [t for t in dict.fromkeys([expected, selected]) if t])
        if user != "none":
            feedback += "\nUser feedback from rated sessions using these tools:\n" + user
        return dspy.Prediction(score=score, feedback=feedback)
    return metric


def _parse_cases(raw: str) -> list[dict]:
    """Parse the LLM's case JSON, tolerating markdown fences and truncation."""
    import re, json
    text = raw.strip()
    match = re.search(r'```json\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            text = match.group(0)
    return json.loads(text)

def _build_tool_selection_dataset(
    tools: list[ToolDescription],
    config: EvolutionConfig,
    signals: Optional[dict] = None,
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
        user_feedback: str = dspy.InputField(
            desc="The real user's /rate feedback on sessions that used these tools ('none' if absent). "
                 "When present, include cases that reproduce the situations the user complained about."
        )
        cases: str = dspy.OutputField(
            desc='JSON array of cases, each with: task_input (str), correct_tool (str), rationale (str), difficulty (easy/medium/hard)'
        )

    lm = get_lm_for_role("judge", override=config.judge_model)
    generator = dspy.ChainOfThought(GenerateToolSelectionCases)

    n_per_tool = max(5, 30 // max(1, len(tools)))
    n_confusion_per_batch = max(1, 20 // max(1, len(tools)))
    batch_size = 5  # tools per LLM call — keeps response within token limits

    cases_raw = []
    # Generate per-batch so each LLM response stays well under the token limit
    # (one huge call for all 56 tools overflows ~4096 tokens).
    with dspy.context(lm=lm):
        for i in range(0, len(tools), batch_size):
            tool_batch = tools[i : i + batch_size]
            tools_json = format_tools_for_prompt(tool_batch)
            feedback = feedback_for_tools(signals or {}, [t.name for t in tool_batch])
            result = generator(
                tools_json=tools_json,
                num_cases_per_tool=n_per_tool,
                # Rated-session complaints get extra cross-tool confusion cases.
                num_confusion_cases=n_confusion_per_batch + (2 if feedback != "none" else 0),
                user_feedback=feedback,
            )
            batch_raw = _parse_cases(result.cases)
            cases_raw.extend(batch_raw)

    examples = []
    for c in cases_raw:
        task = c.get("task_input", "")
        correct = c.get("correct_tool", "")
        difficulty = c.get("difficulty", "medium")
        if not task or not correct:
            continue
        examples.append(EvalExample(
            task_input=task,
            expected_behavior=f"{_SELECT_PREFIX} {correct}",
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
    promote: bool = False,
    min_effect: float = 0.05,
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

    # ── 1b. User /rate signal ────────────────────────────────────────────
    try:
        signals = tool_rating_signals()
    except Exception as e:  # a missing/locked state.db must not block evolution
        console.print(f"  [yellow]Rating signal unavailable: {e}[/yellow]")
        signals = {}
    flagged = flagged_tools(signals)
    console.print(f"  Ratings: {len(signals)} tools in rated sessions, {len(flagged)} in low-rated ones"
                  + (f" ({', '.join(sorted(flagged)[:8])})" if flagged else ""))

    # ── 2. Build evaluation dataset ──────────────────────────────────────
    console.print(f"\n[bold]Building tool selection dataset[/bold]")

    # Global default LM — GEPA uses dspy.settings.lm for student evaluation
    configure_default_lm("eval", override=eval_model, temperature=0.0)
    lm = get_lm_for_role("eval", override=eval_model, temperature=0.0)

    if dataset_path:
        dataset = EvalDataset.load(Path(dataset_path))
        console.print(f"  Loaded dataset: {len(dataset.all_examples)} examples")
    else:
        dataset = _build_tool_selection_dataset(tools, config, signals)
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

    trainset, valset = _bounded_splits(_to_dspy_with_tool(dataset.train), _to_dspy_with_tool(dataset.val),
                                       flagged)
    budget = _gepa_budget(iterations, len(trainset), len(valset))
    console.print(f"  GEPA: {len(trainset)} train / {len(valset)} val, budget {budget} metric calls")

    # ── 5. Run GEPA ─────────────────────────────────────────────────────
    console.print(f"\n[bold cyan]Running GEPA optimization ({iterations} iterations)...[/bold cyan]\n")
    start_time = time.time()

    try:
        reflection_lm = get_lm_for_role("optimizer", override=optimizer_model)
        optimizer = dspy.GEPA(
            metric=make_gepa_metric(signals),
            max_metric_calls=budget,
            num_threads=GEPA_THREADS,
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

    # ── 7a. Statistical gate on paired per-example accuracy ─────────────
    # Tool selection has an exact, grader-free metric, so we gate on the paired
    # difference: per-example score = 0.5 + (evolved - baseline) / 2 (0.5 = tie).
    from evolution.core import staging, stats_gate
    paired = [0.5 + (e - b_) / 2 for b_, e in zip(baseline_scores, evolved_scores)]
    gate = stats_gate.decide(paired, min_effect=min_effect, min_pairs=8)
    console.print(f"  stats gate: paired score={gate.mean:.3f} CI=[{gate.ci_low:.3f}, {gate.ci_high:.3f}] "
                  f"p={gate.p_value:.4f} n={gate.n_pairs} decision={gate.decision}")

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

    # ── 8. Stage (default) ───────────────────────────────────────────────
    stats = gate.to_dict() | {"baseline_score": avg_baseline, "evolved_score": avg_evolved,
                              "accuracy_improvement": improvement, "tools_evolved": len(tools),
                              "elapsed_seconds": elapsed, "iterations": iterations,
                              "metric_call_budget": budget, "rated_tools": len(signals),
                              "low_rated_tools": sorted(flagged)}
    staged_paths = []
    try:
        evolved_raw = json.loads(optimized_module.tools_text)
    except (json.JSONDecodeError, TypeError) as e:
        console.print(f"[red]Failed to parse evolved tool descriptions: {e}[/red]")
        evolved_raw = []
    evolved_tools = [ToolDescription(name=td.get("name", ""), description=td.get("description", ""),
                                     parameters=td.get("parameters", {})) for td in evolved_raw]
    changes = apply_evolved_descriptions(tools, evolved_tools, config.hermes_agent_path) if evolved_tools else []
    # Each changed source file is staged as its own artifact (own backup/rollback).
    for rel_path, new_content in changes or []:
        abs_path = config.hermes_agent_path / rel_path
        art = "tool_" + str(rel_path).replace("/", "_")
        m = staging.finalize(art, "tool_source", abs_path, new_content, stats,
                             apply=apply, force_promote=promote,
                             extra_files={"evolved_tools.json": evolved_raw, "metrics.json": stats})
        staged_paths.append(m["staged_path"])
        status = m.get("status")
        console.print(f"  {rel_path}: {status}" + (f" (abort: {m['promotion_error']})" if m.get("promotion_error") else ""))
    if not changes:
        console.print("[yellow]No description changes detected.[/yellow]")

    if improvement > 0 and gate.decision != "reject":
        console.print(f"\n[bold green]✓ Tool descriptions improved by {improvement:+.3f} (decision={gate.decision})[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ No improvement (change: {improvement:+.3f})[/yellow]")
    print("EVOLUTION_RESULT " + json.dumps(dict(
        success=True, improvement=improvement, ci_low=gate.ci_low, ci_high=gate.ci_high,
        p_value=gate.p_value, decision=gate.decision,
        staged_path=staged_paths[0] if staged_paths else None,
        staged_paths=staged_paths, n_pairs=gate.n_pairs)), flush=True)


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
@click.option("--apply", is_flag=True, help="Apply to source ONLY if the stats gate says promote")
@click.option("--stage-only", is_flag=True, help="Default: stage only, never write live")
@click.option("--promote", "force_promote", is_flag=True, help="Explicitly promote staged files")
@click.option("--min-effect", default=0.05, show_default=True)
def main(iterations, eval_source, dataset_path, optimizer_model, eval_model,
         hermes_repo, run_tests, benchmark_gate, dry_run, create_pr, apply, stage_only,
         force_promote, min_effect):
    """Evolve hermes-agent tool descriptions using DSPy + GEPA."""
    if stage_only:
        apply, force_promote = False, False
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
        promote=force_promote,
        min_effect=min_effect,
    )


if __name__ == "__main__":
    main()
