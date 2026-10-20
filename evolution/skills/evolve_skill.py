"""Evolve a Hermes Agent skill with Propose → Probe → Compose + pairwise judging.

Decision pipeline (no keyword-overlap metric in the keep/reject path):
  * K candidate rewrites are proposed (general / domain / lessons-seeded ...),
  * each is probed pairwise vs the baseline on a small VAL slice,
  * the top two are merged into a composite,
  * the best contender (baseline always included) is judged pairwise on
    HOLDOUT, then gated by a paired bootstrap (:mod:`evolution.core.stats_gate`).

Deployment is STAGED by default (output/<skill>/<ts>/ + staging/<skill>.json).
The live SKILL.md is only overwritten with ``--promote`` or when
``decision == 'promote'`` and ``--apply`` — with backup and sha checks
(:mod:`evolution.core.staging`).

Usage:
    python -m evolution.skills.evolve_skill --skill github-workflow --iterations 2 --candidates 2
    python -m evolution.skills.evolve_skill --skill arxiv --eval-source golden --dataset-path datasets/skills/arxiv/
"""

from __future__ import annotations

import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console

from evolution.core import staging, stats_gate
from evolution.core.compose import (load_failure_context, load_lessons, propose_probe_compose,
                                    record_lesson)
from evolution.core.config import EvolutionConfig, resolve_hermes_agent_path
from evolution.core.constraints import ConstraintValidator
from evolution.core.pairwise import CallBudget, LLMClient, PairwiseJudge, summarize
from evolution.skills.skill_module import find_skill, load_skill, reassemble_skill

console = Console()

EVAL_SOURCES = ["synthetic", "golden", "sessiondb", "mixed"]


def _examples_as_dicts(exs) -> list[dict]:
    out = []
    for e in exs:
        if isinstance(e, dict):
            out.append(e)
        elif hasattr(e, "to_dict"):
            out.append(e.to_dict())
        else:
            out.append(dict(vars(e)))
    return out


def _load_dataset(skill_name, skill, eval_source, dataset_path, config, eval_model, min_total):
    from evolution.core.dataset_builder import (EvalDataset, GoldenDatasetLoader,
                                                SyntheticDatasetBuilder)

    if dataset_path and Path(dataset_path).exists() and eval_source in ("golden", "synthetic", "mixed"):
        ds = GoldenDatasetLoader.load(Path(dataset_path)) if eval_source == "golden" \
            else EvalDataset.load(Path(dataset_path))
        if ds.all_examples:
            return ds
    if eval_source == "golden":
        raise click.UsageError("--eval-source golden requires an existing --dataset-path")
    if eval_source == "sessiondb":
        from evolution.core.external_importers import build_dataset_from_external
        save = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        return build_dataset_from_external(skill_name=skill_name, skill_text=skill["raw"],
                                           sources=["claude-code", "copilot", "hermes"],
                                           output_path=save, model=eval_model)
    if eval_source == "mixed":
        from evolution.core.dataset_builder import build_mixed_dataset
        save = Path(dataset_path) if dataset_path else Path("datasets") / "skills" / skill_name
        return build_mixed_dataset(skill_name, skill["raw"], save, config, minimum=min_total)
    builder = SyntheticDatasetBuilder(config)
    ds = builder.generate(artifact_text=skill["raw"], artifact_type="skill", num_cases=min_total)
    save = Path("datasets") / "skills" / skill_name
    ds.save(save)
    console.print(f"  dataset saved to {save}/")
    return ds


def _resplit(ds, n_val: int, n_holdout_min: int, seed: int):
    """Leakage-safe split adjustment.

    * Real-session rows keep the builder's session/chronological split; they are never
      moved into holdout here (that could put an older session in holdout and a newer one
      of the same task in train).
    * Only session-less (synthetic) rows may be moved from train/val to top up holdout.
    * Any train/val row sharing a session_id or identical task_input with holdout is
      dropped (no leakage).
    """
    train, val, hold = (_examples_as_dicts(ds.train), _examples_as_dicts(ds.val),
                        _examples_as_dicts(ds.holdout))
    rng = random.Random(seed)
    movable = [e for e in train + val if not e.get("session_id")]
    while len(hold) < n_holdout_min and len(movable) > 2:
        e = movable.pop(rng.randrange(len(movable)))
        hold.append(e)
        (train if e in train else val).remove(e)
    hold_sids = {e.get("session_id") for e in hold if e.get("session_id")}
    hold_tasks = {str(e.get("task_input", "")).strip() for e in hold}

    def clean(xs):
        return [e for e in xs if e.get("session_id") not in hold_sids
                and str(e.get("task_input", "")).strip() not in hold_tasks]
    train, val = clean(train), clean(val)
    if len(val) < 2 and len(train) > 2:
        val += [train.pop() for _ in range(min(2, len(train) - 1))]
    return train, val[:n_val], hold


def _emit(result: dict) -> None:
    print("EVOLUTION_RESULT " + json.dumps(result, default=str), flush=True)


def evolve(
    skill_name: str,
    iterations: int = 10,
    eval_source: str = "synthetic",
    dataset_path: Optional[str] = None,
    optimizer_model: Optional[str] = None,
    eval_model: Optional[str] = None,
    hermes_repo: Optional[str] = None,
    run_tests: bool = False,
    dry_run: bool = False,
    apply: bool = False,
    candidates: int = 3,
    promote: bool = False,
    min_effect: float = 0.05,
    repeats: int = 2,
    max_calls: int = 400,
    seed: int = 0,
    val_size: int = 6,
    holdout_min: int = 10,
    noise_check: bool = True,
) -> dict:
    start = time.time()
    config = EvolutionConfig(
        hermes_agent_path=resolve_hermes_agent_path(hermes_repo),
        iterations=iterations,
        optimizer_model=optimizer_model,
        eval_model=eval_model,
        judge_model=None,
        run_pytest=run_tests,
    )
    console.print(f"\n[bold cyan]🧬 Hermes Agent Self-Evolution[/bold cyan] — skill: [bold]{skill_name}[/bold]")
    skill_path = find_skill(skill_name, config.hermes_agent_path)
    if not skill_path:
        console.print(f"[red]✗ Skill '{skill_name}' not found[/red]")
        sys.exit(1)
    skill = load_skill(skill_path)
    console.print(f"  Loaded: {skill_path} ({len(skill['raw']):,} chars)")
    console.print(f"  candidates={candidates} repeats={repeats} min_effect={min_effect} "
                  f"max_calls={max_calls} mode={'PROMOTE' if promote else ('apply-if-significant' if apply else 'stage-only')}")
    if dry_run:
        console.print("[bold green]DRY RUN — setup validated.[/bold green]")
        return {}

    budget = CallBudget(max_calls=max_calls)
    # Rough bound on proposal rounds so --iterations still means something: K proposals per run.
    k = max(1, min(candidates, max(1, iterations) * 2))

    # ── dataset ──
    min_total = max(config.eval_dataset_size, holdout_min + val_size + 4)
    console.print(f"\n[bold]Dataset[/bold] (source: {eval_source})")
    dataset = _load_dataset(skill_name, skill, eval_source, dataset_path, config, eval_model, min_total)
    budget.spend("dataset")  # synthetic/mixed generation ≈ 1 call (approximate, counted)
    train, val, holdout = _resplit(dataset, val_size, holdout_min, seed)
    n_real_h = sum(1 for e in holdout if stats_gate.is_real_example(e))
    console.print(f"  train={len(train)} val={len(val)} holdout={len(holdout)} "
                  f"(holdout real={n_real_h}, synthetic={len(holdout) - n_real_h})")
    if not holdout or not val:
        console.print("[red]✗ dataset too small[/red]")
        res = dict(success=False, improvement=0.0, ci_low=None, ci_high=None, p_value=None,
                   decision="reject", staged_path=None, n_pairs=0, error="dataset too small")
        _emit(res)
        return res

    # ── LLM roles ──
    agent = LLMClient(budget, "eval", temperature=0.0, max_tokens=2048, model=None)
    noise_agent = LLMClient(budget, "eval", temperature=0.7, max_tokens=2048, cache=False)
    optimizer = LLMClient(budget, "optimizer", temperature=0.7, max_tokens=12000)
    judge_llm = LLMClient(budget, "judge", temperature=0.0, max_tokens=2048, fallback_role="eval")
    judge = PairwiseJudge(judge_llm, repeats=repeats, seed=seed)
    console.print(f"  models: agent={agent.model} optimizer={optimizer.model} judge={judge_llm.model}")

    task_classes = sorted({e.get("task_class") or e.get("category") or "" for e in train + val} | {skill_name})
    lessons = load_lessons(task_classes)
    failure_ctx = load_failure_context()
    console.print(f"  lessons={len(lessons)} failure_context={'yes' if failure_ctx else 'no'}")

    validator = ConstraintValidator(config)

    def _validate(body: str) -> Optional[str]:
        full = reassemble_skill(skill["frontmatter"], body)
        bad = [c for c in validator.validate_all(full, "skill", baseline_text=skill["raw"]) if not c.passed]
        if bad:
            return "; ".join(c.message for c in bad)
        ok, why = staging.validate_frontmatter(full, skill["raw"])
        return None if ok else why

    console.print(f"\n[bold cyan]Propose → Probe → Compose (K={k})[/bold cyan]")
    result = propose_probe_compose(
        baseline_body=skill["body"], train=train, val=val, holdout=holdout,
        agent_chat=agent, optimizer_chat=optimizer, judge=judge, budget=budget, k=k,
        kind="skill", lessons=lessons, failure_context=failure_ctx,
        max_growth=0.3, min_effect=min_effect, min_pairs=8, seed=seed,
        noise_agent_chat=noise_agent if noise_check else None,
        validate=_validate, log=lambda s: console.print(s),
    )
    gate = stats_gate.apply_provenance_guard(result.gate, holdout)
    n_real = sum(1 for e in holdout if stats_gate.is_real_example(e))
    st = summarize(result.holdout_results)
    elapsed = time.time() - start

    console.print(f"\n[bold]Holdout pairwise[/bold]: n={gate.n_pairs} W{st.wins}/T{st.ties}/L{st.losses} "
                  f"(order-inconsistent={st.order_inconsistent}, judge failures={st.judge_failures})")
    console.print(f"  mean win score={gate.mean:.3f}  95% CI=[{gate.ci_low:.3f}, {gate.ci_high:.3f}]  "
                  f"p={gate.p_value:.4f}  decision={gate.decision.upper()}")
    if gate.reasons:
        console.print(f"  reasons: {'; '.join(gate.reasons)}")
    if result.noise:
        console.print(f"  noise check (baseline vs baseline, n={result.noise['n']}): "
                      f"mean={result.noise['mean']:.3f} CI=[{result.noise['ci_low']:.3f}, {result.noise['ci_high']:.3f}]")
    console.print(f"  LLM calls: {budget.used}/{budget.max_calls} {budget.by_role}")

    improvement = gate.improvement
    staged_path = None
    promoted = False
    stats = gate.to_dict() | {
        "decision": gate.decision,
        "wins": st.wins, "ties": st.ties, "losses": st.losses,
        "order_inconsistent": st.order_inconsistent, "judge_failures": st.judge_failures,
        "llm_calls": budget.summary(), "elapsed_seconds": elapsed,
        "winner": result.winner.name if result.winner else None,
        "candidates": [{"name": c.name, "probe_score": c.probe_score, "error": c.error}
                       for c in result.candidates]
                      + ([{"name": result.composite.name, "probe_score": result.composite.probe_score,
                           "error": result.composite.error}] if result.composite else []),
        "notes": result.notes,
        "holdout_real": n_real, "holdout_synthetic": len(holdout) - n_real,
    }
    if result.winner is not None:
        evolved_full = reassemble_skill(skill["frontmatter"], result.winner.body)
        metrics = {"timestamp": datetime.now().isoformat(), "skill_name": skill_name,
                   "baseline_score": 0.5, "evolved_score": gate.mean, "improvement": improvement,
                   **{k_: stats[k_] for k_ in ("decision", "ci_low", "ci_high", "p_value", "n_pairs")}}
        manifest = staging.finalize(
            skill_name, "skill", skill_path, evolved_full, stats,
            apply=apply, force_promote=promote,
            extra_files={"baseline_skill.md": skill["raw"], "metrics.json": metrics,
                         "holdout_pairs.json": [vars(r) for r in result.holdout_results]})
        staged_path = manifest["staged_path"]
        promoted = manifest.get("status") == "promoted"
        console.print(f"\n  staged → {staged_path}")
        console.print(f"  manifest → {staging.manifest_path(skill_name)}")
        if manifest.get("promotion_error"):
            console.print(f"[red]  promotion aborted: {manifest['promotion_error']}[/red]")
        elif promoted:
            console.print(f"[bold green]  ✓ promoted to {skill_path} (backup {manifest.get('backup_path')})[/bold green]")
            console.print(f"    rollback: python -m evolution.core.staging rollback --artifact {skill_name}")
        if gate.decision == "promote":
            record_lesson(skill_name, cue=skill_name,
                          lesson=f"Skill '{skill_name}' improved via {result.winner.name}: "
                                 f"{result.winner.strategy[:200]}",
                          evidence=f"holdout pairwise mean={gate.mean:.3f} CI=[{gate.ci_low:.3f},"
                                   f"{gate.ci_high:.3f}] n={gate.n_pairs} p={gate.p_value:.4f}")

    if improvement > 0 and gate.decision != "reject":
        console.print(f"\n[bold green]✓ Evolution improved skill by {improvement:+.3f} (pairwise win score "
                      f"{gate.mean:.3f}); decision={gate.decision}[/bold green]")
    else:
        console.print(f"\n[yellow]⚠ Evolution did not improve skill (change: {improvement:+.3f})[/yellow]")

    res = dict(success=True, improvement=improvement, ci_low=gate.ci_low, ci_high=gate.ci_high,
               p_value=gate.p_value, decision=gate.decision, staged_path=staged_path,
               n_pairs=gate.n_pairs, promoted=promoted, llm_calls=budget.used,
               noise=result.noise, holdout_real=n_real, holdout_synthetic=len(holdout) - n_real)
    _emit(res)
    return res


@click.command()
@click.option("--skill", required=True, help="Name of the skill to evolve")
@click.option("--iterations", default=10, help="Search effort hint (bounds proposals); budget is --max-calls")
@click.option("--eval-source", default="synthetic", type=click.Choice(EVAL_SOURCES))
@click.option("--dataset-path", default=None, help="Path to existing eval dataset dir (JSONL splits)")
@click.option("--optimizer-model", default=None)
@click.option("--eval-model", default=None)
@click.option("--hermes-repo", default=None)
@click.option("--run-tests", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--apply", is_flag=True, help="Promote to live ONLY if the stats gate says 'promote'")
@click.option("--candidates", default=3, show_default=True, help="K candidate proposals")
@click.option("--stage-only", is_flag=True, default=False, help="Default behaviour: stage, never write live")
@click.option("--promote", "force_promote", is_flag=True, help="Explicitly promote the staged winner")
@click.option("--min-effect", default=0.05, show_default=True)
@click.option("--repeats", default=2, show_default=True, help="Judge orders per pair (>=2 runs both)")
@click.option("--max-calls", default=400, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--no-noise-check", is_flag=True)
def main(skill, iterations, eval_source, dataset_path, optimizer_model, eval_model, hermes_repo,
         run_tests, dry_run, apply, candidates, stage_only, force_promote, min_effect, repeats,
         max_calls, seed, no_noise_check):
    if stage_only:
        apply, force_promote = False, False
    try:
        evolve(skill_name=skill, iterations=iterations, eval_source=eval_source,
               dataset_path=dataset_path, optimizer_model=optimizer_model, eval_model=eval_model,
               hermes_repo=hermes_repo, run_tests=run_tests, dry_run=dry_run, apply=apply,
               candidates=candidates, promote=force_promote, min_effect=min_effect,
               repeats=repeats, max_calls=max_calls, seed=seed, noise_check=not no_noise_check)
    except (SystemExit, click.UsageError):
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        _emit(dict(success=False, improvement=0.0, ci_low=None, ci_high=None, p_value=None,
                   decision="reject", staged_path=None, n_pairs=0, error=str(e)))
        sys.exit(2)


if __name__ == "__main__":
    main()
