"""Bound self-evolution disk usage without touching staging.

``housekeeping`` first builds a plan (list of actions) and only executes it when
``dry_run`` is False, so dry runs are strictly read-only and report what *would* happen.

Protected (never rotated/deleted):
  * any path passed in ``protect`` (e.g. the current runner log, current output run),
  * targets of ``*_latest.log`` symlinks in logs/,
  * the newest run directory of every artifact in output/,
  * anything under staging/ (not scanned at all).
"""
import gzip
import shutil
from pathlib import Path

MB = 1024 * 1024
LOG_ROTATE = 5 * MB
LOG_KEEP_GZ = 3
LOG_CAP = 100 * MB
OUTPUT_KEEP = 3
OUTPUT_CAP = 150 * MB
DATASET_CAP = 50 * MB
PLAN = {'logs': 'rotate >5 MiB; keep 3 gz; cap 100 MiB',
        'output': 'newest 3 runs/artifact; cap 150 MiB', 'datasets': 'cap 50 MiB'}


def size(path):
    path = Path(path)
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def _files(root):
    return [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]


def _is_protected(path, protect):
    path = path.resolve()
    return any(path == q or q in path.parents or path in q.parents for q in protect)


def plan_housekeeping(repo, protect=()):
    """Return (actions, projected_after). Pure: performs no writes."""
    root = Path(repo)
    protect = {Path(p).resolve() for p in protect}
    logs, out, datasets = root / 'logs', root / 'output', root / 'datasets'
    if logs.exists():
        protect |= {p.resolve() for p in logs.glob('*_latest.log') if p.is_symlink() and p.exists()}
    actions = []
    sizes = {'logs': size(logs), 'output': size(out), 'datasets': size(datasets)}
    gone = set()

    # logs: rotate big plain logs, then cap total oldest-first
    if logs.exists():
        for p in sorted(logs.iterdir()):
            if (p.is_file() and not p.is_symlink() and not p.name.endswith('.gz')
                    and p.stat().st_size > LOG_ROTATE and not _is_protected(p, protect)):
                actions.append(('rotate_log', p))
                # gzip roughly 10:1 for logs; projected only
                sizes['logs'] -= p.stat().st_size - p.stat().st_size // 10
                gone.add(p)
        for p in sorted(logs.glob('*.gz')):
            stem = p.name.split('.')
            if len(stem) > 2 and stem[-2].isdigit() and int(stem[-2]) >= LOG_KEEP_GZ:
                actions.append(('delete', p)); sizes['logs'] -= p.stat().st_size; gone.add(p)
        for p in sorted(_files(logs), key=lambda p: p.stat().st_mtime):
            if sizes['logs'] <= LOG_CAP:
                break
            if p in gone or _is_protected(p, protect):
                continue
            actions.append(('delete', p)); sizes['logs'] -= p.stat().st_size; gone.add(p)

    # output: per artifact keep newest OUTPUT_KEEP run dirs; newest always protected
    if out.exists():
        runs_all = []
        for artifact in sorted(p for p in out.iterdir() if p.is_dir()):
            runs = sorted((p for p in artifact.iterdir() if p.is_dir()),
                          key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
            if runs:
                protect.add(runs[0].resolve())
            for old in runs[OUTPUT_KEEP:]:
                if not _is_protected(old, protect):
                    actions.append(('rmtree', old)); sizes['output'] -= size(old); gone.add(old)
            runs_all += [r for r in runs[1:OUTPUT_KEEP] if r not in gone]
        for old in sorted(runs_all, key=lambda p: p.stat().st_mtime):
            if sizes['output'] <= OUTPUT_CAP:
                break
            if not _is_protected(old, protect):
                actions.append(('rmtree', old)); sizes['output'] -= size(old); gone.add(old)

    # datasets: oldest files first
    if datasets.exists():
        for p in sorted(_files(datasets), key=lambda p: p.stat().st_mtime):
            if sizes['datasets'] <= DATASET_CAP:
                break
            if p.name == '.gitkeep' or _is_protected(p, protect):
                continue
            actions.append(('delete', p)); sizes['datasets'] -= p.stat().st_size
    return actions, {k: max(0, v) for k, v in sizes.items()}


def _rotate(p):
    """Newest archive is ``name.gz``; older ones ``name.1.gz`` .. ``name.{KEEP-1}.gz``."""
    oldest = p.with_name(f'{p.name}.{LOG_KEEP_GZ - 1}.gz')
    oldest.unlink(missing_ok=True)
    for idx in range(LOG_KEEP_GZ - 2, 0, -1):
        old = p.with_name(f'{p.name}.{idx}.gz')
        if old.exists():
            old.replace(p.with_name(f'{p.name}.{idx + 1}.gz'))
    newest = p.with_name(p.name + '.gz')
    if newest.exists():
        newest.replace(p.with_name(f'{p.name}.1.gz'))
    with p.open('rb') as source, gzip.open(newest, 'wb') as dest:
        shutil.copyfileobj(source, dest)
    p.unlink()


def housekeeping(repo, dry_run=False, protect=()):
    root = Path(repo)
    before = {name: size(root / name) for name in ('logs', 'output', 'datasets')}
    actions, projected = plan_housekeeping(root, protect)
    summary = [f'{kind} {path.relative_to(root)}' for kind, path in actions]
    if dry_run:
        return {'before': before, 'after': before, 'projected': projected,
                'plan': PLAN, 'actions': summary}
    for kind, path in actions:
        if not path.exists():
            continue
        if kind == 'rotate_log':
            _rotate(path)
        elif kind == 'rmtree':
            shutil.rmtree(path)
        elif kind == 'delete':
            path.unlink()
    return {'before': before, 'after': {name: size(root / name) for name in before},
            'plan': PLAN, 'actions': summary}
