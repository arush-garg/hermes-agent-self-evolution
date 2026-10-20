"""Bound self-evolution disk usage without touching staging."""
import gzip
import shutil
from pathlib import Path

MB = 1024 * 1024


def size(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink()) if path.exists() else 0


def _prune(root, cap):
    files = sorted((p for p in root.rglob('*') if p.is_file() and not p.is_symlink()), key=lambda p: p.stat().st_mtime)
    while size(root) > cap and files:
        files.pop(0).unlink()


def housekeeping(repo, dry_run=False):
    root = Path(repo)
    before = {name: size(root / name) for name in ('logs', 'output', 'datasets')}
    plan = {'logs': 'rotate >5 MiB; keep 3 gz; cap 100 MiB',
            'output': 'newest 3 runs/artifact; cap 150 MiB', 'datasets': 'cap 50 MiB'}
    if dry_run:
        return {'before': before, 'after': before, 'plan': plan}
    logs = root / 'logs'
    if logs.exists():
        for p in list(logs.iterdir()):
            if p.is_file() and p.suffix != '.gz' and p.stat().st_size > 5 * MB:
                archived = p.with_name(p.name + '.gz')
                for idx in (3, 2, 1):
                    old = p.with_name(p.name + f'.{idx}.gz')
                    if idx == 3:
                        old.unlink(missing_ok=True)
                    elif old.exists():
                        old.rename(p.with_name(p.name + f'.{idx + 1}.gz'))
                if archived.exists():
                    archived.rename(p.with_name(p.name + '.1.gz'))
                with p.open('rb') as source, gzip.open(archived, 'wb') as dest:
                    shutil.copyfileobj(source, dest)
                p.unlink()
        _prune(logs, 100 * MB)
    out = root / 'output'
    if out.exists():
        for artifact in out.iterdir():
            if artifact.is_dir():
                runs = sorted((p for p in artifact.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
                for old in runs[3:]:
                    shutil.rmtree(old)
        # Preserve newest run metadata when possible; remove oldest runs first.
        run_dirs = sorted((p for p in out.rglob('*') if p.is_dir() and any(x.is_file() for x in p.rglob('*'))), key=lambda p: p.stat().st_mtime)
        while size(out) > 150 * MB and run_dirs:
            old = run_dirs.pop(0)
            if old.exists():
                shutil.rmtree(old)
        if size(out) > 150 * MB:
            _prune(out, 150 * MB)
    datasets = root / 'datasets'
    if datasets.exists():
        _prune(datasets, 50 * MB)
    return {'before': before, 'after': {name: size(root / name) for name in before}, 'plan': plan}
