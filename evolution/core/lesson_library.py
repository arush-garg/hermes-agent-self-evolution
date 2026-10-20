"""Shared evidence-backed cross-task lessons; atomic locked JSONL storage."""
import fcntl
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .trajectory_replay import redact

SPEC = """# Hermes lessons JSONL v1
One file per lowercase slug task class (`[a-z0-9-]{1,40}.jsonl`). Each row:
`id, task_class, cue, lesson, evidence, source, status, session_ids, created_at, hits`.
Status: candidate, confirmed, retired. Maximum 200 rows/class, 2 MiB total.
Writers acquire `.lock` with fcntl; write temp file and os.replace atomically.
Use HERMES_LESSONS_DIR to override the default ~/.hermes/lessons/. Redact secrets.
"""


def _dir():
    p = Path(os.getenv('HERMES_LESSONS_DIR', str(Path.home() / '.hermes/lessons')))
    p.mkdir(parents=True, exist_ok=True)
    readme = p / 'README.md'
    if not readme.exists():
        readme.write_text(SPEC)
    return p


def _slug(task_class):
    slug = re.sub(r'[^a-z0-9-]+', '-', str(task_class).lower()).strip('-')[:40].strip('-')
    if not slug:
        raise ValueError('empty task class')
    return slug


def _read(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def retrieve(task_class: str, k: int = 3) -> list[dict]:
    rows = _read(_dir() / f'{_slug(task_class)}.jsonl')
    return sorted((r for r in rows if r.get('status') != 'retired'),
                  key=lambda r: (r.get('status') == 'confirmed', r.get('hits', 0), r.get('created_at', '')), reverse=True)[:max(0, k)]


def append(task_class, cue, lesson, evidence, source='self-evolution', status='candidate') -> dict:
    if status not in ('candidate', 'confirmed', 'retired'):
        raise ValueError('invalid lesson status')
    root = _dir()
    slug = _slug(task_class)
    path = root / f'{slug}.jsonl'
    row = dict(id=str(uuid.uuid4()), task_class=slug, cue=redact(cue), lesson=redact(lesson),
               evidence=redact(evidence), source=redact(source), status=status, session_ids=[],
               created_at=datetime.now(timezone.utc).isoformat(), hits=0)
    with (root / '.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = _read(path) + [row]
        while len(rows) > 200:
            index = next((i for i, r in enumerate(rows) if r.get('status') != 'confirmed'), 0)
            rows.pop(index)
        content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        total = sum(p.stat().st_size for p in root.glob('*.jsonl') if p != path)
        while len(content.encode()) + total > 2 * 1024 * 1024 and len(rows) > 1:
            index = next((i for i, r in enumerate(rows) if r.get('status') != 'confirmed'), 0)
            rows.pop(index)
            content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        if len(content.encode()) + total > 2 * 1024 * 1024:
            raise ValueError('lesson store full')
        fd, tmp = tempfile.mkstemp(dir=root, prefix='.lesson-')
        try:
            with os.fdopen(fd, 'w') as out:
                out.write(content)
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
    return row
