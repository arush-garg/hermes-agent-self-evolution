"""Failure pattern mining from read-only Hermes trajectories."""
import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from evolution.core.trajectory_replay import sessions, _calls, CORRECTION, redact

ERROR_RE = re.compile(r'(?i)\b(error|exception|traceback|timed out|timeout|permission denied|403|404|not found)\b')


def mine(days=30, db_path=None):
    scored = defaultdict(lambda: {'score': 0, 'failures': [], 'sessions': set()})
    for sid, messages in sessions(db_path, days):
        skills = {str(args).split('"')[3] for m in messages for name, args in _calls(m)
                  if name == 'skill_view' and '"name"' in str(args) and len(str(args).split('"')) > 3}
        calls = {}
        seen_errors = Counter()
        for i, m in enumerate(messages):
            for name, args in _calls(m):
                digest = hashlib.sha256(str(args).encode()).hexdigest()[:12]
                calls[m.get('tool_call_id') or (i, name)] = (name, digest)
            if m['role'] != 'tool':
                continue
            text = str(m['content'] or '')[:500]
            error = ERROR_RE.search(text)
            if not error:
                continue
            name = m['tool_name'] or 'unknown'
            key = (name, hashlib.sha256(text.encode()).hexdigest()[:12], error.group(1).lower())
            seen_errors[key] += 1
            for artifact in skills or {name}:
                entry = scored[artifact]
                entry['score'] += 3 if error.group(1).lower() in ('timeout', 'timed out', '403', '404') else 1
                entry['sessions'].add(sid)
                if len(entry['failures']) < 10:
                    entry['failures'].append({'tool': name, 'signal': error.group(1).lower(), 'sample': redact(text[:180])})
        for key, count in seen_errors.items():
            if count >= 2:
                for artifact in skills or {key[0]}:
                    scored[artifact]['score'] += count * 2
                    scored[artifact]['failures'].append({'tool': key[0], 'signal': 'repeated_error', 'count': count})
        if any(m['role'] == 'user' and CORRECTION.search(m['content'] or '') for m in messages[1:]):
            for artifact in skills:
                scored[artifact]['score'] += 4
                scored[artifact]['sessions'].add(sid)
                scored[artifact]['failures'].append({'signal': 'user_correction'})
    results = [{'artifact': k, 'score': v['score'], 'session_count': len(v['sessions']),
                'failures': v['failures'][:10]} for k, v in scored.items()]
    return sorted(results, key=lambda row: (-row['score'], row['artifact']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--db-path')
    args = parser.parse_args()
    print(json.dumps(mine(args.days, args.db_path), indent=2))
