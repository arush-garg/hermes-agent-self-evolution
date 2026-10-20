"""Failure pattern mining from read-only Hermes trajectories.

Signals (per artifact = skill loaded via skill_view *before* the signal in the session):
  * tool failures: explicit JSON ``error`` field, nonzero ``exit_code``, ``success: false`` /
    failed status, or non-JSON text whose line *starts* with a real failure marker
    (see ``trajectory_replay.tool_failure``). ``exit_code: 0`` output that merely contains
    the word "ERROR" is not a failure.
  * user corrections: the next user turn opens with a correction cue (``is_correction``).

Ranking counts distinct sessions, not raw messages, so one noisy session cannot inflate
the score: score = 3*correction_sessions + failure_sessions + recurring-signature bonus.
Context is short ``error_signature`` strings (<=200 chars, redacted), never raw output,
and each artifact's serialized context is capped at MAX_CONTEXT_CHARS.
"""
import argparse
import json
import re
from collections import defaultdict

from evolution.core.trajectory_replay import sessions, _calls, redact, tool_failure, is_correction

MAX_CONTEXT_CHARS = 2000
_NORM = re.compile(r'0x[0-9a-f]+|\d+|/[^\s:\'"]+')


def _skill_name(args):
    try:
        return str(json.loads(args or '{}').get('name') or '').strip()
    except (ValueError, TypeError, AttributeError):
        m = re.search(r'"name"\s*:\s*"([^"]+)"', str(args))
        return m.group(1) if m else ''


def _bound(row, limit=MAX_CONTEXT_CHARS):
    """Drop trailing examples until the JSON-serialized row fits in ``limit`` chars."""
    while len(json.dumps(row)) > limit and (row['failures'] or row['corrections']):
        (row['failures'] if len(row['failures']) >= len(row['corrections']) else row['corrections']).pop()
    return row


# Harness approval/guard refusals are policy outcomes, not skill defects.
_POLICY_BLOCK = re.compile(r'(?i)BLOCKED:|flagged as dangerous|\[harness-guard\]|approval (?:denied|required)')


def mine(days=30, db_path=None):
    data = defaultdict(lambda: {'fail_sessions': set(), 'corr_sessions': set(),
                                'sigs': defaultdict(set), 'tools': {}, 'corrections': []})
    for sid, messages in sessions(db_path, days):
        active = []  # skills loaded so far in this session (order preserved)
        for m in messages:
            for name, args in _calls(m):
                if name == 'skill_view':
                    skill = _skill_name(args)
                    if skill and skill not in active:
                        active.append(skill)
            if not active:
                continue
            if m['role'] == 'tool':
                sig = tool_failure(m['content'])
                if not sig or _POLICY_BLOCK.search(sig):
                    continue
                tool = m.get('tool_name') or 'unknown'
                key = (tool, _NORM.sub('#', sig.lower())[:120])
                for skill in active:
                    entry = data[skill]
                    entry['fail_sessions'].add(sid)
                    entry['sigs'][key].add(sid)
                    entry['tools'].setdefault(key, sig[:200])
            elif m['role'] == 'user' and is_correction(m['content']):
                for skill in active:
                    entry = data[skill]
                    if sid not in entry['corr_sessions'] and len(entry['corrections']) < 5:
                        entry['corrections'].append(redact(str(m['content']).strip())[:200])
                    entry['corr_sessions'].add(sid)
    result = []
    for artifact, entry in data.items():
        recurring = sorted(((len(s), k) for k, s in entry['sigs'].items()), key=lambda x: (-x[0], x[1]))
        bonus = sum(min(n, 5) for n, _ in recurring if n >= 2)
        score = 3 * len(entry['corr_sessions']) + len(entry['fail_sessions']) + bonus
        row = {
            'artifact': artifact,
            'score': score,
            'session_count': len(entry['fail_sessions'] | entry['corr_sessions']),
            'failure_sessions': len(entry['fail_sessions']),
            'correction_sessions': len(entry['corr_sessions']),
            'failures': [{'tool': k[0], 'sessions': n, 'error_signature': entry['tools'][k]}
                         for n, k in recurring[:8]],
            'corrections': list(entry['corrections']),
        }
        result.append(_bound(row))
    result.sort(key=lambda r: (-r['score'], -r['correction_sessions'], r['artifact']))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=30)
    parser.add_argument('--db', type=str, default=None)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--top', type=int, default=20)
    args = parser.parse_args()
    out = mine(args.days, args.db)[:args.top]
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        for item in out:
            print(f"{item['artifact']:30s} score={item['score']:4d} fail_sessions={item['failure_sessions']:3d} "
                  f"corrections={item['correction_sessions']:3d}")
            for f in item['failures'][:3]:
                print(f"  - {f['tool']:14s} x{f['sessions']:<3d} {f['error_signature'][:90]}")
