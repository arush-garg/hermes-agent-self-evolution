"""Read-only Hermes trajectory replay for real evaluation cases."""
import json
import os
import re
import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = Path.home() / '.hermes/state.db'
SECRET = re.compile(r'(?i)(?:sk-(?:ant-|or-|)[A-Za-z0-9_-]{16,}|(?:ghp_|ghu_|xoxb-|xapp-|ntn_)[A-Za-z0-9_-]{12,}|Bearer\s+\S+|(?:password|token|api[_-]?key|secret)\s*[=:]\s*\S+)')
CORRECTION = re.compile(r"(?i)\b(?:wrong|no,|still|again|don't|stop|incorrect|not what|you missed|actually)\b")
NOISE = re.compile(r'(?i)^\s*(?:\[context compaction|\[system note:|\[important: background|\[note: model|\[cron|<summary|\[out-of-band|heartbeat)')

# Anchored correction cue: must open the user's message (after stripping quotes/code),
# so quoted text or incidental "again"/"no" deep inside a request is not a correction.
_CORRECTION_START = re.compile(
    r"(?i)^\W*(?:no\b|nope\b|wrong\b|that'?s (?:wrong|not|incorrect)|incorrect\b|not what\b|"
    r"you missed\b|still (?:wrong|broken|failing|not|doesn'?t|isn'?t)|stop\b|don'?t\b|"
    r"actually\b|try again\b|that didn'?t\b|it didn'?t\b|doesn'?t work|didn'?t work)")
_QUOTED = re.compile(r"```.*?```|`[^`]*`|^>.*$|\"[^\"]{20,}\"", re.S | re.M)


def is_correction(text):
    """True only when a user turn *opens* with a correction cue (quotes/code stripped)."""
    body = _QUOTED.sub(' ', str(text or '')).strip()
    if not body or len(body) > 1500 or NOISE.search(body):
        return False
    return bool(_CORRECTION_START.search(body[:200]))


def redact(text):
    return SECRET.sub('[REDACTED]', str(text or ''))


def connect(db_path=None):
    path = Path(db_path or os.getenv('HERMES_STATE_DB', DB_PATH)).expanduser().resolve()
    return sqlite3.connect(f'{path.as_uri()}?mode=ro', uri=True)


def _calls(row):
    try:
        calls = json.loads(row.get('tool_calls') or '[]')
        return [(c.get('function', {}).get('name', c.get('name', '')), c.get('function', {}).get('arguments', '')) for c in calls]
    except (ValueError, TypeError, AttributeError):
        return []


def sessions(db_path=None, days=None):
    """Yield parent sessions with chronological messages; never mutate/copy state.db."""
    with connect(db_path) as db:
        db.row_factory = sqlite3.Row
        clause = 'AND s.started_at >= strftime("%s", "now") - ?' if days else ''
        params = (days * 86400,) if days else ()
        rows = db.execute(f'''SELECT m.session_id,m.role,m.content,m.tool_name,m.tool_calls,m.timestamp,
                    s.end_reason,s.started_at,s.title
                    FROM messages m JOIN sessions s ON m.session_id=s.id
                    WHERE s.parent_session_id IS NULL {clause}
                    ORDER BY s.started_at,m.id''', params)
        grouped = defaultdict(list)
        for row in rows:
            r = dict(row)
            grouped[r['session_id']].append(r)
        for sid, messages in grouped.items():
            yield sid, messages

_FAIL_LINE = re.compile(r"(?im)^\s*(?:Traceback \(most recent call last\)|(?:[A-Za-z_.]*Error|Exception|fatal|FATAL|ERROR|Error)\s*:|command not found|Permission denied|timed out after)")


def tool_failure(content):
    """Return a short failure signature (<=200 chars) or ''.

    Hermes stores tool results as JSON in messages.content (e.g. terminal
    {"output","exit_code","error"}, patch {"success"}, others {"error"}).
    Explicit status wins: exit_code==0 / success==true with no error field is NOT a
    failure even if the output contains the word ERROR. Non-JSON text only counts when a
    line *starts* with a real failure marker.
    """
    text = str(content or '')
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        data = None
    if isinstance(data, dict):
        err = data.get('error')
        if err not in (None, '', False):
            return redact(f'error: {str(err).splitlines()[0] if str(err).strip() else err}')[:200]
        code = data.get('exit_code', data.get('returncode'))
        if isinstance(code, int) and code not in (0,):
            # signature = exit code + first anchored failure line, not raw output
            m = _FAIL_LINE.search(str(data.get('output') or ''))
            line = str(data.get('output') or '')[m.start():].strip().splitlines()[0] if m else ''
            return redact(f'exit_code={code}' + (f' {line}' if line else ''))[:200]
        if data.get('success') is False or str(data.get('status', '')).lower() in ('error', 'failed', 'failure'):
            return redact('status=failed ' + str(data.get('message') or data.get('detail') or '').splitlines()[0] if (data.get('message') or data.get('detail')) else 'status=failed')[:200]
        return ''
    m = _FAIL_LINE.search(text[:4000])
    if not m:
        return ''
    return redact(text[m.start():].strip().splitlines()[0])[:200]


def replay(skill, db_path=None, limit=200):
    """Construct user task / outcome / corrected final answer from genuine skill use."""
    from evolution.core.dataset_builder import EvalExample
    skill = skill.lower()
    found = []
    for sid, messages in sessions(db_path):
        skill_positions = [i for i, m in enumerate(messages) if any(
            name == 'skill_view' and skill in str(args).lower() for name, args in _calls(m))]
        # Explicit skill mention in request also counts if the assistant used that skill.
        if not skill_positions:
            continue
        for pos in skill_positions:
            prior = [m for m in messages[:pos + 1] if m['role'] == 'user' and len(str(m['content'] or '')) >= 12 and not NOISE.search(m['content'] or '')]
            if not prior:
                continue
            user = prior[-1]
            start = messages.index(user)
            next_users = [(i, m) for i, m in enumerate(messages[pos + 1:], pos + 1) if m['role'] == 'user' and not NOISE.search(m['content'] or '')]
            end = next_users[0][0] if next_users else len(messages)
            correction = next_users[0][1]['content'] if next_users and is_correction(next_users[0][1]['content']) else ''
            window = messages[start:end]
            answers = [m['content'] for m in window if m['role'] == 'assistant' and m['content'] and not m['tool_calls']]
            tools = [name for m in window for name, _ in _calls(m)]
            errors = [m for m in window if m['role'] == 'tool' and tool_failure(m['content'])]
            if not answers and not correction:
                continue
            if NOISE.search(user['content'] or '') or re.search(r'(?i)^select the hermes skill most needing refinement', user['content'] or ''):
                continue
            if not correction and len(str(answers[-1]).strip()) < 40:
                continue
            if correction:
                outcome = 'corrected'
            elif len(errors) > len(tools) / 2 and tools or messages[-1]['end_reason'] not in (None, '', 'complete', 'completed', 'stop', 'end_turn', 'user_message'):
                outcome = 'failed'
            elif next_users or answers:
                outcome = 'success'
            else:
                outcome = 'unknown'
            if correction:
                # The answer before the correction was rejected by the user: never use it as
                # a target. Use the assistant's reply AFTER the correction (up to the next user
                # turn) only if it is substantive; otherwise the correction alone is the rubric.
                c_idx = next_users[0][0]
                c_end = next_users[1][0] if len(next_users) > 1 else len(messages)
                after = [m['content'] for m in messages[c_idx + 1:c_end]
                         if m['role'] == 'assistant' and m['content'] and not m['tool_calls']]
                fixed = after[-1] if after and len(str(after[-1]).strip()) >= 40 else ''
                expected = redact('Correction takes priority: ' + correction[:700]
                                  + ('\nCorrected response: ' + fixed[:1100] if fixed else ''))
            else:
                expected = redact((answers[-1] if answers else '')[:1100])
            found.append((user['timestamp'] or 0, EvalExample(
                task_input=redact(user['content'])[:1800], expected_behavior=expected,
                category=skill, source='sessiondb', task_class=skill, session_id=sid,
                tool_trace_summary=redact(' → '.join(tools[:35]))[:1500], outcome=outcome,
                failure_signals=['user_correction'] if correction else (['tool_errors'] if errors else []))))
            break  # one task per session avoids split leakage
    if not found:
        return []
    found.sort(key=lambda pair: pair[0])
    return [x[1] for x in found[-limit:]]
