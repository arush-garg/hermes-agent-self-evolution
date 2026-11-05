"""Per-tool signal from the user's ``/rate`` verdicts, for the tool-description pipeline.

A rated session says how the whole session went, not which tool caused it, so each tool *called*
in that session shares the verdict. Failures of a tool inside a low-rated session are counted
separately because they are the likeliest culprit. Feedback text is redacted and capped.
"""
from collections import defaultdict

from evolution.core.trajectory_replay import _calls, redact, sessions, tool_failure
from evolution.core.user_ratings import HIGH_RATING_MIN, LOW_RATING_MAX, load_latest_ratings
from evolution.core.failure_mining import _default_ratings_path

MAX_FEEDBACK_PER_TOOL = 5
MAX_FEEDBACK_CHARS = 300


def _tools_in(messages):
    """(tools called, tools whose result was a failure) in one session."""
    called, failed = set(), set()
    for m in messages:
        called.update(name for name, _ in _calls(m) if name)
        if m['role'] == 'tool' and m.get('tool_name') and tool_failure(m['content']):
            failed.add(m['tool_name'])
    return called, failed


def tool_rating_signals(db_path=None, ratings_path=None, days=30):
    """``{tool: {low_rated_sessions, high_rated_sessions, failures_in_low_rated, feedback}}``.

    Only tools used in at least one rated session appear; ``{}`` when nothing is rated.
    """
    ratings = load_latest_ratings(ratings_path or _default_ratings_path(db_path))
    if not ratings:
        return {}
    data = defaultdict(lambda: {'low_rated_sessions': 0, 'high_rated_sessions': 0,
                                'failures_in_low_rated': 0, 'feedback': []})
    for sid, messages in sessions(db_path, days):
        rating = ratings.get(sid)
        if rating is None:
            continue
        called, failed = _tools_in(messages)
        score = rating['rating']
        comment = redact(str(rating.get('comment') or '').strip())[:MAX_FEEDBACK_CHARS]
        for tool in called:
            entry = data[tool]
            if score <= LOW_RATING_MAX:
                entry['low_rated_sessions'] += 1
                entry['failures_in_low_rated'] += tool in failed
            elif score >= HIGH_RATING_MIN:
                entry['high_rated_sessions'] += 1
            if comment and len(entry['feedback']) < MAX_FEEDBACK_PER_TOOL:
                entry['feedback'].append({'rating': score, 'comment': comment})
    return dict(data)


def flagged_tools(signals):
    """Tools the user's ratings point at: used in a low-rated session."""
    return {tool for tool, s in signals.items() if s['low_rated_sessions']}


def feedback_for_tools(signals, names):
    """Plain-text digest of the user's feedback on ``names`` (``'none'`` when there is none)."""
    lines = []
    for name in names:
        s = signals.get(name)
        if not s or not (s['feedback'] or s['low_rated_sessions']):
            continue
        lines.append(f"- {name}: {s['low_rated_sessions']} low-rated / {s['high_rated_sessions']} high-rated "
                     f"sessions, {s['failures_in_low_rated']} failing in low-rated ones")
        lines.extend(f"    /rate {f['rating']}/5: {f['comment']}" for f in s['feedback'])
    return '\n'.join(lines) or 'none'
