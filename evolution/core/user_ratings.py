"""Read-only access to the user's explicit ``/rate`` verdicts on Hermes sessions.

Hermes appends one JSON row per ``/rate <1-5> [feedback]`` to ``ratings.jsonl`` in the same
Hermes home as ``state.db`` (``{ts, session_id, rating, comment, surface, model?}``). A session
may be re-rated; the latest row wins. These are the strongest signal the pipeline has: the user
said, in their own words, what the harness did well or badly in that session.
"""
import json
from pathlib import Path

RATINGS_FILENAME = 'ratings.jsonl'
MIN_RATING, MAX_RATING = 1, 5
LOW_RATING_MAX = 2    # 1-2: the session went badly; its loaded skills are evolution targets
HIGH_RATING_MIN = 4   # 4-5: the session went well; its feedback says what to preserve


def ratings_path_for_db(db_path):
    """Ratings live beside ``state.db`` in the same Hermes home (profile)."""
    return Path(db_path).expanduser().parent / RATINGS_FILENAME


def _valid(row):
    rating = row.get('rating') if isinstance(row, dict) else None
    return (bool(row.get('session_id')) and isinstance(rating, int) and not isinstance(rating, bool)
            and MIN_RATING <= rating <= MAX_RATING) if isinstance(row, dict) else False


def load_latest_ratings(path):
    """``{session_id: row}`` with the latest valid row per session; ``{}`` when the file is absent."""
    try:
        text = Path(path).read_text(encoding='utf-8', errors='replace')
    except (FileNotFoundError, NotADirectoryError):
        return {}
    latest = {}
    for line in text.splitlines():
        try:
            row = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            continue
        if _valid(row):
            latest[str(row['session_id'])] = row
    return latest
