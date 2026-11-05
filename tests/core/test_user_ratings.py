"""User /rate feedback (ratings.jsonl beside state.db) flows into failure mining."""
import json

from evolution.core.failure_mining import mine
from evolution.core.user_ratings import load_latest_ratings, ratings_path_for_db
from tests.core.test_w1b_data import make_db, task_session


def _write_ratings(home, rows):
    (home / 'ratings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


def test_ratings_file_resolved_beside_state_db(tmp_path):
    assert ratings_path_for_db(tmp_path / 'state.db') == tmp_path / 'ratings.jsonl'


def test_latest_rating_per_session_wins_and_bad_rows_skipped(tmp_path):
    _write_ratings(tmp_path, [
        {'session_id': 's1', 'rating': 1, 'comment': 'bad'},
        {'session_id': 's1', 'rating': 4, 'comment': 'fine after all'},
        {'session_id': 's2', 'rating': 7},
        {'rating': 3},
    ])
    with (tmp_path / 'ratings.jsonl').open('a') as fh:
        fh.write('{oops\n')
    latest = load_latest_ratings(tmp_path / 'ratings.jsonl')
    assert set(latest) == {'s1'}
    assert latest['s1']['rating'] == 4


def test_missing_ratings_file_is_empty(tmp_path):
    assert load_latest_ratings(tmp_path / 'ratings.jsonl') == {}


def test_low_rating_ranks_skill_and_carries_feedback(tmp_path):
    db = tmp_path / 'state.db'
    make_db(db, [task_session('s1', 1700000000, 'alpha', 'do the thing', 'done'),
                 task_session('s2', 1700000100, 'beta', 'other thing', 'done')])
    _write_ratings(tmp_path, [{'session_id': 's1', 'rating': 2, 'comment': 'ignored my token=abc123secret ask'},
                              {'session_id': 's2', 'rating': 5, 'comment': 'great job'}])

    rows = {r['artifact']: r for r in mine(days=None, db_path=db)}

    alpha, beta = rows['alpha'], rows['beta']
    assert alpha['low_rated_sessions'] == 1
    assert alpha['score'] > beta['score']
    assert alpha['user_feedback'][0]['rating'] == 2
    assert 'abc123secret' not in alpha['user_feedback'][0]['comment']  # redacted
    assert beta['high_rated_sessions'] == 1
    assert beta['user_feedback'] == [{'rating': 5, 'comment': 'great job'}]
    assert beta['score'] == 0  # praise never makes a skill a target


def test_no_ratings_file_leaves_mining_unchanged(tmp_path):
    db = tmp_path / 'state.db'
    make_db(db, [task_session('s1', 1700000000, 'alpha', 'do the thing', 'done')])
    assert mine(days=None, db_path=db) == []
