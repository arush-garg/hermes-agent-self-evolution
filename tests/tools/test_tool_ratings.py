"""User /rate feedback reaches the tool-description pipeline, and its GEPA budget stays bounded."""
import json

import dspy

from evolution.tools.evolve_tool_descriptions import (
    MAX_METRIC_CALLS, MAX_TRAIN_EXAMPLES, MAX_VAL_EXAMPLES, _bounded_splits, _gepa_budget, make_gepa_metric)
from evolution.tools.tool_ratings import feedback_for_tools, tool_rating_signals
from tests.core.test_w1b_data import make_db


def _call(name):
    return json.dumps([{'function': {'name': name, 'arguments': '{}'}}])


def _session(sid, started, tools, fail_tool=None):
    msgs = [('user', f'task {sid}', None, None)]
    for tool in tools:
        msgs.append(('assistant', None, None, _call(tool)))
        content = '{"error": "boom"}' if tool == fail_tool else '{"ok": true}'
        msgs.append(('tool', content, tool, None))
    msgs.append(('assistant', 'done', None, None))
    return sid, started, msgs


def _home(tmp_path, sessions, ratings):
    db = tmp_path / 'state.db'
    make_db(db, sessions)
    (tmp_path / 'ratings.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in ratings))
    return db


def test_signals_attribute_ratings_and_failures_to_tools_used(tmp_path):
    db = _home(tmp_path,
               [_session('s1', 1700000000, ['terminal', 'read_file'], fail_tool='terminal'),
                _session('s2', 1700000100, ['read_file']),
                _session('s3', 1700000200, ['web_search'])],
               [{'session_id': 's1', 'rating': 1, 'comment': 'terminal kept failing, token=abc123secret'},
                {'session_id': 's2', 'rating': 5, 'comment': 'fast file reads'}])

    signals = tool_rating_signals(db_path=db, days=None)

    assert signals['terminal']['low_rated_sessions'] == 1
    assert signals['terminal']['failures_in_low_rated'] == 1
    assert signals['read_file'] == {'low_rated_sessions': 1, 'high_rated_sessions': 1,
                                    'failures_in_low_rated': 0, 'feedback': signals['read_file']['feedback']}
    assert {f['rating'] for f in signals['read_file']['feedback']} == {1, 5}
    assert 'abc123secret' not in json.dumps(signals)
    assert 'web_search' not in signals  # unrated session


def test_no_ratings_means_no_signals(tmp_path):
    db = tmp_path / 'state.db'
    make_db(db, [_session('s1', 1700000000, ['terminal'])])
    assert tool_rating_signals(db_path=db, days=None) == {}


def test_feedback_text_only_covers_requested_tools():
    signals = {'terminal': {'low_rated_sessions': 2, 'high_rated_sessions': 0, 'failures_in_low_rated': 1,
                            'feedback': [{'rating': 1, 'comment': 'loops forever'}]},
               'read_file': {'low_rated_sessions': 0, 'high_rated_sessions': 1, 'failures_in_low_rated': 0,
                             'feedback': [{'rating': 5, 'comment': 'great'}]}}
    text = feedback_for_tools(signals, ['terminal'])
    assert 'terminal' in text and 'loops forever' in text and 'great' not in text
    assert feedback_for_tools(signals, ['web_search']) == 'none'


def test_gepa_budget_is_capped_regardless_of_dataset_size():
    assert _gepa_budget(5, n_train=10, n_val=5) == 75
    assert _gepa_budget(5, n_train=174, n_val=58) == MAX_METRIC_CALLS


def _ex(tool, i):
    return dspy.Example(task_input=f'{tool} {i}', correct_tool=tool).with_inputs('task_input')


def test_bounded_splits_cap_sizes_and_keep_rated_tools_in_val():
    train = [_ex('other', i) for i in range(200)]
    val = [_ex('other', i) for i in range(80)] + [_ex('terminal', i) for i in range(3)]
    t, v = _bounded_splits(train, val, flagged={'terminal'})
    assert len(t) == MAX_TRAIN_EXAMPLES and len(v) == MAX_VAL_EXAMPLES
    assert sum(e.correct_tool == 'terminal' for e in v) == 3


def test_gepa_metric_feedback_includes_user_comments_for_gold_tool():
    signals = {'terminal': {'low_rated_sessions': 1, 'high_rated_sessions': 0, 'failures_in_low_rated': 0,
                            'feedback': [{'rating': 2, 'comment': 'used grep via terminal instead of search'}]}}
    metric = make_gepa_metric(signals)
    gold = dspy.Example(task_input='x', correct_tool='terminal')
    wrong = metric(gold, dspy.Prediction(selected_tool='search_files', rationale=''), None, None, None)
    right = metric(gold, dspy.Prediction(selected_tool='terminal', rationale=''), None, None, None)
    assert wrong.score == 0.0 and 'expected terminal' in wrong.feedback.lower()
    assert 'used grep via terminal' in wrong.feedback
    assert right.score == 1.0


def test_reloaded_dataset_recovers_gold_tool_from_expected_behavior():
    from evolution.core.dataset_builder import EvalExample
    from evolution.tools.evolve_tool_descriptions import _to_dspy_with_tool
    reloaded = EvalExample.from_dict({'task_input': 'show lines', 'expected_behavior': 'Select tool: read_file'})
    assert _to_dspy_with_tool([reloaded])[0].correct_tool == 'read_file'
