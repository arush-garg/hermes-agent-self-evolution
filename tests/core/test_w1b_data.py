import json
import os
import sqlite3
from pathlib import Path

from evolution.core.trajectory_replay import replay
from evolution.core.failure_mining import mine
from evolution.core.lesson_library import append, retrieve
from evolution.core.housekeeping import housekeeping, MB, size
from evolution.core import dataset_builder
from evolution.core.dataset_builder import build_mixed_dataset, EvalExample, EvalDataset

import pytest

SCHEMA = ('CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, started_at INTEGER, end_reason TEXT, title TEXT);'
          'CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, tool_calls TEXT, timestamp INTEGER);')


def _skill_call(name):
    return json.dumps([{'function': {'name': 'skill_view', 'arguments': json.dumps({'name': name})}}])


def make_db(path, sessions):
    """sessions: list of (sid, started_at, [(role, content, tool_name, tool_calls), ...])."""
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    for sid, started, msgs in sessions:
        db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)', (sid, None, started, 'completed', 't'))
        for j, (role, content, tool_name, tool_calls) in enumerate(msgs):
            db.execute('INSERT INTO messages(session_id,role,content,tool_name,tool_calls,timestamp) VALUES (?,?,?,?,?,?)',
                       (sid, role, content, tool_name, tool_calls, started + j))
    db.commit(); db.close()


def task_session(sid, started, skill, request, answer):
    return (sid, started, [('user', request, None, None), ('assistant', None, None, _skill_call(skill)),
                           ('assistant', answer, None, None)])


class _FakeSynth:
    def __init__(self, config):
        pass

    def generate(self, text, num_cases=12):
        ex = [EvalExample(task_input=f'synthetic task {i}', expected_behavior='synthetic rubric', source='synthetic')
              for i in range(num_cases)]
        return EvalDataset(train=ex)


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(dataset_builder, 'SyntheticDatasetBuilder', _FakeSynth)


def fixture_db(path):
    db = sqlite3.connect(path)
    db.executescript('CREATE TABLE sessions(id TEXT PRIMARY KEY, parent_session_id TEXT, started_at INTEGER, end_reason TEXT, title TEXT); CREATE TABLE messages(id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, tool_calls TEXT, timestamp INTEGER);')
    for i in range(12):
        sid = f's{i:02}'
        db.execute('INSERT INTO sessions VALUES (?,?,?,?,?)', (sid, None, 1700000000+i, 'completed', 'test'))
        db.execute('INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)', (sid, 'user', 'Please fix this troublesome bug', i))
        db.execute('INSERT INTO messages(session_id,role,tool_calls,timestamp) VALUES (?,?,?,?)', (sid, 'assistant', json.dumps([{'function':{'name':'skill_view','arguments':'{"name":"debugging"}'}}]), i))
        db.execute('INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)', (sid, 'assistant', 'Fixed bug and verified the complete test suite without errors.', i))
        if i == 11:
            db.execute('INSERT INTO messages(session_id,role,content,timestamp) VALUES (?,?,?,?)', (sid, 'user', 'No, still wrong. Run the test again.', i))
    db.commit(); db.close()


def test_replay_corrections_and_read_only(tmp_path):
    path = tmp_path/'state.db'; fixture_db(path)
    cases = replay('debugging', path)
    assert len(cases) == 12
    assert cases[-1].outcome == 'corrected'
    assert cases[-1].expected_behavior.startswith('Correction takes priority:')
    assert cases[0].session_id != cases[-1].session_id


def test_lesson_roundtrip_caps_and_redaction(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_LESSONS_DIR', str(tmp_path))
    for i in range(205):
        append('My Task', 'token=secret-value', f'lesson {i}', 'test')
    rows = (tmp_path/'my-task.jsonl').read_text().splitlines()
    assert len(rows) == 200
    assert 'secret-value' not in ''.join(rows)
    assert retrieve('My Task', k=3)[0]['lesson'] == 'lesson 204'
    assert (tmp_path/'README.md').exists()


def test_housekeeping_caps(tmp_path):
    logs=tmp_path/'logs'; logs.mkdir(); (logs/'large.log').write_bytes(b'a'*(5*MB+1))
    output=tmp_path/'output'/'artifact'; output.mkdir(parents=True)
    for i in range(5): (output/f'run_{i}').mkdir()
    data=tmp_path/'datasets'; data.mkdir(); (data/'huge.bin').write_bytes(b'0'*(51*MB))
    planned=housekeeping(tmp_path, dry_run=True)
    assert (logs/'large.log').exists() and planned['before']==planned['after']
    housekeeping(tmp_path)
    assert (logs/'large.log.gz').exists()
    assert len(list(output.iterdir())) <= 3
    assert size(data) <= 50*MB
    assert size(logs) <= 100*MB and size(tmp_path/'output') <= 150*MB


LONG = 'Completed the task thoroughly with verification and a clear final summary.'


def test_corrected_expected_uses_following_answer_never_superseded(tmp_path):
    path = tmp_path / 'state.db'
    make_db(path, [('s0', 1700000000, [
        ('user', 'Write a hello world program please', None, None),
        ('assistant', None, None, _skill_call('coding')),
        ('assistant', 'WRONG_ANSWER: here is hello world in Python print(hello) as requested', None, None),
        ('user', 'No, that is wrong. Use Java instead.', None, None),
        ('assistant', 'FIXED_ANSWER: public class Hello { System.out.println("hello"); } in Java', None, None),
        ('user', 'thanks, now something unrelated entirely', None, None),
        ('assistant', 'LATER_ANSWER: unrelated follow-up content that must not leak in here', None, None),
    ])])
    [case] = replay('coding', path)
    assert case.outcome == 'corrected'
    assert 'Use Java instead' in case.expected_behavior
    assert 'FIXED_ANSWER' in case.expected_behavior
    assert 'WRONG_ANSWER' not in case.expected_behavior
    assert 'LATER_ANSWER' not in case.expected_behavior


def test_corrected_without_followup_uses_correction_only(tmp_path):
    path = tmp_path / 'state.db'
    make_db(path, [('s0', 1700000000, [
        ('user', 'Summarize the meeting notes', None, None),
        ('assistant', None, None, _skill_call('notes')),
        ('assistant', 'WRONG_ANSWER: a long but incorrect summary of the meeting notes here', None, None),
        ('user', "That's wrong, you missed the action items.", None, None),
    ])])
    [case] = replay('notes', path)
    assert case.outcome == 'corrected'
    assert 'WRONG_ANSWER' not in case.expected_behavior
    assert case.expected_behavior.startswith('Correction takes priority:')


def test_quoted_or_incidental_correction_words_not_corrections(tmp_path):
    path = tmp_path / 'state.db'
    make_db(path, [('s0', 1700000000, [
        ('user', 'Draft an email for me', None, None),
        ('assistant', None, None, _skill_call('email')),
        ('assistant', LONG, None, None),
        ('user', 'Great. Please send it again to the other list, and actually cc Bob.', None, None),
        ('assistant', LONG, None, None),
        ('user', '> no, this is wrong\nThat was the customer reply; draft a response.', None, None),
    ])])
    assert [c.outcome for c in replay('email', path)] == ['success']
    assert all(r['correction_sessions'] == 0 for r in mine(days=None, db_path=path))


def _sessions_for(skill, n, base=1700000000):
    return [task_session(f'{skill}{i:02}', base + i * 100, skill, f'Real request number {i} for {skill}', LONG)
            for i in range(n)]


def test_mixed_holdout_never_synthetic_low_real_data(tmp_path, monkeypatch, no_llm):
    path = tmp_path / 'state.db'
    make_db(path, _sessions_for('sparse', 2))
    monkeypatch.setenv('HERMES_STATE_DB', str(path))
    ds = build_mixed_dataset('sparse', '# s', tmp_path / 'out', None, minimum=12)
    assert ds.holdout and all(e.source == 'sessiondb' for e in ds.holdout)
    assert len(ds.holdout) < 8  # too few real cases: stage-only, never padded
    assert any(e.source == 'synthetic' for e in ds.train + ds.val)
    saved = [json.loads(l) for l in (tmp_path / 'out' / 'holdout.jsonl').read_text().splitlines() if l]
    assert all(r['source'] == 'sessiondb' for r in saved)


def test_mixed_no_real_data_gives_empty_holdout(tmp_path, monkeypatch, no_llm):
    path = tmp_path / 'state.db'
    make_db(path, [])
    monkeypatch.setenv('HERMES_STATE_DB', str(path))
    ds = build_mixed_dataset('nothing', '# s', tmp_path / 'out', None, minimum=12)
    assert ds.holdout == []
    assert len(ds.train) + len(ds.val) == 12


def test_mixed_session_and_time_boundaries(tmp_path, monkeypatch, no_llm):
    path = tmp_path / 'state.db'
    # insert newest first to prove ordering uses started_at, not insertion order
    sess = list(reversed(_sessions_for('timed', 10)))
    # one session with two uses of the skill -> 2 cases that must stay in one split
    sess[0] = ('timed09', 1700000000 + 900, sess[0][2] + [('user', 'Second real request in same session', None, None),
               ('assistant', None, None, _skill_call('timed')), ('assistant', LONG, None, None)])
    make_db(path, sess)
    monkeypatch.setenv('HERMES_STATE_DB', str(path))
    ds = build_mixed_dataset('timed', '# s', tmp_path / 'out', None, minimum=5)
    tr, va, ho = ({e.session_id for e in split} for split in (ds.train, ds.val, ds.holdout))
    assert not (tr & va) and not (tr & ho) and not (va & ho)
    assert ho == {'timed07', 'timed08', 'timed09'}
    assert len([e for e in ds.all_examples if e.session_id == 'timed09']) == 1  # one case per session
    assert max(tr | va) < min(ho)
    assert all(e.source == 'sessiondb' for e in ds.all_examples)


def _tool(content, name='terminal'):
    return ('tool', content, name, None)


def test_exit_code_zero_with_error_text_is_not_failure(tmp_path):
    path = tmp_path / 'state.db'
    ok = json.dumps({'output': 'ERROR: kokoro MISSING\nfile metadata ok; error count 0', 'exit_code': 0, 'error': None})
    make_db(path, [('s0', 1700000000, [('user', 'check media deps', None, None),
                                         ('assistant', None, None, _skill_call('media')), _tool(ok),
                                         _tool('git status: nothing to commit, 0 errors found')])])
    assert mine(days=None, db_path=path) == []


def test_recurring_real_errors_rank_by_distinct_sessions(tmp_path):
    path = tmp_path / 'state.db'
    fail = json.dumps({'output': 'ERROR: ffmpeg exited with code 8', 'exit_code': 1, 'error': None})
    sessions = []
    for i in range(4):  # recurring across 4 sessions
        sessions.append((f'r{i}', 1700000000 + i, [('user', 'render video', None, None),
                         ('assistant', None, None, _skill_call('recurring')), _tool(fail)]))
    # one noisy session with 50 failures must not outrank it
    sessions.append(('n0', 1700000100, [('user', 'render', None, None), ('assistant', None, None, _skill_call('noisy'))]
                     + [_tool(json.dumps({'output': f'boom {i}', 'exit_code': 2, 'error': None})) for i in range(50)]))
    make_db(path, sessions)
    rows = mine(days=None, db_path=path)
    assert [r['artifact'] for r in rows][:2] == ['recurring', 'noisy']
    top = rows[0]
    assert top['failure_sessions'] == 4 and top['failures'][0]['sessions'] == 4
    assert top['failures'][0]['error_signature'].startswith('exit_code=1')
    assert rows[1]['score'] <= 2
    for r in rows:
        assert len(json.dumps(r)) <= 2000
        assert all(len(f['error_signature']) <= 200 and 'sample' not in f for f in r['failures'])


def test_corrections_weighted_and_redacted(tmp_path):
    path = tmp_path / 'state.db'
    sessions = [(f'c{i}', 1700000000 + i, [('user', 'do the thing', None, None), ('assistant', None, None, _skill_call('corrected')),
                 ('assistant', LONG, None, None), ('user', 'No, wrong token=abc123secret, try again', None, None)])
                for i in range(2)]
    fail = json.dumps({'error': 'HTTP 404'})
    sessions += [(f'f{i}', 1700000100 + i, [('user', 'fetch it', None, None), ('assistant', None, None, _skill_call('failing')), _tool(fail, 'web')])
                 for i in range(3)]
    make_db(path, sessions)
    ranked = mine(days=None, db_path=path)
    rows = {r['artifact']: r for r in ranked}
    # 2 corrected sessions outrank (tie-break) 3 sessions of the same recurring tool error
    assert ranked[0]['artifact'] == 'corrected'
    assert rows['corrected']['correction_sessions'] == 2
    assert 'abc123secret' not in json.dumps(rows)


def _mtimes(root):
    return {str(p): (p.lstat().st_mtime_ns, p.lstat().st_size) for p in Path(root).rglob('*')}


def test_housekeeping_dry_run_is_read_only_and_reports_actions(tmp_path):
    for i in range(5):
        d = tmp_path / 'output' / 'skill' / f'run{i}'
        d.mkdir(parents=True); (d / 'metrics.json').write_text('{}')
        os.utime(d, (1700000000 + i, 1700000000 + i))
    (tmp_path / 'logs').mkdir(); (tmp_path / 'logs' / 'big.log').write_bytes(b'x' * (6 * MB))
    before = _mtimes(tmp_path)
    res = housekeeping(tmp_path, dry_run=True)
    assert _mtimes(tmp_path) == before
    assert any(a.startswith('rotate_log') for a in res['actions'])
    assert sum(a.startswith('rmtree') for a in res['actions']) == 2


def test_housekeeping_never_deletes_current_run_or_newest(tmp_path, monkeypatch):
    import evolution.core.housekeeping as hk
    monkeypatch.setattr(hk, 'OUTPUT_CAP', 1)
    monkeypatch.setattr(hk, 'LOG_CAP', 1)
    runs = []
    for i in range(5):
        d = tmp_path / 'output' / 'skill' / f'run{i}'
        d.mkdir(parents=True); (d / 'blob').write_bytes(b'x' * 1000)
        os.utime(d, (1700000000 + i, 1700000000 + i)); runs.append(d)
    logs = tmp_path / 'logs'; logs.mkdir()
    current = logs / 'daily_runner_now.log'; current.write_bytes(b'y' * 1000)
    os.utime(current, (1600000000, 1600000000))  # oldest by mtime, still protected
    (logs / 'old.log').write_bytes(b'z' * 1000)
    housekeeping(tmp_path, dry_run=False, protect=[current, runs[0]])
    assert current.exists() and runs[0].exists()  # explicitly protected current run
    assert runs[4].exists()  # newest run per artifact is always kept
    assert not runs[3].exists() and not (logs / 'old.log').exists()  # caps enforced
