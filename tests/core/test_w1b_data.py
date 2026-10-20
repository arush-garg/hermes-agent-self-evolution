import json
import os
import sqlite3
from pathlib import Path

from evolution.core.trajectory_replay import replay
from evolution.core.failure_mining import mine
from evolution.core.lesson_library import append, retrieve
from evolution.core.housekeeping import housekeeping, MB, size


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
