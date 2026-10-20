"""Tests for staged deployment, promotion, rollback and caps."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from evolution.core import staging

SKILL = "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\nDo the thing.\n"
EVOLVED = "---\nname: demo\ndescription: Demo skill\n---\n\n# Demo\nDo the thing carefully, then verify.\n"


@pytest.fixture
def env(tmp_path):
    live = tmp_path / "skills" / "demo" / "SKILL.md"
    live.parent.mkdir(parents=True)
    live.write_text(SKILL)
    root = tmp_path / "repo"
    root.mkdir()
    return root, live


class TestStage:
    def test_stage_never_touches_live(self, env):
        root, live = env
        m = staging.stage("demo", "skill", live, EVOLVED, {"decision": "promote"}, root=root)
        assert live.read_text() == SKILL
        assert Path(m["staged_path"]).read_text() == EVOLVED
        assert Path(m["staged_path"]).name == "evolved_skill.md"
        assert Path(m["staged_path"]).parent.parent == root / "output" / "demo"
        man = json.loads((root / "staging" / "demo.json").read_text())
        assert man["live_sha256_at_staging"] == staging.sha256_text(SKILL)
        assert man["status"] == "staged" and man["stats"]["decision"] == "promote"

    def test_finalize_default_stages_only(self, env):
        root, live = env
        m = staging.finalize("demo", "skill", live, EVOLVED, {"decision": "promote"}, root=root)
        assert m["status"] == "staged" and live.read_text() == SKILL

    def test_finalize_apply_requires_promote_decision(self, env):
        root, live = env
        m = staging.finalize("demo", "skill", live, EVOLVED, {"decision": "stage"}, apply=True, root=root)
        assert m["status"] == "staged" and live.read_text() == SKILL
        m = staging.finalize("demo", "skill", live, EVOLVED, {"decision": "promote"}, apply=True, root=root)
        assert m["status"] == "promoted" and live.read_text() == EVOLVED

    def test_force_promote(self, env):
        root, live = env
        m = staging.finalize("demo", "skill", live, EVOLVED, {"decision": "reject"},
                             force_promote=True, root=root)
        assert m["status"] == "promoted"


class TestPromoteRollback:
    def test_promote_backup_and_rollback(self, env):
        root, live = env
        staging.stage("demo", "skill", live, EVOLVED, {}, root=root)
        m = staging.promote("demo", root=root)
        assert live.read_text() == EVOLVED
        assert Path(m["backup_path"]).read_text() == SKILL
        staging.rollback("demo", root=root)
        assert live.read_text() == SKILL
        assert staging.load_manifest("demo", root=root)["status"] == "rolled_back"

    def test_concurrent_edit_aborts(self, env):
        root, live = env
        staging.stage("demo", "skill", live, EVOLVED, {}, root=root)
        live.write_text(SKILL + "\nhand edit\n")
        with pytest.raises(staging.PromotionError, match="concurrent"):
            staging.promote("demo", root=root)
        assert "hand edit" in live.read_text()

    def test_finalize_reports_promotion_error(self, env):
        root, live = env
        bad = "---\nname: other\ndescription: Demo skill\n---\n\nbody\n"
        m = staging.finalize("demo", "skill", live, bad, {"decision": "promote"}, apply=True, root=root)
        assert "promotion_error" in m and live.read_text() == SKILL

    def test_double_promote_refused(self, env):
        root, live = env
        staging.stage("demo", "skill", live, EVOLVED, {}, root=root)
        staging.promote("demo", root=root)
        with pytest.raises(staging.PromotionError):
            staging.promote("demo", root=root)

    def test_rollback_without_backup(self, env):
        root, live = env
        staging.stage("demo", "skill", live, EVOLVED, {}, root=root)
        with pytest.raises(staging.PromotionError):
            staging.rollback("demo", root=root)

    def test_keep_last_five_backups(self, env):
        root, live = env
        for i in range(8):
            text = EVOLVED.replace("verify.", f"verify {i}.")
            staging.stage("demo", "skill", live, text, {}, root=root)
            staging.promote("demo", root=root)
        backups = list((root / "staging" / "backups" / "demo").iterdir())
        assert len(backups) == 5
        # newest backup = version 6 (live is 7)
        newest = sorted(backups, key=lambda p: p.name)[-1]
        assert "verify 6." in newest.read_text()


class TestValidation:
    def test_valid(self):
        assert staging.validate_frontmatter(EVOLVED, SKILL)[0]

    @pytest.mark.parametrize("text,why", [
        ("no frontmatter", "frontmatter"),
        ("---\nname: [unclosed\n---\nbody", "YAML"),
        ("---\nname: demo\n---\nbody", "description"),
        ("---\nname: demo2\ndescription: Demo skill\n---\nbody", "name"),
        ("---\nname: demo\ndescription: changed\n---\nbody", "description"),
        ("---\nname: demo\ndescription: Demo skill\n---\n   \n", "empty"),
    ])
    def test_invalid(self, text, why):
        ok, msg = staging.validate_frontmatter(text, SKILL)
        assert not ok and why.lower() in msg.lower()

    def test_payload_json(self):
        assert not staging.validate_payload("tool_descriptions", "{bad", None)[0]
        assert staging.validate_payload("tool_descriptions", "[]", None)[0]


class TestCaps:
    def test_size_cap_prunes_oldest(self, tmp_path):
        root = tmp_path
        od = root / "output" / "x"
        files = []
        for i in range(6):
            p = od / f"run{i}" / "blob.bin"
            p.parent.mkdir(parents=True)
            p.write_bytes(b"0" * 1000)
            t = time.time() - 1000 + i
            os.utime(p, (t, t))
            files.append(p)
        removed = staging.enforce_size_cap(max_bytes=3500, root=root)
        assert removed == files[:3]
        assert all(p.exists() for p in files[3:])
        assert not (od / "run0").exists()  # empty dir cleaned

    def test_size_cap_protects_manifests_and_staged(self, env):
        root, live = env
        m = staging.stage("demo", "skill", live, EVOLVED, {}, root=root)
        old = root / "output" / "junk.bin"
        old.write_bytes(b"0" * 5000)
        os.utime(old, (1, 1))
        os.utime(m["staged_path"], (0, 0))
        staging.enforce_size_cap(max_bytes=100, root=root)
        assert Path(m["staged_path"]).exists() and staging.manifest_path("demo", root).exists()
        assert not old.exists()

    def test_default_cap_is_50mb(self):
        assert staging.MAX_TOTAL_BYTES == 50 * 1024 * 1024


def test_cli_list_and_rollback(env):
    root, live = env
    staging.stage("demo", "skill", live, EVOLVED, {"improvement": 0.2}, root=root)
    staging.promote("demo", root=root)
    envv = dict(os.environ, EVOLUTION_STAGING_ROOT=str(root))
    repo = Path(staging.__file__).resolve().parents[2]
    out = subprocess.run([sys.executable, "-m", "evolution.core.staging", "list"], env=envv,
                         capture_output=True, text=True, cwd=repo)
    assert out.returncode == 0 and '"artifact": "demo"' in out.stdout
    out = subprocess.run([sys.executable, "-m", "evolution.core.staging", "rollback", "--artifact", "demo"],
                         env=envv, capture_output=True, text=True, cwd=repo)
    assert out.returncode == 0, out.stderr
    assert live.read_text() == SKILL


def test_gitignored():
    repo = Path(staging.__file__).resolve().parents[2]
    gi = (repo / ".gitignore").read_text().split()
    assert any(l.strip("/") == "staging" for l in gi)
    assert any(l.strip("/") == "output" for l in gi)
