"""Tests for daily evolution runner orchestration."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from evolution.monitor.daily_runner import (
    discover_skills,
    load_past_metrics,
    select_skill_via_hermes,
    write_log_entry,
)


@pytest.fixture
def skills_dir(tmp_path):
    """Create a minimal skills directory structure."""
    skills = tmp_path / "skills"
    for name in ["github-code-review", "systematic-debugging", "arxiv"]:
        skill_dir = skills / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {name}\ndescription: Test\n---\nContent\n")
    return tmp_path


@pytest.fixture
def output_dir(tmp_path):
    """Create an output directory with past metrics."""
    output = tmp_path / "output"
    for skill, score in [("github-code-review", 0.7), ("systematic-debugging", 0.5)]:
        run_dir = output / skill / "20240101_120000"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(json.dumps({
            "skill_name": skill,
            "timestamp": "20240101_120000",
            "baseline_score": 0.6,
            "evolved_score": score,
            "improvement": score - 0.6,
        }))
    return tmp_path


class TestDiscoverSkills:
    def test_discovers_skills_from_directory(self, skills_dir):
        # get_hermes_agent_path is imported inside the function body
        with patch("evolution.core.config.get_hermes_agent_path") as mock_get_path:
            mock_get_path.side_effect = FileNotFoundError("no hermes-agent")
            skills = discover_skills()
            assert isinstance(skills, list)

    def test_returns_sorted_list(self, tmp_path):
        """discover_skills returns a sorted list."""
        with patch("evolution.core.config.get_hermes_agent_path") as mock_get:
            mock_get.side_effect = FileNotFoundError
            with patch("pathlib.Path.home") as mock_home:
                # Point home to a directory with no .hermes/skills
                mock_home.return_value = tmp_path
                result = discover_skills()
                assert result == sorted(result)


class TestLoadPastMetrics:
    def test_loads_metrics_from_output_dir(self, output_dir, tmp_path):
        """Test that load_past_metrics reads the most recent metrics.json."""
        import evolution.monitor.daily_runner as runner_module

        original_repo = runner_module.REPO_ROOT
        try:
            runner_module.REPO_ROOT = output_dir
            metrics = load_past_metrics()
            assert "github-code-review" in metrics
            assert "systematic-debugging" in metrics
        finally:
            runner_module.REPO_ROOT = original_repo

    def test_returns_empty_when_no_output(self, tmp_path):
        import evolution.monitor.daily_runner as runner_module

        original_repo = runner_module.REPO_ROOT
        try:
            runner_module.REPO_ROOT = tmp_path  # no output/ subdir
            metrics = load_past_metrics()
            assert metrics == {}
        finally:
            runner_module.REPO_ROOT = original_repo

    def test_extracts_correct_fields(self, output_dir):
        import evolution.monitor.daily_runner as runner_module

        original_repo = runner_module.REPO_ROOT
        try:
            runner_module.REPO_ROOT = output_dir
            metrics = load_past_metrics()
            skill_metrics = metrics["systematic-debugging"]
            assert "baseline_score" in skill_metrics
            assert "evolved_score" in skill_metrics
            assert "improvement" in skill_metrics
        finally:
            runner_module.REPO_ROOT = original_repo


class TestSelectSkillViaHermes:
    def test_dry_run_returns_valid_skill(self):
        skills = ["arxiv", "github-code-review", "systematic-debugging"]
        past = {"github-code-review": {"evolved_score": 0.8, "baseline_score": 0.7, "improvement": 0.1}}

        result = select_skill_via_hermes(skills, past, dry_run=True)
        assert result in skills

    def test_dry_run_with_no_past_metrics_returns_first(self):
        skills = ["arxiv", "github-code-review"]
        result = select_skill_via_hermes(skills, {}, dry_run=True)
        assert result in skills

    def test_dry_run_prefers_lowest_scored_skill(self):
        skills = ["a", "b", "c"]
        past = {
            "a": {"evolved_score": 0.9, "baseline_score": 0.8, "improvement": 0.1, "timestamp": "t"},
            "b": {"evolved_score": 0.3, "baseline_score": 0.4, "improvement": -0.1, "timestamp": "t"},
        }
        result = select_skill_via_hermes(skills, past, dry_run=True)
        assert result == "b"  # lowest evolved_score

    def test_raises_when_no_skills(self):
        with pytest.raises(ValueError):
            select_skill_via_hermes([], {})

    def test_fuzzy_fallback_when_hermes_returns_unknown(self):
        skills = ["github-code-review", "systematic-debugging"]
        past = {}

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="I recommend github-code-review skill\n",
            )
            result = select_skill_via_hermes(skills, past)
            assert result == "github-code-review"

    def test_hard_fallback_when_hermes_returns_garbage(self):
        skills = ["github-code-review", "systematic-debugging"]
        past = {
            "github-code-review": {"evolved_score": 0.5, "baseline_score": 0.4,
                                    "improvement": 0.1, "timestamp": "t"},
            "systematic-debugging": {"evolved_score": 0.3, "baseline_score": 0.4,
                                      "improvement": -0.1, "timestamp": "t"},
        }

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="COMPLETELY_UNKNOWN_SKILL\n")
            result = select_skill_via_hermes(skills, past)
            # Falls back to lowest evolved_score
            assert result == "systematic-debugging"


class TestWriteLogEntry:
    def test_writes_json_log(self, tmp_path):
        import evolution.monitor.daily_runner as runner_module

        original_repo = runner_module.REPO_ROOT
        try:
            runner_module.REPO_ROOT = tmp_path
            write_log_entry("test-skill", True, 0.15)

            log_file = tmp_path / "logs" / "daily_evolution.log"
            assert log_file.exists()

            entry = json.loads(log_file.read_text().strip())
            assert entry["skill"] == "test-skill"
            assert entry["success"] is True
            assert entry["improvement"] == 0.15
        finally:
            runner_module.REPO_ROOT = original_repo

    def test_appends_multiple_entries(self, tmp_path):
        import evolution.monitor.daily_runner as runner_module

        original_repo = runner_module.REPO_ROOT
        try:
            runner_module.REPO_ROOT = tmp_path
            write_log_entry("skill-a", True, 0.1)
            write_log_entry("skill-b", False, None)

            log_file = tmp_path / "logs" / "daily_evolution.log"
            lines = [l for l in log_file.read_text().strip().split("\n") if l]
            assert len(lines) == 2
        finally:
            runner_module.REPO_ROOT = original_repo
