"""Tests for PR builder."""

import pytest
from unittest.mock import MagicMock, patch, call
from pathlib import Path

from evolution.core.pr_builder import PRBuilder, PRMetrics, _build_commit_message, _build_pr_body


@pytest.fixture
def metrics():
    return PRMetrics(
        target_name="github-code-review",
        artifact_type="skill",
        baseline_score=0.65,
        evolved_score=0.78,
        improvement=0.13,
        optimizer="GEPA",
        iterations=10,
        elapsed_seconds=120.0,
        eval_dataset_size=20,
        holdout_score=0.76,
        constraint_violations_rejected=3,
    )


class TestBuildCommitMessage:
    def test_contains_target_name(self, metrics):
        msg = _build_commit_message(metrics)
        assert "github-code-review" in msg

    def test_contains_scores(self, metrics):
        msg = _build_commit_message(metrics)
        assert "0.650" in msg
        assert "0.780" in msg

    def test_contains_optimizer_info(self, metrics):
        msg = _build_commit_message(metrics)
        assert "GEPA" in msg
        assert "10" in msg

    def test_contains_dataset_size(self, metrics):
        msg = _build_commit_message(metrics)
        assert "20" in msg

    def test_contains_holdout_score(self, metrics):
        msg = _build_commit_message(metrics)
        assert "0.760" in msg

    def test_contains_rejected_count(self, metrics):
        msg = _build_commit_message(metrics)
        assert "3" in msg

    def test_zero_improvement_no_crash(self):
        m = PRMetrics(
            target_name="test",
            artifact_type="skill",
            baseline_score=0.0,
            evolved_score=0.0,
            improvement=0.0,
            optimizer="GEPA",
            iterations=5,
            elapsed_seconds=60.0,
        )
        msg = _build_commit_message(m)
        assert "test" in msg


class TestBuildPRBody:
    def test_contains_summary_section(self, metrics):
        body = _build_pr_body(metrics)
        assert "## Evolution Summary" in body

    def test_contains_target_name(self, metrics):
        body = _build_pr_body(metrics)
        assert "github-code-review" in body

    def test_contains_improvement_percentage(self, metrics):
        body = _build_pr_body(metrics)
        assert "%" in body

    def test_contains_review_checklist(self, metrics):
        body = _build_pr_body(metrics)
        assert "## Review Checklist" in body
        assert "- [ ]" in body

    def test_contains_attribution(self, metrics):
        body = _build_pr_body(metrics)
        assert "hermes-agent-self-evolution" in body

    def test_contains_holdout_score_when_provided(self, metrics):
        body = _build_pr_body(metrics)
        assert "0.760" in body

    def test_no_holdout_when_not_provided(self):
        m = PRMetrics(
            target_name="test",
            artifact_type="skill",
            baseline_score=0.5,
            evolved_score=0.6,
            improvement=0.1,
            optimizer="GEPA",
            iterations=5,
            elapsed_seconds=60.0,
            holdout_score=None,
        )
        body = _build_pr_body(m)
        assert "Holdout" not in body


class TestPRBuilderCreatePR:
    def test_create_pr_git_failure_raises(self, tmp_path):
        """PR creation fails gracefully when git is not available."""
        builder = PRBuilder()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="not a git repo")

            metrics = PRMetrics(
                target_name="test",
                artifact_type="skill",
                baseline_score=0.5,
                evolved_score=0.6,
                improvement=0.1,
                optimizer="GEPA",
                iterations=5,
                elapsed_seconds=10.0,
            )

            with pytest.raises(RuntimeError):
                builder.create_pr(
                    hermes_repo=tmp_path,
                    changed_files=[],
                    metrics=metrics,
                )
