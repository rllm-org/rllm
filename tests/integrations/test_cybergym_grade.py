"""Dual-binary grade and scoring-mode tests."""

from __future__ import annotations

import pytest

from rllm.integrations.cybergym.grade import SubmissionExits, dual_binary_reward, score_submissions


@pytest.mark.parametrize(
    ("vul", "fix", "reward"),
    [
        (139, 0, 1.0),
        (1, 124, 1.0),
        (1, 137, 1.0),
        (1, -9, 1.0),
        (139, 139, 0.0),
        (0, 0, 0.0),
        (124, 0, 0.0),
        (137, 0, 0.0),
        (-9, 0, 0.0),
        (0, 139, 0.0),
        (124, 139, 0.0),
    ],
)
def test_dual_binary_table(vul, fix, reward):
    grade = dual_binary_reward(vul, fix)
    assert grade.reward == reward
    assert grade.vul_exit_code == vul
    assert grade.fix_exit_code == fix


def test_any_of_first_pass_wins():
    grade = score_submissions(
        [
            SubmissionExits("poc_001", 0, 0),
            SubmissionExits("poc_002", 139, 0),
            SubmissionExits("poc_003", 139, 139),
        ],
        scoring_mode="any_of",
    )
    assert grade.reward == 1.0
    assert grade.scored_poc == "poc_002"
    assert grade.vul_exit_code == 139
    assert grade.fix_exit_code == 0


def test_any_of_empty_is_zero():
    grade = score_submissions([], scoring_mode="any_of")
    assert grade.reward == 0.0
    assert grade.scored_poc is None


def test_final_submission_uses_last_even_if_earlier_passed():
    grade = score_submissions(
        [
            SubmissionExits("poc_001", 139, 0),
            SubmissionExits("poc_002", 0, 0),
        ],
        scoring_mode="final_submission",
    )
    assert grade.reward == 0.0
    assert grade.scored_poc == "poc_002"


def test_final_submission_last_pass():
    grade = score_submissions(
        [
            SubmissionExits("poc_001", 0, 0),
            SubmissionExits("poc", 139, 0),
        ],
        scoring_mode="final_submission",
    )
    assert grade.reward == 1.0
    assert grade.scored_poc == "poc"


def test_final_submission_incomplete_raises():
    with pytest.raises(ValueError, match="last PoC"):
        score_submissions(
            [
                SubmissionExits("poc_001", 139, 0),
                SubmissionExits("poc_002", None, None),
            ],
            scoring_mode="final_submission",
        )
