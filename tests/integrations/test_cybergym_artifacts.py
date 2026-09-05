"""Parse Harbor test.sh / reward.json into CyberGym scores."""

from __future__ import annotations

import pytest

from rllm.integrations.cybergym.artifacts import parse_reward_json, parse_test_output, score_from_harbor_result

_TEST_OUTPUT = """\
Verifying 2 PoC submission(s) via task-server

--- poc_001 ---
  vul_exit=0
  fix_exit=0

--- poc_002 ---
  vul_exit=139
  fix_exit=0
  PASS

Result: PASS (vul_exit=139, fix_exit=0)
{
  "reward": 1.0,
  "vul_exit_code": 139,
  "fix_exit_code": 0,
  "vul_crashed": true,
  "fix_safe": true
}
"""


def test_parse_test_output_blocks():
    subs = parse_test_output(_TEST_OUTPUT)
    assert [s.name for s in subs] == ["poc_001", "poc_002"]
    assert subs[0].vul_exit_code == 0
    assert subs[1].vul_exit_code == 139
    assert subs[1].fix_exit_code == 0


def test_parse_reward_json():
    data = parse_reward_json(_TEST_OUTPUT)
    assert data is not None
    assert data["reward"] == 1.0
    assert data["vul_exit_code"] == 139


def test_any_of_uses_harbor_reward_and_exits():
    score = score_from_harbor_result(
        raw_result={"verifier_result": {"stdout": _TEST_OUTPUT, "rewards": {"reward": 1.0}}},
        harbor_reward=1.0,
        scoring_mode="any_of",
    )
    assert score.reward == 1.0
    assert score.vul_exit_code == 139
    assert score.fix_exit_code == 0
    assert score.n_submissions == 2
    assert score.scoring_mode == "any_of"


def test_reward_json_zero_still_keeps_exit_codes():
    score = score_from_harbor_result(
        raw_result={"verifier_result": {"stdout": '{"reward": 0.0, "vul_exit_code": 0, "fix_exit_code": 0}'}},
        harbor_reward=0.0,
        scoring_mode="any_of",
    )
    assert score.reward == 0.0
    assert score.vul_exit_code == 0
    assert score.fix_exit_code == 0
    assert score.n_submissions == 1


def test_no_poc_is_zero():
    score = score_from_harbor_result(
        raw_result={"verifier_result": {"stdout": "No PoC files found\n"}},
        harbor_reward=0.0,
        scoring_mode="any_of",
    )
    assert score.reward == 0.0
    assert score.n_submissions == 0


def test_final_submission_last_fail_even_if_earlier_pass():
    score = score_from_harbor_result(
        raw_result={
            "verifier_result": {
                "stdout": """\
Verifying 2 PoC submission(s)
--- poc_001 ---
  vul_exit=139
  fix_exit=0
  PASS
--- poc_002 ---
  vul_exit=0
  fix_exit=0
"""
            }
        },
        harbor_reward=1.0,
        scoring_mode="final_submission",
    )
    assert score.reward == 0.0
    assert score.scored_poc == "poc_002"
    assert score.harbor_reward == 1.0


def test_final_submission_incomplete_raises():
    with pytest.raises(ValueError, match="last PoC"):
        score_from_harbor_result(
            raw_result={
                "verifier_result": {
                    "stdout": """\
--- poc_001 ---
  vul_exit=139
  fix_exit=0
  PASS
--- poc_002 ---
"""
                }
            },
            harbor_reward=1.0,
            scoring_mode="final_submission",
        )
