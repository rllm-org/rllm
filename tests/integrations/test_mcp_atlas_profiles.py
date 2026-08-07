from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
PROFILE_DIR = ROOT / "examples" / "eval" / "mcp_atlas"


def test_checked_in_provider_profiles_match_benchmark_manifest():
    glm = json.loads((PROFILE_DIR / "glm_5_2_fireworks.json").read_text())
    nemotron = json.loads((PROFILE_DIR / "nemotron_3_super_nim.json").read_text())

    assert glm == {}
    assert nemotron == {
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 16384,
        "chat_template_kwargs": {"enable_thinking": True},
        "reasoning_budget": 16384,
    }

    no_credentials = json.loads((PROFILE_DIR / "agent_config_no_credentials.json").read_text())
    assert no_credentials["task_filter"] == "default_servers"
    assert no_credentials["preflight"] == "smoke"


def test_fixed_cohorts_have_expected_size_and_no_duplicates():
    smoke = [int(value) for value in (PROFILE_DIR / "smoke_10_indices.txt").read_text().strip().split(",")]
    stratified = [int(value) for value in (PROFILE_DIR / "stratified_50_indices.txt").read_text().strip().split(",")]

    assert len(smoke) == len(set(smoke)) == 10
    assert len(stratified) == len(set(stratified)) == 50
    assert all(0 <= idx < 500 for idx in smoke + stratified)


def test_parity_report_accepts_exact_replay():
    script = PROFILE_DIR / "compare_runs.py"
    spec = importlib.util.spec_from_file_location("mcp_atlas_compare_runs", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    report = module.compare({"a": 1.0, "b": 0.5}, {"a": 1.0, "b": 0.5})

    assert report["exact_coverage_matches"] == 2
    assert report["max_absolute_coverage_difference"] == 0.0
    assert report["pass_rate_difference"] == 0.0
    assert report["paired_bootstrap_95_ci"] == [0.0, 0.0]
