import json

from rllm.types import Episode

from swe.agent_flow import SWEAgentFlow
from swe.flow_config import SWEAgentFlowConfig


def test_flow_config_keeps_trajectory_save_keys(tmp_path):
    config = SWEAgentFlowConfig.from_config(
        {
            "save_trajectories": True,
            "trajectory_output_dir": str(tmp_path),
        }
    )

    assert config.save_trajectories is True
    assert config.trajectory_output_dir == str(tmp_path)


def test_save_trajectory_writes_clean_episode_json(tmp_path):
    flow = SWEAgentFlow(
        SWEAgentFlowConfig(
            save_trajectories=True,
            trajectory_output_dir=str(tmp_path),
        )
    )
    episode = Episode(
        id="episode:0",
        task={"instance_id": "owner/repo:123"},
        artifacts={
            "patch": "diff --git a/file.py b/file.py\n",
            "exit_status": "Submitted",
            "messages": [
                {
                    "role": "assistant",
                    "content": "done",
                    "extra": {
                        "actions": [{"command": "pwd"}],
                        "response": {"large": "raw API response should not be saved"},
                    },
                }
            ],
            "segments": [
                {
                    "kind": "solver",
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "segment",
                            "extra": {"format_error": "bad tool call"},
                        }
                    ],
                }
            ],
        },
    )

    flow._save_trajectory(episode)

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())

    assert data["episode_id"] == "episode:0"
    assert data["instance_id"] == "owner/repo:123"
    assert data["exit_status"] == "Submitted"
    assert data["patch"].startswith("diff --git")
    assert data["messages"][0]["extra"] == {"actions": [{"command": "pwd"}]}
    assert data["segments"][0]["messages"][0]["extra"] == {"format_error": "bad tool call"}
