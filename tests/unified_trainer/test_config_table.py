"""Tests for ``print_config_table`` / ``_flatten_config`` in visualization."""

from omegaconf import OmegaConf

from rllm.trainer.algorithms.visualization import _flatten_config, print_config_table


def test_flatten_nested_config_to_dotted_paths():
    flat = _flatten_config(
        {
            "data": {"max_prompt_length": 57344, "val_batch_size": -1},
            "rllm": {"data": {"max_prompt_length": 57344, "seed": 42}},
            "model": {"name": "m"},
        }
    )
    assert flat == {
        "data.max_prompt_length": 57344,
        "data.val_batch_size": -1,
        "rllm.data.max_prompt_length": 57344,
        "rllm.data.seed": 42,
        "model.name": "m",
    }


def test_flatten_keeps_leaf_lists_and_empty_dicts_whole():
    flat = _flatten_config({"trainer": {"logger": ["console", "wandb"], "extra": {}}})
    assert flat["trainer.logger"] == ["console", "wandb"]
    assert flat["trainer.extra"] == {}


def test_print_config_table_runs_on_dictconfig(capsys):
    cfg = OmegaConf.create(
        {
            "data": {"max_prompt_length": 57344},
            "rllm": {"data": {"max_prompt_length": 57344, "seed": 42}},
        }
    )
    print_config_table(cfg, title="Training Config (TestBackend)")
    out = capsys.readouterr().out
    assert "Training Config (TestBackend)" in out
    assert "rllm.data.max_prompt_length" in out
    assert "57344" in out


def test_print_config_table_resolves_interpolations(capsys):
    cfg = OmegaConf.create(
        {
            "rllm": {"data": {"max_response_length": 8192}},
            "rollout": {"max_tokens": "${rllm.data.max_response_length}"},
        }
    )
    print_config_table(cfg)
    out = capsys.readouterr().out
    # The interpolation is resolved to the concrete value, not printed verbatim.
    assert "${rllm.data.max_response_length}" not in out
    assert "8192" in out
