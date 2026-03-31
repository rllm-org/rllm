import importlib.util
import sys
import types
from pathlib import Path


def _load_workflow_engine_module(monkeypatch):
    # Stub verl + torch bits imported inside the function.
    verl = types.ModuleType("verl")

    class FakeDataProto:
        def __init__(self, meta_info):
            self.meta_info = meta_info

        @classmethod
        def from_dict(cls, tensors=None, non_tensors=None, meta_info=None):  # noqa: ARG002
            return cls(meta_info=meta_info)

    verl.DataProto = FakeDataProto

    verl_utils = types.ModuleType("verl.utils")
    verl_tf = types.ModuleType("verl.utils.torch_functional")
    verl_tf.pad_sequence_to_length = lambda x, *a, **k: x

    sys.modules["verl"] = verl
    sys.modules["verl.utils"] = verl_utils
    sys.modules["verl.utils.torch_functional"] = verl_tf

    # Stub torch so type annotations don't crash import in minimal env.
    torch = types.ModuleType("torch")
    torch.Tensor = object
    torch.long = "long"
    torch.float32 = "float32"
    torch.nn = types.SimpleNamespace(utils=types.SimpleNamespace(rnn=types.SimpleNamespace(pad_sequence=lambda *a, **k: [])))
    torch.zeros_like = lambda *a, **k: []
    torch.as_tensor = lambda *a, **k: []
    torch.arange = lambda *a, **k: []
    torch.cumsum = lambda *a, **k: []
    torch.concat = lambda *a, **k: []
    sys.modules["torch"] = torch

    module_path = Path(__file__).resolve().parents[2] / "rllm" / "engine" / "agent_workflow_engine.py"
    spec = importlib.util.spec_from_file_location("rllm_agent_workflow_engine_test", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_transform_results_includes_dropped_episodes_meta(monkeypatch):
    mod = _load_workflow_engine_module(monkeypatch)

    # Minimal episode stubs
    class Ep:
        def __init__(self, id, termination_reason=None, trajectories=None):
            self.id = id
            self.termination_reason = termination_reason
            self.trajectories = trajectories or []
            self.metrics = {}
            self.is_correct = False

    class Term:
        def __init__(self, value):
            self.value = value

    episodes = [
        None,
        Ep("t1:0", termination_reason=Term("max_prompt_length_exceeded"), trajectories=[type("T", (), {"steps": []})()]),
    ]

    # Instantiate engine without running __init__
    engine = object.__new__(mod.AgentWorkflowEngine)
    engine.config = type("Cfg", (), {"rllm": type("R", (), {"stepwise_advantage": type("S", (), {"enable": False})()})()})()
    engine.rollout_engine = type("RE", (), {"chat_parser": type("CP", (), {})()})()

    out = mod.AgentWorkflowEngine.transform_results_for_verl(engine, episodes, ["t0", "t1"])

    assert "dropped_episodes" in out.meta_info
    assert len(out.meta_info["dropped_episodes"]) == 2
