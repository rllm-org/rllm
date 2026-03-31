"""Tests for geo3k flow."""

from evaluator import geo3k_evaluator
from geo3k_flow import _build_vlm_content, _detect_mime

from rllm.types import Episode, Step, Trajectory


def test_detect_mime_png():
    assert _detect_mime(b"\x89PNG\r\n\x1a\n") == "image/png"


def test_detect_mime_jpeg():
    assert _detect_mime(b"\xff\xd8\xff\xe0") == "image/jpeg"


def test_build_vlm_content_with_bytes():
    # Minimal PNG header
    fake_png = b"\x89PNG" + b"\x00" * 20
    content = _build_vlm_content("What is x?", [fake_png])
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "What is x?"


def test_build_vlm_content_no_images():
    content = _build_vlm_content("What is x?", [])
    assert len(content) == 1
    assert content[0]["type"] == "text"


def test_evaluator_correct():
    task = {"question": "Find x", "ground_truth": "48"}

    episode = Episode(
        trajectories=[
            Trajectory(name="solver", steps=[Step(action="The answer is \\boxed{48}")]),
        ],
        artifacts={"answer": "The answer is \\boxed{48}"},
    )

    result = geo3k_evaluator.evaluate(task, episode)
    assert result.is_correct is True
    assert result.reward == 1.0


def test_evaluator_wrong():
    task = {"question": "Find x", "ground_truth": "48"}

    episode = Episode(
        trajectories=[
            Trajectory(name="solver", steps=[Step(action="The answer is \\boxed{24}")]),
        ],
        artifacts={"answer": "The answer is \\boxed{24}"},
    )

    result = geo3k_evaluator.evaluate(task, episode)
    assert result.is_correct is False
    assert result.reward == 0.0


def test_evaluator_no_boxed():
    task = {"question": "Find x", "ground_truth": "48"}

    episode = Episode(
        trajectories=[
            Trajectory(name="solver", steps=[Step(action="I think 48")]),
        ],
        artifacts={"answer": "I think 48"},
    )

    result = geo3k_evaluator.evaluate(task, episode)
    assert result.is_correct is False
    assert result.reward == 0.0
