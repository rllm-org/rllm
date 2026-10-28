"""CPU preprocessing must preserve inputs and leave the event loop available."""

import asyncio
import importlib
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, sentinel

import pytest


@pytest.fixture(params=["verl", "tinker-parser", "tinker-renderer"])
def engine(request, monkeypatch):
    backend = request.param
    module = pytest.importorskip(f"rllm.engine.rollout.{backend.split('-')[0]}_engine")
    parser = Mock()
    parser.parse.return_value = "rendered prompt"
    parser.process_image_data.side_effect = lambda messages: [image.copy() for message in messages for image in message.get("images", [])]
    monkeypatch.setattr(module.ChatTemplateParser, "get_parser", Mock(return_value=parser))
    tokenizer = Mock()
    tokenizer.encode.return_value = [10, 99, 11]
    if backend == "verl":
        from omegaconf import OmegaConf

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {"rollout": {"name": "vllm", "do_sample": True, "temperature": 1, "top_k": -1, "top_p": 1, "val_kwargs": {"do_sample": False, "top_k": -1, "top_p": 1}}},
                "data": {"max_prompt_length": 4096, "max_response_length": 32},
                "rllm": {"workflow": {"n_processor_workers": 2}},
            }
        )
        value = module.VerlEngine(config, Mock(), tokenizer)
    else:
        renderer = Mock()
        renderer.build_generation_prompt.return_value = SimpleNamespace(chunks=[sentinel.image_chunk, 42])
        renderer.create_conversation_prefix_with_tools.return_value = [{"role": "system", "content": "tools"}]
        monkeypatch.setattr(module.renderers, "get_renderer", Mock(return_value=renderer))
        value = module.TinkerEngine(
            "http://unused.invalid", "test-model", tokenizer, Mock(), renderer_name="qwen3_vl_instruct", bypass_render_with_parser=backend == "tinker-parser", n_processor_workers=2
        )
    value.get_token_output_from_token_input = AsyncMock(return_value=sentinel.completion)
    value.assemble_model_output = Mock(return_value=SimpleNamespace(weight_version=None))
    try:
        yield value
    finally:
        value.shutdown()


@pytest.mark.parametrize("with_image", [False, True])
def test_request_inputs_and_options_are_preserved(engine, with_image):
    messages = [{"role": "user", "content": "question"}]
    if with_image:
        from PIL import Image

        messages[0]["images"] = [Image.new("RGB", (8, 8), (1, 2, 3))]
    tools = [{"type": "function", "function": {"name": "inspect", "description": "Inspect", "parameters": {"type": "object"}}}]
    if hasattr(engine, "processor"):
        torch = pytest.importorskip("torch")
        engine.processor = Mock(
            return_value={
                "input_ids": torch.tensor([[10, 99, 99, 11]]),
                "attention_mask": torch.ones(1, 4),
                "pixel_values": torch.tensor([[1.0, 2.0, 3.0]]),
                "image_grid_thw": torch.tensor([[1, 2, 4]]),
            }
        )

    result = asyncio.run(engine.get_model_response(messages, tools=tools, accumulate_reasoning=True, reasoning_effort="high", max_tokens=17, application_id="request-1"))

    assert result.weight_version == 0
    call = engine.get_token_output_from_token_input.call_args.kwargs
    assert call["max_tokens"] == 17
    if engine.chat_parser is not None:
        engine.chat_parser.parse.assert_called_once_with(messages, add_generation_prompt=True, is_first_msg=True, tools=tools, accumulate_reasoning=True, reasoning_effort="high")
        assert call["token_input"] == [10, 99, 11]
    else:
        assert call["token_input"] == [sentinel.image_chunk, 42]
        rendered_messages = engine.renderer.build_generation_prompt.call_args.args[0]
        content = rendered_messages[-1]["content"]
        if with_image:
            assert content[0]["type"] == "image"
            assert content[0]["image"].tobytes() == messages[0]["images"][0].tobytes()
            assert content[1] == {"type": "text", "text": "question"}
        else:
            assert content == "question"
    assembled = engine.assemble_model_output.call_args.kwargs
    assert assembled["token_output"] is sentinel.completion
    if hasattr(engine, "processor"):
        assert call["application_id"] == "request-1"
        if with_image:
            assert call["image_data"][0].tobytes() == messages[0]["images"][0].tobytes()
            assert assembled["prompt_ids"].tolist() == [10, 99, 99, 11]
            assert assembled["multi_modal_inputs"]["pixel_values"].tolist() == [[1, 2, 3]]
            assert assembled["multi_modal_inputs"]["image_grid_thw"].tolist() == [[1, 2, 4]]
            assert assembled["multi_modal_inputs"]["images_seqlens"].tolist() == [8]
            assert "input_ids" not in assembled["multi_modal_inputs"]
            assert "attention_mask" not in assembled["multi_modal_inputs"]
        else:
            assert call["image_data"] is None
            assert assembled["prompt_ids"] == [10, 99, 11]
            assert assembled["multi_modal_inputs"] is None
            engine.processor.assert_not_called()
    else:
        assert "application_id" not in call
    assert messages[0]["content"] == "question"
    assert ("images" in messages[0]) == with_image


def test_preprocessing_is_bounded_and_does_not_block_loop(engine):
    release = threading.Event()
    lock = threading.Lock()
    thread_ids = set()

    async def exercise():
        loop = asyncio.get_running_loop()
        workers_started = asyncio.Event()

        def prepare(*args):
            with lock:
                thread_ids.add(threading.get_ident())
                if len(thread_ids) == 2:
                    loop.call_soon_threadsafe(workers_started.set)
            if not release.wait(5):
                raise TimeoutError("Event loop did not release preprocessing workers")
            if hasattr(engine, "processor"):
                return [10], None, [10], None
            return [10]

        engine._prepare_request_inputs = prepare
        tasks = [asyncio.create_task(engine.get_model_response([])) for _ in range(4)]
        try:
            await asyncio.wait_for(workers_started.wait(), timeout=2)
            assert all(not task.done() for task in tasks)
            assert threading.get_ident() not in thread_ids
            assert len(thread_ids) == 2
            engine.weight_version = 7
        finally:
            release.set()
            results = await asyncio.gather(*tasks)
        assert all(result.weight_version == 0 for result in results)

    asyncio.run(exercise())
    assert len(thread_ids) == 2


def test_preprocessing_exception_propagates_without_sampling(engine):
    engine._prepare_request_inputs = Mock(side_effect=ValueError("invalid image"))
    with pytest.raises(ValueError, match="invalid image"):
        asyncio.run(engine.get_model_response([]))
    engine.get_token_output_from_token_input.assert_not_called()


def test_cancelled_request_does_not_sample(engine):
    release = threading.Event()

    async def exercise():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()

        def prepare(*args):
            loop.call_soon_threadsafe(started.set)
            if not release.wait(5):
                raise TimeoutError("Preprocessing was not released")
            return ([10], None, [10], None) if hasattr(engine, "processor") else [10]

        engine._prepare_request_inputs = prepare
        task = asyncio.create_task(engine.get_model_response([]))
        try:
            await asyncio.wait_for(started.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)
        engine.get_token_output_from_token_input.assert_not_called()

    asyncio.run(exercise())


def test_shutdown_waits_for_active_work_and_cancels_queued_work(engine):
    started = threading.Barrier(3)
    release = threading.Event()
    cancelled = threading.Event()
    stopped = threading.Event()

    def work():
        started.wait(timeout=2)
        return release.wait(5)

    active = [engine.processor_executor.submit(work) for _ in range(2)]
    started.wait(timeout=2)
    queued = engine.processor_executor.submit(lambda: None)
    queued.add_done_callback(lambda future: cancelled.set())

    def shutdown():
        try:
            engine.shutdown()
        finally:
            stopped.set()

    thread = threading.Thread(target=shutdown)
    thread.start()
    try:
        assert cancelled.wait(2)
        assert queued.cancelled()
        assert not stopped.is_set()
    finally:
        release.set()
        thread.join(timeout=2)
    assert stopped.is_set()
    assert all(future.result(timeout=2) for future in active)


def test_shutdown_is_idempotent_and_prevents_new_submissions(engine):
    engine.processor_executor.submit(lambda: None).result(timeout=2)
    engine.shutdown()
    engine.shutdown()
    with pytest.raises(RuntimeError, match="shutdown"):
        asyncio.run(engine.get_model_response([]))
    assert all(not thread.is_alive() for thread in engine.processor_executor._threads)


@pytest.mark.parametrize("workers", [0, -1])
def test_rejects_nonpositive_worker_count(engine, workers):
    with pytest.raises(ValueError, match="max_workers"):
        if hasattr(engine, "config"):
            engine.config.rllm.workflow.n_processor_workers = workers
            type(engine)(engine.config, Mock(), engine.tokenizer)
        else:
            type(engine)("http://unused.invalid", "test-model", engine.tokenizer, Mock(), renderer_name="qwen3_vl_instruct", n_processor_workers=workers)


def test_fireworks_shutdown_without_preprocessing_pool(monkeypatch):
    module = pytest.importorskip("rllm.engine.rollout.fireworks_engine")
    from rllm.parser import ChatTemplateParser

    monkeypatch.setattr(ChatTemplateParser, "get_parser", Mock())
    engine = module.FireworksEngine(tokenizer=Mock(), sampler=Mock())
    engine.shutdown()
    engine.shutdown()


@pytest.mark.parametrize("has_shutdown", [False, True])
def test_workflow_shutdown_releases_both_executors(has_shutdown):
    module = importlib.import_module("rllm.engine.agent_workflow_engine")
    rollout_engine = SimpleNamespace()
    if has_shutdown:
        rollout_engine.shutdown = Mock()
    engine = module.AgentWorkflowEngine(Mock(), {}, rollout_engine, n_parallel_tasks=1)
    executor = engine.executor
    executor.submit(lambda: None).result(timeout=2)
    engine.shutdown()
    engine.shutdown()
    assert engine.executor is None
    assert all(not thread.is_alive() for thread in executor._threads)
    if has_shutdown:
        assert rollout_engine.shutdown.call_count == 2
