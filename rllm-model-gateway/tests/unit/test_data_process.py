"""Tests for token/logprob extraction and trace record building."""

from rllm_model_gateway.data_process import (
    build_trace_record,
    build_trace_record_from_chunks,
    extract_completion_token_ids,
    extract_delta_logprobs,
    extract_delta_token_ids,
    extract_logprobs,
    extract_prompt_token_ids,
    strip_vllm_fields,
)

# ------------------------------------------------------------------
# Extraction helpers
# ------------------------------------------------------------------


class TestExtractPromptTokenIds:
    def test_present(self):
        assert extract_prompt_token_ids({"prompt_token_ids": [1, 2, 3]}) == [1, 2, 3]

    def test_missing(self):
        assert extract_prompt_token_ids({}) == []

    def test_none(self):
        assert extract_prompt_token_ids({"prompt_token_ids": None}) == []


class TestExtractCompletionTokenIds:
    def test_from_direct_token_ids(self):
        """vLLM 0.11+ format: token_ids directly in choice."""
        resp = {"choices": [{"token_ids": [10, 11, 12]}]}
        assert extract_completion_token_ids(resp) == [10, 11, 12]

    def test_no_choices(self):
        assert extract_completion_token_ids({}) == []

    def test_no_token_ids(self):
        assert extract_completion_token_ids({"choices": [{}]}) == []


class TestExtractLogprobs:
    def test_from_logprobs_content(self):
        resp = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "Hi", "logprob": -0.5},
                            {"token": " there", "logprob": -0.3},
                        ]
                    }
                }
            ]
        }
        assert extract_logprobs(resp) == [-0.5, -0.3]

    def test_no_logprobs(self):
        assert extract_logprobs({"choices": [{}]}) == []

    def test_no_choices(self):
        assert extract_logprobs({}) == []


class TestExtractDeltaTokenIds:
    def test_from_direct_token_ids(self):
        """vLLM 0.11+ format: token_ids directly in choice."""
        chunk = {"choices": [{"token_ids": [42, 43]}]}
        assert extract_delta_token_ids(chunk) == [42, 43]

    def test_empty_chunk(self):
        assert extract_delta_token_ids({"choices": [{}]}) == []


class TestExtractDeltaLogprobs:
    def test_from_chunk(self):
        chunk = {
            "choices": [
                {
                    "logprobs": {
                        "content": [
                            {"token": "a", "logprob": -0.1},
                            {"token": "b", "logprob": -0.2},
                        ]
                    }
                }
            ]
        }
        assert extract_delta_logprobs(chunk) == [-0.1, -0.2]

    def test_no_logprobs(self):
        assert extract_delta_logprobs({"choices": [{}]}) == []


# ------------------------------------------------------------------
# Response sanitisation
# ------------------------------------------------------------------


class TestStripVllmFields:
    def test_strips_root_level_fields(self):
        resp = {
            "prompt_token_ids": [1, 2],
            "prompt_logprobs": None,
            "kv_transfer_params": None,
            "choices": [],
        }
        sanitized = strip_vllm_fields(resp)
        assert "prompt_token_ids" not in sanitized
        assert "prompt_logprobs" not in sanitized
        assert "kv_transfer_params" not in sanitized

    def test_strips_choice_level_fields(self):
        resp = {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "token_ids": [10, 11],
                    "stop_reason": None,
                }
            ]
        }
        sanitized = strip_vllm_fields(resp)
        choice = sanitized["choices"][0]
        assert "token_ids" not in choice
        assert "stop_reason" not in choice
        assert choice["message"]["content"] == "hi"

    def test_preserves_other_fields(self):
        resp = {"id": "123", "model": "m", "choices": []}
        sanitized = strip_vllm_fields(resp)
        assert sanitized["id"] == "123"
        assert sanitized["model"] == "m"


# ------------------------------------------------------------------
# Trace record building
# ------------------------------------------------------------------


class TestBuildTraceRecord:
    def test_non_streaming(self):
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        response_body = {
            "model": "test-model",
            "prompt_token_ids": [1, 2, 3],
            "choices": [
                {
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                    "stop_reason": None,
                    "token_ids": [10, 11],
                    "logprobs": {
                        "content": [
                            {"token": "hi", "logprob": -0.5, "bytes": None, "top_logprobs": []},
                            {"token": "!", "logprob": -0.1, "bytes": None, "top_logprobs": []},
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2},
        }
        trace = build_trace_record("session-1", request_body, response_body, 100.0)

        assert trace.session_id == "session-1"
        assert trace.model == "test-model"
        assert trace.prompt_token_ids == [1, 2, 3]
        assert trace.completion_token_ids == [10, 11]
        assert trace.logprobs == [-0.5, -0.1]
        assert trace.finish_reason == "stop"
        assert trace.token_counts == {"prompt": 3, "completion": 2}
        assert trace.messages == [{"role": "user", "content": "hello"}]
        assert trace.response_message == {"role": "assistant", "content": "hi"}
        assert trace.latency_ms == 100.0
        assert trace.trace_id  # non-empty UUID

    def test_streaming_chunks(self):
        request_body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        chunks = [
            {
                "model": "test-model",
                "prompt_token_ids": [1, 2, 3],
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "logprobs": None,
                        "finish_reason": None,
                    }
                ],
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "Hi"},
                        "token_ids": [10],
                        "logprobs": {"content": [{"token": "Hi", "logprob": -0.5}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": " there"},
                        "token_ids": [11],
                        "logprobs": {"content": [{"token": " there", "logprob": -0.3}]},
                        "finish_reason": None,
                    }
                ]
            },
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        ]
        trace = build_trace_record_from_chunks("session-2", request_body, chunks, 200.0)

        assert trace.session_id == "session-2"
        assert trace.prompt_token_ids == [1, 2, 3]
        assert trace.completion_token_ids == [10, 11]
        assert trace.logprobs == [-0.5, -0.3]
        assert trace.response_message["content"] == "Hi there"
        assert trace.finish_reason == "stop"
        assert trace.token_counts == {"prompt": 3, "completion": 2}
