import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest

from rllm_model_gateway.proxy import ReverseProxy


def _response(body):
    return httpx.Response(200, json=body, request=httpx.Request("POST", "http://worker/v1/completions"))


@pytest.mark.parametrize(
    "error_type",
    [
        httpx.ReadError,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
        httpx.TimeoutException,
    ],
)
@pytest.mark.asyncio
async def test_send_with_retry_retries_transient_http_errors(error_type, monkeypatch):
    request = httpx.Request("POST", "http://worker/v1/completions")
    transient_error = error_type("transient failure", request=request)
    expected = _response({"choices": []})
    proxy = ReverseProxy(router=AsyncMock(), store=AsyncMock(), max_retries=1)
    proxy._http = AsyncMock()
    proxy._http.request = AsyncMock(side_effect=transient_error)
    retry_client = AsyncMock()
    retry_client.request = AsyncMock(return_value=expected)
    monkeypatch.setattr("rllm_model_gateway.proxy.httpx.AsyncClient", lambda **_: retry_client)

    response = await proxy._send_with_retry(
        method="POST",
        url=str(request.url),
        content=b"{}",
        headers={},
    )

    assert response is expected
    proxy._http.request.assert_awaited_once()
    retry_client.request.assert_awaited_once()
    retry_client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_non_streaming_abort_resumes_from_partial_token_ids(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    first = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "message": {"role": "assistant", "content": "hel"},
                "token_ids": [10],
                "logprobs": {"content": [{"logprob": -0.1}]},
                "finish_reason": "abort",
            }
        ],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    resumed = {
        "prompt_token_ids": [1, 2, 10],
        "choices": [
            {
                "text": "lo",
                "token_ids": [11],
                "logprobs": {"token_logprobs": [-0.2]},
                "finish_reason": "stop",
            }
        ],
    }

    proxy = ReverseProxy(
        router=AsyncMock(),
        store=AsyncMock(),
        cumulative_token_mode=True,
        max_retries=2,
    )
    proxy._http = AsyncMock()
    proxy._http.request = AsyncMock(side_effect=[_response(first), _response(resumed)])

    request_body = {
        "model": "model",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "return_token_ids": True,
        "logprobs": True,
    }
    result, status = await proxy._send_with_abort_resume(
        method="POST",
        url="http://worker/v1/chat/completions",
        content=json.dumps(request_body).encode(),
        headers={},
        request_body=request_body,
        worker_url="http://worker/v1",
        chat_response=True,
    )

    assert status == 200
    assert result["choices"][0]["message"]["content"] == "hello"
    assert result["choices"][0]["token_ids"] == [10, 11]
    assert result["choices"][0]["finish_reason"] == "stop"
    assert result["prompt_token_ids"] == [1, 2]

    resume_request = json.loads(proxy._http.request.await_args_list[1].kwargs["content"])
    assert resume_request["prompt"] == [1, 2, 10]
    assert resume_request["max_tokens"] == 4
    assert "messages" not in resume_request


@pytest.mark.asyncio
async def test_repeated_aborts_stop_after_three_resumes(monkeypatch):
    """A worker that keeps aborting must not loop forever, even with no progress."""
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())
    aborted_chat = {
        "prompt_token_ids": [1, 2],
        "choices": [{"message": {"role": "assistant", "content": "hel"}, "token_ids": [10], "finish_reason": "abort"}],
    }
    aborted_completion = {
        "choices": [{"text": "", "token_ids": [], "finish_reason": "abort"}],
    }

    proxy = ReverseProxy(
        router=AsyncMock(),
        store=AsyncMock(),
        cumulative_token_mode=True,
        max_retries=2,
    )
    proxy._http = AsyncMock()
    proxy._http.request = AsyncMock(side_effect=[_response(aborted_chat)] + [_response(aborted_completion)] * 5)

    request_body = {"model": "model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
    result, status = await proxy._send_with_abort_resume(
        method="POST",
        url="http://worker/v1/chat/completions",
        content=json.dumps(request_body).encode(),
        headers={},
        request_body=request_body,
        worker_url="http://worker/v1",
        chat_response=True,
    )

    assert status == 200
    assert proxy._http.request.await_count == 4  # original + 3 resumes
    assert result["choices"][0]["finish_reason"] == "abort"
    assert result["choices"][0]["token_ids"] == [10]
