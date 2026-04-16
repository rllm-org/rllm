"""Integration tests for AgentCoreRuntime against a live ACR deployment.

Skipped unless all env vars are set: AGENTCORE_AGENT_ARN, AGENTCORE_S3_BUCKET,
AGENTCORE_BASE_URL, and AGENTCORE_MODEL_ID.
"""

import uuid

import pytest

from rllm.experimental.engine.remote_runtime.agentcore_runtime import AgentCoreRuntime
from rllm.experimental.engine.remote_runtime.protocol import (
    RemoteRuntimeConfig,
    TaskSubmission,
)

from .conftest import AGENT_ARN, BASE_URL, MODEL_ID, S3_BUCKET, requires_agentcore


def _make_runtime() -> AgentCoreRuntime:
    config = RemoteRuntimeConfig(
        enabled=True,
        backend="agentcore",
        agentcore={
            "agent_runtime_arn": AGENT_ARN,
            "s3_bucket": S3_BUCKET,
        },
    )
    runtime = AgentCoreRuntime(config, exp_id=f"integ-{uuid.uuid4().hex[:8]}", model_id=MODEL_ID)
    runtime.initialize()
    return runtime


def _make_submission(prompt: str, answer: str) -> TaskSubmission:
    return TaskSubmission(
        task={"prompt": prompt, "answer": answer},
        session_id=str(uuid.uuid4()),
        task_id=str(uuid.uuid4()),
        inference_url=BASE_URL,
    )


@requires_agentcore
class TestSingleTask:
    @pytest.mark.asyncio
    async def test_single_task(self):
        """Submit a GSM8K problem, verify success=True, reward is not None."""
        runtime = _make_runtime()
        sub = _make_submission(
            prompt=(
                "Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 dozen mini cupcakes which cost $80 per dozen, and 6 dozen mini cheesecakes for $55 per dozen. How much was the total cost?"
            ),
            answer="694",
        )

        results = await runtime.execute_tasks([sub], timeout=300.0)

        assert len(results) == 1
        result = results[0]
        assert result.success is True
        assert result.reward is not None
        assert result.elapsed > 0
        runtime.shutdown()


@requires_agentcore
class TestBatchTasks:
    @pytest.mark.asyncio
    async def test_batch_tasks(self):
        """Submit 3 tasks, verify all results returned with correct session_ids."""
        runtime = _make_runtime()
        problems = [
            ("What is 2 + 2?", "4"),
            ("What is 10 * 5?", "50"),
            ("What is 100 / 4?", "25"),
        ]
        subs = [_make_submission(p, a) for p, a in problems]
        expected_session_ids = {s.session_id for s in subs}

        results = await runtime.execute_tasks(subs, timeout=300.0)

        assert len(results) == 3
        returned_session_ids = {r.session_id for r in results}
        assert returned_session_ids == expected_session_ids
        runtime.shutdown()


@requires_agentcore
class TestTimeoutHandling:
    @pytest.mark.asyncio
    async def test_timeout(self):
        """Timeout of 0.01s should fail — not enough time for ACR round-trip."""
        runtime = _make_runtime()
        sub = _make_submission("What is 1 + 1?", "2")

        results = await runtime.execute_tasks([sub], timeout=0.01)

        assert len(results) == 1
        assert results[0].success is False
        runtime.shutdown()
