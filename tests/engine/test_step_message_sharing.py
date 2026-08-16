"""Gateway-produced Steps must share message objects end-to-end.

The full integration path the review demanded: compact store -> compact
payload over a JSON wire round trip -> client expansion -> TraceRecords
(model_construct, no validation copies) -> trace_record_to_step. Sharing
must survive every hop; user-constructed Steps keep the defensive copy —
there is deliberately NO public flag to bypass it.
"""

import asyncio
import json
import sys

sys.path.insert(0, "rllm-model-gateway/src")

from rllm_model_gateway.client import _expand_compact_traces
from rllm_model_gateway.models import TraceRecord
from rllm_model_gateway.store.memory_store import MemoryTraceStore

from rllm.engine.trace_converter import trace_record_to_step
from rllm.types import Step


def _payload():
    async def build():
        store = MemoryTraceStore(compact=True)
        base = [{"role": "system", "content": "sys"}]
        for i in range(3):
            base = base + [{"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"}]
            await store.store_trace(f"t{i}", "s", {"trace_id": f"t{i}", "session_id": "s", "messages": list(base[:-1]), "response_message": base[-1]})
        return json.loads(json.dumps(await store.get_session_traces_compact("s"), default=str))

    return asyncio.run(build())


def test_sharing_survives_wire_expansion_records_and_steps():
    records = [TraceRecord.model_construct(**t) for t in _expand_compact_traces(_payload())]
    # cross-record sharing after expansion + model_construct
    assert records[1].messages[0] is records[0].messages[0]
    assert records[2].messages[0] is records[0].messages[0]
    steps = [trace_record_to_step(r) for r in records]
    # cross-STEP sharing: the same system-message dict object in every step
    assert steps[1].chat_completions[0] is steps[0].chat_completions[0]
    assert steps[2].chat_completions[0] is steps[0].chat_completions[0]
    # and steps share with their source records (no copy anywhere on the path)
    assert steps[0].chat_completions[0] is records[0].messages[0]


def test_user_steps_keep_defensive_copy_and_no_bypass_exists():
    msgs = [{"role": "user", "content": "x"}]
    step = Step(chat_completions=list(msgs), metadata={"shared_messages": True})  # flag must be inert
    msgs[0]["content"] = "mutated"
    assert step.chat_completions[0]["content"] == "x"
