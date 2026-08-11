"""Live check that sampling params litellm doesn't recognise still reach the provider.

The unit tests in ``tests/eval/test_eval_proxy.py`` only assert the generated
config carries ``extra_body``. They would keep passing if litellm stopped
forwarding it — which is the exact silent-drop this passthrough exists to
prevent, so the guarantee needs one test that talks to a real provider.

Requires FIREWORKS_API_KEY.
"""

import json
import os
import urllib.request

import pytest

from rllm.eval.proxy import EvalProxyManager

FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY")

requires_fireworks = pytest.mark.skipif(
    not FIREWORKS_API_KEY,
    reason="FIREWORKS_API_KEY env var required",
)

MODEL = "accounts/fireworks/models/deepseek-v4-flash-0731"
# Hard enough that effort actually changes how much the model thinks; a trivial
# prompt reasons about the same amount at every level and the assert goes flaky.
PROMPT = "Prove that for every positive integer n, 2^(2n)+1 has no prime factor congruent to 3 mod 4. Show full reasoning."


def _reasoning_chars(effort: str) -> int:
    """Round-trip one completion through a proxy configured for *effort*."""
    pm = EvalProxyManager(provider="fireworks", model_name=MODEL, api_key=FIREWORKS_API_KEY, sampling_extra={"reasoning_effort": effort})
    pm.start_proxy_subprocess(pm.build_proxy_config())
    try:
        url = pm.get_proxy_url().rstrip("/") + "/chat/completions"
        body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 6000}).encode()
        req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json", "Authorization": "Bearer sk-placeholder"})
        with urllib.request.urlopen(req, timeout=300) as response:
            payload = json.loads(response.read())
        return len(payload["choices"][0]["message"].get("reasoning_content") or "")
    finally:
        pm.shutdown_proxy()


@requires_fireworks
def test_reasoning_effort_reaches_the_provider():
    """``max`` must visibly out-think ``low``.

    Fireworks honours ``reasoning_effort`` but litellm's fireworks_ai allowlist
    has no entry for it, so without the ``extra_body`` passthrough both calls
    silently run at the model's default and come back the same size. Observed
    ~3.9x; asserting 2x leaves room for sampling variance.
    """
    low = _reasoning_chars("low")
    high = _reasoning_chars("max")

    assert high > 2 * low, f"reasoning_effort looks dropped: low={low} chars, max={high} chars"
