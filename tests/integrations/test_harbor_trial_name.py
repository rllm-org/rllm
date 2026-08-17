"""Harbor trial names must satisfy Modal's sandbox naming rule.

rLLM session ids are ``<task_id>:<attempt>``; Modal rejects the colon, so every
trial failed at sandbox creation. Harbor's Docker backend sanitizes internally
and hid the problem.
"""

import re

from rllm.integrations.harbor.runtime import _sanitize_trial_name

# Modal's rule, quoted from its InvalidError message.
MODAL_ALLOWED = re.compile(r"^[a-zA-Z0-9._-]+$")


def test_reported_failure_is_fixed():
    sid = "django__django-14534:0"  # from a swebench-verified run that failed
    assert not MODAL_ALLOWED.match(sid)
    assert _sanitize_trial_name(sid) == "django__django-14534-0"


def test_attempts_stay_distinct():
    """Retries must not collide onto one sandbox name."""
    assert len({_sanitize_trial_name(f"t:{i}") for i in range(3)}) == 3


def test_safe_names_are_unchanged():
    for name in ("swebench-verified", "task_1.2-3"):
        assert _sanitize_trial_name(name) == name
