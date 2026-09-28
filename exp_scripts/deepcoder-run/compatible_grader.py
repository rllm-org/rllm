"""Process-start compatibility only; delegate every score to upstream evaluator.

The change is scoped to code_reward's multiprocessing references, not Python's
global start method, Ray, vLLM, reward rules, timeouts, or test selection.
Warm the forkserver during evaluator construction/deserialization, BEFORE the
gateway/executor starts concurrent filelock transitions. No fallback to fork.
"""
import multiprocessing as mp
import os
import threading

from deepcoder_eval import deepcoder_evaluator
from rllm.rewards import code_reward

_LOCK = threading.Lock()
_PID = None
_CONTEXT = None


def install_grader_context():
    global _PID, _CONTEXT
    with _LOCK:
        if _PID == os.getpid():
            assert code_reward.multiprocessing is _CONTEXT
            return
        if "forkserver" not in mp.get_all_start_methods():
            raise RuntimeError("DeepCoder grading requires forkserver on this host")
        context = mp.get_context("forkserver")
        # Avoid importing the training launcher/data/model in the server. The
        # server imports only the CPU grader; children share its imports via COW.
        # grader_preload caps RLIMIT_AS in the forkserver, inherited by every grading child.
        context.set_forkserver_preload(["rllm.rewards.code_reward", "grader_preload"])
        manager = context.Manager()
        try:
            assert list(manager.list([1])) == [1]
        finally:
            manager.shutdown()
        code_reward.multiprocessing = context
        code_reward.Manager = context.Manager
        _CONTEXT = context
        _PID = os.getpid()
        print(f"[grader-process] pid={_PID} start_method=forkserver warmed=True", flush=True)


class GraderTimeout(RuntimeError):
    """Raised (training only) when the verifier hit a wall-clock timeout.

    On .24 the CPU grader stalls under concurrency and 3-6% of rollouts are
    judged "Time Limit Exceeded" for reasons unrelated to the submitted code.
    Raising turns the episode into TerminationReason.ERROR, which
    rllm.compact_filtering (mask_error=True) drops from the GRPO group instead
    of scoring it 0. Genuine slow-code timeouts are dropped too; on .16 that
    background rate is 0.06%, so the lost negative signal is negligible.
    """


def _timed_out(metadata) -> bool:
    """True if the verifier ended by a wall-clock timeout, in any of its shapes.

    livecodebench.run_test reports timeouts three ways:
      * per-test alarm inside the graded call  -> error_code -3, "Time Limit Exceeded"
      * child never reported (parent join cap) -> error "global timeout"
      * alarm firing outside the per-test try (compile/compare) is swallowed by
        run_test's outer `except Exception` -> error_code -4,
        "Error during testing: " (TimeoutException has an empty str())
    """
    if not isinstance(metadata, dict):
        return False
    if metadata.get("error") == "global timeout":
        return True
    for t in metadata.get("test_results") or []:
        if not isinstance(t, dict):
            continue
        err = str(t.get("error") or "")
        msg = str(t.get("error_message") or "")
        if t.get("error") == "global timeout" or t.get("error_code") == -3:
            return True
        if "Time Limit Exceeded" in msg or "timeout" in err.lower() or "timeout" in msg.lower():
            return True
        if msg.strip() == "Error during testing:" or (msg.startswith("Error during testing") and not msg.split(":", 1)[-1].strip()):
            return True
    return False


class ForkserverEvaluator:
    """Ray transports this adapter; the worker warms its own server on load.

    drop_timeouts=False keeps the upstream behaviour exactly (a timeout is a
    scored failure). It must stay False for validation and standalone eval.
    """

    def __init__(self, drop_timeouts: bool = False):
        self.drop_timeouts = bool(drop_timeouts)
        install_grader_context()

    def __getstate__(self):
        return {"drop_timeouts": self.drop_timeouts}

    def __setstate__(self, state):
        self.drop_timeouts = bool((state or {}).get("drop_timeouts", False))
        install_grader_context()

    def evaluate(self, task, episode):
        install_grader_context()
        out = deepcoder_evaluator.evaluate(task, episode)
        md = getattr(out, "metadata", None)
        if self.drop_timeouts and not getattr(out, "is_correct", False):
            # One-line signature of every non-"Wrong Answer" failure so the training
            # log itself shows which verifier error shapes occur (audit for the drop rule).
            try:
                tests = (md or {}).get("test_results") or []
                bad = [t for t in tests if isinstance(t, dict) and not t.get("passed", True)]
                t0 = bad[0] if bad else {}
                code = t0.get("error_code"); msg = str(t0.get("error_message") or "")[:60]; err = str(t0.get("error") or "")[:40]
                if code != -2 and (code is not None or err or (md or {}).get("error")):
                    print(f"[grader-sig] code={code} err={err!r} msg={msg!r} top_err={str((md or {}).get('error'))[:40]!r} detected={_timed_out(md)}", flush=True)
            except Exception:
                pass
        if self.drop_timeouts and _timed_out(md):
            print("[grader-drop] verifier timeout -> episode marked ERROR and excluded from training", flush=True)
            raise GraderTimeout("verifier timeout; rollout excluded from the GRPO group")
        return out

    __call__ = evaluate
