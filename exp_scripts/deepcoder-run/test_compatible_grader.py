import concurrent.futures
import json
import multiprocessing as mp
import pickle
import unittest
import os
import cloudpickle

from filelock import _api as lock_api
from rllm.rewards import code_reward
from rllm.types import Episode

from compatible_grader import ForkserverEvaluator, install_grader_context
from deepcoder_eval import deepcoder_evaluator


STDIN = [
    {"input": "2 3\n", "output": "5\n", "testtype": "stdin_stdout"},
    {"input": "-4 7\n", "output": "3\n", "testtype": "stdin_stdout"},
]
FUNCTION = [{"input": "2\n3", "output": "5", "testtype": "functional",
             "metadata": {"func_name": "add"}}]
CASES = [
    (STDIN, "```python\na,b=map(int,input().split());print(a+b)\n```", 1),
    (STDIN, "```python\nprint(0)\n```", 0),
    (STDIN, "```python\nraise ValueError('bad')\n```", 0),
    (STDIN, "```python\ndef broken(\n```", 0),
    (STDIN, "no fenced solution", 0),
    (STDIN, "```python\nprint(0)\n```\n```python\na,b=map(int,input().split());print(a+b)\n```", 1),
    (FUNCTION, "```python\nclass Solution:\n def add(self,a,b): return a+b\n```", 1),
    (FUNCTION, "```python\nclass Solution:\n def add(self,a,b): return a-b\n```", 0),
]


def score(case, evaluator):
    tests, answer, expected = case
    task = {"data_source": "livecodebench", "ground_truth": json.dumps(tests)}
    result = evaluator.evaluate(task, Episode(artifacts={"answer": answer}))
    assert result.reward == expected
    assert result.is_correct == bool(expected)
    return result.reward, result.is_correct


def test_sequential_rewards_match_original_fork():
    # Baseline in this single-threaded test process, before compatibility install.
    assert mp.get_context("fork").get_start_method() == "fork"
    original_mp, original_manager = code_reward.multiprocessing, code_reward.Manager
    try:
        code_reward.multiprocessing = mp.get_context("fork")
        code_reward.Manager = code_reward.multiprocessing.Manager
        baseline = [score(case, deepcoder_evaluator) for case in CASES]
    finally:
        code_reward.multiprocessing, code_reward.Manager = original_mp, original_manager
    adapter = ForkserverEvaluator()
    assert [score(case, adapter) for case in CASES] == baseline
    # Do not change the backend/global multiprocessing start-method policy.
    assert mp.get_start_method() == "fork"


def test_deterministic_filelock_transition_and_parallel_scoring():
    adapter = ForkserverEvaluator()
    # Reproduce the guard deterministically: fork is refused, warmed forkserver
    # is not. Use filelock's private test hook ONLY in this regression test.
    with lock_api._fork_transition():
        with unittest.TestCase().assertRaisesRegex(RuntimeError, "unsafe while filelock"):
            mp.get_context("fork").Manager()
        assert score(CASES[0], adapter) == (1, True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda c: score(c, adapter), CASES * 4))
    assert len(results) == 32


def test_pickle_reinitializes_worker_and_context_is_idempotent():
    adapter = ForkserverEvaluator()
    restored = pickle.loads(pickle.dumps(adapter))
    install_grader_context()
    assert code_reward.multiprocessing.get_start_method() == "forkserver"
    assert score(CASES[0], restored) == (1, True)


def worker_probe(serialized, connection):
    try:
        adapter = cloudpickle.loads(serialized)
        connection.send((os.getpid(), mp.get_start_method(),
                         code_reward.multiprocessing.get_start_method(),
                         score(CASES[0], adapter)))
    finally:
        connection.close()


def test_fresh_process_cloudpickle_worker():
    adapter = ForkserverEvaluator()
    context = mp.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=worker_probe, args=(cloudpickle.dumps(adapter), send))
    process.start()
    send.close()
    try:
        assert receive.poll(40), "worker failed to initialize its own grading context"
        pid, global_method, grader_method, reward = receive.recv()
        assert pid != os.getpid()
        assert global_method == "spawn" # compatibility does not override worker policy
        assert grader_method == "forkserver"
        assert reward == (1, True)
        process.join(10)
        assert process.exitcode == 0
    finally:
        receive.close()
        if process.is_alive():
            process.kill()
            process.join()


def test_timeout_remains_failure():
    install_grader_context()
    success, metadata = code_reward.lcb_check_correctness_v2(
        STDIN[:1], "while True: pass", timeout=1,
    )
    assert success is False


if __name__ == "__main__":
    for test in [test_sequential_rewards_match_original_fork,
                 test_deterministic_filelock_transition_and_parallel_scoring,
                 test_pickle_reinitializes_worker_and_context_is_idempotent,
                 test_fresh_process_cloudpickle_worker,
                 test_timeout_remains_failure]:
        test()
        print(f"PASS {test.__name__}", flush=True)
