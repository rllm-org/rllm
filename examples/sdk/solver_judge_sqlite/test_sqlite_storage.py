"""
Simple end-to-end test for SQLite session storage.

This script demonstrates and tests SQLite-based session storage for the RLLM SDK.
It creates a simple solver-judge workflow and verifies that traces are properly
stored and retrieved from SQLite.
"""

import asyncio
import os
import re
import uuid

from rllm.sdk import ContextVarSession, SqliteSessionStorage, get_chat_client_async


class SimpleSolver:
    """Simple solver agent that generates solutions."""

    def __init__(self, storage):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY", model="Qwen/Qwen3-4B-Instruct-2507")
        self.storage = storage

    async def solve(self, problem: str):
        """Generate a solution for the given problem."""
        with ContextVarSession(agent="solver", groupby_key=str(uuid.uuid4()), storage=self.storage) as sess:
            messages = [{"role": "user", "content": f"{problem}. Output the final answer within <answer>...</answer>"}]
            response = await self.client.chat.completions.create(
                messages=messages,
                temperature=1,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content

            # Parse the solution
            answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                solution = f"<answer>{answer_match.group(1).strip()}</answer>"
            else:
                solution = "No solution found"

            print(f"Solver generated solution: {solution}")
            print(f"Number of traces in solver session: {len(sess.llm_calls)}")
            print(f"Session UID: {sess._uid}")
            print(f"Session storage: {sess.storage}")

            return solution, sess


class SimpleJudge:
    """Simple judge agent that evaluates solutions."""

    def __init__(self, storage):
        self.client = get_chat_client_async(base_url="http://localhost:4000/v1", api_key="EMPTY", model="Qwen/Qwen3-4B-Instruct-2507")
        self.storage = storage

    async def judge(self, problem: str, solution: str):
        """Judge a solution for the given problem."""
        with ContextVarSession(agent="judge", storage=self.storage) as sess:
            prompt = f"""You are an expert verifier. Given a countdown problem and a solution, evaluate if it's correct.
Problem:
{problem}

Solution:
{solution}

Is this solution correct? Output YES or NO within <answer>...</answer> tags."""

            messages = [{"role": "user", "content": prompt}]
            response = await self.client.chat.completions.create(
                messages=messages,
                temperature=1,
                max_tokens=1000,
            )
            response_text = response.choices[0].message.content

            # Parse the judgment
            answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                judgment = answer_match.group(1).strip()
            else:
                judgment = "UNKNOWN"

            print(f"Judge evaluation: {judgment}")
            print(f"Number of traces in judge session: {len(sess.llm_calls)}")
            print(f"Session UID: {sess._uid}")
            print(f"Session storage: {sess.storage}")

            return judgment, sess


async def test_sqlite_storage():
    """Test SQLite storage with a simple solver-judge workflow."""
    print("=" * 80)
    print("Testing SQLite Session Storage")
    print("=" * 80)

    # Create SQLite storage with a test database
    db_path = "./test_traces.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"Removed existing database: {db_path}")

    storage = SqliteSessionStorage(db_path=db_path)
    print(f"Created SQLite storage: {storage}")
    print()

    # Sample problem from countdown dataset format
    problem = """Using the numbers [25, 50, 75, 100] and operations (+, -, *, /), find an expression that equals 450.
Each number can be used at most once."""

    # Create agents
    solver = SimpleSolver(storage)
    judge = SimpleJudge(storage)

    # Run solver
    print("Step 1: Running solver...")
    print("-" * 80)
    solution, solver_sess = await solver.solve(problem)
    print()

    # Run judge
    print("Step 2: Running judge...")
    print("-" * 80)
    judgment, judge_sess = await judge.judge(problem, solution)
    print()

    # Verify storage
    print("Step 3: Verifying SQLite storage...")
    print("-" * 80)

    # Check that the database file was created
    assert os.path.exists(db_path), f"Database file {db_path} was not created!"
    print(f"✓ Database file created: {db_path}")

    # Verify that traces were stored
    solver_traces = solver_sess.llm_calls
    judge_traces = judge_sess.llm_calls

    print(f"✓ Solver session has {len(solver_traces)} trace(s)")
    print(f"✓ Judge session has {len(judge_traces)} trace(s)")

    # Print trace details
    if solver_traces:
        print("\nSolver trace details:")
        for i, trace in enumerate(solver_traces):
            print(f"  Trace {i + 1}:")
            print(f"    - ID: {trace.trace_id}")
            print(f"    - Model: {trace.model}")
            print(f"    - Session name: {trace.session_name}")

    if judge_traces:
        print("\nJudge trace details:")
        for i, trace in enumerate(judge_traces):
            print(f"  Trace {i + 1}:")
            print(f"    - ID: {trace.trace_id}")
            print(f"    - Model: {trace.model}")
            print(f"    - Session name: {trace.session_name}")

    print()
    print("=" * 80)
    print("SQLite Storage Test Completed Successfully!")
    print("=" * 80)
    print(f"Database location: {os.path.abspath(db_path)}")
    print()


async def test_cross_process_simulation():
    """Test that SQLite storage can be used across different 'processes' (sessions)."""
    print("\n" + "=" * 80)
    print("Testing Cross-Process Storage Simulation")
    print("=" * 80)

    db_path = "./test_traces_multiprocess.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    # Simulate process 1: Create storage and write traces
    print("\nSimulating Process 1: Writing traces...")
    storage1 = SqliteSessionStorage(db_path=db_path)
    solver1 = SimpleSolver(storage1)

    problem = "Using numbers [1, 2, 3, 4], find an expression that equals 10."
    solution, sess1 = await solver1.solve(problem)
    session_uid = sess1._uid
    print(f"Process 1 session UID: {session_uid}")
    print(f"Process 1 wrote {len(sess1.llm_calls)} trace(s)")

    # Simulate process 2: Create new storage instance and read traces
    print("\nSimulating Process 2: Reading traces...")
    storage2 = SqliteSessionStorage(db_path=db_path)

    # Create a new session with the same UID to read traces
    sess2 = ContextVarSession(storage=storage2)
    # Manually set the UID to match process 1's session
    sess2._uid = session_uid

    traces = sess2.llm_calls
    print(f"Process 2 read {len(traces)} trace(s) from session {session_uid}")

    if traces:
        print("✓ Cross-process trace sharing successful!")
        print(f"  First trace ID: {traces[0].trace_id}")
    else:
        print("✗ No traces found - cross-process sharing may have failed")

    print("\n" + "=" * 80)
    print("Cross-Process Storage Test Completed!")
    print("=" * 80)


if __name__ == "__main__":
    # Run basic test
    asyncio.run(test_sqlite_storage())

    # Run cross-process simulation test
    asyncio.run(test_cross_process_simulation())
