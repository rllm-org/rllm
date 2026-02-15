"""
Remote Agent Server for the Solver-Judge Workflow.

This server implements the remote agent endpoint protocol. It receives tasks
from the rLLM trainer, runs the SolverJudgeWorkflow using the trainer's
inference API for model calls, and returns Episode objects.

The server uses ``RemoteRolloutEngine`` which calls the trainer's native
``POST /v1/model_response`` endpoint.  This preserves the full ``ModelOutput``
(including prompt_ids, completion_ids, logprobs) needed for RL training --
unlike the OpenAI chat completions format which strips these fields.

Usage:
    # Start the remote agent server (in a separate terminal or container):
    uvicorn examples.remote_solver_judge.remote_agent_server:app --host 0.0.0.0 --port 5100

    # Or run directly:
    python -m examples.remote_solver_judge.remote_agent_server --port 5100
"""

from examples.solver_judge.solver_judge_flow import SolverJudgeWorkflow
from rllm.experimental.remote.remote_agent_app import create_remote_agent_app
from rllm.rewards.countdown_reward import countdown_reward_fn

# Create the remote agent FastAPI app.
# The inference_api_url (pointing to the trainer's model) is provided
# dynamically in each request, so the RemoteRolloutEngine is created on-the-fly.
app = create_remote_agent_app(
    workflow_cls=SolverJudgeWorkflow,
    workflow_args={
        "n_solutions": 2,
        "reward_function": countdown_reward_fn,
    },
    n_parallel=64,
    engine_kwargs={
        # These are passed to RemoteRolloutEngine.
        # The inference_api_url is set dynamically per request.
        "timeout": 300.0,
        "max_retries": 3,
    },
)

if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Remote Solver-Judge Agent Server")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5100)
    args = parser.parse_args()

    print(f"Starting remote agent server at http://{args.host}:{args.port}")
    print("Endpoints:")
    print(f"  POST http://{args.host}:{args.port}/generate_episode")
    print(f"  GET  http://{args.host}:{args.port}/health")
    uvicorn.run(app, host=args.host, port=args.port)
