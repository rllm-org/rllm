import asyncio
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rllm.engine.rollout_engine import RolloutEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.workflows.strands_workflow import StrandsWorkflow
from rllm.integrations.strands_session import make_session_factory
from strands_tools import http_request, file_read, calculator, python_repl
from strands_tools.browser import LocalChromiumBrowser


async def main() -> None:
    load_dotenv(find_dotenv())

    tokenizer_model = os.getenv("TOKENIZER_MODEL", "Qwen/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    together_api_key = os.getenv("TOGETHER_AI_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if together_api_key:
        openai_kwargs = {"api_key": together_api_key, "base_url": "https://api.together.xyz/v1"}
        model_id = os.getenv("TOGETHER_AI_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct-Turbo")
    elif openai_api_key:
        openai_kwargs = {"api_key": openai_api_key}
        if os.getenv("OPENAI_BASE_URL"):
            openai_kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
        model_id = os.getenv("MODEL_NAME", "gpt-4o")
    else:
        raise ValueError("API key required")

    rollout = RolloutEngine(
        engine_name="openai",
        tokenizer=tokenizer,
        openai_kwargs=openai_kwargs,
        sampling_params={"model": model_id},
    )

    browser = LocalChromiumBrowser()


    # Real Strands session factory (requires `strands` installed)
    session_factory = make_session_factory(
        default_tools=[browser.browser, http_request, file_read, calculator, python_repl],
        default_system_prompt="You are a helpful agent that uses tools when beneficial.",
    )

    awe = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "strands_session_factory": session_factory,
            "system_prompt": "You are a helpful agent that uses tools when beneficial. Use the tools to answer the question.",
            "tools": [browser.browser, http_request, file_read, calculator, python_repl],
            "max_steps": 30,
            "reward_fn": None,
        },
        rollout_engine=rollout,
        config=None,
        n_parallel_tasks=2,
        retry_limit=1,
    )

    tasks = ["加州最古老的扑克室是什么？"]
    ids = ["ep-0"]
    episodes = await awe.execute_tasks(tasks, task_ids=ids, workflow_id="strands-example")
    print("done:", [ep.id for ep in episodes])
    # Print last assistant message if present
    for ep in episodes:
        for name, traj in ep.trajectories:
            if traj.steps and traj.steps[-1].chat_completions:
                msgs = traj.steps[-1].chat_completions
                last = msgs[-1]
                if last.get("role") == "assistant":
                    print("assistant:", last.get("content"))

    # Print the whole episode(s) JSON and save to file
    episodes_json = [ep.to_dict() for ep in episodes]
    print(json.dumps(episodes_json, ensure_ascii=False, indent=2))

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episodes-{ids[0]}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_json, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())


