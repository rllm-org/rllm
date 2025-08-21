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

# Silence OTEL SDK - no effect
import os, logging

os.environ.setdefault("OTEL_SDK_DISABLED", "true")   # disable OTEL SDK
os.environ.setdefault("OTEL_TRACES_EXPORTER", "none")
os.environ.setdefault("OTEL_METRICS_EXPORTER", "none")
os.environ.setdefault("OTEL_LOGS_EXPORTER", "none")
os.environ.setdefault("OTEL_PYTHON_LOG_LEVEL", "ERROR")

logging.getLogger("opentelemetry").setLevel(logging.ERROR)


from rllm.engine.rollout_engine import RolloutEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.workflows.strands_workflow import StrandsWorkflow
from rllm.integrations.strands_session import make_session_factory
from strands_tools import http_request, file_read, calculator, python_repl
from strands_tools.browser import LocalChromiumBrowser


async def main() -> None:
    load_dotenv(find_dotenv())

    tokenizer_model = os.getenv("TOKENIZER_MODEL", None)
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
            "system_prompt": """
            You are a research agent. Always ground answers in tool-based evidence. Do not guess. Use the registered tools via function calls. Do not simulate tool calls in plain text. Cite sources by returning the URL/title in your final answer.            
            
            Tool selection rules:
            - browser: general navigation and reading.
            - First action: init_session with a kebab-case session_name (e.g., "research-1"); reuse it.
            - After navigate: usually read with get_text  (e.g., selectors "body", "main", "h1,h2,h3", "p"), if you need to interact, use click items or type things. Avoid looping the same read on a page.
            - If you are stucked by Captcha, use https://duckduckgo.com to do the search instead.
            - http_request: prefer for structured/fast sources (GitHub/Wikipedia/USGS APIs or light HTML).
            - file_read: open local/remote files (PDF/CSV/text). For PDFs, extract essential text snippets.
            - calculator: arithmetic/unit conversions; for complex code use python_repl.

            Answer policy:
            - Only output final_answer after at least one Observation supports it (prefer two). Keep answers concise.
            - If the current approach fails twice, change strategy (different query/tool) rather than repeating the same navigate/get_text.
""",
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


