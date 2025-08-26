import asyncio
import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from transformers import AutoTokenizer

# Ensure repo root is on sys.path before importing rllm modules
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rllm.tools.web_tools import FirecrawlTool
from rllm.tools.web_tools.gsearch_tool import GoogleSearchTool

from strands import tool

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

    # Optional browser tool (graceful fallback if runtime not installed)
    try:
        browser = LocalChromiumBrowser()
        browser_tool = browser.browser
    except Exception as e:
        logging.warning(f"Browser disabled: {e}")
        browser = None
        browser_tool = None

    # Strands-compliant tools using @tool decorator (Example of how to use rllm tools in SRTANDS)
    @tool(name="firecrawl", description="Fetch page content as markdown + links")
    async def firecrawl(url: str) -> dict:
        try:
            fc = FirecrawlTool()
            res = await fc(url=url, use_async=True)
            if res.error:
                return {"status": "error", "content": [{"text": res.error}]}
            return {"status": "success", "content": [{"json": res.output}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": str(e)}]}

    @tool(name="gsearch", description="Search Google and return top results with snippets")
    def gsearch(query: str) -> dict:
        try:
            gs = GoogleSearchTool()
            r = gs(query=query)
            if r.error:
                return {"status": "error", "content": [{"text": r.error}]}
            return {"status": "success", "content": [{"json": r.output}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": str(e)}]}


    # Real Strands session factory (requires `strands` installed)
    # Single source of truth: define tools once and pass only via workflow args. gsearch is not used in this example.
    tools = [firecrawl, http_request, file_read, calculator, python_repl]
    if browser_tool:
        tools.insert(1, browser_tool)

    session_factory = make_session_factory(
        default_tools=[],
        default_system_prompt="You are a helpful agent that uses tools when beneficial.",
    )

    awe = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "strands_session_factory": session_factory,
            "system_prompt": """
            You are a research agent. Always ground answers in tool-based evidence. Do not guess. Use the registered tools via function calls. Do not simulate tool calls in plain text. Cite sources by returning the URL/title in your final answer.            
            
            Tool selection rules:
            - firecrawl: preferred for fetching page content; returns markdown + links (low tokens).
            - browser: use for web search and interactive flows. For search, navigate to DuckDuckGo with the query (e.g., https://duckduckgo.com/?q=<query>), then read result titles/snippets. Also use for dynamic pages.
            - http_request: prefer only for structured/fast APIs (GitHub/Wikipedia/USGS) or light HTML.
            - file_read: open local/remote files (PDF/CSV/text). For PDFs, extract essential text snippets.
            - calculator: arithmetic/unit conversions; for complex code use python_repl.

            Answer policy:
            - Only output final_answer after at least one Observation supports it (prefer two). Keep answers concise.
            - If the current approach fails twice, change strategy (different search terms/tool) rather than repeating the same navigate/get_text.
""",
            "tools": tools,
            "max_steps": 10,
            "reward_fn": None,
        },
        rollout_engine=rollout,
        config=None,
        n_parallel_tasks=2,
        retry_limit=1,
    )

    tasks = ["Who won the recent Tennis Grand Slam?", "What is the GDP of Shenzhen vs Shanghai most recently?"]
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

    # Prepare the whole episode(s) JSON and save to file (no terminal print)
    episodes_json = [ep.to_dict() for ep in episodes]

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"episodes-{ids[0]}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(episodes_json, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_path}")
    awe.shutdown()
    print("shutdown")

    # Diagnostics: show lingering threads and asyncio tasks; try to close HTTP client
    try:
        import threading
        # Enumerate active threads after shutdown (use non-deprecated daemon flag)
        active_threads = [(t.name, t.daemon, t.ident) for t in threading.enumerate()]
        print("active_threads:", active_threads)
    except Exception:
        pass

    try:
        loop = asyncio.get_running_loop()
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if pending:
            print("pending_asyncio_tasks:", [repr(t) for t in pending])
    except Exception:
        pass

    # Close OpenAI/Together HTTP client if present to release network resources
    try:
        import inspect
        client = getattr(rollout, "client", None)
        if client is not None:
            close_fn = getattr(client, "aclose", None) or getattr(client, "close", None)
            if close_fn is not None:
                if inspect.iscoroutinefunction(close_fn):
                    await close_fn()
                else:
                    result = close_fn()
                    if inspect.iscoroutine(result):
                        await result
            print("rollout_client_closed")
    except Exception as e:
        print("rollout_client_close_error:", e)

    # Close LocalChromiumBrowser explicitly (minimal, use provided helpers)
    try:
        if hasattr(browser, 'browser') and callable(getattr(browser, 'browser')):
            # Use the browser method with close action
            close_action = {"action": {"type": "close", "session_name": "main-session"}}
            browser.browser(close_action)
        elif hasattr(browser, 'quit'):
            browser.quit()
    except Exception as e:
        print(f"Error closing browser: {e}")
    
    # Force exit the program
    os._exit(0)

if __name__ == "__main__":
    asyncio.run(main())