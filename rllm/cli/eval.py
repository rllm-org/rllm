"""Eval CLI command.

``rllm eval <benchmark> --agent <name> [--evaluator <name>] [--base-url <url>] [--model <name>]``

When ``--base-url`` is omitted, a LiteLLM proxy is auto-started using the
configuration from ``rllm setup`` (stored in ``~/.rllm/config.json``).
"""

from __future__ import annotations

import asyncio
import logging
import os

import click
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.theme import Theme

from rllm.cli._pull import load_dataset_catalog, pull_dataset

logger = logging.getLogger(__name__)

theme = Theme({"label": "dim", "success": "bold green", "error": "bold red", "val": "bold", "key": "yellow"})
console = Console(theme=theme)


def _suggest_benchmarks(name: str, catalog_names: list[str], max_suggestions: int = 3) -> list[str]:
    """Return catalog names similar to *name*, ordered by edit distance."""
    from difflib import get_close_matches

    return get_close_matches(name, catalog_names, n=max_suggestions, cutoff=0.5)


def _dict_rows_to_tasks(rows: list[dict]) -> list:
    """Wrap dict-rows from a catalog dataset as Task objects.

    Harbor rows carry ``task_path`` pointing at their own Harbor task
    directory; we root the Task there so HarborRuntime and per-task
    verifier resolution find ``task.toml`` / ``tests/`` natively. Other
    rows get ``dataset_dir=Path(".")`` and rely on ``evaluator_override``.
    """
    from pathlib import Path

    from rllm.types import Task

    tasks: list[Task] = []
    for idx, row in enumerate(rows):
        instruction = row.get("instruction") or row.get("question") or ""
        task_id = str(row.get("id") or row.get("task_id") or idx)
        task_path = row.get("task_path")
        dataset_dir = Path(task_path) if task_path else Path(".")
        tasks.append(
            Task(
                id=task_id,
                instruction=str(instruction),
                metadata=dict(row),
                dataset_dir=dataset_dir,
                sub_dir=None,
            )
        )
    return tasks


def _run_eval(
    benchmark: str,
    agent_name: str,
    evaluator_name: str | None,
    base_url: str,
    model: str,
    split: str,
    concurrency: int,
    max_examples: int | None,
    task_indices: list[int] | None,
    output_path: str | None,
    agent_metadata: dict | None = None,
    enable_ui: bool = False,
    save_episodes: bool = True,
    episodes_dir: str | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int | None = None,
):
    """Core eval logic, extracted for clean proxy lifecycle management."""
    from rllm.data import DatasetRegistry
    from rllm.eval.agent_loader import load_agent
    from rllm.eval.evaluator_loader import load_evaluator, resolve_evaluator_from_catalog

    # ------------------------------------------------------------------
    # Local benchmark path: directory with dataset.toml / task.toml
    # ------------------------------------------------------------------
    from rllm.tasks.loader import BenchmarkLoader

    _is_local = BenchmarkLoader.is_local_benchmark(benchmark)

    # If `benchmark` is a bare name (not a path) and a materialised dir
    # exists under ~/.rllm/datasets/<name>/, transparently use that
    # whenever --agent is set to anything other than a harbor scaffold
    # (harbor catalog datasets carry their verifier inside each task dir,
    # so they go through the catalog branch's row-wrapping path instead).
    _materialised_path_override = None
    if not _is_local and not benchmark.startswith(("./", "../", "/", "~", "harbor:")):
        _materialised = os.path.expanduser(os.path.join(os.environ.get("RLLM_HOME", "~/.rllm"), "datasets", benchmark))
        if os.path.isfile(os.path.join(_materialised, "dataset.toml")):
            _can_redirect = bool(agent_name) and not agent_name.startswith("harbor:")
            if _can_redirect and BenchmarkLoader.is_local_benchmark(_materialised):
                console.print(f"  [dim]Using materialised dataset at {_materialised}[/]")
                _materialised_path_override = _materialised
                _is_local = True

    # Helpful error when user clearly intended a path but it doesn't resolve.
    if not _is_local and (benchmark.startswith("./") or benchmark.startswith("../") or benchmark.startswith("/") or benchmark.startswith("~")):
        import os as _os

        resolved = _os.path.expanduser(benchmark)
        console.print(f"  [error]Path-like benchmark '{benchmark}' did not resolve to a benchmark directory.[/]")
        console.print(f"  [dim]Resolved to: {_os.path.abspath(resolved)}[/]")
        console.print(f"  [dim]Exists: {_os.path.exists(resolved)}, is dir: {_os.path.isdir(resolved) if _os.path.exists(resolved) else 'N/A'}[/]")
        console.print("  [dim]A benchmark directory must contain dataset.toml, task.toml, or sub-dirs with task.toml.[/]")
        raise SystemExit(1)

    if _is_local:
        sandbox_backend = (agent_metadata or {}).get("sandbox_backend")
        # For local benchmarks, --agent picks the AgentFlow ("react",
        # "claude-code", ...) or a module:Class import path.
        _load_path = _materialised_path_override or benchmark
        bench_result = BenchmarkLoader.load(
            _load_path,
            sandbox_backend=sandbox_backend,
            harness_name=agent_name,
        )
        catalog_entry = {
            "description": bench_result.description,
            "category": bench_result.category,
        }

        if split is None:
            split = bench_result.split or "test"

        # Resolve agent name (display + harness lookup)
        if agent_name is None:
            agent_name = bench_result.harness_name or "react"

        # Construct the AgentFlow: catalog covers built-in harnesses
        # (react/bash/claude-code) and any user-registered or plugin agents.
        try:
            agent = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Cannot load agent '{agent_name}': {e}[/]")
            raise SystemExit(1) from None

        # Apply CLI sandbox overrides
        if agent_metadata:
            from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

            if isinstance(agent, SandboxedAgentFlow):
                if "sandbox_backend" in agent_metadata:
                    agent.sandbox_backend = agent_metadata["sandbox_backend"]
                if "sandbox_concurrency" in agent_metadata:
                    agent.max_concurrent = agent_metadata["sandbox_concurrency"]

        # Evaluator: comes from each Task's [verifier] config by default.
        # CLI --evaluator (when provided) overrides for every task.
        evaluator = None
        evaluator_display = f"per-task ({len(bench_result.tasks)} tasks)"
        if evaluator_name is not None:
            try:
                evaluator = load_evaluator(evaluator_name)
                evaluator_display = f"{evaluator_name} (overrides per-task verifier)"
            except (KeyError, ImportError, AttributeError, TypeError) as e:
                console.print(f"  [error]Error loading evaluator '{evaluator_name}': {e}[/]")
                raise SystemExit(1) from None

        # Wrap tasks in a Dataset so the existing CLI filter code (select, len) works
        from rllm.data.dataset import Dataset

        dataset = Dataset(data=list(bench_result.tasks), name=bench_result.name, split=bench_result.split)

    # ------------------------------------------------------------------
    # Catalog / Harbor path (existing behavior)
    # ------------------------------------------------------------------
    else:
        # Load catalog for defaults
        catalog = load_dataset_catalog()
        all_datasets = catalog.get("datasets", {})
        catalog_entry = all_datasets.get(benchmark)

        # Explicit Harbor prefix: "harbor:<name>" resolves from Harbor registry
        if catalog_entry is None and benchmark.startswith("harbor:"):
            from rllm.cli._pull import resolve_harbor_catalog_entry

            harbor_name = benchmark.removeprefix("harbor:")
            with Status(f"[dim]Looking up '{harbor_name}' in Harbor registry...[/]", console=console):
                catalog_entry = resolve_harbor_catalog_entry(harbor_name)
            if catalog_entry:
                console.print(f"  [success]Found Harbor dataset:[/] [val]{harbor_name}[/]")
                benchmark = harbor_name  # Use the clean name for display and registry

        # Resolve agent
        if agent_name is None:
            if catalog_entry and "default_agent" in catalog_entry:
                agent_name = catalog_entry["default_agent"]
            elif not catalog_entry:
                msg = f"  [error]Benchmark '{benchmark}' not found.[/]"
                suggestions = _suggest_benchmarks(benchmark, list(all_datasets.keys()))
                if suggestions:
                    msg += f"\n\n  Did you mean: [bold]{', '.join(suggestions)}[/]?"
                msg += "\n\n  Run [bold]rllm dataset list[/] to see available benchmarks."
                msg += "\n  Use [bold]harbor:[/] prefix for Harbor datasets (e.g., [bold]rllm eval harbor:swebench-verified[/])."
                console.print(msg)
                raise SystemExit(1)
            else:
                console.print(f"  [error]No --agent specified and no default_agent in catalog for '{benchmark}'.[/]")
                raise SystemExit(1)

        # Resolve split
        if split is None:
            if catalog_entry:
                split = catalog_entry.get("eval_split", "test")
            else:
                split = "test"

        # Docker check for Harbor tasks
        if (agent_name and agent_name.startswith("harbor:")) or (catalog_entry and catalog_entry.get("source", "").startswith("harbor:")):
            from rllm.integrations.harbor.utils import diagnose_docker

            ok, reason, hint = diagnose_docker()
            if not ok:
                console.print(f"  [error]Harbor tasks require Docker — {reason}.[/]")
                if hint:
                    console.print(f"  [dim]{hint}[/]")
                raise SystemExit(1)

        _is_harbor_agent = bool(agent_name) and agent_name.startswith("harbor:")

        # Load the agent. Catalog covers built-in flows (react/bash/claude-code)
        # plus user-registered + plugin agents; ``harbor:<scaffold>`` resolves
        # to a HarborRuntime via the harbor: prefix branch in load_agent.
        try:
            agent = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Error loading agent '{agent_name}': {e}[/]")
            raise SystemExit(1) from None

        # Apply sandbox CLI overrides
        if agent_metadata:
            from rllm.integrations.harbor.runtime import HarborRuntime
            from rllm.sandbox.sandboxed_flow import SandboxedAgentFlow

            if isinstance(agent, HarborRuntime):
                if "sandbox_backend" in agent_metadata:
                    agent.environment_type = agent_metadata["sandbox_backend"]
            elif isinstance(agent, SandboxedAgentFlow):
                if "sandbox_backend" in agent_metadata:
                    agent.sandbox_backend = agent_metadata["sandbox_backend"]
                if "sandbox_concurrency" in agent_metadata:
                    agent.max_concurrent = agent_metadata["sandbox_concurrency"]

        # Resolve evaluator: explicit --evaluator > catalog auto-resolve >
        # catalog reward_fn > None. When ``evaluator`` is None ``EvalHooks``
        # falls back to per-task verifier resolution (works only when each
        # Task has its verifier config — i.e. materialised dir or harbor
        # task.toml).
        evaluator = None
        evaluator_display = "per-task (from dataset.toml)"
        if evaluator_name is not None:
            try:
                evaluator = load_evaluator(evaluator_name)
                evaluator_display = f"{evaluator_name} (overrides per-task verifier)"
            except (KeyError, ImportError, AttributeError, TypeError) as e:
                console.print(f"  [error]Error loading evaluator '{evaluator_name}': {e}[/]")
                raise SystemExit(1) from None
        else:
            # ``harbor_reward_fn`` reads ``episode.artifacts['harbor_reward']``,
            # which is only populated by HarborRuntime. When the user runs a
            # harbor-sourced dataset through an rllm-native harness (oracle,
            # opencode, mini-swe-agent, …), the artifact never gets set and
            # the eval would always score zero. Skip the catalog evaluator in
            # that case and let ``EvalHooks`` resolve a per-task verifier
            # (typically ``tests/test.sh`` inside the harbor task dir).
            _harbor_reward_fn_skipped = catalog_entry and catalog_entry.get("reward_fn") == "harbor_reward_fn" and not _is_harbor_agent
            if _harbor_reward_fn_skipped:
                evaluator_display = "per-task (rllm runtime on harbor task)"
            else:
                evaluator = resolve_evaluator_from_catalog(benchmark)
                if evaluator is not None:
                    reward_fn_name = catalog_entry.get("reward_fn", "") if catalog_entry else ""
                    evaluator_display = reward_fn_name or type(evaluator).__name__
                elif catalog_entry and catalog_entry.get("reward_fn"):
                    try:
                        evaluator = load_evaluator(catalog_entry["reward_fn"])
                        evaluator_display = catalog_entry["reward_fn"]
                    except (KeyError, ImportError):
                        pass

        # Load dataset — auto-pull if not available locally
        dataset = DatasetRegistry.load_dataset(benchmark, split)
        if dataset is None and catalog_entry:
            with Status(f"[dim]Pulling {benchmark} from {catalog_entry['source']}...[/]", console=console):
                pull_dataset(benchmark, catalog_entry)
            dataset = DatasetRegistry.load_dataset(benchmark, split)

        if dataset is None:
            console.print(f"  [error]Could not load dataset '{benchmark}' split '{split}'.[/]")
            raise SystemExit(1)

        # Prefer a materialised local dir for non-harbor catalog datasets so
        # each Task carries its per-task verifier from dataset.toml. Harbor
        # datasets carry their verifier inside the harbor task dir (read via
        # task.metadata["task_path"]), so we wrap rows directly.
        bench_result = None
        if not _is_harbor_agent:
            _materialised = os.path.expanduser(os.path.join(os.environ.get("RLLM_HOME", "~/.rllm"), "datasets", benchmark))
            if not os.path.isfile(os.path.join(_materialised, "dataset.toml")):
                try:
                    from rllm.eval.materialize import materialize_benchmark as _materialize_benchmark

                    _materialize_benchmark(name=benchmark, split=split, rows=list(dataset.data), catalog_entry=catalog_entry or {})
                    console.print(f"  [dim]Materialised existing dataset on the fly at {_materialised}[/]")
                except Exception as e:
                    logger.warning("Could not materialise %s on the fly: %s", benchmark, e)

            if os.path.isfile(os.path.join(_materialised, "dataset.toml")) and BenchmarkLoader.is_local_benchmark(_materialised):
                console.print(f"  [dim]Using materialised dataset at {_materialised}[/]")
                bench_result = BenchmarkLoader.load(_materialised, harness_name=agent_name)
                from rllm.data.dataset import Dataset

                dataset = Dataset(data=list(bench_result.tasks), name=bench_result.name, split=bench_result.split or split)

        if bench_result is None:
            # Wrap dict-rows as Tasks; rely on evaluator_override for scoring.
            if evaluator is None:
                console.print(f"  [error]No evaluator found for '{benchmark}'. Specify --evaluator explicitly.[/]")
                raise SystemExit(1)
            from rllm.data.dataset import Dataset

            tasks = _dict_rows_to_tasks(list(dataset.data))
            dataset = Dataset(data=tasks, name=benchmark, split=split)

    # Filter to specific task indices if requested
    if task_indices is not None:
        out_of_range = [i for i in task_indices if i < 0 or i >= len(dataset)]
        if out_of_range:
            console.print(f"  [error]Task indices out of range (dataset has {len(dataset)} examples): {out_of_range}[/]")
            raise SystemExit(1)
        dataset = dataset.select(task_indices)
    elif max_examples is not None and max_examples < len(dataset):
        dataset = dataset.select(range(max_examples))

    # Resolve agent description
    agent_desc = ""
    if ":" not in agent_name:
        from rllm.cli._pull import load_agent_catalog

        agent_catalog = load_agent_catalog()
        agent_entry = agent_catalog.get("agents", {}).get(agent_name, {})
        agent_desc = agent_entry.get("description", "")

    # Print eval header
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="label", width=12)
    table.add_column()
    table.add_row("Benchmark", f"[val]{benchmark}[/]  [dim]({split}, {len(dataset)} examples)[/]")
    table.add_row("Model", f"[val]{model}[/]")
    agent_text = f"[val]{agent_name}[/]"
    if agent_desc:
        agent_text += f"  [dim]{agent_desc}[/]"
    table.add_row("Agent", agent_text)
    table.add_row("Evaluator", f"[dim]{evaluator_display}[/]")
    console.print()
    console.print(Panel(table, border_style="cyan", expand=False))
    console.print()

    # Single timestamp shared by UI session, results.json, and episodes dir.
    from datetime import datetime, timezone

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Set up the run directory. Layout:
    #   <run_dir>/
    #     meta.json        — always
    #     results.json     — written below, after the run completes
    #     episodes/        — populated only when save_episodes is True
    from rllm.eval.episode_store import EvalEpisodeStore

    if episodes_dir is not None:
        run_dir = os.path.expanduser(episodes_dir)
    else:
        rllm_home = os.path.expanduser(os.environ.get("RLLM_HOME", "~/.rllm"))
        model_safe = model.replace("/", "_").replace("\\", "_")
        bench_safe = benchmark.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(rllm_home, "eval_results", f"{bench_safe}_{model_safe}_{timestamp}")
    run_store = EvalEpisodeStore(run_dir)
    run_store.write_meta(
        {
            "benchmark": benchmark,
            "model": model,
            "agent": agent_name,
            "split": split,
            "timestamp": timestamp,
        }
    )
    episode_store = run_store if save_episodes else None

    # Create UI logger before run for progressive episode uploads
    ui_logger = None
    on_episode_complete = None
    _flush_episode_buffer = None
    _ui_callback = None
    if enable_ui:
        from rllm.utils.tracking import UILogger

        experiment = f"{model}_{agent_name}_{timestamp}".replace("/", "_")
        ui_logger = UILogger(
            project_name=benchmark,
            experiment_name=experiment,
            config={"model": model, "agent": agent_name, "benchmark": benchmark, "split": split},
            session_type="eval",
        )
        if ui_logger.session_id:
            import threading

            _episode_buffer = []
            _buffer_lock = threading.Lock()
            _BATCH_SIZE = 50

            def _flush_episode_buffer():
                with _buffer_lock:
                    if _episode_buffer:
                        ui_logger.log(data={}, step=0, episodes=list(_episode_buffer))
                        _episode_buffer.clear()

            def _ui_callback(episode):
                with _buffer_lock:
                    _episode_buffer.append(episode)
                    should_flush = len(_episode_buffer) >= _BATCH_SIZE
                    batch = list(_episode_buffer) if should_flush else None
                    if should_flush:
                        _episode_buffer.clear()
                if batch:
                    ui_logger.log(data={}, step=0, episodes=batch)

    # Wrap UI streaming and per-file dump into a single callback.
    if episode_store is not None or _ui_callback is not None:

        def on_episode_complete(idx, episode):
            if episode_store is not None:
                try:
                    episode_store.write(idx, episode)
                except Exception:
                    logger.debug("episode_store.write failed", exc_info=True)
            if _ui_callback is not None:
                _ui_callback(episode)

    # Single execution path: every Task goes through ``AgentFlowEngine``
    # via ``EvalHooks``. The engine fronts every LLM call with the rLLM
    # model gateway (so flows that ``return None`` get their Steps
    # populated from gateway-captured traces, exactly as in training).
    # ``evaluator`` (when set) overrides per-task verifier resolution;
    # otherwise ``EvalHooks`` reads each Task's [verifier] config.
    from rllm.eval.runner import run_dataset

    sampling_params: dict = {"temperature": temperature, "top_p": top_p}
    if max_tokens is not None:
        sampling_params["max_tokens"] = max_tokens

    result, episodes = asyncio.run(
        run_dataset(
            tasks=list(dataset.data),
            agent_flow=agent,
            base_url=base_url,
            model=model,
            concurrency=concurrency,
            sandbox_backend=(agent_metadata or {}).get("sandbox_backend"),
            agent_name=agent_name,
            dataset_name=getattr(dataset, "name", benchmark) or benchmark,
            on_episode_complete=on_episode_complete,
            evaluator_override=evaluator,
            sampling_params=sampling_params,
        )
    )

    # Flush remaining buffered episodes BEFORE posting eval result / finishing
    # session.  The flush enqueues episodes onto the UILogger background
    # thread; we must give the thread time to drain before finish() shuts it
    # down, otherwise episodes arrive after the session is closed (500).
    if _flush_episode_buffer is not None:
        _flush_episode_buffer()

    # Print results
    pct = f"{result.score * 100:.1f}%"
    res_table = Table(show_header=False, box=None, padding=(0, 2))
    res_table.add_column(style="label", width=12)
    res_table.add_column()
    score_style = "bold green" if result.score >= 0.5 else "bold yellow" if result.score >= 0.2 else "bold red"
    res_table.add_row("Accuracy", f"[{score_style}]{pct}[/]  [dim]({result.correct}/{result.total})[/]")
    error_style = "dim" if result.errors == 0 else "bold red"
    res_table.add_row("Errors", f"[{error_style}]{result.errors}[/]")

    # Display signal breakdown if any
    if result.signal_averages:
        for sig_name, sig_avg in result.signal_averages.items():
            res_table.add_row(sig_name.title(), f"[dim]{sig_avg:.3f}[/]")

    console.print(Panel(res_table, title="[bold]Results[/]", border_style="green" if result.score >= 0.5 else "yellow", expand=False))

    # Print error details so the user knows what went wrong
    if result.errors > 0:
        error_items = [item for item in result.items if item.error]
        console.print()
        for item in error_items[:5]:
            console.print(f"  [bold red]Task {item.idx}:[/] {item.error}")
        if len(error_items) > 5:
            console.print(f"  [dim]... and {len(error_items) - 5} more errors (see JSON output for details)[/]")

    # Save aggregate results inside the run dir so a single path holds
    # everything: meta.json, results.json, and (optionally) episodes/.
    result.timestamp = timestamp
    if output_path is None:
        result.save(os.path.join(run_dir, "results.json"))
        run_id = os.path.basename(os.path.normpath(run_dir))
        console.print(f"\n  [dim]Saved to {run_dir}[/]")
        console.print(f"  [dim]View with: rllm view {run_id}[/]")
    else:
        saved_path = result.save(output_path)
        console.print(f"\n  [dim]Saved to {saved_path}[/]")

    # Send eval result and finish UI session
    if ui_logger is not None and ui_logger.session_id:
        ui_logger.log_eval_result(result)
        ui_logger.finish()

    console.print()


@click.command("eval")
@click.argument("benchmark")
@click.option("--agent", "agent_name", default=None, help="Agent scaffold: registry name or module:object path.")
@click.option("--evaluator", "evaluator_name", default=None, help="Evaluator: registry name or module:class path.")
@click.option("--base-url", default=None, help="OpenAI-compatible API endpoint URL. If omitted, a proxy is auto-started using 'rllm setup' config.")
@click.option("--model", default=None, help="Model name to evaluate. Defaults to configured model from 'rllm setup'.")
@click.option("--split", default=None, help="Dataset split (default: from catalog eval_split).")
@click.option("--concurrency", default=64, type=int, help="Number of parallel requests.")
@click.option("--temperature", default=1.0, type=float, help="Sampling temperature passed to the agent flow's LLM calls.")
@click.option("--top-p", "top_p", default=1.0, type=float, help="Nucleus sampling top_p passed to the agent flow's LLM calls.")
@click.option("--max-tokens", "max_tokens", default=None, type=int, help="Max response tokens per LLM call (default: server's own budget).")
@click.option("--max-examples", default=None, type=int, help="Limit number of examples (for dev/testing).")
@click.option("--task-indices", default=None, type=str, help="Comma-separated task indices to evaluate (e.g., '0', '3,7,12', '0-9').")
@click.option("--output", "output_path", default=None, help="Output file path for results JSON.")
@click.option(
    "--search-backend",
    "search_backend",
    default=None,
    type=click.Choice(["serper", "brave"], case_sensitive=False),
    help="Search backend for the search agent (auto-detected from API keys if omitted).",
)
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    default=None,
    type=click.Choice(["docker", "local", "modal", "daytona", "e2b", "runloop", "gke", "apple-container"], case_sensitive=False),
    help="Sandbox/environment backend. For Harbor agents: docker, daytona, modal, e2b, etc. For sandboxed agents: docker, local, modal.",
)
@click.option("--sandbox-concurrency", "sandbox_concurrency", default=None, type=int, help="Override max concurrent sandboxes (default: agent's max_concurrent).")
@click.option("--ui/--no-ui", "enable_ui", default=None, help="Enable/disable live UI logging. Default: auto-enabled when logged in (see 'rllm login').")
@click.option("--save-episodes/--no-save-episodes", "save_episodes", default=True, help="Save each Episode as its own JSON file for later visualization (default: enabled).")
@click.option("--episodes-dir", "episodes_dir", default=None, help="Directory to write the episode JSONs into. Default: ~/.rllm/eval_results/<bench>_<model>_<timestamp>/.")
def eval_cmd(
    benchmark: str,
    agent_name: str | None,
    evaluator_name: str | None,
    base_url: str | None,
    model: str | None,
    split: str | None,
    concurrency: int,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    max_examples: int | None,
    task_indices: str | None,
    output_path: str | None,
    search_backend: str | None,
    sandbox_backend: str | None,
    sandbox_concurrency: int | None,
    enable_ui: bool | None,
    save_episodes: bool,
    episodes_dir: str | None,
):
    """Evaluate a model on a benchmark dataset."""
    # Auto-detect UI logging: enable if user is logged in (has ui_api_key or RLLM_API_KEY)
    _ui_explicit = enable_ui is not None
    if enable_ui is None:
        import os

        from rllm.eval.config import load_ui_config

        ui_config = load_ui_config()
        enable_ui = bool(os.environ.get("RLLM_API_KEY") or ui_config.get("ui_api_key"))

    if not enable_ui and not _ui_explicit:
        console.print("  [blue]Tip: Try rllm UI for live monitoring! Run [bold]rllm login[/bold] to get started.[/]")

    proxy_manager = None

    if base_url is not None:
        # Direct mode: user provided --base-url, require --model too
        if model is None:
            console.print("  [error]--model is required when --base-url is provided.[/]")
            raise SystemExit(1)
    else:
        # Proxy mode: auto-start LiteLLM proxy from config
        from rllm.eval.config import load_config

        config = load_config()
        if not config.is_configured():
            console.print()
            console.print("  [error]No configuration found.[/] Run [bold]rllm setup[/] first to configure your provider and API key.")
            console.print()
            raise SystemExit(1)

        # --model overrides configured model
        if model is None:
            model = config.model

        if config.provider == "custom":
            # Custom provider: skip LiteLLM proxy, use base_url directly
            import os as _os

            base_url = config.base_url
            if config.api_key:
                _os.environ.setdefault("OPENAI_API_KEY", config.api_key)
            console.print(f"  [success]Using custom endpoint[/] at [dim]{base_url}[/]")
        else:
            from rllm.eval.proxy import EvalProxyManager

            proxy_manager = EvalProxyManager(
                provider=config.provider,
                model_name=model,
                api_key=config.api_key,
            )
            with Status(f"[dim]Starting LiteLLM proxy for [bold]{config.provider}/{model}[/bold]...[/]", console=console):
                try:
                    proxy_manager.start_proxy_subprocess(proxy_manager.build_proxy_config())
                except (RuntimeError, TimeoutError) as e:
                    console.print(f"\n  [error]Failed to start LiteLLM proxy.[/]\n\n  {e}")
                    console.print("\n  [dim]Make sure litellm is installed:[/] [bold]pip install litellm\\[proxy][/]")
                    console.print()
                    raise SystemExit(1) from None
            base_url = proxy_manager.get_proxy_url()
            console.print(f"  [success]Proxy ready[/] at [dim]{base_url}[/]")

    # Build agent metadata from CLI options
    agent_metadata = {}
    if search_backend:
        agent_metadata["search_backend"] = search_backend
    if sandbox_backend:
        agent_metadata["sandbox_backend"] = sandbox_backend
    if sandbox_concurrency is not None:
        agent_metadata["sandbox_concurrency"] = sandbox_concurrency

    # Parse --task-indices: "5", "3,7,12", "0-9", or "2,5-8,11"
    parsed_indices = None
    if task_indices is not None:
        parsed_indices = []
        for part in task_indices.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                parsed_indices.extend(range(int(lo), int(hi) + 1))
            else:
                parsed_indices.append(int(part))

    try:
        _run_eval(
            benchmark,
            agent_name,
            evaluator_name,
            base_url,
            model,
            split,
            concurrency,
            max_examples,
            parsed_indices,
            output_path,
            agent_metadata=agent_metadata,
            enable_ui=enable_ui,
            save_episodes=save_episodes,
            episodes_dir=episodes_dir,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    finally:
        if proxy_manager is not None:
            proxy_manager.shutdown_proxy()
