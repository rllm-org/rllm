"""Eval CLI command.

``rllm eval <benchmark> --agent <name> [--evaluator <name>] [--base-url <url>] [--model <name>]``

When ``--base-url`` is omitted, a LiteLLM proxy is auto-started using the
configuration from ``rllm setup`` (stored in ``~/.rllm/config.json``).
"""

from __future__ import annotations

import asyncio

import click
from rich.console import Console
from rich.panel import Panel
from rich.status import Status
from rich.table import Table
from rich.theme import Theme

from rllm.experimental.cli._pull import load_dataset_catalog, pull_dataset

theme = Theme({"label": "dim", "success": "bold green", "error": "bold red", "val": "bold", "key": "yellow"})
console = Console(theme=theme)


def _suggest_benchmarks(name: str, catalog_names: list[str], max_suggestions: int = 3) -> list[str]:
    """Return catalog names similar to *name*, ordered by edit distance."""
    from difflib import get_close_matches

    return get_close_matches(name, catalog_names, n=max_suggestions, cutoff=0.5)


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
):
    """Core eval logic, extracted for clean proxy lifecycle management."""
    from rllm.data import DatasetRegistry
    from rllm.experimental.eval.agent_loader import load_agent
    from rllm.experimental.eval.evaluator_loader import load_evaluator, resolve_evaluator_from_catalog
    from rllm.experimental.eval.runner import EvalRunner

    # ------------------------------------------------------------------
    # Local benchmark path: directory with dataset.toml / task.toml
    # ------------------------------------------------------------------
    from rllm.tasks.loader import BenchmarkLoader

    if BenchmarkLoader.is_local_benchmark(benchmark):
        sandbox_backend = (agent_metadata or {}).get("sandbox_backend")
        bench_result = BenchmarkLoader.load(benchmark, sandbox_backend=sandbox_backend)
        dataset = bench_result.dataset
        catalog_entry = bench_result.catalog_entry

        if split is None:
            split = dataset.split or "test"

        # Agent: CLI --agent overrides loader default
        if agent_name is not None:
            agent = load_agent(agent_name)
        elif bench_result.agent is not None:
            agent = bench_result.agent
            agent_name = catalog_entry.get("default_agent", "task-executor")
        else:
            console.print(f"  [error]No --agent specified and no default agent for local benchmark '{benchmark}'.[/]")
            raise SystemExit(1)

        # Evaluator: CLI --evaluator overrides loader default
        evaluator_display = "N/A"
        if evaluator_name is not None:
            evaluator = load_evaluator(evaluator_name)
            evaluator_display = evaluator_name
        elif bench_result.evaluator is not None:
            evaluator = bench_result.evaluator
            evaluator_display = type(evaluator).__name__
        else:
            console.print(f"  [error]No --evaluator specified and no default evaluator for local benchmark '{benchmark}'.[/]")
            raise SystemExit(1)

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
            from rllm.experimental.cli._pull import resolve_harbor_catalog_entry

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
            from rllm.integrations.harbor.utils import check_docker_available

            if not check_docker_available():
                console.print("  [error]Harbor tasks require Docker. Make sure Docker is installed and running.[/]")
                raise SystemExit(1)

        # Load agent (now returns AgentFlow)
        try:
            agent = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            console.print(f"  [error]Error loading agent '{agent_name}': {e}[/]")
            raise SystemExit(1) from None

        # Apply sandbox CLI overrides to agent
        if agent_metadata:
            from rllm.integrations.harbor.runtime import HarborRuntime

            if isinstance(agent, HarborRuntime):
                if "sandbox_backend" in agent_metadata:
                    agent.environment_type = agent_metadata["sandbox_backend"]
            else:
                from rllm.experimental.agents.sandboxed_agent import SandboxedAgentFlow

                if isinstance(agent, SandboxedAgentFlow):
                    if "sandbox_backend" in agent_metadata:
                        agent.sandbox_backend = agent_metadata["sandbox_backend"]
                    if "sandbox_concurrency" in agent_metadata:
                        agent.max_concurrent = agent_metadata["sandbox_concurrency"]

        # Load evaluator
        evaluator = None
        evaluator_display = "N/A"
        if evaluator_name is not None:
            try:
                evaluator = load_evaluator(evaluator_name)
                evaluator_display = evaluator_name
            except (KeyError, ImportError, AttributeError, TypeError) as e:
                console.print(f"  [error]Error loading evaluator '{evaluator_name}': {e}[/]")
                raise SystemExit(1) from None
        else:
            # Auto-resolve from catalog
            evaluator = resolve_evaluator_from_catalog(benchmark)
            if evaluator is not None:
                reward_fn_name = catalog_entry.get("reward_fn", "") if catalog_entry else ""
                evaluator_display = reward_fn_name or type(evaluator).__name__
            elif catalog_entry and catalog_entry.get("reward_fn"):
                # Catalog entry exists (possibly synthesized for Harbor) but
                # resolve_evaluator_from_catalog missed it — try loading directly.
                try:
                    evaluator = load_evaluator(catalog_entry["reward_fn"])
                    evaluator_display = catalog_entry["reward_fn"]
                except (KeyError, ImportError):
                    pass

        if evaluator is None:
            console.print(f"  [error]No evaluator found for '{benchmark}'. Specify --evaluator explicitly.[/]")
            raise SystemExit(1)

        # Load dataset — auto-pull if not available locally
        dataset = DatasetRegistry.load_dataset(benchmark, split)
        if dataset is None:
            if catalog_entry:
                with Status(f"[dim]Pulling {benchmark} from {catalog_entry['source']}...[/]", console=console):
                    pull_dataset(benchmark, catalog_entry)
                dataset = DatasetRegistry.load_dataset(benchmark, split)

        if dataset is None:
            console.print(f"  [error]Could not load dataset '{benchmark}' split '{split}'.[/]")
            raise SystemExit(1)

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
        from rllm.experimental.cli._pull import load_agent_catalog

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

    # Run evaluation
    runner = EvalRunner(
        base_url=base_url,
        model=model,
        concurrency=concurrency,
        agent_metadata=agent_metadata or {},
        catalog_entry=catalog_entry,
        benchmark_name=benchmark,
    )

    # Create UI logger before run for progressive episode uploads
    ui_logger = None
    on_episode_complete = None
    _flush_episode_buffer = None
    if enable_ui:
        from datetime import datetime, timezone

        from rllm.utils.tracking import UILogger

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
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

            def on_episode_complete(episode):
                with _buffer_lock:
                    _episode_buffer.append(episode)
                    should_flush = len(_episode_buffer) >= _BATCH_SIZE
                    batch = list(_episode_buffer) if should_flush else None
                    if should_flush:
                        _episode_buffer.clear()
                if batch:
                    ui_logger.log(data={}, step=0, episodes=batch)

    result, episodes = asyncio.run(runner.run(dataset, agent, evaluator, agent_name=agent_name, on_episode_complete=on_episode_complete))

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

    # Save results
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
def eval_cmd(
    benchmark: str,
    agent_name: str | None,
    evaluator_name: str | None,
    base_url: str | None,
    model: str | None,
    split: str | None,
    concurrency: int,
    max_examples: int | None,
    task_indices: str | None,
    output_path: str | None,
    search_backend: str | None,
    sandbox_backend: str | None,
    sandbox_concurrency: int | None,
    enable_ui: bool | None,
):
    """Evaluate a model on a benchmark dataset."""
    # Auto-detect UI logging: enable if user is logged in (has ui_api_key or RLLM_API_KEY)
    _ui_explicit = enable_ui is not None
    if enable_ui is None:
        import os

        from rllm.experimental.eval.config import load_ui_config

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
        from rllm.experimental.eval.config import load_config

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
            from rllm.experimental.eval.proxy import EvalProxyManager

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
        _run_eval(benchmark, agent_name, evaluator_name, base_url, model, split, concurrency, max_examples, parsed_indices, output_path, agent_metadata=agent_metadata, enable_ui=enable_ui)
    finally:
        if proxy_manager is not None:
            proxy_manager.shutdown_proxy()
