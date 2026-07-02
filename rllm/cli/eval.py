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
from rich.status import Status

from rllm import paths
from rllm.cli._pull import load_dataset_catalog, pull_dataset
from rllm.cli._sampling import SAMPLING_PARAMS_HELP as _SAMPLING_PARAMS_HELP
from rllm.cli._ui import console, fail, info_panel, not_found, parse_index_spec
from rllm.types import Task

logger = logging.getLogger(__name__)


def _suggest_benchmarks(name: str, catalog_names: list[str], max_suggestions: int = 3) -> list[str]:
    """Return catalog names similar to *name*, ordered by edit distance."""
    from difflib import get_close_matches

    return get_close_matches(name, catalog_names, n=max_suggestions, cutoff=0.5)


def _apply_sandbox_overrides(agent, agent_metadata: dict | None) -> None:
    """Route CLI sandbox flags through the flow's ``configure()`` and warn about leftovers."""
    if not agent_metadata:
        return
    configure = getattr(agent, "configure", None)
    leftovers = dict(configure(dict(agent_metadata)) if callable(configure) else agent_metadata)
    # run_dataset / the sandbox hooks consume the backend regardless of the flow.
    leftovers.pop("sandbox_backend", None)
    for flag in leftovers:
        logger.warning("--%s has no effect for agent %s", flag.replace("_", "-"), type(agent).__name__)


# Agent knobs surfaced in the eval header, in display order. Each entry is
# (attribute, label, formatter); a flow that doesn't expose an attribute (or
# leaves it ``None``) simply omits that pair, so this works across harnesses.
# NB: temperature is intentionally omitted — sampling params (temperature,
# top_p, …) are shown in the gateway-enforced "Sampling" row, which is
# authoritative; listing the harness's requested temperature here too just
# duplicated it with a (usually different, since the gateway wins) value.
_AGENT_CONFIG_SPECS = (
    ("max_turns", "max turns", str),
    ("max_steps", "max steps", str),
    ("max_concurrent", "max concurrent", str),
    ("run_timeout", "run timeout", lambda v: f"{v}s"),
    ("install_timeout", "install timeout", lambda v: f"{v}s"),
)


def _agent_config_rows(agent) -> list[tuple[str, str]]:
    """Build the ``Config`` panel row from the loaded flow's effective knobs.

    Reads the curated attributes off the agent instance (max turns, temperature,
    timeouts, …) and renders each present one as a ``key value`` chip — dim key,
    bold value — with chips separated by a dim ``·``. Spaces *within* a chip are
    non-breaking (`` ``) so a pair never splits across a line wrap; wraps
    fall on the separators instead. Returns an empty list when the flow exposes
    none of the knobs.
    """
    chips = []
    for attr, label, fmt in _AGENT_CONFIG_SPECS:
        value = getattr(agent, attr, None)
        if value is not None:
            key = label.replace(" ", " ")
            chips.append(f"[dim]{key}[/] [val]{fmt(value)}[/]")
    if not chips:
        return []
    return [("Config", "  [dim]·[/]  ".join(chips))]


def _dict_rows_to_tasks(rows: list[dict]) -> list[Task]:
    """Wrap dict-rows from a catalog dataset as Task objects.

    Harbor rows carry ``task_path``; we root the Task there and merge the task's
    ``task.toml`` + ``environment/Dockerfile`` metadata (workdir, verifier
    timeout, per-task image) so per-task verifier resolution and an rllm-native
    harness get the right sandbox. Other rows get ``dataset_dir=Path(".")`` and
    rely on the fixed-evaluator policy.
    """
    from pathlib import Path

    from rllm.tasks.loader import _merge_task_toml_metadata

    tasks: list[Task] = []
    for idx, row in enumerate(rows):
        instruction = row.get("instruction") or row.get("question") or ""
        task_id = str(row.get("id") or row.get("task_id") or idx)
        task_path = row.get("task_path")
        dataset_dir = Path(task_path) if task_path else Path(".")
        metadata = dict(row)
        if task_path:
            metadata = _merge_task_toml_metadata(dataset_dir, metadata)
        tasks.append(
            Task(
                id=task_id,
                instruction=str(instruction),
                metadata=metadata,
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
    use_snapshot: bool = True,
    warm_queue_size: int = 0,
    sampling_config=None,
    attempts: int = 1,
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
        _materialised = paths.rllm_path("datasets", benchmark)
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
        console.print(f"  [dim]Resolved to: {_os.path.abspath(resolved)}[/]")
        console.print(f"  [dim]Exists: {_os.path.exists(resolved)}, is dir: {_os.path.isdir(resolved) if _os.path.exists(resolved) else 'N/A'}[/]")
        console.print("  [dim]A benchmark directory must contain dataset.toml, task.toml, or sub-dirs with task.toml.[/]")
        fail(f"Path-like benchmark '{benchmark}' did not resolve to a benchmark directory.")

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

        # dataset.toml's default_sandbox applies when --sandbox-backend wasn't
        # given; it reaches the agent and hooks through agent_metadata like the
        # CLI flag would.
        if not sandbox_backend and bench_result.sandbox_backend:
            agent_metadata = dict(agent_metadata or {})
            agent_metadata["sandbox_backend"] = bench_result.sandbox_backend

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
            fail(f"Cannot load agent '{agent_name}': {e}")

        _apply_sandbox_overrides(agent, agent_metadata)

        # Evaluator: comes from each Task's [verifier] config by default.
        # CLI --evaluator (when provided) overrides for every task.
        evaluator = None
        evaluator_display = f"per-task ({len(bench_result.tasks)} tasks)"
        if evaluator_name is not None:
            try:
                evaluator = load_evaluator(evaluator_name)
                evaluator_display = f"{evaluator_name} (overrides per-task verifier)"
            except (KeyError, ImportError, AttributeError, TypeError) as e:
                fail(f"Error loading evaluator '{evaluator_name}': {e}")

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
                hint = ""
                suggestions = _suggest_benchmarks(benchmark, list(all_datasets.keys()))
                if suggestions:
                    hint += f"\n\n  Did you mean: [bold]{', '.join(suggestions)}[/]?"
                hint += "\n\n  Run [bold]rllm dataset list[/] to see available benchmarks."
                hint += "\n  Use [bold]harbor:[/] prefix for Harbor datasets (e.g., [bold]rllm eval harbor:swebench-verified[/])."
                not_found("Benchmark", benchmark, hint=hint.lstrip())
            else:
                fail(f"No --agent specified and no default_agent in catalog for '{benchmark}'.")

        # Resolve split
        if split is None:
            if catalog_entry:
                split = catalog_entry.get("eval_split", "test")
            else:
                split = "test"

        _is_harbor_agent = bool(agent_name) and agent_name.startswith("harbor:")
        _is_harbor_source = bool(catalog_entry) and catalog_entry.get("source", "").startswith("harbor:")

        # Require Docker only when execution lands on the local daemon; a remote
        # backend (modal/daytona) pulls the task's image in the cloud.
        _eff_backend = (agent_metadata or {}).get("sandbox_backend")
        _runs_on_local_docker = _eff_backend in (None, "docker")
        if (_is_harbor_agent or _is_harbor_source) and _runs_on_local_docker:
            from rllm.integrations.harbor.utils import diagnose_docker

            ok, reason, hint = diagnose_docker()
            if not ok:
                if hint:
                    console.print(f"  [dim]{hint}[/]")
                console.print("  [dim]Or run on a remote backend, e.g. [bold]--sandbox-backend modal[/].[/]")
                fail(f"Harbor tasks require Docker — {reason}.")

        # Load the agent. Catalog covers built-in flows (react/bash/claude-code)
        # plus user-registered + plugin agents; ``harbor:<scaffold>`` resolves
        # to a HarborRuntime via the harbor: prefix branch in load_agent.
        try:
            agent = load_agent(agent_name)
        except (KeyError, ImportError, AttributeError, TypeError) as e:
            fail(f"Error loading agent '{agent_name}': {e}")

        _apply_sandbox_overrides(agent, agent_metadata)

        # Resolve evaluator: explicit --evaluator > catalog auto-resolve >
        # catalog reward_fn > None. When ``evaluator`` is None ``SandboxTaskHooks``
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
                fail(f"Error loading evaluator '{evaluator_name}': {e}")
        else:
            # ``harbor_reward_fn`` reads ``episode.artifacts['harbor_reward']``,
            # which is only populated by HarborRuntime. When the user runs a
            # harbor-sourced dataset through an rllm-native harness (oracle,
            # opencode, mini-swe-agent, …), the artifact never gets set and
            # the eval would always score zero. Skip the catalog evaluator in
            # that case and let ``SandboxTaskHooks`` resolve a per-task verifier
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
            fail(f"Could not load dataset '{benchmark}' split '{split}'.")

        # Materialise non-harbor catalog datasets so each Task carries its
        # per-task verifier from dataset.toml. Harbor datasets are
        # task-per-directory; materialising would drop their tests/ +
        # environment/, so wrap their rows directly instead.
        bench_result = None
        if not _is_harbor_agent and not _is_harbor_source:
            _materialised = paths.rllm_path("datasets", benchmark)
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
            # Harbor rows root at their task dir, so SandboxTaskHooks resolves
            # each task's own tests/test.sh — no dataset-wide evaluator needed.
            # Non-harbor rows carry no per-task verifier, so still require one.
            if evaluator is None and not _is_harbor_source:
                fail(f"No evaluator found for '{benchmark}'. Specify --evaluator explicitly.")
            from rllm.data.dataset import Dataset

            tasks = _dict_rows_to_tasks(list(dataset.data))
            dataset = Dataset(data=tasks, name=benchmark, split=split)

    # Filter to specific task indices if requested
    if task_indices is not None:
        out_of_range = [i for i in task_indices if i < 0 or i >= len(dataset)]
        if out_of_range:
            fail(f"Task indices out of range (dataset has {len(dataset)} examples): {out_of_range}")
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
    agent_text = f"[val]{agent_name}[/]"
    if agent_desc:
        agent_text += f"  [dim]{agent_desc}[/]"
    rows = [
        ("Benchmark", f"[val]{benchmark}[/]  [dim]({split}, {len(dataset)} examples)[/]"),
        ("Model", f"[val]{model}[/]"),
        ("Agent", agent_text),
        *_agent_config_rows(agent),
        ("Evaluator", f"[dim]{evaluator_display}[/]"),
    ]
    if not use_snapshot:
        rows.append(("Snapshots", "[dim]disabled (--no-snapshot, cold start)[/]"))
    if sampling_config is not None and not sampling_config.is_empty:
        rows.append(("Sampling", f"[dim]{sampling_config.as_dict()} (gateway-enforced)[/]"))
    console.print()
    console.print(info_panel(rows, border="brand"))
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
        model_safe = model.replace("/", "_").replace("\\", "_")
        bench_safe = benchmark.replace("/", "_").replace("\\", "_")
        run_dir = paths.rllm_path("eval_results", f"{bench_safe}_{model_safe}_{timestamp}")
    run_store = EvalEpisodeStore(run_dir)
    run_store.write_meta(
        {
            "benchmark": benchmark,
            "model": model,
            "agent": agent_name,
            "split": split,
            "timestamp": timestamp,
            "attempts": attempts,
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
    # via ``SandboxTaskHooks``. The engine fronts every LLM call with the rLLM
    # model gateway (so flows that ``return None`` get their Steps
    # populated from gateway-captured traces, exactly as in training).
    # ``evaluator`` (when set) overrides per-task verifier resolution;
    # otherwise ``SandboxTaskHooks`` reads each Task's [verifier] config.
    from rllm.eval.runner import run_dataset

    result, episodes = asyncio.run(
        run_dataset(
            tasks=list(dataset.data),
            agent_flow=agent,
            base_url=base_url,
            model=model,
            concurrency=concurrency,
            sandbox_backend=(agent_metadata or {}).get("sandbox_backend"),
            use_snapshot=use_snapshot,
            warm_queue_size=warm_queue_size,
            agent_name=agent_name,
            dataset_name=getattr(dataset, "name", benchmark) or benchmark,
            on_episode_complete=on_episode_complete,
            evaluator=evaluator,
            sampling_params=(sampling_config.as_dict() if sampling_config is not None else None),
            attempts=attempts,
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
    score_style = "bold green" if result.score >= 0.5 else "bold yellow" if result.score >= 0.2 else "bold red"
    error_style = "dim" if result.errors == 0 else "bold red"
    res_rows = [
        ("Accuracy", f"[{score_style}]{pct}[/]  [dim]({result.correct}/{result.total})[/]"),
        ("Errors", f"[{error_style}]{result.errors}[/]"),
    ]
    for k, v in sorted(result.pass_at.items()):
        res_rows.append((f"pass@{k}", f"[bold]{v * 100:.1f}%[/]"))

    # Per-category completion breakdown (env_done / timeout / verifier_timeout / ...),
    # infra-error reasons highlighted so a flaky endpoint or broken verifier is
    # visible at a glance instead of hiding inside a single "Errors" count.
    if result.termination_breakdown:
        from rllm.types import INFRA_ERROR_REASONS

        infra_vals = {r.value for r in INFRA_ERROR_REASONS}
        parts = [f"[{'red' if reason in infra_vals else 'dim'}]{reason} {count}[/]" for reason, count in sorted(result.termination_breakdown.items(), key=lambda kv: (-kv[1], kv[0]))]
        res_rows.append(("Terminations", "  ".join(parts)))

    # Display signal breakdown if any
    if result.signal_averages:
        for sig_name, sig_avg in result.signal_averages.items():
            res_rows.append((sig_name.title(), f"[dim]{sig_avg:.3f}[/]"))

    console.print(info_panel(res_rows, title="[bold]Results[/]", border="green" if result.score >= 0.5 else "yellow"))

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
@click.option(
    "--proxy-port", "proxy_port", default=None, type=int, help="Pin the auto-started LiteLLM proxy to this port. Default: a free port is picked automatically (so concurrent eval jobs don't collide)."
)
@click.option("--model", default=None, help="Model name to evaluate. Defaults to configured model from 'rllm setup'.")
@click.option("--split", default=None, help="Dataset split (default: from catalog eval_split).")
@click.option("--concurrency", default=64, type=int, help="Number of parallel requests.")
@click.option("--attempts", default=1, type=int, help="Independent rollouts per task; reports pass@k for k=1..N (default: 1).")
@click.option("--max-examples", default=None, type=int, help="Limit number of examples (for dev/testing).")
@click.option("--task-indices", default=None, type=str, help="Comma-separated task indices to evaluate (e.g., '0', '3,7,12', '0-9').")
@click.option("--output", "output_path", default=None, help="Output file path for results JSON.")
@click.option(
    "--sandbox-backend",
    "sandbox_backend",
    default=None,
    type=click.Choice(["docker", "local", "modal", "daytona", "e2b", "runloop", "gke", "apple-container"], case_sensitive=False),
    help="Sandbox/environment backend. For Harbor agents: docker, daytona, modal, e2b, etc. For sandboxed agents: docker, local, modal.",
)
@click.option("--sandbox-concurrency", "sandbox_concurrency", default=None, type=int, help="Override max concurrent sandboxes (default: agent's max_concurrent).")
@click.option(
    "--snapshot/--no-snapshot",
    "use_snapshot",
    default=True,
    help="Boot each task from a pre-built environment snapshot when one exists (default). Use --no-snapshot to force the cold path (e.g. A/B timing). Build snapshots with 'rllm snapshot create'.",
)
@click.option(
    "--warm-queue-size",
    "warm_queue_size",
    default=0,
    type=int,
    help="Prefetch up to N sandboxes ahead of consumption to overlap creation with rollout (0 = off; -1 = match --concurrency). Requires --sandbox-backend.",
)
@click.option("--ui/--no-ui", "enable_ui", default=None, help="Enable/disable live UI logging. Default: auto-enabled when logged in (see 'rllm login').")
@click.option("--save-episodes/--no-save-episodes", "save_episodes", default=True, help="Save each Episode as its own JSON file for later visualization (default: enabled).")
@click.option("--episodes-dir", "episodes_dir", default=None, help="Directory to write the episode JSONs into. Default: ~/.rllm/eval_results/<bench>_<model>_<timestamp>/.")
@click.option("--sampling-params", "sampling_params", default=None, help=_SAMPLING_PARAMS_HELP)
@click.option("--temperature", default=None, type=float, help="Sampling temperature (shortcut for --sampling-params temperature=...).")
@click.option("--top-p", "top_p", default=None, type=float, help="Nucleus sampling top_p (shortcut for --sampling-params top_p=...).")
@click.option("--max-tokens", "max_tokens", default=None, type=int, help="Max generated tokens per call (shortcut for --sampling-params max_tokens=...).")
@click.option(
    "--agent-timeout",
    "agent_timeout",
    default=None,
    type=int,
    metavar="SECONDS",
    help=(
        "Per-rollout agent wall-clock timeout in seconds for sandboxed CLI harnesses (e.g. terminus2). "
        "Default 3600. Sandbox lifetimes are sized to outlast this, so the environment isn't torn down mid-rollout."
    ),
)
def eval_cmd(
    benchmark: str,
    agent_name: str | None,
    evaluator_name: str | None,
    base_url: str | None,
    proxy_port: int | None,
    model: str | None,
    split: str | None,
    concurrency: int,
    attempts: int,
    max_examples: int | None,
    task_indices: str | None,
    output_path: str | None,
    sandbox_backend: str | None,
    sandbox_concurrency: int | None,
    use_snapshot: bool,
    warm_queue_size: int,
    enable_ui: bool | None,
    save_episodes: bool,
    episodes_dir: str | None,
    sampling_params: str | None,
    temperature: float | None,
    top_p: float | None,
    max_tokens: int | None,
    agent_timeout: int | None,
):
    """Evaluate a model on a benchmark dataset."""
    from rllm.cli._sampling import resolve_eval_sampling

    try:
        sampling_config = resolve_eval_sampling(sampling_params, temperature, top_p, max_tokens)
    except (ValueError, FileNotFoundError, TypeError) as e:
        fail(f"Invalid --sampling-params: {e}")
    if attempts < 1:
        fail("--attempts must be >= 1.")
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
            fail("--model is required when --base-url is provided.")
    else:
        # Proxy mode: auto-start LiteLLM proxy from config
        from rllm.eval.config import load_config

        config = load_config()
        if not config.is_configured():
            fail("No configuration found. Run `rllm setup` first to configure your provider and API key.")

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
                proxy_port=proxy_port,
            )
            with Status(f"[dim]Starting LiteLLM proxy for [bold]{config.provider}/{model}[/bold]...[/]", console=console):
                try:
                    proxy_manager.start_proxy_subprocess(proxy_manager.build_proxy_config())
                except (RuntimeError, TimeoutError) as e:
                    console.print("\n  [dim]Make sure litellm is installed:[/] [bold]pip install litellm\\[proxy][/]")
                    console.print()
                    fail(f"Failed to start LiteLLM proxy.\n\n  {e}")
            base_url = proxy_manager.get_proxy_url()
            console.print(f"  [success]Proxy ready[/] at [dim]{base_url}[/]")

    # Build agent metadata from CLI options
    agent_metadata = {}
    if sandbox_backend:
        agent_metadata["sandbox_backend"] = sandbox_backend
    if sandbox_concurrency is not None:
        agent_metadata["sandbox_concurrency"] = sandbox_concurrency
    if agent_timeout is not None:
        if agent_timeout < 1:
            fail("--agent-timeout must be >= 1 second.")
        # Consumed by BaseCliHarness.configure() → sets the harness run_timeout.
        agent_metadata["agent_timeout"] = agent_timeout
        # Also publish it as the canonical env knob so the Modal backend derives
        # a sandbox lifetime with headroom over it (else a raised timeout would
        # still be reaped at the default lifetime).
        import os

        os.environ["RLLM_HARNESS_RUN_TIMEOUT_S"] = str(agent_timeout)

    parsed_indices = parse_index_spec(task_indices) if task_indices is not None else None

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
            use_snapshot=use_snapshot,
            warm_queue_size=warm_queue_size,
            sampling_config=sampling_config,
            attempts=attempts,
        )
    finally:
        if proxy_manager is not None:
            proxy_manager.shutdown_proxy()
