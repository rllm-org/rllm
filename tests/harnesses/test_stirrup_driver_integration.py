"""Exercise the GDPval driver against a real Stirrup install.

These assertions cover the parts of AA's contract that only the real library
can answer — the tool inventory the model actually sees, the search result
count, the image downscaling limit. Stirrup is not an rLLM dependency (it is
installed into its own virtualenv inside the sandbox), so the whole module
skips when it is absent.

Stirrup requires Python 3.12+, so run them under a matching interpreter. Pass
``--with pytest`` too, or pytest resolves from the project venv and imports a
different Stirrup than the one under test::

    uv run --python 3.12 --with stirrup==0.2.0 --with pytest \\
        pytest tests/harnesses/test_stirrup_driver_integration.py
"""

from __future__ import annotations

import asyncio
import json

import pytest

from rllm.data import gdpval_aa as aa
from rllm.harnesses.stirrup import STIRRUP_VERSION

stirrup = pytest.importorskip("stirrup", reason="Stirrup runs in the sandbox venv, not the rLLM venv")

# The project venv inherits system site-packages, so an unrelated Stirrup can
# shadow the pinned one. Assert against the version the harness actually
# installs rather than reporting failures for a copy nobody is shipping.
_installed = __import__("importlib.metadata", fromlist=["version"]).version("stirrup")
if tuple(int(p) for p in _installed.split(".")) < tuple(int(p) for p in STIRRUP_VERSION.split(".")):
    pytest.skip(
        f"installed Stirrup {_installed} predates the pinned {STIRRUP_VERSION}",
        allow_module_level=True,
    )

from stirrup.clients.chat_completions_client import ChatCompletionsClient  # noqa: E402
from stirrup.tools.view_image import ViewImageToolProvider  # noqa: E402
from stirrup.tools.web import WebToolProvider  # noqa: E402

#: AA names the tools in prose ("Web Fetch"); these are the wire names Stirrup
#: actually registers, which is what the model is offered.
EXPECTED_TOOLS = {"code_exec", "fetch_web_page", "web_search", "view_image", "finish", "abandon_task_finish"}


def _agent(driver, monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "brave-test-key")
    exec_env = driver.SandboxCodeExecToolProvider(driver.WORKDIR, shell_timeout=driver.SHELL_TIMEOUT, env={"PATH": "/usr/bin:/bin"})
    return stirrup.Agent(
        # 0.2.0 separates the budgets: max_tokens caps generation,
        # context_window_tokens is what the compaction threshold applies to.
        client=ChatCompletionsClient(model="test-model", base_url="http://127.0.0.1:1/v1", api_key="sk-test", max_tokens=16_384, context_window_tokens=200_000),
        name="gdpval-aa-solver",
        max_turns=aa.AA_MAX_TURNS,
        system_prompt=aa.AA_GDPVAL_SYSTEM_PROMPT,
        tools=[exec_env, WebToolProvider(), ViewImageToolProvider(exec_env)],
        finish_tool=[driver.FINISH_TOOL, driver.ABANDON_TOOL],
        context_summarization_cutoff=aa.AA_CONTEXT_SUMMARIZATION_CUTOFF,
    ), exec_env


def test_exactly_the_six_aa_tools_are_exposed(driver, monkeypatch):
    agent, _ = _agent(driver, monkeypatch)

    async def run():
        async with agent.session(cache_on_interrupt=False) as session:
            return set(session.tools)

    assert asyncio.run(run()) == EXPECTED_TOOLS


def test_finish_and_abandon_schemas_match_the_prompt(driver, monkeypatch):
    agent, _ = _agent(driver, monkeypatch)

    async def run():
        async with agent.session(cache_on_interrupt=False) as session:
            return (
                session.tools["finish"].parameters.model_json_schema(),
                session.tools["abandon_task_finish"].parameters.model_json_schema(),
            )

    finish, abandon = asyncio.run(run())

    assert set(finish["properties"]) == {"summary", "paths"}
    assert "ABSOLUTE" in finish["properties"]["paths"]["description"]
    assert set(abandon["properties"]) == {"reason"}


def test_each_command_runs_in_a_fresh_shell(driver, monkeypatch):
    _, exec_env = _agent(driver, monkeypatch)

    async def run():
        await exec_env.run_command("mkdir -p nested")
        moved = await exec_env.run_command("cd nested && pwd")
        after_cd = await exec_env.run_command("pwd")
        await exec_env.run_command("export LEAKED=yes")
        after_export = await exec_env.run_command('echo "[${LEAKED:-unset}]"')
        return moved, after_cd, after_export

    moved, after_cd, after_export = asyncio.run(run())

    assert moved.stdout.strip().endswith("/nested")
    assert after_cd.stdout.strip() == str(driver.WORKDIR.resolve())
    assert after_export.stdout.strip() == "[unset]"


def test_absolute_paths_are_usable(driver, monkeypatch):
    """Stirrup's local backend rejects these outright; AA's prompt requires them."""
    _, exec_env = _agent(driver, monkeypatch)

    target = driver.WORKDIR / "report.txt"
    result = asyncio.run(exec_env.run_command(f"echo hello > {target} && cat {target}"))

    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"
    assert result.error_kind is None


def test_command_timeout_is_enforced(driver, monkeypatch):
    _, exec_env = _agent(driver, monkeypatch)

    result = asyncio.run(exec_env.run_command("sleep 5", timeout=1))

    assert result.error_kind == "timeout"
    assert "timed out after 1 seconds" in result.stderr


def test_brave_search_requests_the_top_five_results():
    from stirrup.tools import web

    source = __import__("inspect").getsource(web)

    assert '"count": 5' in source
    assert "top 5 results" in source.lower()


def test_images_are_downscaled_to_one_megapixel():
    from stirrup.constants import RESOLUTION_1MP
    from stirrup.core.models import ImageContentBlock

    assert RESOLUTION_1MP == aa.AA_MAX_IMAGE_PIXELS == 1_000_000
    default = __import__("inspect").signature(ImageContentBlock.to_base64_url).parameters["max_pixels"].default
    assert default == RESOLUTION_1MP


def test_context_compaction_triggers_at_seventy_percent():
    from stirrup.constants import CONTEXT_SUMMARIZATION_CUTOFF

    assert CONTEXT_SUMMARIZATION_CUTOFF == aa.AA_CONTEXT_SUMMARIZATION_CUTOFF == 0.7


def test_finish_rejects_invalid_submissions_through_the_tool_executor(driver):
    work = driver.WORKDIR
    good = work / "report.docx"
    good.write_text("report")

    async def call(paths):
        return await driver.finish_executor(driver.FinishParams(summary="done", paths=paths))

    assert asyncio.run(call([str(good)])).success is True
    for bad in [["report.docx"], [str(work)], [str(work / "missing.pdf")], ["/etc/passwd"], []]:
        result = asyncio.run(call(bad))
        assert result.success is False, bad
        assert "ERROR" in str(result.content)


def test_abandon_ends_the_task_without_files(driver):
    result = asyncio.run(driver.abandon_executor(driver.AbandonParams(reason="required input is missing")))

    assert result.success is True
    assert result.content == "required input is missing"
    assert json.loads(json.dumps(driver.stage_submission([])[0])) == []
