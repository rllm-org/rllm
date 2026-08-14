"""Lock Artificial Analysis' published GDPval-AA v2 specification.

The prompt literals below are an independent transcription of AA's published
methodology, deliberately *not* re-read from ``rllm/data/gdpval_aa/*.txt`` — a
test that loads the same file it validates cannot detect drift in that file.

Source: https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdpval-aa
"""

from __future__ import annotations

import pytest

from rllm.data import gdpval_aa as aa

EXPECTED_SYSTEM_PROMPT = "You are an AI agent completing a standalone professional task. Your job is to use the provided tools to produce the requested deliverables within 250 steps, then submit your work.\n\nWhen you are done, call the `finish` tool as your final step with:\n1. A brief summary of what you accomplished.\n2. Absolute paths to every deliverable file.\n\nIf you have genuinely concluded that the task cannot be completed because required inputs are missing, a hard dependency is unavailable, or the request is incoherent, call the `abandon_task_finish` tool with a brief reason instead. Do not use it to escape difficulty.\n\nYou cannot interact with the user during the task. Make reasonable assumptions when needed and record them in your finish summary."

EXPECTED_TASK_PROMPT_NO_REFERENCES = "## Runtime\n\nYou are running in an isolated Linux sandbox. Use the `code_exec` tool to read, create, and modify files. Commands run as the non-root user `user` (UID 1000). Default working directory is `/home/user`.\n\nEvery command runs independently: no working directory, environment variable, or other shell state carries over from one call to the next. Prefer absolute paths for both files and commands, and do not navigate with `cd` across calls — a `cd` in one command is gone by the next, so relying on it leaves you silently operating in the wrong place. When a step genuinely needs a different directory, chain it into the same command (e.g. `cd /home/user/work && python build.py`).\n\nA broad scientific-computing and document-processing stack is already installed, so confirm what is present before assuming a gap:\n- Python 3.13 with the usual data stack (numpy, pandas, polars, scipy), plotting (matplotlib, plotly), the scikit-learn ML family, and document tooling (python-docx, python-pptx, openpyxl, PyMuPDF, pdfplumber, reportlab, weasyprint, Pillow, opencv), plus Playwright.\n- System tools include LibreOffice, Pandoc, Tesseract, FFmpeg, ImageMagick, Ghostscript, TeX Live, OpenJDK, Chromium, jq, and git.\n- Commands are terminated after 10 minutes. Keep them bounded, persist intermediate results to disk, and split long jobs into smaller steps.\n\n## Completing Your Work\n\nIn order to complete the task you must use the `finish` tool to submit your work. If you do not use the `finish` tool you will fail this task!\n\nAs a last resort if you really cannot make any meaningful progress, use `abandon_task_finish` with a brief reason instead of submitting files.\n\n**Required in your finish call:**\n1. A brief summary of what you accomplished\n2. A list of **ABSOLUTE file paths** for the required output files (Do not submit folders).\n\n## Task\n\nHere is the task you need to complete:\n\nTASK BODY\n\nPlease begin working on the task now."

EXPECTED_TASK_PROMPT_ONE_REFERENCE = "## Runtime\n\nYou are running in an isolated Linux sandbox. Use the `code_exec` tool to read, create, and modify files. Commands run as the non-root user `user` (UID 1000). Default working directory is `/home/user`.\n\nEvery command runs independently: no working directory, environment variable, or other shell state carries over from one call to the next. Prefer absolute paths for both files and commands, and do not navigate with `cd` across calls — a `cd` in one command is gone by the next, so relying on it leaves you silently operating in the wrong place. When a step genuinely needs a different directory, chain it into the same command (e.g. `cd /home/user/work && python build.py`).\n\nA broad scientific-computing and document-processing stack is already installed, so confirm what is present before assuming a gap:\n- Python 3.13 with the usual data stack (numpy, pandas, polars, scipy), plotting (matplotlib, plotly), the scikit-learn ML family, and document tooling (python-docx, python-pptx, openpyxl, PyMuPDF, pdfplumber, reportlab, weasyprint, Pillow, opencv), plus Playwright.\n- System tools include LibreOffice, Pandoc, Tesseract, FFmpeg, ImageMagick, Ghostscript, TeX Live, OpenJDK, Chromium, jq, and git.\n- Commands are terminated after 10 minutes. Keep them bounded, persist intermediate results to disk, and split long jobs into smaller steps.\n\n## Reference Files Location\n\nThe reference files for the task are available in your environment's file system.\n\nHere are their paths:\n\n- /home/user/input.xlsx\n\n## Completing Your Work\n\nIn order to complete the task you must use the `finish` tool to submit your work. If you do not use the `finish` tool you will fail this task!\n\nAs a last resort if you really cannot make any meaningful progress, use `abandon_task_finish` with a brief reason instead of submitting files.\n\n**Required in your finish call:**\n1. A brief summary of what you accomplished\n2. A list of **ABSOLUTE file paths** for the required output files (Do not submit folders).\n\n## Task\n\nHere is the task you need to complete:\n\nTASK BODY\n\nPlease begin working on the task now."

EXPECTED_TASK_PROMPT_MANY_REFERENCES = "## Runtime\n\nYou are running in an isolated Linux sandbox. Use the `code_exec` tool to read, create, and modify files. Commands run as the non-root user `user` (UID 1000). Default working directory is `/home/user`.\n\nEvery command runs independently: no working directory, environment variable, or other shell state carries over from one call to the next. Prefer absolute paths for both files and commands, and do not navigate with `cd` across calls — a `cd` in one command is gone by the next, so relying on it leaves you silently operating in the wrong place. When a step genuinely needs a different directory, chain it into the same command (e.g. `cd /home/user/work && python build.py`).\n\nA broad scientific-computing and document-processing stack is already installed, so confirm what is present before assuming a gap:\n- Python 3.13 with the usual data stack (numpy, pandas, polars, scipy), plotting (matplotlib, plotly), the scikit-learn ML family, and document tooling (python-docx, python-pptx, openpyxl, PyMuPDF, pdfplumber, reportlab, weasyprint, Pillow, opencv), plus Playwright.\n- System tools include LibreOffice, Pandoc, Tesseract, FFmpeg, ImageMagick, Ghostscript, TeX Live, OpenJDK, Chromium, jq, and git.\n- Commands are terminated after 10 minutes. Keep them bounded, persist intermediate results to disk, and split long jobs into smaller steps.\n\n## Reference Files Location\n\nThe reference files for the task are available in your environment's file system.\n\nHere are their paths:\n\n- /home/user/a.xlsx\n- /home/user/b.docx\n\n## Completing Your Work\n\nIn order to complete the task you must use the `finish` tool to submit your work. If you do not use the `finish` tool you will fail this task!\n\nAs a last resort if you really cannot make any meaningful progress, use `abandon_task_finish` with a brief reason instead of submitting files.\n\n**Required in your finish call:**\n1. A brief summary of what you accomplished\n2. A list of **ABSOLUTE file paths** for the required output files (Do not submit folders).\n\n## Task\n\nHere is the task you need to complete:\n\nTASK BODY\n\nPlease begin working on the task now."

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


def test_system_prompt_matches_aa_verbatim():
    assert aa.AA_GDPVAL_SYSTEM_PROMPT == EXPECTED_SYSTEM_PROMPT


def test_task_prompt_without_reference_files_omits_the_section():
    rendered = aa.render_aa_gdpval_task_prompt("TASK BODY", [])

    assert rendered == EXPECTED_TASK_PROMPT_NO_REFERENCES
    assert "## Reference Files Location" not in rendered


def test_task_prompt_with_one_reference_file():
    rendered = aa.render_aa_gdpval_task_prompt("TASK BODY", ["/home/user/input.xlsx"])

    assert rendered == EXPECTED_TASK_PROMPT_ONE_REFERENCE
    assert "- /home/user/input.xlsx" in rendered


def test_task_prompt_with_several_reference_files():
    rendered = aa.render_aa_gdpval_task_prompt("TASK BODY", ["/home/user/a.xlsx", "/home/user/b.docx"])

    assert rendered == EXPECTED_TASK_PROMPT_MANY_REFERENCES
    assert "- /home/user/a.xlsx\n- /home/user/b.docx" in rendered


def test_aa_editorial_annotation_is_not_sent_to_the_model():
    """AA marks the reference section as conditional; that note is not prompt text."""
    rendered = aa.render_aa_gdpval_task_prompt("TASK BODY", ["/home/user/a.xlsx"])

    assert "This section appears only when" not in rendered


def test_task_description_is_interpolated_verbatim():
    body = "Line one.\n\n  Indented line.\tTabbed.\n- bullet — em dash\n"

    rendered = aa.render_aa_gdpval_task_prompt(body, [])

    assert f"Here is the task you need to complete:\n\n{body}\n\nPlease begin working on the task now." in rendered


def test_prompt_whitespace_is_stable():
    rendered = aa.render_aa_gdpval_task_prompt("TASK BODY", ["/home/user/a.xlsx"])

    assert not rendered.startswith("\n")
    assert not rendered.endswith("\n")
    assert "\r" not in rendered
    assert "\n\n\n" not in rendered
    assert "\n\n" + "## Completing Your Work" in rendered
    assert aa.AA_GDPVAL_SYSTEM_PROMPT.strip() == aa.AA_GDPVAL_SYSTEM_PROMPT


def test_relative_reference_paths_are_rejected():
    with pytest.raises(ValueError, match="must be absolute"):
        aa.render_aa_gdpval_task_prompt("TASK BODY", ["input.xlsx"])


def test_prompts_never_leak_expected_deliverable_filenames():
    """Only the task text and reference paths are interpolated."""
    rendered = aa.render_aa_gdpval_task_prompt("Produce a report.", ["/home/user/input.xlsx"])

    assert "expert" not in rendered.lower()
    assert "deliverable_files" not in rendered


# ---------------------------------------------------------------------------
# Runtime contract
# ---------------------------------------------------------------------------


def test_runtime_constants_match_published_limits():
    assert aa.AA_MAX_TURNS == 250
    assert aa.AA_SHELL_TIMEOUT_SEC == 600
    assert aa.AA_CONTEXT_SUMMARIZATION_CUTOFF == 0.7
    assert aa.AA_WEB_SEARCH_RESULTS == 5
    assert aa.AA_MAX_IMAGE_PIXELS == 1_000_000
    assert (aa.AA_AGENT_USER, aa.AA_AGENT_UID) == ("user", 1000)
    assert aa.AA_WORKDIR == "/home/user"


# ---------------------------------------------------------------------------
# Package manifests
# ---------------------------------------------------------------------------


def test_published_manifest_sizes():
    assert len(aa.python_package_pins()) == 419
    assert len(aa.system_package_pins()) == 762


@pytest.mark.parametrize("pins", [aa.python_package_pins(), aa.system_package_pins()])
def test_every_package_is_pinned_exactly_once(pins):
    names = [name for name, _ in pins]

    assert len(set(names)) == len(names)
    assert all(name and version for name, version in pins)


def test_manifests_cover_the_stack_the_prompt_promises():
    python_pins = dict(aa.python_package_pins())
    system_pins = dict(aa.system_package_pins())

    # Named explicitly in AA's task prompt.
    for name in [
        "numpy",
        "pandas",
        "polars",
        "scipy",
        "matplotlib",
        "plotly",
        "scikit-learn",
        "python-docx",
        "python-pptx",
        "openpyxl",
        "PyMuPDF",
        "pdfplumber",
        "reportlab",
        "weasyprint",
        "playwright",
    ]:
        assert name in python_pins, name
    for name in ["libreoffice-core", "pandoc", "tesseract-ocr", "ffmpeg", "imagemagick", "ghostscript", "texlive-latex-extra", "chromium", "jq", "git"]:
        assert name in system_pins, name
    assert system_pins["python3.13"].startswith("3.13")


def test_requirements_rendering_round_trips():
    lines = aa.python_requirements().splitlines()

    assert len(lines) == 419
    assert lines[0] == "{}=={}".format(*aa.python_package_pins()[0])
    assert aa.python_requirements().endswith("\n")


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def test_dockerfile_pins_every_layer():
    dockerfile = aa.render_dockerfile()

    assert f"FROM --platform={aa.AA_PLATFORM} {aa.AA_BASE_IMAGE}@{aa.AA_BASE_IMAGE_DIGEST}" in dockerfile
    assert aa.AA_BASE_IMAGE_DIGEST.startswith("sha256:")
    assert aa.AA_PLATFORM == "linux/amd64"
    sources = aa.apt_sources_list()
    assert f"snapshot.debian.org/archive/debian/{aa.AA_DEBIAN_SNAPSHOT}/ {aa.AA_DEBIAN_SUITE} main" in sources
    assert f"snapshot.debian.org/archive/debian-security/{aa.AA_DEBIAN_SNAPSHOT}/ {aa.AA_DEBIAN_SUITE}-security main" in sources
    # A rolling suite would silently drift off the published version set.
    assert "deb.debian.org" not in sources
    # The base image has no ca-certificates until this very apt run installs
    # it, so https would fail its own certificate check on the first fetch.
    # Signature verification against the archive keyring is what secures this.
    assert "https://snapshot.debian.org" not in sources


def test_dockerfile_creates_the_solver_identity_and_workdir():
    dockerfile = aa.render_dockerfile()

    assert f"--uid {aa.AA_AGENT_UID} --gid {aa.AA_AGENT_GID}" in dockerfile
    assert f"WORKDIR {aa.AA_WORKDIR}" in dockerfile
    # The image must stay root so rLLM can stage files and mount the verifier;
    # the solver is dropped to `user` per-task via task.toml.
    assert "\nUSER " not in dockerfile


def test_dockerfile_verifies_the_manifest_at_build_time():
    dockerfile = aa.render_dockerfile()

    assert "verify_environment.py" in dockerfile
    assert "--system-manifest" in dockerfile and "--python-manifest" in dockerfile
    # Versions come from the manifest, never from a resolver's own choice.
    assert "uv pip install --requirement" in dockerfile


def test_build_toolchain_does_not_survive_into_the_image():
    """AA's closure has no compiler, but some pins have no cp313 wheel."""
    dockerfile = aa.render_dockerfile()

    assert "build-essential" in dockerfile
    assert "apt-get purge -y $(cat /opt/gdpval-aa/packages-build-only.txt)" in dockerfile
    # The removal is the recorded delta, so autoremove cannot take a package
    # the manifest requires.
    assert "comm -13 /opt/gdpval-aa/packages-before-build.txt /opt/gdpval-aa/packages-after-build.txt" in dockerfile

    commands = aa.dockerfile_run_commands()
    installed = next(i for i, c in enumerate(commands) if "build-essential" in c and "apt-get install" in c)
    built = next(i for i, c in enumerate(commands) if "uv pip install --requirement" in c)
    purged = next(i for i, c in enumerate(commands) if "apt-get purge" in c)
    verified = next(i for i, c in enumerate(commands) if "verify_environment.py --system-manifest" in c)
    assert installed < built < purged < verified


def test_dockerfile_is_replayable_on_non_docker_backends():
    """Modal/Daytona replay only single-line RUN steps and skip COPY."""
    dockerfile = aa.render_dockerfile()

    assert "COPY " not in dockerfile
    for line in dockerfile.splitlines():
        if line.startswith("RUN "):
            assert not line.rstrip().endswith("\\"), "RUN steps must not use line continuations"
    assert len(aa.dockerfile_run_commands()) == sum(1 for line in dockerfile.splitlines() if line.startswith("RUN "))
