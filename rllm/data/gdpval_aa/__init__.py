"""Artificial Analysis' published GDPval-AA v2 specification.

This package holds the parts of the GDPval-AA v2 methodology that must be
reproduced byte-for-byte rather than approximated:

* ``system_prompt.txt`` / ``task_prompt_template.txt`` — AA's published solver
  prompts, transcribed verbatim.  The only edits are the two interpolation
  points AA documents (the reference-file paths and the original GDPval task
  description) and the removal of AA's own editorial note marking the
  reference-file section as conditional.
* ``python_packages.txt`` / ``system_packages.txt`` — the 419 pinned Python
  packages and 762 pinned Debian packages AA publishes for the v2 sandbox.

Everything here describes AA's environment; nothing here judges output.  The
resulting image is *AA-compatible*: AA publishes the installed package closure
rather than its E2B build recipe, so the package set is reproduced exactly
while the build steps that produce it are our own.

Source: https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdpval-aa
"""

from __future__ import annotations

import base64
import hashlib
import os
import shlex
from functools import lru_cache
from importlib.resources import files

_DATA = files("rllm.data.gdpval_aa")

# --------------------------------------------------------------------------
# Runtime contract (AA "Task Submission Details" / "Execution Limits")
# --------------------------------------------------------------------------

#: Turns allowed per task. One turn is an assistant message plus its tool calls.
AA_MAX_TURNS = 250
#: Individual shell commands are terminated after ten minutes.
AA_SHELL_TIMEOUT_SEC = 600
#: Fraction of the context window that triggers a summarization request.
AA_CONTEXT_SUMMARIZATION_CUTOFF = 0.7
#: Brave Web Search returns the top five results.
AA_WEB_SEARCH_RESULTS = 5
#: Images are downscaled to at most one megapixel before reaching the model.
AA_MAX_IMAGE_PIXELS = 1_000_000

#: Solver identity and default working directory inside the sandbox.
AA_AGENT_USER = "user"
AA_AGENT_UID = 1000
AA_AGENT_GID = 1000
AA_WORKDIR = "/home/user"

#: Roots the solver may submit files from. Everything else is rejected.
AA_SUBMITTABLE_ROOTS = (AA_WORKDIR, "/tmp")

# --------------------------------------------------------------------------
# Image provenance
# --------------------------------------------------------------------------

#: AA's sandbox is amd64: ``aspose-words`` and ``cadquery-ocp`` publish
#: x86_64-only wheels with no sdist, and 11 pinned Debian packages have no
#: arm64 build. Building for another platform cannot reproduce the manifest.
AA_PLATFORM = "linux/amd64"
AA_BASE_IMAGE = "debian:trixie-20260518-slim"
AA_BASE_IMAGE_DIGEST = "sha256:b6e2a152f22a40ff69d92cb397223c906017e1391a73c952b588e51af8883bf8"
#: Dated Debian snapshot that still carries every pinned version. Verified by
#: ``tests/data/test_gdpval_aa_spec.py`` against the published manifest.
AA_DEBIAN_SNAPSHOT = "20260528T000000Z"
AA_DEBIAN_SUITE = "trixie"
AA_PYTHON_VERSION = "3.13"
#: The Python manifest is the closure of a clean virtualenv, not of Debian's
#: ``python3-*`` packages (which pin different versions of numpy and friends).
AA_VENV_DIR = "/opt/aa-venv"

#: ``pip`` and ``wheel`` bootstrap a virtualenv and are not part of AA's
#: published freeze. Everything else must match exactly.
AA_PIP_BOOTSTRAP_PACKAGES = frozenset({"pip", "wheel"})


#: Published sandbox image. Tasks reference this instead of rebuilding the
#: closure, which is how every other rLLM benchmark works: the heavy image is
#: prebuilt and pulled, and the task's Dockerfile is a thin wrapper.
#:
#: The tag is for humans; the digest is the pin. Tags move on re-push, so a task
#: that referenced only ``:aa-v2`` would silently change environment without any
#: change in the repo. ``RLLM_GDPVAL_IMAGE`` overrides both, for a private
#: mirror or a locally built image.
AA_PUBLISHED_IMAGE = "ghcr.io/rllm-org/gdpval-aa-compat:aa-v2"
AA_PUBLISHED_IMAGE_DIGEST = "sha256:44a31f31f416bfe0292b8ce07181a04d4aec3942c8ff5366148bee34a5237a75"


def published_image_ref() -> str | None:
    """Pinned reference for the published sandbox image, or ``None``.

    ``None`` means no published image is available, so the task must build the
    closure from :func:`render_dockerfile` instead.
    """
    override = os.environ.get("RLLM_GDPVAL_IMAGE")
    if override:
        return override
    if not AA_PUBLISHED_IMAGE_DIGEST:
        return None
    return f"{AA_PUBLISHED_IMAGE}@{AA_PUBLISHED_IMAGE_DIGEST}"


def render_task_dockerfile() -> str:
    """Thin wrapper over the published image, for a task's ``environment/``.

    Mirrors what swebench_pro and friends emit: ``FROM`` a prebuilt image plus a
    workdir, no ``RUN`` steps. :func:`render_dockerfile` remains the recipe the
    published image is *built* from, with its manifest check as the release gate.
    """
    ref = published_image_ref()
    if ref is None:
        raise RuntimeError("no published GDPval image is pinned; set AA_PUBLISHED_IMAGE_DIGEST or RLLM_GDPVAL_IMAGE")
    return "\n".join(
        [
            "# GDPval-AA v2 compatible sandbox — pulled, not built.",
            "#",
            "# The environment is the published image below; see",
            "# rllm.data.gdpval_aa.render_dockerfile for the recipe it was built",
            "# from, which verifies the closure against AA's published manifests.",
            "#",
            # The published manifest is an amd64-only index, and AA's closure has
            # no arm64 equivalent (aspose-words and cadquery-ocp ship x86_64
            # wheels only). Without --platform, Docker on an arm64 host resolves
            # the index against its own arch and fails with "no match for
            # platform in manifest" instead of emulating.
            f"FROM --platform={AA_PLATFORM} {ref}",
            f"WORKDIR {AA_WORKDIR}",
            'CMD ["sleep", "infinity"]',
            "",
        ]
    )


def _read(name: str) -> str:
    return (_DATA / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

#: AA's published solver system prompt, verbatim.
AA_GDPVAL_SYSTEM_PROMPT = _read("system_prompt.txt")

_TASK_PROMPT_TEMPLATE = _read("task_prompt_template.txt")
_REFERENCE_FILES_SECTION_TEMPLATE = _read("reference_files_section_template.txt")


def render_aa_gdpval_task_prompt(task_description: str, reference_paths: list[str] | tuple[str, ...] = ()) -> str:
    """Render AA's published task prompt for one GDPval task.

    ``task_description`` is the original GDPval prompt, interpolated verbatim.
    ``reference_paths`` must be absolute paths inside the sandbox; when empty,
    the whole reference-file section is omitted, exactly as AA documents.
    """
    paths = list(reference_paths)
    relative = [path for path in paths if not path.startswith("/")]
    if relative:
        raise ValueError(f"reference paths must be absolute: {relative}")

    if paths:
        section = _REFERENCE_FILES_SECTION_TEMPLATE.format(reference_file_lines="\n".join(f"- {path}" for path in paths))
    else:
        section = ""

    return _TASK_PROMPT_TEMPLATE.format(reference_files_section=section, task_description=task_description)


def sha256_text(text: str) -> str:
    """Hex SHA-256 of *text* encoded as UTF-8 (used for prompt provenance)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Package manifests
# --------------------------------------------------------------------------


def _pins(name: str, separator: str) -> tuple[tuple[str, str], ...]:
    entries = []
    for line in _read(name).splitlines():
        line = line.strip()
        if not line:
            continue
        package, found, version = line.partition(separator)
        if not found:
            raise ValueError(f"malformed pin in {name}: {line!r}")
        entries.append((package, version))
    return tuple(entries)


@lru_cache(maxsize=1)
def python_package_pins() -> tuple[tuple[str, str], ...]:
    """AA's 419 pinned Python packages as ``(name, version)`` pairs."""
    return _pins("python_packages.txt", "==")


@lru_cache(maxsize=1)
def system_package_pins() -> tuple[tuple[str, str], ...]:
    """AA's 762 pinned Debian packages as ``(name, version)`` pairs."""
    return _pins("system_packages.txt", "=")


def python_requirements() -> str:
    """The Python manifest as a ``requirements.txt`` body."""
    return "".join(f"{name}=={version}\n" for name, version in python_package_pins())


def system_package_specs() -> str:
    """The Debian manifest as newline-separated ``name=version`` apt specs."""
    return "".join(f"{name}={version}\n" for name, version in system_package_pins())


def normalize_python_package_name(name: str) -> str:
    """Normalize a distribution name for comparison (PEP 503)."""
    return name.lower().replace("_", "-").replace(".", "-")


# --------------------------------------------------------------------------
# Image definition
# --------------------------------------------------------------------------

#: uv provisions the solver virtualenv. AA's Debian manifest carries no
#: ``python3-venv``/``python3-pip``, so the Python closure cannot come from
#: Debian; uv installs outside dpkg and therefore leaves the manifest exact.
AA_UV_VERSION = "0.9.24"

_MANIFEST_DIR = "/opt/gdpval-aa"
_SYSTEM_MANIFEST_PATH = f"{_MANIFEST_DIR}/system_packages.txt"
_PYTHON_MANIFEST_PATH = f"{_MANIFEST_DIR}/python_packages.txt"
_VERIFY_SCRIPT_PATH = f"{_MANIFEST_DIR}/verify_environment.py"
_PACKAGES_BEFORE_PATH = f"{_MANIFEST_DIR}/packages-before-build.txt"
_PACKAGES_AFTER_PATH = f"{_MANIFEST_DIR}/packages-after-build.txt"
_BUILD_ONLY_PACKAGES_PATH = f"{_MANIFEST_DIR}/packages-build-only.txt"

#: Toolchain needed to compile the pins that publish no cp313 wheel. Removed
#: again once the venv exists, so it never appears in the final closure.
_BUILD_PACKAGES = "build-essential python3.13-dev libcairo2-dev pkg-config"

#: These are ``http``, not ``https``, on purpose. ``debian:trixie-slim`` ships
#: without ``ca-certificates`` — that package is installed *by* this apt run —
#: so an https source cannot verify its own certificate on the very first
#: fetch. Authenticity does not depend on TLS: apt verifies the InRelease
#: signature against the bundled Debian archive keyring, every package hash
#: comes from that signed index, and each version is pinned besides.
_APT_SOURCES = "\n".join(
    [
        f"deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/{AA_DEBIAN_SNAPSHOT}/ {AA_DEBIAN_SUITE} main",
        f"deb [check-valid-until=no] http://snapshot.debian.org/archive/debian/{AA_DEBIAN_SNAPSHOT}/ {AA_DEBIAN_SUITE}-updates main",
        f"deb [check-valid-until=no] http://snapshot.debian.org/archive/debian-security/{AA_DEBIAN_SNAPSHOT}/ {AA_DEBIAN_SUITE}-security main",
        "",
    ]
)

_APT_CONF = "\n".join(
    [
        'Acquire::Check-Valid-Until "false";',
        'Acquire::Retries "10";',
        'APT::Install-Recommends "false";',
        'APT::Install-Suggests "false";',
        "",
    ]
)


def apt_sources_list() -> str:
    """The dated-snapshot ``sources.list`` the image installs packages from.

    Exposed separately because it is base64-encoded into a ``RUN`` step, so it
    is not greppable in the rendered Dockerfile.
    """
    return _APT_SOURCES


def _write_file_command(path: str, content: str) -> str:
    """A single-line shell command that writes *content* to *path*.

    Deliberately avoids heredocs and ``COPY``. Non-docker backends replay the
    Dockerfile by extracting ``RUN`` lines and joining backslash continuations
    with spaces (``rllm.eval._resolution._dockerfile_run_commands``), which
    destroys any embedded newline and skips ``COPY`` entirely. Base64 on one
    line survives both ``docker build`` and that replay path unchanged.
    """
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    parent = shlex.quote(path.rsplit("/", 1)[0] or "/")
    quoted = shlex.quote(path)
    return f"mkdir -p {parent} && printf %s {encoded} | base64 -d > {quoted}"


def dockerfile_run_commands() -> list[str]:
    """The image's build steps, in order, each a single-line shell command."""
    packages = " ".join(f"{name}={version}" for name, version in system_package_pins())
    return [
        # Rolling trixie moves within weeks; the published pins only resolve
        # against a dated snapshot.
        f"rm -f /etc/apt/sources.list && rm -rf /etc/apt/sources.list.d/* && {_write_file_command('/etc/apt/sources.list', _APT_SOURCES)}",
        _write_file_command("/etc/apt/apt.conf.d/99-rllm-gdpval", _APT_CONF),
        _write_file_command(_SYSTEM_MANIFEST_PATH, system_package_specs()),
        # One apt invocation: the pinned set is a full transitive closure, so
        # splitting it across calls would let apt pick its own dependencies.
        f"apt-get update && apt-get install -y --no-install-recommends --allow-downgrades --allow-change-held-packages {packages} && fc-cache -f && rm -rf /var/lib/apt/lists/*",
        f"curl -LsSf https://astral.sh/uv/{AA_UV_VERSION}/install.sh | env UV_INSTALL_DIR=/usr/local/bin INSTALLER_NO_MODIFY_PATH=1 sh && uv --version",
        _write_file_command(_PYTHON_MANIFEST_PATH, python_requirements()),
        # A toolchain is needed to *build* the venv but must not survive into
        # it: several pins (pyswisseph, pycairo) publish no cp313 wheel and are
        # compiled from source, yet AA's published closure contains no compiler
        # — so AA builds with one and drops it. Record the package set first so
        # exactly what the toolchain added can be removed again afterwards.
        (f"dpkg-query -W -f='${{Package}}\\n' | sort > {_PACKAGES_BEFORE_PATH} && apt-get update && apt-get install -y --no-install-recommends {_BUILD_PACKAGES}"),
        # UV_NO_CACHE keeps ~5.5GB of downloaded wheels out of the image — a
        # third of its size, paid on every pull. It has to be avoided *here*
        # rather than deleted later: layers are additive, so an `rm` in a
        # subsequent RUN would leave the bytes in this one.
        f"UV_NO_CACHE=1 uv venv --seed --python {AA_PYTHON_VERSION} {AA_VENV_DIR}"
        f" && UV_NO_CACHE=1 VIRTUAL_ENV={AA_VENV_DIR} uv pip install --requirement {_PYTHON_MANIFEST_PATH}"
        " && rm -rf /root/.cache/uv",
        # Purge precisely the delta, not a hand-written list: --auto-remove
        # alone would be free to take packages the manifest requires.
        (
            f"dpkg-query -W -f='${{Package}}\\n' | sort > {_PACKAGES_AFTER_PATH}"
            f" && comm -13 {_PACKAGES_BEFORE_PATH} {_PACKAGES_AFTER_PATH} > {_BUILD_ONLY_PACKAGES_PATH}"
            f" && apt-get purge -y $(cat {_BUILD_ONLY_PACKAGES_PATH})"
            " && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*"
        ),
        f"PLAYWRIGHT_BROWSERS_PATH=/opt/playwright {AA_VENV_DIR}/bin/playwright install chromium",
        # `ENV PATH` covers non-login shells and `su ... -c`, which is every
        # path the harness and code-exec provider use. A *login* shell is the
        # exception: /etc/profile resets PATH, so a model running `bash -lc`
        # would silently get Debian's python without AA's 419 packages.
        _write_file_command("/etc/profile.d/10-gdpval-aa.sh", f'export PATH="{AA_VENV_DIR}/bin:$PATH"\nexport VIRTUAL_ENV="{AA_VENV_DIR}"\n'),
        (
            f"(getent group {AA_AGENT_GID} || groupadd --gid {AA_AGENT_GID} {AA_AGENT_USER})"
            f" && (getent passwd {AA_AGENT_UID} || useradd --uid {AA_AGENT_UID} --gid {AA_AGENT_GID} --create-home --shell /bin/bash {AA_AGENT_USER})"
            f" && mkdir -p {AA_WORKDIR} && chown -R {AA_AGENT_UID}:{AA_AGENT_GID} {AA_WORKDIR} && chmod 1777 /tmp"
        ),
        _write_file_command(_VERIFY_SCRIPT_PATH, _read("verify_environment.py")),
        f"{AA_VENV_DIR}/bin/python {_VERIFY_SCRIPT_PATH} --system-manifest {_SYSTEM_MANIFEST_PATH} --python-manifest {_PYTHON_MANIFEST_PATH}",
    ]


def render_dockerfile() -> str:
    """Render the AA-compatible GDPval sandbox Dockerfile.

    Every layer is pinned: the base image by digest, Debian packages by a dated
    snapshot plus explicit versions, Python packages by the published freeze.
    The build fails if the resulting closure does not match AA's manifest.
    """
    lines = [
        "# GDPval-AA v2 compatible sandbox.",
        "#",
        "# Reproduces the package closure Artificial Analysis publishes for",
        "# GDPval-AA v2. AA discloses the closure rather than its E2B build",
        "# recipe, so this image is AA-compatible, not byte-identical.",
        "#",
        "# Generated by rllm.data.gdpval_aa.render_dockerfile — edit that, not this.",
        f"# Platform is fixed to {AA_PLATFORM}: aspose-words and cadquery-ocp publish",
        "# x86_64-only wheels with no sdist, and 11 pinned Debian packages have no",
        "# arm64 build, so no other architecture can satisfy the manifest.",
        f"FROM --platform={AA_PLATFORM} {AA_BASE_IMAGE}@{AA_BASE_IMAGE_DIGEST}",
        "",
        # AA publishes no image and no build recipe — only the package closure.
        # Say so on the artifact, so a published copy cannot be mistaken for an
        # Artificial Analysis release.
        'LABEL org.opencontainers.image.title="GDPval-AA v2 compatible sandbox (unofficial reproduction)"',
        'LABEL org.opencontainers.image.description="Independent reproduction of the package closure Artificial Analysis publishes for GDPval-AA v2. AA publishes no image and no build recipe: only the pinned Python and Debian versions. The build steps here are rLLM\'s own, verified against those manifests at build time. Not an Artificial Analysis artifact."',
        'LABEL org.opencontainers.image.source="https://github.com/rllm-org/rllm"',
        'LABEL ai.artificialanalysis.methodology="GDPval-AA v2"',
        'LABEL ai.artificialanalysis.methodology-url="https://artificialanalysis.ai/methodology/intelligence-benchmarking#gdpval-aa"',
        f'LABEL org.rllm.gdpval.debian-packages="{len(system_package_pins())}"',
        f'LABEL org.rllm.gdpval.python-packages="{len(python_package_pins())}"',
        f'LABEL org.rllm.gdpval.debian-snapshot="{AA_DEBIAN_SNAPSHOT}"',
        "",
        "ENV DEBIAN_FRONTEND=noninteractive",
        "ENV LANG=C.UTF-8",
        "ENV LC_ALL=C.UTF-8",
        f"ENV VIRTUAL_ENV={AA_VENV_DIR}",
        f"ENV PATH={AA_VENV_DIR}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "ENV PLAYWRIGHT_BROWSERS_PATH=/opt/playwright",
        "",
    ]
    for command in dockerfile_run_commands():
        lines.append(f"RUN {command}")
        lines.append("")
    lines += [
        f"WORKDIR {AA_WORKDIR}",
        "",
        "# The image stays root so rLLM can stage reference files, chown the",
        "# workdir, and mount the verifier. The solver itself runs as",
        f"# {AA_AGENT_USER} (UID {AA_AGENT_UID}) via task.toml's [agent] user.",
        'CMD ["sleep", "infinity"]',
        "",
    ]
    return "\n".join(lines)
