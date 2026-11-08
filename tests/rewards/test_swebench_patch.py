import shlex
import subprocess
from pathlib import Path

from rllm.rewards.code_utils.patch_utils import GIT_APPLY_CMDS


def _run(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True)


def test_patch_fallback_does_not_reverse_applied_changes(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.email", "test@example.com"], repo)
    _run(["git", "config", "user.name", "test"], repo)

    (repo / "code.py").write_text("def answer():\n    return 1  # bug\n")
    (repo / "tox.ini").write_text("[tox]\nenvlist = py\n\n[testenv]\ncommands = pytest\n")
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-qm", "base"], repo)

    (repo / "code.py").write_text("def answer():\n    return 2  # fixed\n")
    (repo / "tox.ini").write_text("[tox]\nenvlist = py\n\n[testenv]\ncommands = pytest -x\n")
    patch = tmp_path / "model.patch"
    patch.write_text(_run(["git", "diff"], repo).stdout)
    _run(["git", "restore", "."], repo)
    (repo / "tox.ini").write_text("[tox]\nenvlist = py311\n\n[testenv]\ncommands = pytest\n")

    attempts = []
    for command in GIT_APPLY_CMDS:
        result = _run([*shlex.split(command), str(patch)], repo, check=False)
        attempts.append((command, result.returncode, result.stdout + result.stderr))
        if result.returncode == 0:
            break

    assert attempts[-1][1] == 0, attempts
    assert "return 2  # fixed" in (repo / "code.py").read_text(), attempts
    assert "envlist = py311" in (repo / "tox.ini").read_text()
    assert "commands = pytest -x" in (repo / "tox.ini").read_text()
