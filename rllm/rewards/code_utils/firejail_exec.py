# https://github.com/ganler/code-r1/blob/main/verl/utils/reward_score/coder1/firejail_exec.py
import os
import subprocess

from tempfile import NamedTemporaryFile, TemporaryDirectory
from .utils import BASE_IMPORTS, BASE_LEETCODE_IMPORTS


# sudo add-apt-repository ppa:deki/firejail
# sudo apt-get update
# sudo apt-get install firejail firejail-profiles

CLI_ARG_SIZE_LIMIT = 1024 * 3

_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 30


def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = os.environ.copy()
    env["OPENBLAS_NUM_THREADS"] = "1"

    # Build the firejail command with resource limits and cleanup options
    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=2097152",  # Limit file size
        "--rlimit-as=4294967296",
        f"--timeout=00:00:{timeout}",
    ]

    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["python3", "-m", "pytest", tmpdir])
            result = subprocess.run(
                command,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                check=False,
            )
    else:
        code = BASE_IMPORTS + "\n" + BASE_LEETCODE_IMPORTS + "\n" + code
        if len(code) < CLI_ARG_SIZE_LIMIT:
            command.extend(["python3", "-c", code])
            result = subprocess.run(command,
                                    input=stdin.encode() if stdin else None,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env,
                                    check=False)
        else:
            with NamedTemporaryFile(dir="/tmp", suffix=".py") as tmp:
                tmp.write(code.encode())
                tmp.flush()
                command.insert(4, f"--whitelist={tmp.name}")
                command.extend(["python3", tmp.name])
                result = subprocess.run(command,
                                        input=stdin.encode() if stdin else None,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE,
                                        env=env,
                                        check=False)

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()
    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"