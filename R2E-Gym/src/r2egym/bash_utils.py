import subprocess
from logging import Logger
from subprocess import CompletedProcess


def run_subprocess_shell(
    command: str,
    cwd: str,
    logger: Logger | None = None,
    capture_output=True,
    timeout: int = 120,
    **kwargs,
) -> CompletedProcess[str]:
    try:
        result = subprocess.run(
            command,
            executable="/bin/bash",
            shell=True,
            capture_output=capture_output,
            text=True,
            cwd=cwd,
            **kwargs,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        if logger:
            logger.error(f"Timeout expired for {command}")
        result = CompletedProcess(
            args=command,
            returncode=1,
            stderr="Timeout expired",
            stdout="Timeout",
        )
    except subprocess.CalledProcessError as e:
        if logger:
            logger.error(f"CalledProcessError running {command} -- {e}")
        result = CompletedProcess(
            args=command,
            returncode=1,
        )
    except Exception as e:
        if logger:
            logger.error(f"Error running {command} -- {e}")
        result = CompletedProcess(
            args=command,
            returncode=1,
            stdout="",
            stderr=str(e),
        )
    return result
