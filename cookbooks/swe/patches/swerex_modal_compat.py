"""SwerexModalEnvironment patches for correct env-var injection and cleanup.

Upstream mini-swe-agent v2.2+ added v2 protocol support (dict actions,
serialize, Submitted handling) but still passes env vars via
``RexCommand(env=dict)`` which calls ``subprocess.run(env=dict)``.
This **replaces** the container's entire environment, stripping
GOPATH, CARGO_HOME, PATH additions, etc. from the Docker image.

This patch overrides ``execute()`` to inject env vars via ``export``
prefix with ``env=None``, inheriting the container's full environment
and adding our vars on top.

Also adds ``close()`` as alias for ``stop()`` (used by some code paths).
"""

from __future__ import annotations

import asyncio
import shlex
from typing import Any


def apply_swerex_modal_compat_patch():
    """Patch SwerexModalEnvironment for correct env-var handling."""
    from minisweagent.environments.extra.swerex_modal import SwerexModalEnvironment
    from minisweagent.exceptions import Submitted
    from swerex.exceptions import CommandTimeoutError
    from swerex.runtime.abstract import Command as RexCommand

    if getattr(SwerexModalEnvironment, "_compat_patch_applied", False):
        return

    # Save upstream execute for delegation when no env vars are configured.
    _upstream_execute = SwerexModalEnvironment.execute

    def _execute_with_env_exports(self: Any, action: Any, cwd: str = "", *, timeout: int | None = None):
        """Override execute() to inject env vars via shell export prefix.

        When ``self.config.env`` is non-empty, prepend ``export K=V && ...``
        and pass ``env=None`` to ``subprocess.run()`` so the container's
        native environment (GOPATH, CARGO_HOME, PATH, etc.) is preserved.

        When ``self.config.env`` is empty, delegate to upstream unchanged.
        """
        env_vars = self.config.env
        if not env_vars:
            return _upstream_execute(self, action, cwd=cwd, timeout=timeout)

        command = action.get("command", "") if isinstance(action, dict) else str(action)

        exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_vars.items())
        command = f"export {exports} && {command}"

        effective_timeout = timeout or self.config.timeout
        try:
            result = asyncio.run(
                self.deployment.runtime.execute(
                    RexCommand(
                        command=command,
                        shell=True,
                        check=False,
                        cwd=cwd or self.config.cwd,
                        timeout=effective_timeout,
                        merge_output_streams=True,
                        env=None,  # Inherit container's full environment
                    )
                )
            )
            output = {
                "output": result.stdout,
                "returncode": result.exit_code,
                "exception_info": "",
            }
        except CommandTimeoutError:
            output = {
                "output": f"Command timed out after {effective_timeout}s. Try a faster command or add a timeout.",
                "returncode": 124,
                "exception_info": "",
            }
        except Exception as e:
            output = {
                "output": str(e) if str(e) else "",
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
            }

        # Handle completion marker (same logic as upstream _check_finished)
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output.get("returncode") == 0:
            submission = "".join(lines[1:])
            raise Submitted({
                "role": "exit",
                "content": submission,
                "extra": {"exit_status": "Submitted", "submission": submission},
            })

        return output

    def _close(self: Any):
        return self.stop()

    SwerexModalEnvironment.execute = _execute_with_env_exports  # type: ignore
    SwerexModalEnvironment.close = _close  # type: ignore
    SwerexModalEnvironment._compat_patch_applied = True  # type: ignore
