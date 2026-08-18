"""GLM-5.2 reproduction profile for the llm-stats SWE-bench Pro result.

llm-stats reports a 0.621 resolve rate and Z.ai discloses OpenHands, a
tailored prompt, temperature 1, top-p 1, 32K output tokens, and a 400K context
window. The exact private prompt and original task-level traces are unavailable,
so this profile records the remaining provenance gap instead of claiming exact
parity.
"""

# ruff: noqa: E501 -- keep the public OpenHands prompt readable and auditable.

from __future__ import annotations

from rllm.harnesses.openhands import (
    OPENHANDS_EXTENSIONS_REVISION,
    OPENHANDS_LIBTMUX_VERSION,
    OPENHANDS_LOCALE,
    OPENHANDS_SDK_REVISION,
    OpenHandsHarness,
)
from rllm.types import AgentConfig, Episode, Task, TerminationReason

LLM_STATS_REFERENCE_URL = "https://llm-stats.com/benchmarks/swe-bench-pro"
LLM_STATS_TARGET_SCORE = 0.621
TARGET_MODEL = "GLM-5.2"
OPENROUTER_MODEL_ID = "z-ai/glm-5.2"
FIREWORKS_MODEL_ID = "accounts/fireworks/models/glm-5p2"

# Public OpenHands SWE-bench Pro baseline inspected on 2026-08-08. The generic
# harness owns the SDK/runtime pins; this revision identifies only the public
# benchmark prompt used as the closest available substitute for Z.ai's private
# tailored prompt.
OPENHANDS_BENCHMARKS_REVISION = "1411fe96666e2c00b958cd30055ad232e2a64ca1"

TEMPERATURE = 1.0
TOP_P = 1.0
MAX_OUTPUT_TOKENS = 32_768
CONTEXT_WINDOW_TOKENS = 400_000
MAX_ITERATIONS = 500

REPRODUCTION_PROFILE = {
    "status": "best_effort",
    "score_source": LLM_STATS_REFERENCE_URL,
    "target_model": TARGET_MODEL,
    "target_score": LLM_STATS_TARGET_SCORE,
    "score_status": "self_reported_unverified",
    "openhands_benchmarks_revision": OPENHANDS_BENCHMARKS_REVISION,
    "openhands_sdk_revision": OPENHANDS_SDK_REVISION,
    "openhands_extensions_revision": OPENHANDS_EXTENSIONS_REVISION,
    "libtmux_version": OPENHANDS_LIBTMUX_VERSION,
    "locale": OPENHANDS_LOCALE,
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "context_window_tokens": CONTEXT_WINDOW_TOKENS,
    "max_iterations": MAX_ITERATIONS,
    "prompt_provenance": "public OpenHands SWE-bench Pro baseline; Z.ai tailored prompt undisclosed",
    "prompt_input": "rLLM task instruction wrapped by the public OpenHands prompt",
    "workspace": "official task image /app exposed through OpenHands local workspace",
}


def _render_prompt(instruction: str, workdir: str, base_commit: str) -> str:
    """Render the pinned public OpenHands SWE-bench Pro prompt baseline."""
    return f"""I have access to a code repository in the directory {workdir} . You can explore and modify files using the available tools. Consider the following issue description:

<issue_description>
{instruction}
</issue_description>

Can you help me implement the necessary changes to the repository so that the requirements specified in the <issue_description> are met?
I've already taken care of all changes to any of the test files described in the <issue_description>. This means you DON'T have to modify the testing logic or any of the tests in any way.
The benchmark image already includes the repository and its baseline dependencies, so prefer using the existing environment and only install missing dependencies when the repository clearly requires it.
Your task is to make the minimal changes to non-test files in the {workdir} directory to ensure the <issue_description> is satisfied.

Follow these phases to resolve the issue:

Phase 1. READING: read the problem and reword it in clearer terms
   1.1 If there are code or config snippets, explain any best practices or conventions they imply.
   1.2 Highlight error messages, method names, variables, file names, stack traces, and technical details.
   1.3 Explain the problem in clear terms.
   1.4 Enumerate the steps to reproduce the problem.
   1.5 Highlight any best practices to take into account when testing and fixing the issue.

Phase 2. RUNNING: understand how the repository is built and tested
   2.1 Read the repository docs and relevant config files to understand the expected workflow.
   2.2 Identify the project language, package manager, test runner, and any required services.
   2.3 Run the most relevant tests or reproduction steps for this issue.

Phase 3. EXPLORATION: find the files that are related to the problem and possible solutions
   3.1 Use search tools to locate relevant methods, classes, keywords, and error messages.
   3.2 Identify all files related to the problem statement.
   3.3 Propose the most likely files and functions to change, and explain why.
   3.4 Select the best fix location before editing.

Phase 4. TEST CREATION: before implementing any fix, create a script or command sequence to reproduce and verify the issue.
   4.1 Look at existing tests to understand the expected style and structure.
   4.2 Create a minimal reproduction that demonstrates the issue.
   4.3 Run it to confirm you are reproducing the problem.
   4.4 Refine it as needed.

Phase 5. FIX ANALYSIS: state clearly the problem and how to fix it
   5.1 State clearly what the problem is.
   5.2 State clearly where the problem is located.
   5.3 State clearly how the reproduction proves the issue.
   5.4 State clearly any best practices to preserve in the fix.
   5.5 State clearly how you will fix the problem.

Phase 6. FIX IMPLEMENTATION: edit the source code to implement your chosen solution.
   6.1 Make minimal, focused changes to fix the issue.

Phase 7. VERIFICATION: test your implementation thoroughly.
   7.1 Re-run your reproduction to verify the fix works.
   7.2 Add edge cases when useful.
   7.3 Run existing tests related to the modified code to ensure you have not broken anything else.

Phase 8. FINAL REVIEW: carefully re-read the problem description and compare your changes with the base commit {base_commit}.
   8.1 Ensure you've fully addressed all requirements.
   8.2 Run any relevant tests for the issue, the files you modified, and the functions you changed.
   8.3 If any tests fail, revise your implementation until all relevant tests pass.

Be thorough in your exploration, testing, and reasoning. It is fine if your thinking process is lengthy: quality and completeness are more important than brevity.
"""


class SwebenchProOpenHandsGLM52Harness(OpenHandsHarness):
    """Apply the public GLM-5.2 reproduction settings to OpenHands."""

    name = "swebench-pro-openhands-glm52"
    stdout_log_path = "/tmp/swebench-pro-openhands-glm52.log"

    temperature = TEMPERATURE
    top_p = TOP_P
    max_input_tokens = CONTEXT_WINDOW_TOKENS
    max_output_tokens = MAX_OUTPUT_TOKENS
    max_iterations = MAX_ITERATIONS
    target_score = LLM_STATS_TARGET_SCORE

    def build_env(self, task: Task, config: AgentConfig) -> dict[str, str]:
        model = config.model.lower()
        if "glm-5.2" not in model and model != FIREWORKS_MODEL_ID:
            raise ValueError(f"{self.name} requires a GLM-5.2 model, got {config.model!r}")
        return super().build_env(task, config)

    def render_prompt(self, task: Task, workdir: str) -> str:
        metadata = task.metadata.get("metadata", {}) or {}
        base_commit = str(task.metadata.get("base_commit") or metadata.get("base_commit") or "the task base commit")
        return _render_prompt(str(task.instruction).strip(), workdir, base_commit)

    def _outcome_episode(
        self,
        task: Task,
        termination_reason: TerminationReason | None = None,
        error: dict | None = None,
    ) -> Episode:
        episode = super()._outcome_episode(task, termination_reason, error)
        episode.metadata = dict(episode.metadata or {})
        episode.metadata["reproduction_profile"] = dict(REPRODUCTION_PROFILE)
        return episode
