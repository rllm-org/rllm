# MCP-Atlas evaluation

This integration runs the public 500-task MCP-Atlas release through Scale's pinned TypeScript agent harness and Python claim scorer. It pins source commit `ab35dcd10cf94985d709265927eec951f5d9faa0`, image `ghcr.io/scaleapi/mcp-atlas:1.2.7`, dataset revision `b5bcde2`, and Gemini 2.5 Pro judging at temperature 0. The judge defaults to OpenRouter at `https://openrouter.ai/api/v1` using model `openrouter/google/gemini-2.5-pro`.

Install the integration dependencies and configure an inference provider:

```bash
uv sync --extra mcp-atlas
rllm model setup
export OPENROUTER_API_KEY=...
```

The same OpenRouter key can serve the evaluated model and the Gemini judge. To
use Gemini directly instead, set `GEMINI_API_KEY` without any
`MCP_ATLAS_JUDGE_*` or `OPENROUTER_API_KEY` variables. Advanced overrides are
`MCP_ATLAS_JUDGE_MODEL`, `MCP_ATLAS_JUDGE_BASE_URL`, and
`MCP_ATLAS_JUDGE_API_KEY`; their resolved non-secret values are recorded in the
run manifest.

Copy MCP credentials into a local file that is not committed, then point `env_file` at it in a copy of `agent_config.json`. Full strict runs fail before inference if a server needed by the selected tasks is offline or fails its official representative probe.

Pull and smoke-test the benchmark:

```bash
rllm dataset pull mcp_atlas
rllm eval mcp_atlas \
  --model accounts/fireworks/models/glm-5p2 \
  --task-indices "$(cat examples/eval/mcp_atlas/smoke_10_indices.txt)" \
  --agent-config @examples/eval/mcp_atlas/agent_config.json \
  --sampling-params @examples/eval/mcp_atlas/glm_5_2_fireworks.json \
  --concurrency 5
```

To run every public task whose complete tool allowlist is covered by the 20
credential-free default servers, use the checked-in filter profile. The filter
also excludes `e2b-server`, which upstream currently misclassifies as keyless
even though it needs `E2B_API_KEY`:

```bash
rllm eval mcp_atlas \
  --model accounts/fireworks/models/glm-5p2 \
  --agent-config @examples/eval/mcp_atlas/agent_config_no_credentials.json \
  --sampling-params @examples/eval/mcp_atlas/glm_5_2_fireworks.json \
  --concurrency 5
```

`task_filter: default_servers` is applied before service startup and model
calls. The manifest records the filter, allowed servers, excluded servers, and
the exact selected task IDs. If combined with `--task-indices` or
`--max-examples`, those selectors apply to the already-filtered cohort.

The checked-in 10-task cohort is a smaller fixed smoke sample. The checked-in
50-task cohort covers every server prefix represented in public revision
`b5bcde2`:

```bash
rllm eval mcp_atlas \
  --model nvidia/nemotron-3-super-120b-a12b \
  --task-indices "$(cat examples/eval/mcp_atlas/stratified_50_indices.txt)" \
  --agent-config @examples/eval/mcp_atlas/agent_config.json \
  --sampling-params @examples/eval/mcp_atlas/nemotron_3_super_nim.json \
  --concurrency 5
```

Run all 500 by omitting `--task-indices`. Every rollout is atomically checkpointed under the printed run directory. Resume missing, timed-out, judge-error, or harness-error items with the same arguments plus:

```bash
--resume-run ~/.rllm/eval_results/<run-directory>
```

To compare with an official `score_claims.py` CSV produced from the same captured responses or a parallel live run:

```bash
python examples/eval/mcp_atlas/compare_runs.py \
  --rllm ~/.rllm/eval_results/<run-directory>/results.json \
  --official official_scored.csv \
  --mode replay
```

Use `--mode live` for the ≤5 percentage-point pass-rate and paired-bootstrap acceptance check. The optional Docker contract test is gated to avoid accidental model spend:

```bash
MCP_ATLAS_LIVE=1 \
MCP_ATLAS_LIVE_LLM_BASE_URL=https://.../v1 \
MCP_ATLAS_LIVE_LLM_API_KEY=... \
MCP_ATLAS_LIVE_MODEL=openai/gpt-4o \
uv run --extra dev --extra mcp-atlas pytest tests/integration/test_mcp_atlas_live.py -q
```

The public-500 GLM result is not an exact reproduction of Scale's published `77.8% ±2.6`, which includes 500 private tasks. Nemotron 3 Super has no published Scale MCP-Atlas target; readiness is established by protocol, replay-scoring, and live official-runner parity.
