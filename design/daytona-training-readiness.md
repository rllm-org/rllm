# Design: Making Daytona training-ready & Harbor-eval-faithful for the tmax cookbook

- **Status:** P0 + P1 implemented (`rllm/sandbox/backends/daytona.py`, `rllm/eval/_resolution.py`, `tests/sandbox/test_daytona_backend.py`); P2 still proposed.
- **Goal:** Bring the rLLM **Daytona** sandbox backend to parity with what `e4e91cf6` ("align native Path B task execution with Harbor (Modal)") + PR #684 ("eval reliability: concurrent-eval proxy ports, sandbox isolation, proxy resilience") did for **Modal**, so `TERMINAL_SANDBOX_BACKEND=daytona` in `cookbooks/tmax/train_fireworks.sh` reproduces Harbor's Daytona eval performance and trains stably at GRPO scale.
- **Related code:**
  - rLLM backend: `rllm/sandbox/backends/daytona.py`, `rllm/sandbox/backends/modal_backend.py`, `rllm/sandbox/protocol.py`
  - Glue: `rllm/sandbox/sandboxed_flow.py`, `rllm/sandbox/snapshot.py`, `rllm/sandbox/warm_queue.py`, `rllm/eval/_resolution.py`, `rllm/gateway/{tunnel,manager}.py`
  - Path-B execution: `rllm/harnesses/cli_harness.py`, `rllm/harnesses/terminus2.py`, `rllm/hooks.py`
  - Reference: `harbor/src/harbor/environments/daytona/{environment,snapshots,utils}.py`, `harbor/src/harbor/environments/modal.py`
  - Memory: `terminal-bench-eval-sandbox-reaping`, `eval-serving-backend-confound`

---

## Summary

tmax trains Harbor's Terminus-2 terminal agent. The agent runs **inside** the sandbox (`llm_inside_env=True`), reaches the rLLM model gateway over a public tunnel, and is scored by an rLLM per-task verifier (`tests/test.sh` → `/logs/verifier/reward.txt`) — this is the **Path B** (rLLM-native sandbox) flow, *not* Harbor's own runtime. To make eval reward on Daytona match Harbor's Daytona eval, Path B must reproduce how Harbor's Daytona environment *starts and configures* a task; to train at scale, the backend must survive hundreds of concurrent sandbox creates per GRPO step.

The good news: Daytona is **already a first-class backend** in the factory, snapshot system, resource mapping, and network classification — the structural plumbing is done. The remaining gaps are all *inside* `rllm/sandbox/backends/daytona.py` (plus a couple of `_resolution.py` lines), and each one has a working reference in either rLLM's Modal backend or Harbor's own Daytona environment.

Three gaps are **fidelity blockers** (P0): no per-exec user switching, no `SandboxCommandTimeout` (and unverified hard-kill on timeout), and no `set_env` for `[environment].env`. Two are **training-scale stability** (P1): no create retry/rate-limiting, and lifetime governed only by a default idle auto-stop. The rest are operational polish (P2).

---

## 1. What already works for Daytona (no change needed)

These are done — Daytona is not a green-field backend:

| Capability | Where | Status |
|---|---|---|
| Backend factory dispatch (`create_sandbox`) | `sandboxed_flow.py:72-91` | ✅ `daytona` branch present |
| Snapshot build / delete / liveness probe + `SnapshotNotFound` cold-fallback | `daytona.py:322-413`, dispatched from `sandboxed_flow.py:94-141` | ✅ declarative bake, idempotent, `replay_dockerfile` honored |
| Per-backend resource mapping (CPU, mem→GB, disk→GB, `create_timeout`) | `_resolution.py:243-281` (`daytona` branch) | ✅ matches `test_harbor_parity` intent |
| Network classification → remote → public tunnel | `tunnel.py:28-35` (`daytona` ∉ `LOCAL_SANDBOX_BACKENDS`); `eval/runner.py:98-100` | ✅ Daytona auto-wires cloudflared (eval) / uses pinned ngrok (`train_fireworks.sh:118`) |
| File transfer (single tar.gz round-trip, matches Harbor) | `daytona.py:215-265` | ✅ mirrors `harbor/.../tar_transfer.py` |
| `is_alive()` for warm-queue reuse (never raises) | `daytona.py:267-283` | ✅ |
| Leak cleanup (`_LIVE_SANDBOXES` + atexit) | `daytona.py:42-56,174-175` | ✅ |
| Agent→gateway connectivity | `cli_harness._exec_agent` injects gateway env via **per-exec `export`** (`cli_harness.py:112-115`), not `set_env` | ✅ works on Daytona today |

> **Important nuance:** the Terminus-2 agent's gateway URL/key are injected per-exec by `_exec_agent` (`export … ; <cmd>`), so the agent reaches the gateway on Daytona *today*. The `set_env` gap (P0-3 below) does **not** break agent connectivity — it breaks `[environment].env` task-declared vars that the **verifier** depends on.

---

## 2. Three-way comparison

| Dimension | Harbor Daytona (reference) | rLLM Modal (training-ready) | rLLM Daytona (current) | Gap |
|---|---|---|---|---|
| Per-exec user switch (agent vs verifier) | `su <user> -s /bin/bash -c`, UID via `getent` (`environment.py:1346-1366`) | emulated via `su` in `_build_exec_command` (`modal_backend.py:138-159`) | **`user` arg ignored** (`daytona.py:184-189`) | **P0-1** |
| Command-timeout semantics | shell `timeout <sec>` wrapper + poll (`environment.py:1352-1353`) | maps SIGKILL→`SandboxCommandTimeout` (`modal_backend.py:292-301`) | plain `RuntimeError` on any non-zero; SDK-timeout kill unverified (`daytona.py:204-212`) | **P0-2** |
| `[environment].env` on every exec | per-exec `env K=V` / compose env (`environment.py:1348-1350`) | `set_env` + per-exec export (`modal_backend.py:251-259`) | no `set_env` → one-shot `export` that doesn't survive (`_resolution.py:419-424`) | **P0-3** |
| Create retry / rate-limit | tenacity: 10 retries linear backoff on transient/429 (`utils.py:100-134`) | `_CreateRateLimiter` token bucket (`modal_backend.py:63-118`) | **single `create()`, no retry/limit** (`daytona.py:144-170`) | **P1-4** |
| Lifetime vs reaping mid-rollout | ephemeral + `auto_stop`/`auto_delete` knobs (`environment.py:1122-1123`) | hard lifetime sized to agent+verifier+install+slack (`_resolution.py:261-269`) | default `auto_stop_interval=30` min idle, not task-sized (`daytona.py:35,118`) | **P1-5** |
| Per-run isolation labels | `harbor.managed` / `session_id` labels (`environment.py:1128-1137`) | `rllm_run_id` tags + per-run App name (`modal_backend.py:195,220-221`) | accepts `labels` kwarg but doesn't auto-stamp run id | **P2-6** |
| Explicit network egress | `network_block_all` derived from task policy (`environment.py:914-923`) | `block_network`/allowlist | no network param → SDK default | **P2-7** |
| stderr | returned separately | demuxed | discarded on success (`daytona.py:201`) | **P2-8** |

---

## 3. Proposed changes (prioritized)

### P0 — fidelity blockers (without these, Daytona reward ≠ Harbor reward) — ✅ implemented

**P0-1. Per-exec user switching (`su` emulation).**
`DaytonaSandbox.exec` ignores `user` and runs everything as the create-time `os_user` (root). Harbor runs the **agent** as `agent_user` and the **verifier** as `verifier_user`; `_resolution.py:400-411` already sets up verifier-dir ownership *assuming the backend will switch users*. On Daytona that assumption is false today, so (a) agent/verifier isolation is lost and (b) permission-sensitive Harbor tasks diverge.
**Change:** mirror Modal's `_build_exec_command` (`modal_backend.py:138-159`) — when `user` is set, wrap the command in `su <user> -s /bin/bash -c <cmd>`, resolving an int UID via `getent passwd`. Harbor's own Daytona env does exactly this (`environment.py:1346-1366`), so it's known-good on the platform.
*tmax impact:* lower (terminal-bench tasks often run as root), but required for general Harbor parity and any task that sets `agent_user`.

**P0-2. `SandboxCommandTimeout` + guaranteed hard-kill on timeout.**
`exec` passes `timeout` to the SDK's `process.exec` and raises a plain `RuntimeError` on any non-zero exit. Two problems: (1) a budget-exhausted agent is indistinguishable from a real failure, so `cli_harness.run` (`cli_harness.py:328`) can't take the "expected, score anyway" path and instead logs a misleading WARNING; (2) it's unverified whether the SDK's `timeout` actually *kills* a runaway process — if it doesn't, the rollout hangs until the sandbox idle-stops.
**Change:** adopt Harbor's robust pattern — wrap long execs in a shell `timeout <sec> bash -c …` (`environment.py:1352-1353`) and map exit code `124` (and the SDK timeout error, if any) to `protocol.SandboxCommandTimeout`. This both guarantees the kill and gives `cli_harness` the signal it already handles.

**P0-3. `set_env` for `[environment].env` fidelity.**
Daytona has no `set_env`, so `_resolution._setup_task_environment` (`_resolution.py:419-424`) falls back to a one-shot `export` that does **not** survive across Daytona's independent exec shells. Harbor injects `[environment].env` into every command; the **verifier** in particular relies on these vars. Result: verifier reward can diverge on tasks that declare env.
**Change:** add `set_env(dict)` (store in `self._persistent_env`) and prepend `export …` for those vars in every `exec`, exactly like Modal (`modal_backend.py:251-259` + `_build_exec_command`). `_resolution.py` already feature-detects `set_env`, so no glue change.

### P1 — training-scale stability (without these, GRPO steps lose batches / hang) — ✅ implemented

**P1-4. Create retry + rate-limiting.** *(highest-impact training gap)*
rLLM's Daytona backend does a single `Daytona.create()` with no retry (`daytona.py:144-170`). A GRPO step spins up many group copies at once (the same burst that forced Modal's `_CreateRateLimiter`); Daytona's server-side limits will reject creates and lose those rollouts. Both reference impls solve this: Modal with a process-global token bucket (`modal_backend.py:63-118`), Harbor with tenacity (10 retries, linear backoff, transient/429-aware — `utils.py:100-134`).
**Change:** wrap `Daytona.create()` in retry-with-backoff that classifies transient/rate-limit/capacity errors (reuse Harbor's `is_transient_daytona_error` taxonomy), and optionally add a `_CreateRateLimiter`-style local throttle tunable via `RLLM_DAYTONA_SANDBOX_CREATE_RPS` / `_BURST`. Preserve the existing `SnapshotNotFound` translation (don't retry a vanished snapshot — cold-fall back instead).

**P1-5. Task-sized lifetime instead of default idle auto-stop.**
Daytona lifetime is governed by `auto_stop_interval` (default **30 min idle**, `daytona.py:35`). The agent budget in `train_fireworks.sh` is `RLLM_HARNESS_RUN_TIMEOUT_S=1800` (30 min) plus install plus verifier — and a stalled LLM round-trip can *look* idle. This is precisely the "sandbox reaped mid-rollout" failure that deflated Modal eval scores (see memory `terminal-bench-eval-sandbox-reaping`).
**Change:** in `_resolution.py`'s `daytona` branch, set `auto_stop_interval` (minutes) sized from `agent_timeout + verifier_timeout + install_timeout + slack` — the same arithmetic Modal uses for its hard `timeout` — raised to the **provider-agnostic** `RLLM_SANDBOX_TIMEOUT_S` (seconds) floor, converted to minutes. The lifetime floor is computed once and shared by both backends (`_resolution.py` `_sandbox_resource_kwargs`); `RLLM_MODAL_SANDBOX_TIMEOUT_S` remains a deprecated alias (`rllm/env.py` `sandbox_timeout_override_s`). Confirm `close()`'s `delete()`→`stop()` fallback (`daytona.py:285-302`) still frees compute promptly.

### P2 — operational parity / polish

- **P2-6. Per-run isolation labels.** Auto-stamp `rllm_run_id` (from `RLLM_RUN_ID`) into the `labels` Daytona already accepts, mirroring Modal's tags (`modal_backend.py:220-221`), so leaked sandboxes from one run can be listed/reaped without disturbing a co-tenant run on a shared org.
- **P2-7. Explicit network egress.** Pass the network parameter explicitly so outbound internet (gateway tunnel reachability) is guaranteed rather than relying on the SDK default; honor a task `no-network` policy if present (Harbor derives `network_block_all` from task policy, `environment.py:914-923`).
- **P2-8. stderr capture.** Optionally merge/return stderr like `DockerSandbox` for debuggability (low priority — the harness already tees `2>&1`).
- **P2-9. Docstring cleanup.** `daytona.py:73` still claims preview-link port exposure that was removed in #633.

---

## 4. Wiring `train_fireworks.sh` for Daytona

1. `export TERMINAL_SANDBOX_BACKEND=daytona` (the only knob; read at `train.py:63`, threaded into harness + trainer).
2. Provide Daytona auth: `DAYTONA_API_KEY` (or `DAYTONA_JWT_TOKEN` + `DAYTONA_ORGANIZATION_ID`), `DAYTONA_API_URL`, `DAYTONA_TARGET`.
3. Gateway tunnel already correct: `train_fireworks.sh:118` pins `rllm.gateway.tunnel=https://rllm.ngrok.dev`; Daytona is remote, so the in-sandbox Terminus driver gets the public URL automatically. (Confirm ngrok URL is reachable from the Daytona region.)
4. Set the provider-agnostic `RLLM_SANDBOX_TIMEOUT_S` (seconds) — one knob for all backends; it sizes Daytona's `auto_stop_interval` (P1-5) and Modal's hard lifetime alike.
5. Optional warm-start: `rllm snapshot create --sandbox-backend daytona …` (already supported) to pre-bake the Terminus-2 install and avoid per-rollout `uv`/apt cost.

---

## 5. Validation plan

1. **Unit tests** (extend `tests/sandbox/test_daytona_backend.py`): `su` wrapping when `user` set; `set_env` persistence across two execs; `timeout` → `SandboxCommandTimeout` mapping; create-retry on a simulated transient error.
2. **Parity test** (extend `tests/eval/test_harbor_parity.py`): assert the `daytona` resource branch now also emits the `auto_stop_interval` lifetime kwarg.
3. **End-to-end eval parity:** run `rllm eval` on a fixed tmax-15k slice with `--sandbox-backend daytona` and compare reward to Harbor's Daytona eval on the *same* tasks. **Control for the serving backend** (same model server for both) — per memory `eval-serving-backend-confound`, serving differences, not the harness, dominate rLLM-vs-Harbor gaps. The bar: Daytona Path-B reward ≈ Harbor Daytona reward within noise, matching the Modal result from `e4e91cf6`.
4. **Training smoke:** one short Fireworks GRPO run on Daytona at full rollout concurrency to confirm P1-4/P1-5 (no batch-killing create failures, no mid-rollout reaping).

---

## 6. Effort estimate

All P0 + P1 changes are localized to `rllm/sandbox/backends/daytona.py` (~150 lines, every pattern already exists in `modal_backend.py` or `harbor/.../daytona/`) plus ~5 lines in `_resolution.py` for P1-5. No glue/protocol changes required — the optional `set_env` is already feature-detected and the factory/snapshot/network plumbing already treats Daytona as first-class. P0-2 carries a small unknown (Daytona SDK timeout/kill semantics) that the `timeout`-wrapper approach sidesteps deterministically.
