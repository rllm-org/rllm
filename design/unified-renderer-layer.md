# Unified Renderer Layer for rLLM

**Goal:** Make cumulative token mode work for every model a backend supports — including DeepSeek-V4 — without writing a bridge per model.

## Problem

Cumulative token mode needs exactly one renderer method:

```python
bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, *, tools) -> RenderedTokens | None
# B[:len(prev_prompt)+len(prev_completion)] == prev_prompt + prev_completion, else None
```

Only PrimeIntellect's `renderers` (pip) has it — and only for an exact-match model list (no DeepSeek-V4; fine-tunes/local paths fall to `DefaultRenderer`, whose bridge is always `None`). The renderers that *do* cover new models — `tinker_cookbook` and `fw-ai/cookbook` (`DeepseekV4Renderer`) — have `build_generation_prompt`/`build_supervised_example` but **no bridge**. So DeepSeek-V4 in cumulative mode has no path today.

## Key facts

- The renderer is a **training-time, backend-coupled** concern. Tinker backend ⇒ `tinker_cookbook` renderers; Fireworks backend ⇒ fw cookbook renderers. Each backend already ships renderers for all its models — no vendoring decision needed.
- The bridge is **synthesizable** from any deterministic renderer: `trim_to_turn_close(prev) + rendered_delta`, sampled tokens kept verbatim, with a runtime prefix-check. PrimeIntellect already ships the generic helpers (`trim_to_turn_close`, `_common_prefix_len`, `build_trajectory_step`).
- `TokenAccumulator` only calls `render_ids()` and `bridge_to_next_turn()` — that's the whole interface to satisfy.

## Design

Adopt PrimeIntellect's `Renderer` protocol as canonical (gateway already depends on it; `message_indices` gives clean masking). Add a thin layer that wraps each backend's native renderers into it:

```
rllm/renderers/
  __init__.py      # get_renderer(backend, model, tokenizer); re-exports RenderedTokens/Renderer
  bridging.py      # BridgingRendererMixin: synthesized bridge for any deterministic renderer
  adapters.py      # TinkerRendererAdapter (covers tinker_cookbook AND fw cookbook),
                   #   ChatTemplateAdapter (universal fallback)
```

`get_renderer(backend, model, tokenizer)` resolves in priority order:
1. **Native PI renderer** if the model is in `MODEL_RENDERER_MAP` (hand-tuned, parity-tested).
2. **Backend's cookbook renderer** via `TinkerRendererAdapter` — e.g. Fireworks/DeepSeek-V4 → wrap `DeepseekV4Renderer`, bridge from the mixin.
3. **`ChatTemplateAdapter`** fallback (loud warning; covers fine-tunes/local paths).

`TinkerRendererAdapter` maps: `render_ids` ← `build_generation_prompt().to_ints()`; supervised mask ← `build_supervised_example()`; `get_stop_token_ids` ← `get_stop_sequences()`; `parse_response` normalized; `bridge_to_next_turn` from the mixin.

**Safety:** the in-bridge prefix-check means a bad synthesis degrades to a reset (`RENDERER_NO_BRIDGE` in existing logs), never silent corruption.

## DeepSeek-V4 path

Register `(fireworks, deepseek-v4) → TinkerRendererAdapter(DeepseekV4Renderer)`. Gateway `server.py` swaps `create_renderer` → `get_renderer`. `TokenAccumulator` is unchanged. Cumulative mode now works.

## Phase 1 (agreed scope: Gateway + FireworksEngine) — IMPLEMENTED

Built on `main`. Files: `rllm/renderers/{types,bridging,adapters,registry,__init__}.py`,
`tests/test_renderers.py`; edits to gateway `server.py`/`models.py` and `fireworks_engine.py`.

1. `rllm.renderers` + `get_renderer`/`resolve`: prime-rl native → `TinkerRendererAdapter`
   (covers tinker_cookbook AND Fireworks-cookbook, e.g. `deepseek_v4`) → `ChatTemplateAdapter`.
   `BridgingRendererMixin` synthesizes the bridge generically — **verified byte-identical to
   prime-rl's hand-tuned qwen3 bridge** in tests.
2. Gateway `server.py`: `create_renderer` → `resolve`; adds `renderer_name` config. The hard
   "DefaultRenderer" error is now a loud warning — the chat-template fallback gives a best-effort
   bridge that the runtime prefix-check makes safe (drift → reset, never corruption).
3. FireworksEngine: **opt-in** unified renderer via `renderer_name`/`renderer_family` (the
   DeepSeek-V4 path). Default keeps the existing `ChatTemplateParser` path so currently-working
   Fireworks models don't regress. Verified the unified path is token-identical to the chat
   template for qwen-family.

39 tests pass (11 renderer + 28 existing gateway), ruff clean.

**Note on base branch:** this is on `main`, whose gateway `TokenAccumulator` is the simpler
variant (no `plan_turn`/`session_id`/`ResetReason`). The renderer package is branch-agnostic;
porting onto `terminal-rl` only touches the integration edits.

## Risks

- `tinker.ModelInput` dependency for the adapter (already an optional dep); extract IDs via `to_ints()`.
- Delta-render assumes content-independent post-assistant scaffolding; the prefix-check catches violations → such models get a hand-written native renderer.
- Multimodal bridge (`previous_multi_modal_data`) is out of scope for phase 1; VL models stay on native PI renderers.
