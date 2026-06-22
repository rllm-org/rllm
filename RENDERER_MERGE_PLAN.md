# Renderer Merge Plan — unify prime-rl + Fireworks renderers for rLLM training

**Goal:** support more models for rLLM **RL-rollout** training by combining the
model coverage of two renderer ecosystems behind a single rLLM-side interface,
preferring the backend that gives the strongest multi-turn token-identity
guarantee.

**Status:** native renderer package **implemented** (`rllm/renderers/`, tests in
`tests/test_renderers.py`, 13 passing). Engine/gateway repointing (§6 PR 4–5) is
**not yet done** — the new layer is additive and not yet wired into the rollout
engines or the gateway.

### Implemented (this change)
- `rllm/renderers/` — native token-level `Renderer` protocol + `RenderedTokens` /
  `ParsedResponse` / `ToolSpec` (reuses `rllm.tools.tool_base.ToolCall`).
- `resolve()` / `select_backend()` registry with RL-first precedence
  (prime-rl exact-match → tinker/Fireworks adapter → DefaultRenderer), plus
  `renderer_family` (prime) and `renderer_name` (tinker) overrides.
- prime-rl backend wrapper (`_prime.py`) and tinker/Fireworks adapter
  (`_tinker.py`); Fireworks registration shim (`_fw_register.py`) — lazy, so
  `import rllm.renderers` stays ~0.5s and pulls no transformers/torch until first
  use.
- Verified byte-for-byte parity vs `apply_chat_template` for prime (Qwen3-8B,
  Qwen3.5-4B) and the tinker adapter; prime bridge extends the prefix
  byte-for-byte; DeepSeek-V4-Flash (a prime-rl gap) renders via the adapter.

### DeepSeek-V4 (and other Fireworks models): rollout + cumulative mode (done)

**Constraint:** `rllm` depends on `rllm-model-gateway` (pyproject line 21), so the
gateway must **not** import `rllm` (circular) — and it must stay free of
tinker/Fireworks deps ("lightweight" package). All tinker/Fireworks knowledge
therefore lives in `rllm.renderers`; the gateway receives a built renderer by
**injection**.

- **Generic cross-turn bridge** in `rllm/renderers/_tinker.py`
  (`TinkerAdapter.bridge_to_next_turn`) — built from the same primitives
  `build_generation_prompt` uses (`render_message` + `_get_generation_suffix`),
  anchored to the turn-close (inline `trim_to_turn_close`). **Byte-for-byte vs
  prime-rl's hand-coded bridge for Qwen3 / Qwen3.5**; preserves verbatim sampled
  history (incl. thinking) where a full re-render would strip it. Returns `None`
  on assistant-in-new-slice / truncation-without-close / multimodal. `has_bridge`
  is `True`. No gateway dependency.
- **Rollout fix** in `rllm/engine/rollout/fireworks_engine.py`: DeepSeek-V4's
  tokenizer ships **no Jinja `chat_template`**, so `ChatTemplateParser` crashed in
  `__init__`. FireworksEngine now falls back to the tinker/Fireworks renderer
  (TinkerEngine's existing non-bypass render + parse path) when there's no chat
  template. Chat-template models are unchanged.
- **Gateway stays dependency-free**: `create_app(..., renderer=...)` accepts an
  injected renderer; with none, it builds a **prime-rl** renderer from config as
  before (its only renderer dep). Startup guard is `_renderer_has_bridge()`.
- **Injection** (`rllm/gateway/manager.py`): in-process (thread) mode —
  used by the Fireworks backend (`gateway_mode = "process" if verl else "thread"`)
  — builds the renderer via `rllm.renderers.resolve()` and passes it to
  `create_app`. prime-rl models → prime bridge; DeepSeek-V4 etc. → tinker adapter
  bridge. Subprocess (verl) mode keeps the prime-rl-only config build.
- Tests: bridge parity + DeepSeek-V4 in `tests/test_renderers.py`; gateway
  injection contract in `rllm-model-gateway/tests/unit/test_renderer_injection.py`.
  Existing gateway cumulative-mode + tinker-engine tests still pass.

### Tool-call bridge (validated)
The bridge renders the new turn's delta via `build_generation_prompt(sentinel +
new_messages)` and splits on the sentinel's N-th close token, so the renderer
does its own turn-level tool preprocessing (merging consecutive tool results into
one user turn, pairing them with the assistant's calls, role handling). Per-
message rendering was wrong here — DeepSeek-V4's `render_message` rejects raw
`tool` role outright. Validated: DeepSeek-V4 merges consecutive tool results into
a single user turn (history kept verbatim); Qwen3.5 tool-result bridge equals
prime-rl's hand-coded bridge byte-for-byte. Note: tinker's *Qwen* renderer
doesn't group multi-tool, but Qwen routes to prime-rl (which does) — so only the
tinker-served models (DeepSeek-V4, …) rely on this path, and DeepSeek's
`build_generation_prompt` merges correctly.

### Remaining
- Repoint `tinker_engine` at `rllm.renderers.resolve()` for consistency (§6 PR 4).
- Migrate/retire `ChatTemplateParser` once parity is proven in the engines.
- Parity-check the bridge on the other tinker-served FW models (Gemma-4,
  Ministral-3, Kimi-K2.7-code) before relying on cumulative mode for them; the
  bridge returns `None` on anything it can't render (safe full-re-render fallback).

---

## 1. Background — four renderer systems, two abstractions

| System | Import | Abstraction | Key API | Built for |
|---|---|---|---|---|
| **prime-rl `renderers`** | `import renderers` (`PrimeIntellect-ai/renderers`) | **token-level** | `create_renderer`, `render_ids`, `parse_response`, `get_stop_token_ids`, **`bridge_to_next_turn`** | RL rollouts: multi-turn **token identity** |
| **Fireworks cookbook** | `import training.renderer` (`fw-ai/cookbook#training`) | **message-level** | subclasses `tinker_cookbook.renderers.Renderer`; `register_renderer` | SFT: `build_supervised_examples` + disaggregate |
| `tinker_cookbook.renderers` | `import tinker_cookbook.renderers` | message-level | `get_renderer`, `build_generation_prompt`, `parse_response` | SFT + Tinker RL via "extension property" |
| `rllm.parser.ChatTemplateParser` | in-repo | text/token | `parse`, `tokenize_and_mask`, `parse_completion` | de-facto rollout path (`bypass_render_with_parser=True`) |

The two packages named in the request **share no base class**. prime-rl is
token-in/token-out with a `bridge_to_next_turn` contract; Fireworks renderers
are message-level `tinker_cookbook.Renderer` subclasses. So "merge" =
**a facade in rLLM that exposes one interface and routes each model to whichever
backend supports it** — not a code-level union of the two upstreams.

### Current consumption in rLLM (verified)
- **Rollout engines:** `rllm/engine/rollout/fireworks_engine.py` and
  `tinker_engine.py` both default to `bypass_render_with_parser=True` → rollout
  tokens come from `rllm.parser.ChatTemplateParser`, **not** from either renderer
  package. `tinker_engine` also calls `renderers.get_renderer(name, tokenizer, …)`
  (tinker_cookbook) for stop sequences / the non-bypass path.
- **Gateway (prime-rl IS used here):** `rllm-model-gateway` uses prime-rl
  `renderers` for **cumulative token mode** (drift-free multi-turn token
  forwarding), gated behind `cumulative_token_mode` (default `False`):
  - `rllm/gateway/manager.py:361` launches the gateway with
    `--cumulative-token-mode` (+ `--renderer-family`) when enabled.
  - `rllm-model-gateway/.../server.py:161,176` → `from renderers import
    create_renderer`; `create_renderer(tokenizer, renderer=config.renderer_family)`.
    **Hard-fails if it resolves to `DefaultRenderer`** (no bridge).
  - `rllm-model-gateway/.../token_accumulator.py:145` →
    `renderer.bridge_to_next_turn(...)` extends prior `prompt+completion`.
- **`training.renderer` (Fireworks) is imported nowhere** in rLLM.

### Two findings that shape the plan
1. **Fireworks renderers are installed but dead.** `fireworks-training-cookbook`
   is a dependency (pyproject line 96) and `training.renderer` is importable, but
   nothing imports it, so `register_renderer` never fires —
   `get_renderer("deepseek_v4", …)` currently misses. **One import wires up
   DeepSeek-V4 / Gemma4 / Kimi-K2.7-code + the SFT disaggregate fix.**
2. **prime-rl `renderers` is already integrated via the gateway** (not just
   pip-installed). The gateway's `create_renderer(...)` call **hard-errors when a
   model isn't in prime-rl's `MODEL_RENDERER_MAP`** — that error site is the
   natural insertion point for the Fireworks/tinker adapter (§3.4), so unsupported
   models fall back instead of failing. The facade should treat the gateway as the
   *existing* prime-rl integration, not a greenfield one.

---

## 2. Model coverage — who brings what

**Only Fireworks brings (the "more models" target):** DeepSeek-V4-Flash,
Gemma4, Kimi-K2.7-code; plus tinker-style minimax_m2 / glm5 / nemotron and the
multi-turn-SFT disaggregate correctness fix.

**Only prime-rl brings:** GLM-5.1, GLM-4.5, Laguna-XS.2, Nemotron-3-Ultra,
token-level MiniMax-M2.5 — plus `bridge_to_next_turn` for **every** model.

**Neither:** MiniMax-M3 (template diverges hard from M2 — see §7).

---

## 3. Target architecture — `rllm/renderers/` facade

```
rllm/renderers/
  __init__.py     # public: resolve(), Renderer protocol, RendererSpec
  protocol.py     # the token-level Protocol rLLM rollouts speak
  registry.py     # unified model -> backend map + precedence + resolve()
  _prime.py       # passthrough wrapper over prime-rl create_renderer
  _tinker.py      # adapter: tinker_cookbook/Fireworks Renderer -> Protocol
  _default.py     # last-resort apply_chat_template wrapper (bridge -> None)
```

### 3.1 `protocol.py` — the one interface (prime-rl's shape; RL-first)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class Renderer(Protocol):
    def render_ids(self, messages, *, tools=None,
                   add_generation_prompt=False) -> list[int]: ...
    def parse_response(self, token_ids) -> "ParsedResponse": ...
    def get_stop_token_ids(self) -> list[int]: ...
    # RL-critical: return prev_prompt_ids + prev_completion_ids + new turn,
    # byte-for-byte, or None to signal "caller must full-re-render".
    def bridge_to_next_turn(self, previous_prompt_ids, previous_completion_ids,
                            new_messages, *, tools=None) -> list[int] | None: ...

# ParsedResponse = (content, reasoning_content, tool_calls) — mirror prime-rl.
```

This is intentionally prime-rl's contract: RL rollout correctness lives in
`bridge_to_next_turn`, which the message-level abstractions don't natively have.

### 3.2 `_prime.py` — passthrough (no real work)

prime-rl renderers already implement this Protocol exactly. Wrapper just holds
the instance and forwards. Coverage test = membership in
`renderers.base.MODEL_RENDERER_MAP` (exact match on `tokenizer.name_or_path`);
do **not** rely on `create_renderer` "succeeding" — it always returns a
`DefaultRenderer` fallback.

### 3.3 `_tinker.py` — adapter (the real work)

Wrap a `tinker_cookbook.renderers.Renderer` (incl. Fireworks subclasses) to the
Protocol:

| Protocol method | Maps to tinker_cookbook | Notes |
|---|---|---|
| `render_ids(msgs, tools, add_gen)` | `build_generation_prompt(msgs, …)` | extract int token ids from the returned ModelInput; **verify exact return type against the pinned tinker_cookbook version at impl time** |
| `parse_response(ids)` | `renderer.parse_response(ids)` | map tinker `Message` → `ParsedResponse(content, reasoning_content, tool_calls)` |
| `get_stop_token_ids()` | `renderer.get_stop_sequences()` | convert any string stops → ids via tokenizer |
| `bridge_to_next_turn(...)` | — | **return `None`** unless `getattr(renderer, "has_extension_property", False)`; if true, do prefix-extension (render new messages only and append). Conservative `None` is always *correct* (caller full-re-renders) — never emit a wrong bridge. |

The `None`-bridge path means RL multi-turn on tinker-backed models falls back to
full re-render per turn (same behavior as today's `ChatTemplateParser`), so this
is **no regression** — and any model that *also* exists in prime-rl gets the real
bridge via precedence (§3.4).

### 3.4 `registry.py` — precedence (RL-first)

```python
def resolve(model_name, tokenizer, *, renderer_name=None,
            image_processor=None) -> Renderer:
    # 0. explicit override always wins
    if renderer_name:
        return _explicit(renderer_name, tokenizer, image_processor)
    # 1. prime-rl exact match -> token bridge available  (PREFERRED for RL)
    if model_name in renderers.base.MODEL_RENDERER_MAP:
        return PrimeRenderer(create_renderer(tokenizer))
    # 2. Fireworks/tinker covers a model prime-rl lacks (DeepSeek-V4, Gemma4, ...)
    name = _tinker_renderer_name(model_name)   # see §4 — incl. FW models
    if name is not None:
        return TinkerAdapter(get_renderer(name, tokenizer, image_processor))
    # 3. last resort
    return DefaultRenderer(tokenizer)          # apply_chat_template, bridge->None
```

Precedence order is what encodes "RL rollout primary": a model in **both**
ecosystems routes to prime-rl so it gets `bridge_to_next_turn`.

---

## 4. Model → renderer-name resolution (`_tinker_renderer_name`)

`tinker_engine` already uses `model_info.get_recommended_renderer_name(model)`,
but tinker_cookbook's `model_info` lags new models and **does not know the
Fireworks-only renderers**. Add a small rLLM-side override map:

```python
_FW_MODEL_RENDERER = {
    "deepseek-ai/DeepSeek-V4-Flash": "deepseek_v4",
    "google/Gemma4-...":             "gemma4",
    "moonshotai/Kimi-K2.7-...":      "kimi_k27_code",
    # glm5 / minimax_m2 / nemotron also available via FW registrations
}
def _tinker_renderer_name(model_name):
    if model_name in _FW_MODEL_RENDERER:
        return _FW_MODEL_RENDERER[model_name]
    try:
        return model_info.get_recommended_renderer_name(model_name)
    except KeyError:
        return None
```

(Exact HF repo ids for Gemma4 / Kimi-K2.7 to be filled from the FW renderer
docstrings / `MODEL_RENDERER_MAP` analogues at impl time.)

---

## 5. File-by-file changes

| File | Change |
|---|---|
| `rllm/renderers/__init__.py` (new) | export `resolve`, `Renderer`, `ParsedResponse` |
| `rllm/renderers/protocol.py` (new) | Protocol + `ParsedResponse` |
| `rllm/renderers/registry.py` (new) | `resolve()`, precedence, `_tinker_renderer_name`, `_FW_MODEL_RENDERER` |
| `rllm/renderers/_prime.py` (new) | prime-rl passthrough wrapper |
| `rllm/renderers/_tinker.py` (new) | tinker_cookbook/Fireworks adapter |
| `rllm/renderers/_default.py` (new) | apply_chat_template fallback (bridge→None) |
| `rllm/renderers/_fw_register.py` (new) | `import training.renderer` (the dead-code fix), guarded in try/except for envs without the cookbook |
| `rllm/engine/rollout/fireworks_engine.py` | swap `ChatTemplateParser.get_parser(...)` for `rllm.renderers.resolve(...)`; use `render_ids` / `bridge_to_next_turn` for multi-turn rollouts; keep `ChatTemplateParser` as a 4th backend behind a flag during migration |
| `rllm/engine/rollout/tinker_engine.py` | route through `resolve()` instead of bare `renderers.get_renderer`; preserve the existing `renderer_name` override path |
| `rllm-model-gateway/.../server.py` | replace the bare `create_renderer(...)` (which hard-errors off `MODEL_RENDERER_MAP`) with `resolve()` so cumulative token mode falls back to the tinker/FW adapter for models prime-rl lacks, instead of failing. **Highest-leverage integration point** — prime-rl already lives here. |
| `pyproject.toml` | pin `renderers` (prime-rl) as an explicit dep instead of relying on the transitive install |

---

## 6. Rollout (suggested PR sequencing)

1. **PR 1 — quick win (low risk):** add `rllm/renderers/_fw_register.py` and import
   it where renderers are resolved. Unlocks DeepSeek-V4 / Gemma4 / Kimi-K2.7-code
   + the SFT disaggregate fix through the *existing* `get_renderer` path. No
   facade yet.
2. **PR 2 — facade skeleton:** `protocol.py` + `_prime.py` + `_default.py` +
   `registry.resolve()` with prime-rl-only precedence; unit + parity tests.
3. **PR 3 — tinker adapter:** `_tinker.py` + `_tinker_renderer_name`; precedence
   step 2 live. DeepSeek-V4 etc. now reachable through the facade.
4. **PR 4 — repoint engines:** fireworks/tinker engines call `resolve()`; gate
   `ChatTemplateParser` behind a deprecation flag.
5. **PR 5 — migrate/retire `ChatTemplateParser`** once parity is proven.

---

## 7. MiniMax-M3 (explicitly out, but noted)

Neither package supports MiniMax-M3, and its `chat_template.jinja` is a ground-up
redesign vs M2/M2.5 (new `]<]minimax[>[` namespace token, `<mm:think>` tags,
recursive `to_xml` tool-call args, `root`/`developer` role split, `thinking_mode`).
The M2 renderer in **either** ecosystem produces token-for-token wrong output for
M3. Supporting M3 = a new hand-coded renderer; under this plan, write it as a
**prime-rl-shaped** renderer (so it gets `bridge_to_next_turn`) and register it in
the unified registry. Tracked separately.

---

## 8. Testing

- **Parity:** for each model, assert `resolve(...).render_ids(msgs)` ==
  `tokenizer.apply_chat_template(msgs, tokenize=True)` for single-turn (catches
  adapter token-extraction bugs). prime-rl ships such tests; mirror for the
  adapter.
- **Bridge correctness (RL-critical):** assert
  `bridge_to_next_turn(p, c, new)` starts with `p + c` byte-for-byte, or is
  `None`. Never a third outcome.
- **Precedence:** a model in both ecosystems resolves to `PrimeRenderer`
  (i.e. exposes a non-`None` bridge).
- **Env guard:** facade imports succeed when the Fireworks cookbook is absent
  (try/except around `_fw_register`).

---

## 9. Risks / open questions

- **Adapter token extraction:** `build_generation_prompt` return type varies by
  tinker_cookbook version — verify against the pinned version; the int-id
  extraction is the riskiest line in the adapter.
- **Stop-token conversion:** tinker `get_stop_sequences()` may return strings;
  need a tokenizer-based string→id conversion that matches what the engine
  expects.
- **Double tokenizer load:** prime-rl `create_renderer` and the engine both load
  tokenizers; reuse one instance to avoid the startup cost (prime-rl supports
  passing a tokenizer in).
- **`model_info` lag:** Fireworks model ids must be in `_FW_MODEL_RENDERER`
  because tinker `model_info` won't know them — keep that map close to the FW
  cookbook version it targets.
