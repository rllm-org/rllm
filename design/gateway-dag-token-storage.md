# Design: Linear (DAG) token storage for gateway traces — store deltas, not the closure

- **Status:** Phase 1 implemented (transparent delta-chain storage — `token_chain.py`, `TraceRecord.{prompt_delta_token_ids,parent_trace_id}`, `data_process.apply_chain`, proxy `_chain_link`, reconstruction in the read endpoints; tests in `tests/unit/test_token_chain.py`). Phase 2 (end-to-end deltas into the transforms) still proposed.
- **Related:** verifiers v1 "message graph" (https://www.primeintellect.ai/blog/verifiers-v1); `rllm-model-gateway` (`token_accumulator.py`, `proxy.py`, `models.py`, `store/`); `rllm/engine/trace_converter.py`; `rllm/trainer/verl/transform.py`; `rllm/trainer/tinker/transform.py`; cumulative-token-mode PRs (#692 reset classification)
- **Scope:** how the gateway *persists* per-turn token IDs for a multi-turn RL session, and (optionally) how the trainer consumes them. The generation path, the renderer bridge, and the reset taxonomy are unchanged.

---

## Summary

Today every turn of a session is stored as a `TraceRecord` carrying the **full cumulative** `prompt_token_ids`. Turn *k*'s prompt already contains all of turns `0..k-1`, so a session of *N* turns stores ≈ `T·(1+2+…+N) = O(N²·T)` prompt tokens — the same quadratic blowup verifiers v1 removed by moving from prompt/completion pairs to a message **graph** whose size is linear in turns.

The fix is small because rLLM **already enforces the exact invariant** the graph relies on. The gateway's cumulative-token bridge (`renderers.bridge_to_next_turn`) guarantees

```
prompt_ids[k]  ==  prompt_ids[k-1] + completion_ids[k-1] + delta[k]      (byte-for-byte)
```

so each turn only needs to store its **delta** (the newly-rendered tokens since the previous turn) plus a pointer to its predecessor. For a linear conversation the "DAG" is just a **delta chain** (a linked list); a full branching DAG is a later generalization (see §7). Reconstructing any turn's full prompt is a forward walk of the chain.

Two independent facts make this clean rather than a rewrite:

1. **The full per-turn prompts are already redundant on the storage side** — they are a running prefix-sum of `delta + completion`.
2. **They are already redundant on the training side too.** Both the verl and tinker transforms materialize each step's full prompt *only to slice it back down to a delta* against a running accumulator (`prompt_ids[len(full_seq):]`). We store the closure, ship it over the wire, then throw away everything but the delta.

Recommendation: land a **transparent storage-layer** change first (store deltas, reconstruct on read; zero training-side change), then an optional **end-to-end** phase that carries deltas all the way into the transforms (removes the wire O(N²) and deletes the prefix-slicing).

---

## 1. Background — where the bytes go today

One LLM call → one `TraceRecord` (`models.py`), persisted by `store/{sqlite,memory}_store.py` as a JSON blob keyed by `trace_id`, indexed per session. The trainer pulls a whole session with `GET /sessions/{sid}/traces` (`client.py`), turns each record into a `Step` (`trace_converter.trace_record_to_step`), and assembles a `Trajectory`.

Per-turn cost of a session with *N* turns, ~*T* tokens added per turn:

| Field | Turn *k* size | Session total | Waste? |
|---|---|---|---|
| `prompt_token_ids` | ≈ `k·T` (all prior turns) | **`O(N²·T)`** | **yes — dominant** |
| `messages` (chat list) | *k* messages of text | **`O(N²)` text** | yes — secondary |
| `completion_token_ids` | this turn only | `O(N·T)` | no (linear) |
| `logprobs`, `routing_matrices` | this turn only, completion-aligned | `O(N·T)` | no (linear) |

`completion_token_ids` / `logprobs` / `routing_matrices` are already per-turn-only and length-aligned to the completion — untouched by this proposal. The blowup is entirely in `prompt_token_ids` (tokens) and `messages` (text). A 50-turn SWE rollout stores ~25× more prompt-token data than it needs.

## 2. The invariant that makes deltas exact

In cumulative-token mode (`cumulative_token_mode=True`), turns `1..N` are served as `/v1/completions` with a pre-tokenized prompt built by `TokenAccumulator.build_next_prompt` → `renderer.bridge_to_next_turn(prev_prompt_ids, prev_completion_ids, new_messages, tools)`. Its contract (see `token_accumulator.py` module docstring) is that the returned sequence **starts byte-for-byte with `prev_prompt_ids + prev_completion_ids`**. Hence, writing `cumulative[k] = prompt_ids[k] + completion_ids[k]`:

```
prompt_ids[k]  =  cumulative[k-1] + delta[k]
delta[k]       =  prompt_ids[k][len(cumulative[k-1]):]     # newly rendered messages + gen prompt
```

Unrolling, the entire trajectory is

```
prompt_ids[0] + completion[0] + delta[1] + completion[1] + … + delta[k] + completion[k] + …
```

and every intermediate `prompt_ids[k]` is a prefix-sum of the two per-turn lists `{delta, completion}`. **Storing `delta[k]` per turn is lossless** given `delta[0] = prompt_ids[0]` (the full initial prompt). This is precisely verifiers' "each message is a unique node linked to its predecessor; the trace size is linear in turns."

When the bridge *cannot* prove the contract (renderer gap, compaction, session reuse) the accumulator already `reset()`s — that becomes a **segment break** (a new chain root), exactly as it becomes a new training row today (§5).

## 3. The trainer already reduces to deltas

Both transforms merge a cumulative trajectory into **one masked sequence**, computing the delta themselves:

- **tinker** (`trajectory_to_datums`, `transform.py:133-147`): keeps a running `full_sequence`; for each step, `delta = token_input_flat[len(full_sequence):]` when it is a prefix, else emits a Datum and reseeds. Appends `delta` (mask 0) then `response_ids` (mask 1).
- **verl** (`_process_trajectory`, `transform.py:372-401`): keeps `seg["full_seq"]`; `delta_obs = prompt_ids[len(full_seq):]`; appends `delta_obs` (mask 0) then action (mask 1); a failed prefix check emits the segment and reseeds.

`Trajectory.is_cumulative()` (`types.py:393`) is **not** used by either path — the decision is made incrementally per step. What is genuinely load-bearing downstream:

1. `prompt_ids[0]` — the first turn's prompt (verl's row `prompt`; folded into tinker's flat seq),
2. every turn's `response_ids` (+ aligned `logprobs`, `advantage`, R3 `routing_matrices`),
3. every turn's `delta[k]` (k>0),
4. the **segment-break markers**.

That is exactly the chain we propose to store. Today we transmit full prompts and re-derive (1)–(4) by slicing; storing the chain hands the transforms their input directly and lets the prefix-slicing be deleted.

## 4. Design — store the chain

### 4.1 Data model (`TraceRecord`)

Add two fields; keep everything else:

```python
class TraceRecord(BaseModel):
    ...
    prompt_token_ids: list[int] = Field(default_factory=list)   # kept; empty when a delta is present
    # --- linear-storage additions ---
    prompt_delta_token_ids: list[int] | None = None   # new prompt tokens since parent's cumulative; turn 0 == full prompt
    parent_trace_id: str | None = None                # predecessor in this session's chain; None at a segment root
```

- **Backward compatible.** Old traces (delta `None`, full `prompt_token_ids`, no parent) still load and reconstruct trivially (each is its own root: `delta = prompt_token_ids`, `parent = None`). New traces set `prompt_delta_token_ids` + `parent_trace_id` and store `prompt_token_ids = []`.
- **`messages`** gets the same treatment in a parallel step (`messages_delta` = the accumulator's already-computed `new_messages` + the assistant `response_message`); reconstruct the cumulative list on read. Text is cheaper than tokens, so this can lag the token change.

### 4.2 Computing the delta + parent (gateway)

The accumulator already holds the parent's cumulative sequence at ingest time. Extend `TokenAccumulator`:

```python
# new state
self.segment_trace_ids: list[str] = []      # trace ids of the current (unbroken) chain

def delta_and_parent(self, prompt_token_ids: list[int]) -> tuple[list[int], str | None]:
    prev_len = len(self.prev_prompt_ids) + len(self.prev_completion_ids)   # == len(cumulative[k-1]); 0 at root
    delta = list(prompt_token_ids[prev_len:])
    parent = self.segment_trace_ids[-1] if self.segment_trace_ids else None
    return delta, parent

def record_trace_id(self, trace_id: str, *, advance: bool) -> None:
    if advance:
        self.segment_trace_ids.append(trace_id)
    elif self.segment_trace_ids:            # replay: overwrite the current turn's node in place
        self.segment_trace_ids[-1] = trace_id
```

`reset()` clears `segment_trace_ids` (chain break). The trace_id must be known before persisting, so generate it in the proxy and pass it into `build_trace_record(..., trace_id=...)` (currently it mints its own uuid — a one-line optional arg).

Proxy wiring (both the turn-0 chat path and the cumulative turn-1+ path in `proxy.py`), right where `acc.ingest_turn(...)` / `acc.update_prefix(...)` already run:

```python
trace_id = str(uuid.uuid4())
delta, parent = acc.delta_and_parent(prompt_token_ids)          # turn 0: delta == full prompt, parent None
acc.ingest_turn(prompt_token_ids, completion_token_ids, advance=not replay)
acc.record_trace_id(trace_id, advance=not replay)
# build_trace_record stamps prompt_delta_token_ids=delta, parent_trace_id=parent,
# and (linear-storage mode) leaves prompt_token_ids=[]
```

No delta is computed for sessions where the accumulator never engages (cumulative mode off, or token IDs absent, e.g. eval against OpenAI/Anthropic): those keep the full-prompt path and every trace is its own root — behavior identical to today.

### 4.3 Reconstruction

Two boundaries; pick per phase (§6):

```python
def reconstruct_prompt(trace, by_id):          # walk to the segment root, prefix-sum forward
    chain, node = [], trace
    while node is not None:
        chain.append(node)
        node = by_id.get(node.parent_trace_id) if node.parent_trace_id else None
    chain.reverse()
    ids = []
    for n in chain[:-1]:
        ids += n.prompt_delta_token_ids + n.completion_token_ids
    ids += chain[-1].prompt_delta_token_ids
    return ids                                  # == original prompt_token_ids, byte-for-byte
```

For the linear case, `get_session_traces` already returns traces time-ordered ascending, so this is a single forward pass with a running accumulator (no per-trace tree walk). The explicit `parent_trace_id` keeps it correct under replay/out-of-order and generalizes to branching later.

## 5. Edge cases (all already have an analogue today)

- **Segment break / reset** (`PREFIX_CHANGED` compaction, `RENDERER_NO_BRIDGE`, `EMPTY_DELTA`): accumulator resets → next turn is a new root (`parent=None`, `delta=` full prompt). Maps 1:1 to the transforms' "prefix check fails → emit row, reseed" and to a new DAG root. A session may therefore hold several chains; reconstruction restarts at each root.
- **Replay (`DUPLICATE`)**: overwrite the current node in place (`advance=False` → `segment_trace_ids[-1]` replaced; sqlite `store_trace` is already `INSERT OR REPLACE`). Parent unchanged; no sibling is created.
- **Turn 0 / non-cumulative / eval upstreams**: no delta computed; full prompt stored as the root's delta. Graceful degradation to the current representation.
- **Streaming**: identical hooks already exist in `_handle_cumulative_streaming*` finally-blocks; compute delta there too.
- **Multimodal / `EncodedTextChunk`** (tinker prompts): cumulative mode operates in token-id space (`prev_prompt_ids` are ints); image/chunked prompts are not on the cumulative path and keep the full-prompt representation.
- **`delete_session`**: chains are session-local; deleting a session drops its whole chain. No cross-session parents in this proposal (that is §7).

## 6. Rollout plan

**Phase 1 — transparent storage (recommended first).** Store deltas + parent; reconstruct full `prompt_token_ids` inside `get_session_traces` (and `get_trace`) so the client, `trace_record_to_step`, and both transforms see the **exact** `TraceRecord` shape they see today. Fixes the stated problem (DB/disk + gateway-RAM for held traces) with **zero training-side change**. Wire + transient trainer RAM are still O(N²) because we re-inflate on read.

**Phase 2 — end-to-end deltas (optional, behind a flag).** Return deltas as-is; teach `trace_record_to_step` / the transforms to consume `prompt_delta_token_ids` directly (they already compute exactly this). Removes the wire O(N²), the trainer never materializes full prompts, and the transforms' prefix-slicing (`transform.py` verl 372-401 / tinker 133-140) collapses to a straight concat. Must preserve: verl's need for `prompt[0]` + segment boundaries, tinker's chunk handling, and logprob/response length-alignment. Ship gated (`linear_token_traces`), validate Datum/DataProto parity against Phase-1 output, then default on.

No DB migration (JSON blob); fields default so mixed old/new traces coexist.

## 7. Out of scope — the full branching DAG

The linear chain removes the O(N²) the user asked about. A true multi-parent DAG buys two further things, deferred as higher-cost / lower-return:

- **Cross-rollout prefix sharing.** *G* GRPO rollouts of one task share an identical system+instruction prefix but live in *G* separate sessions. A DAG with shared roots would store that prefix once. Payoff is a single shared head vs. *G* growing tails (small next to the per-session quadratic), and it needs cross-session ref-counting + GC + a store-schema change.
- **Compaction branches** (verifiers' "branched trajectories share common roots"). rLLM hard-breaks on compaction today; a DAG could hang the compacted branch off the pre-compaction prefix. Deferred with the reset taxonomy as-is.

## 8. Testing

- **Unit** (`test_token_accumulator.py`): `delta_and_parent` across extend / replay / reset; `record_trace_id` advance vs. overwrite; round-trip `reconstruct_prompt == original prompt_token_ids` byte-for-byte.
- **Store**: multi-turn session stores O(N) token-id bytes, not O(N²); `get_session_traces` reconstruction equals full-mode output.
- **Trainer parity**: verl `transform_episodes_to_dataproto` and tinker `trajectory_to_datums` produce identical outputs from reconstructed (Phase 1) and delta-native (Phase 2) traces.

## 9. Impact

For an *N*-turn session, stored prompt-token data drops from `≈ T·N²/2` to `≈ T·N` — an ~*N*/2× reduction on the dominant term (≈25× at 50 turns), matching verifiers' quadratic→linear result. Completion/logprob storage is unchanged (already linear). Phase 2 extends the same reduction to the wire and to trainer RAM and deletes redundant prefix-slicing in both transforms.
