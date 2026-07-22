"""Per-session cumulative token state for drift-free multi-turn RL training.

Prevents tokenization drift (decode→re-encode producing different token IDs)
by maintaining raw token ID sequences across turns and using /v1/completions
with pre-tokenized prompts instead of re-tokenizing from text each turn.

The cross-turn "bridge" (end-of-assistant-turn + new user/tool messages +
start-of-next-generation) is delegated to the ``renderers`` package
(https://github.com/PrimeIntellect-ai/renderers), whose per-model
``bridge_to_next_turn`` guarantees the returned sequence starts byte-for-byte
with ``previous_prompt + previous_completion`` — exactly the prefix-extension
invariant the training-side merger relies on.

When an incoming request can't be served as a cumulative extension, the
accumulator resets to a fresh turn-0. ``ResetReason`` classifies *why* by the
observable structural relationship between the incoming message list and the
snapshot we last processed — see that enum for the taxonomy and the likely
upstream cause behind each reason.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from rllm_model_gateway.fastjson import dumps_sorted

logger = logging.getLogger(__name__)


class ResetReason(str, Enum):
    """Why a :class:`TokenAccumulator` dropped its state and restarted.

    Each value names the *observable structural relationship* between the
    incoming chat request and the accumulated snapshot — never a guessed root
    cause. The likely cause is documented per value but deliberately kept out of
    the name (a short incoming list, for instance, is not necessarily a retry).
    The reasons are mutually exclusive and total: the only non-reset outcome is
    the healthy ``extend`` path.
    """

    #: Incoming list is byte-identical to the snapshot we already processed —
    #: the conversation did **not advance**. Almost always an upstream retry /
    #: duplicate request; the log's ``age_s`` (seconds since the snapshot) tells
    #: you whether it was a fast resend. Handled as a **replay** (regenerate +
    #: overwrite the turn in place), not a reset — so it's the one classification
    #: carried in :class:`TurnPlan` that does not drop state.
    DUPLICATE = "duplicate"

    #: The messages *covering the snapshot prefix* changed: the list is shorter
    #: than what we processed, or its shared prefix no longer matches what we
    #: recorded. The history under the boundary was rewritten / compacted, or a
    #: ``session_id`` was reused. The log's ``first_divergent_index`` and
    #: ``incoming_at_divergence`` show what changed.
    PREFIX_CHANGED = "prefix_changed"

    #: Incoming list cleanly extends the snapshot (prefix matches, list grew),
    #: but the only new messages are assistant-role — nothing renderable to
    #: bridge (assistant tokens are already captured as completion ids).
    #: Distinct from DUPLICATE: here the conversation *did* move forward and
    #: aligns; it just moved forward by an assistant message. Typically
    #: append-assistant-and-recall.
    EMPTY_DELTA = "empty_delta"

    #: Clean extension with renderable (non-assistant) new messages, but
    #: ``renderer.bridge_to_next_turn`` returned ``None`` — the renderer cannot
    #: prove the prefix-extension contract for this model. A renderer gap, not
    #: an upstream-traffic problem.
    RENDERER_NO_BRIDGE = "renderer_no_bridge"

    #: Explicit/manual reset (tests, shutdown) with no structural cause.
    MANUAL = "manual"


# Structural relationship of an incoming message list to the snapshot prefix.
# Internal classifier output; mapped to a ResetReason (or the extend path) by
# ``plan_turn``.
_REL_FIRST_TURN = "first_turn"
_REL_EXTENDS = "extends"
_REL_DUPLICATE = "duplicate"
_REL_PREFIX_CHANGED = "prefix_changed"


@dataclass
class TurnPlan:
    """How an incoming request should be handled against the current snapshot.

    ``action`` is ``"extend"`` (build a cumulative bridge from ``new_messages``),
    ``"replay"`` (a duplicate resend — regenerate from the accumulated prompt and
    overwrite the current turn in place, no reset), or ``"reset"`` (drop state and
    re-ingest as turn-0, logging ``reason``). ``diagnostics`` carries the
    structural facts for the reset/replay log.
    """

    action: str
    reason: ResetReason | None = None
    new_messages: list[dict[str, Any]] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _message_fingerprint(message: dict[str, Any]) -> str:
    """Stable SHA-256 of a single message (detects any role/content change).

    Uses orjson (sorted keys) → hashes the bytes directly; the exact encoding
    only needs to be stable within a run, and every fingerprint goes through
    this one function, so snapshot and incoming fps stay comparable.
    """
    return hashlib.sha256(dumps_sorted(message)).hexdigest()


def _per_message_fingerprints(messages: list[dict[str, Any]]) -> list[str]:
    """Per-message fingerprints, so divergence can be located by index."""
    return [_message_fingerprint(m) for m in messages]


def _content_preview(message: dict[str, Any], limit: int = 60) -> str:
    """Short, log-safe preview of a message's content."""
    content = message.get("content")
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    return (text or "")[:limit]


def extract_new_messages(messages: list[dict[str, Any]], prev_message_count: int) -> list[dict[str, Any]]:
    """Return the messages added since the last processed turn, minus assistants.

    The agent sends a cumulative message list. We already processed
    ``prev_message_count`` messages. The genuinely new messages are
    ``messages[prev_message_count:]`` — but the first of these is typically the
    assistant turn the model just sampled, which is already captured as
    completion token IDs. ``bridge_to_next_turn`` refuses assistant content
    (re-tokenizing sampled tokens would corrupt training), so we drop every
    assistant message from the slice.

    Returns an empty list if there are no new non-assistant messages.
    """
    if len(messages) <= prev_message_count:
        return []
    new = messages[prev_message_count:]
    return [m for m in new if m.get("role") != "assistant"]


class TokenAccumulator:
    """Maintains per-session cumulative token state for drift-free generation.

    Tracks the exact token IDs of the most recent turn so the next turn's
    prompt can be built by ``renderers.bridge_to_next_turn`` (concatenation +
    new-message rendering) rather than re-tokenization from text.

    Detects non-cumulative message list changes (prefix divergence) and
    automatically resets, so the training-side merger sees a segment break.
    """

    def __init__(self, renderer: Any, session_id: str | None = None) -> None:
        self.renderer = renderer
        self.session_id = session_id or "unknown"
        # Stable id of the conversation lineage this slot serves, assigned by
        # SessionSlots. Stamped on every trace so the trainer can split a
        # session's traces into one trajectory per lineage.
        self.lineage_id: str | None = None
        self.prev_prompt_ids: list[int] = []
        self.prev_completion_ids: list[int] = []
        self.turn_count: int = 0
        self.message_count: int = 0
        # Per-message fingerprints of the snapshot (the message_count messages
        # we last verified). Per-message (not whole-list) so a divergence can be
        # localized to an index for the reset log.
        self._prefix_fps: list[str] = []
        # Monotonic time the snapshot was taken — reported as ``age_s`` on a
        # DUPLICATE reset to show how soon the resend arrived.
        self._snapshot_mono: float | None = None
        # Survives reset(): how many times this session has reset. A climbing
        # count on one session is the signal for a reset storm.
        self.reset_count: int = 0

    @property
    def cumulative_ids(self) -> list[int]:
        """Full token sequence through the most recent completion.

        After turn N this is ``prev_prompt_ids + prev_completion_ids`` — the
        prompt the last turn was sampled from plus the tokens it produced.
        """
        return self.prev_prompt_ids + self.prev_completion_ids

    def should_rewrite(self) -> bool:
        """Return True if this session should use /v1/completions rewriting."""
        return self.turn_count > 0

    def _classify_prefix(self, messages: list[dict[str, Any]]) -> str:
        """Structural relationship of *messages* to the snapshot prefix.

        Returns one of ``_REL_FIRST_TURN``, ``_REL_EXTENDS``,
        ``_REL_DUPLICATE``, ``_REL_PREFIX_CHANGED``.
        """
        if self.turn_count == 0:
            return _REL_FIRST_TURN
        incoming = len(messages)
        if incoming < self.message_count:
            # Can't cover our snapshot prefix at all → history shrank/rewritten.
            return _REL_PREFIX_CHANGED
        if _per_message_fingerprints(messages[: self.message_count]) != self._prefix_fps:
            # Same-or-greater length, but the snapshot-covered messages differ.
            return _REL_PREFIX_CHANGED
        if incoming == self.message_count:
            # Prefix matches and nothing was appended → identical resend.
            return _REL_DUPLICATE
        return _REL_EXTENDS

    def is_cumulative(self, messages: list[dict[str, Any]]) -> bool:
        """Whether *messages* is a (possibly empty) cumulative extension.

        True on the first turn and when the incoming list strictly extends the
        verified snapshot prefix. False when the list does not advance
        (duplicate) or the snapshot-covered prefix changed.
        """
        return self._classify_prefix(messages) in (_REL_FIRST_TURN, _REL_EXTENDS)

    def continues(self, messages: list[dict[str, Any]]) -> bool:
        """Whether *messages* continues THIS established lineage.

        Used by :class:`SessionSlots` to route a request to the slot it belongs
        to. True when the incoming list extends the verified snapshot prefix or
        is an exact resend (a replay of this lineage's current turn); False on a
        fresh (turn-0) accumulator and when the snapshot prefix diverges — the
        latter is precisely a *different* lineage (e.g. a subagent with its own
        system prompt), which should open its own slot rather than reset this
        one. Unlike :meth:`is_cumulative`, this excludes the turn-0 case (a fresh
        slot is never a continuation target) and includes the duplicate/replay
        case (a resend still belongs to this lineage).
        """
        if self.turn_count == 0:
            return False
        return self._classify_prefix(messages) in (_REL_EXTENDS, _REL_DUPLICATE)

    def plan_turn(self, messages: list[dict[str, Any]]) -> TurnPlan:
        """Decide how to handle an incoming chat request for an active session.

        Returns a :class:`TurnPlan`: ``extend`` (bridge the new messages),
        ``replay`` (a duplicate resend — regenerate from the same prompt and
        overwrite the current turn in place, no reset), or ``reset`` (with the
        classified :class:`ResetReason` and diagnostics for the log). Does not
        call the renderer — a structurally valid extension that the renderer
        later declines is surfaced separately by the caller as
        :attr:`ResetReason.RENDERER_NO_BRIDGE`.
        """
        relation = self._classify_prefix(messages)
        diag: dict[str, Any] = {"incoming_len": len(messages), "relation": relation}

        if relation in (_REL_FIRST_TURN, _REL_EXTENDS):
            new_messages = extract_new_messages(messages, self.message_count)
            if new_messages:
                return TurnPlan(action="extend", new_messages=new_messages, diagnostics=diag)
            # Clean extension, but the appended slice is assistant-only.
            new_slice = messages[self.message_count :]
            diag["new_roles"] = [m.get("role") for m in new_slice]
            return TurnPlan(action="reset", reason=ResetReason.EMPTY_DELTA, diagnostics=diag)

        if relation == _REL_DUPLICATE:
            if self._snapshot_mono is not None:
                diag["age_s"] = round(time.monotonic() - self._snapshot_mono, 2)
            # Identical resend (upstream retry): the conversation did not advance.
            # Regenerate from the same prompt and overwrite this turn in place —
            # do NOT reset (that would drop drift-free state and break the segment
            # for a single step).
            return TurnPlan(action="replay", reason=ResetReason.DUPLICATE, diagnostics=diag)

        # _REL_PREFIX_CHANGED
        diag.update(self._divergence_diag(messages))
        # Token count of the (typically compacted) incoming request, so a
        # prefix_changed line shows how far the history shrank — compare
        # snapshot_tokens (pre-reset history) against the model's context limit
        # to confirm a token-limit-triggered compaction.
        diag["incoming_tokens"] = self._count_tokens(messages)
        return TurnPlan(action="reset", reason=ResetReason.PREFIX_CHANGED, diagnostics=diag)

    def _count_tokens(self, messages: list[dict[str, Any]]) -> int | None:
        """Best-effort token count of *messages* via the renderer.

        Returns ``None`` if the renderer can't tokenize the list (e.g. a mock,
        or content it rejects) — diagnostics must never break the reset path.
        """
        try:
            return len(self.renderer.render_ids(messages))
        except Exception:
            return None

    def _divergence_diag(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        """Locate where *messages* first diverges from the snapshot prefix."""
        incoming_fps = _per_message_fingerprints(messages)
        overlap = min(len(incoming_fps), len(self._prefix_fps))
        idx = next((i for i in range(overlap) if incoming_fps[i] != self._prefix_fps[i]), overlap)
        diag: dict[str, Any] = {"first_divergent_index": idx}
        if idx < len(messages):
            diag["incoming_at_divergence"] = {
                "role": messages[idx].get("role"),
                "preview": _content_preview(messages[idx]),
            }
        return diag

    @staticmethod
    def _explain(reason: ResetReason, diag: dict[str, Any]) -> str:
        """A short plain-language note on *why* a reset happened, for the log."""
        if reason is ResetReason.DUPLICATE:
            age = diag.get("age_s")
            when = f", resent {age}s later" if age is not None else ""
            return f"identical to the last processed turn{when} — likely an upstream retry/duplicate"
        if reason is ResetReason.PREFIX_CHANGED:
            idx = diag.get("first_divergent_index")
            at = diag.get("incoming_at_divergence") or {}
            where = f" at msg #{idx}" if idx is not None else ""
            detail = f" (role={at.get('role')!r}: {at.get('preview')!r})" if at else ""
            return f"processed history changed{where}{detail} — compacted/edited or session id reused"
        if reason is ResetReason.EMPTY_DELTA:
            return f"only adds assistant msg(s) {diag.get('new_roles')} — nothing new to render"
        if reason is ResetReason.RENDERER_NO_BRIDGE:
            rdr = diag.get("renderer", "the renderer")
            return f"{rdr} couldn't bridge new msg(s) {diag.get('new_roles')} for this model"
        return "manual reset"

    def reset(self, reason: ResetReason = ResetReason.MANUAL, *, diagnostics: dict[str, Any] | None = None) -> None:
        """Clear all accumulated state, restarting this session's history.

        Logs *why* at INFO: the machine-readable ``reason`` plus a plain-language
        explanation and the structural context. The reason distinguishes benign
        compaction from upstream retries and renderer gaps.
        """
        self.reset_count += 1
        diag = diagnostics or {}
        # snapshot_tokens = size of the accumulated history right before this
        # reset (prompt + completion of the last ingested turn). Compare against
        # the model's context limit to spot token-limit-triggered compaction.
        snapshot_tokens = len(self.prev_prompt_ids) + len(self.prev_completion_ids)
        logger.info(
            "TokenAccumulator reset (session=%s) reason=%s: %s [turn=%d msgs=%d snapshot_tokens=%d incoming=%s incoming_tokens=%s reset_count=%d]",
            self.session_id,
            reason.value if isinstance(reason, ResetReason) else reason,
            self._explain(reason, diag),
            self.turn_count,
            self.message_count,
            snapshot_tokens,
            diag.get("incoming_len", "?"),
            diag.get("incoming_tokens", "?"),
            self.reset_count,
        )
        self.prev_prompt_ids = []
        self.prev_completion_ids = []
        self.turn_count = 0
        self.message_count = 0
        self._prefix_fps = []
        self._snapshot_mono = None

    def ingest_turn(self, prompt_token_ids: list[int], completion_token_ids: list[int], *, advance: bool = True) -> None:
        """Record the token IDs from a completed turn.

        ``prompt_token_ids`` is the full prompt the turn was sampled from (turn
        1: the chat-rendered prompt; later turns: the bridge output we built).
        ``completion_token_ids`` is what the model produced. Together they form
        ``previous_prompt`` / ``previous_completion`` for the next bridge call.

        ``advance=False`` overwrites the most recent turn in place without bumping
        ``turn_count`` — used when replaying a duplicate resend, where the
        conversation did not move forward but the regenerated tokens replace the
        prior sample.
        """
        self.prev_prompt_ids = list(prompt_token_ids)
        self.prev_completion_ids = list(completion_token_ids)
        if advance:
            self.turn_count += 1

    def update_prefix(self, messages: list[dict[str, Any]]) -> None:
        """Snapshot the current message list as the verified prefix."""
        self.message_count = len(messages)
        self._prefix_fps = _per_message_fingerprints(messages)
        self._snapshot_mono = time.monotonic()

    def build_next_prompt(
        self,
        new_messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
    ) -> list[int] | None:
        """Construct the full prompt token IDs for the next turn.

        Delegates to ``renderers.bridge_to_next_turn``, which extends the prior
        ``prev_prompt_ids + prev_completion_ids`` with the rendered new messages
        plus the next generation prompt. Returns the full token-ID sequence, or
        ``None`` if the renderer cannot prove the prefix-extension contract (in
        which case the caller falls back to the normal chat path).
        """
        rendered = self.renderer.bridge_to_next_turn(
            self.prev_prompt_ids,
            self.prev_completion_ids,
            new_messages,
            tools=tools,
        )
        if rendered is None:
            return None
        return list(rendered.token_ids)


class SessionSlots:
    """The open conversation *lineages* of a single gateway session.

    One session URL (``/sessions/<uid>/v1``) can carry several concurrent
    lineages that do **not** share a growing prefix — most importantly a
    subagent (opencode / claude-code ``task`` tool) that runs under the same
    session but with its own system prompt and tool set. A single
    :class:`TokenAccumulator` would ``reset()`` on every switch between the
    parent and a subagent, and after a reset the resumed turn is re-tokenized
    from text (drifting off the pre-subagent tokens) — defeating cumulative
    token mode and fragmenting the training trajectory.

    This holds one accumulator ("slot") per lineage and routes each incoming
    request to the slot it *continues* (:meth:`TokenAccumulator.continues`),
    opening a **new** slot when it continues none instead of resetting an
    existing one. So the parent lineage stays bridged (drift-free) straight
    through a subagent interruption, each subagent lineage is bridged
    internally, and the session's turns form a forest / DAG of prefix chains.

    Requests within a session are issued sequentially by the agent (a subagent
    ``task`` call blocks the parent until it returns), so no locking is needed.
    """

    def __init__(self, renderer: Any, session_id: str | None = None, *, max_slots: int = 64) -> None:
        self._renderer = renderer
        self._session_id = session_id or "unknown"
        self._max_slots = max_slots
        self._slots: list[TokenAccumulator] = []
        self._active: TokenAccumulator | None = None
        # Monotonic selection counter → least-recently-selected eviction.
        self._use_clock: int = 0
        self._last_use: dict[int, int] = {}
        # Monotonic lineage counter → stable per-lineage ids (never reused, so
        # an evicted-then-reappearing lineage still gets a fresh id).
        self._next_lineage_num: int = 0

    def _new_slot(self) -> TokenAccumulator:
        slot = TokenAccumulator(self._renderer, session_id=self._session_id)
        slot.lineage_id = f"{self._session_id}#{self._next_lineage_num}"
        self._next_lineage_num += 1
        self._slots.append(slot)
        return slot

    def select(self, messages: list[dict[str, Any]]) -> TokenAccumulator:
        """Return the slot *messages* continues (deepest match), or a fresh slot.

        The chosen slot is marked active so the request's later ingest
        (``ingest_turn`` / ``update_prefix`` in the proxy) lands on the same
        slot. Prefer the deepest (longest-snapshot) matching lineage; distinct
        lineages have distinct roots, so at most one normally matches.
        """
        best: TokenAccumulator | None = None
        best_depth = -1
        for slot in self._slots:
            if slot.message_count > best_depth and slot.continues(messages):
                best = slot
                best_depth = slot.message_count
        if best is None:
            best = self._new_slot()
            self._evict_if_needed(keep=best)
        self._mark_active(best)
        return best

    @property
    def active(self) -> TokenAccumulator:
        """The slot chosen by the last :meth:`select` (created lazily if none).

        The proxy's turn-0 ingest sites read this to ingest into the same slot
        that ``handle()`` selected for the request. Never ``None``.
        """
        if self._active is None:
            self._mark_active(self._new_slot())
        return self._active  # type: ignore[return-value]

    def _mark_active(self, slot: TokenAccumulator) -> None:
        self._active = slot
        self._use_clock += 1
        self._last_use[id(slot)] = self._use_clock

    def _evict_if_needed(self, *, keep: TokenAccumulator) -> None:
        """Cap the number of open lineages; drop the least-recently-selected.

        The slot count is bounded by the session's turn count (one new slot per
        non-continuing request) and sessions are deleted after each rollout, so
        this practically never fires — it is a defensive bound against a
        pathological reset/lineage storm. The active and just-added slots are
        never evicted, so an in-progress parent lineage survives a subagent.
        """
        if len(self._slots) <= self._max_slots:
            return
        evictable = [s for s in self._slots if s is not keep and s is not self._active]
        if not evictable:
            return
        victim = min(evictable, key=lambda s: self._last_use.get(id(s), 0))
        self._slots.remove(victim)
        self._last_use.pop(id(victim), None)
        logger.warning(
            "SessionSlots(session=%s) exceeded max_slots=%d; evicted least-recently-used lineage (turn=%d, msgs=%d). Possible reset/lineage storm.",
            self._session_id,
            self._max_slots,
            victim.turn_count,
            victim.message_count,
        )

    @property
    def slot_count(self) -> int:
        return len(self._slots)
