"""Monkey-patches for Verl and vLLM within the rLLM unified trainer.

All patches are applied lazily (on first call) and are idempotent — calling
them multiple times is safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VERL_DYNAMIC_BATCH_PATCHED = False
_VLLM_SDK_PATCHED = False
_VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = False
_VERL_TENSORDICT_JAGGED_PATCHED = False


# ---------------------------------------------------------------------------
# Verl dynamic batch: sync micro-batch counts across DP ranks
# ---------------------------------------------------------------------------


def patch_verl_dynamic_batch_sync() -> None:
    """Patch ``prepare_dynamic_batch`` to sync micro-batch counts across DP ranks.

    Fixes `verl#5750 <https://github.com/verl-project/verl/issues/5750>`_:
    when ``use_dynamic_bsz=True``, each DP rank independently calculates
    ``num_micro_batches`` based on its local sequence lengths.  Different
    ranks can end up with different counts, causing NCCL collective
    operations (AllGather/ReduceScatter in FSDP) to deadlock.

    The fix defaults ``dp_group`` to ``torch.distributed.group.WORLD`` so
    that ``prepare_dynamic_batch`` performs an ``all_reduce(MAX)`` across
    ranks, forcing every rank to iterate through the same number of
    micro-batches.  This is the same approach as verl PR #5591.
    """
    global _VERL_DYNAMIC_BATCH_PATCHED
    if _VERL_DYNAMIC_BATCH_PATCHED:
        return

    import verl.utils.seqlen_balancing as sbl

    _original_prepare = sbl.prepare_dynamic_batch

    def _patched_prepare(data, max_token_len, dp_group=None, **kwargs):
        if dp_group is None:
            import torch.distributed

            if torch.distributed.is_initialized():
                dp_group = torch.distributed.group.WORLD
        return _original_prepare(data, max_token_len, dp_group=dp_group, **kwargs)

    sbl.prepare_dynamic_batch = _patched_prepare

    # Also patch the already-imported reference in dp_actor so both
    # compute_log_prob and update_policy use the patched version.
    try:
        from verl.workers.actor import dp_actor

        dp_actor.prepare_dynamic_batch = _patched_prepare
    except (ImportError, AttributeError):
        pass  # dp_actor may not be importable outside GPU workers

    _VERL_DYNAMIC_BATCH_PATCHED = True
    logger.info("Patched prepare_dynamic_batch to sync micro-batch counts across DP ranks (verl#5750)")


# ---------------------------------------------------------------------------
# vLLM SDK instrumentation
# ---------------------------------------------------------------------------


def patch_vllm_for_sdk() -> None:
    """Patch vLLM replicas to add logprob/token-id instrumentation for SDK traces.

    Creates an ``InstrumentedvLLMHttpServer`` Ray actor that loads
    ``rllm/patches/vllm_instrumentation.py`` in an isolated module namespace
    and patches ``vLLMReplica.__init__`` to use it.
    """
    global _VLLM_SDK_PATCHED
    if _VLLM_SDK_PATCHED:
        return

    import ray
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )

    @ray.remote(num_cpus=1)
    class InstrumentedvLLMHttpServer(vLLMHttpServer):
        """vLLM HTTP server with automatic vLLM instrumentation in Ray worker."""

        def __init__(self, *args, **kwargs):
            import importlib.util
            import sys
            from pathlib import Path

            instrumentation_path = Path(__file__).parent.parent.parent / "patches" / "vllm_instrumentation.py"

            spec = importlib.util.spec_from_file_location("rllm_vllm_instrumentation_isolated", str(instrumentation_path))
            vllm_instrumentation = importlib.util.module_from_spec(spec)
            sys.modules["rllm_vllm_instrumentation_isolated"] = vllm_instrumentation
            spec.loader.exec_module(vllm_instrumentation)

            vllm_instrumentation.instrument_vllm(add_response_logprobs=True)
            super().__init__(*args, **kwargs)

    _original_init = vLLMReplica.__init__

    def _patched_init(self, *args, **kwargs):
        _original_init(self, *args, **kwargs)
        self.server_class = InstrumentedvLLMHttpServer

    vLLMReplica.__init__ = _patched_init
    _VLLM_SDK_PATCHED = True
    logger.info("Patched vLLMReplica for SDK instrumentation")


# ---------------------------------------------------------------------------
# Verl qwen3_vl: out-of-place add in dummy visual forward (backport PR #5881)
# ---------------------------------------------------------------------------


def patch_verl_qwen3_vl_dummy_inplace() -> None:
    """Backport `volcengine/verl#5881` for ``verl.models.transformers.qwen3_vl``.

    The dummy/no-image branch of ``_get_input_embeds`` mutates ``inputs_embeds``
    inplace::

        inputs_embeds += 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds += 0.0 * emb.mean()

    ``inputs_embeds`` is produced by ``model.get_input_embeddings()(input_ids)``
    and is a leaf with ``requires_grad=True`` after FSDP wrapping; the inplace
    ``+=`` raises a RuntimeError on the autograd backward and, in our 0.7.1
    pin, also surfaces as ``CUDNN_STATUS_NOT_INITIALIZED`` on the preceding
    ``model.visual(...)`` call due to the way the failure propagates inside
    Conv3d's cuDNN workspace setup. PR #5881 (merged main 2026-04-07; not in
    0.7.1) replaces both lines with out-of-place addition.

    We patch by re-exec'ing a fixed source of ``_get_input_embeds`` in the
    module's global namespace; ``qwen3_vl_base_forward`` looks up
    ``_get_input_embeds`` from module globals at call time, so the rebound
    function is picked up automatically.
    """
    global _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED
    if _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED:
        return

    import inspect

    from verl.models.transformers import qwen3_vl as mod

    src = inspect.getsource(mod._get_input_embeds)
    new_src = src.replace(
        "inputs_embeds += 0.0 * image_embeds.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()",
    ).replace(
        "inputs_embeds += 0.0 * emb.mean()",
        "inputs_embeds = inputs_embeds + 0.0 * emb.mean()",
    )

    if new_src == src:
        # Upstream already fixed (e.g. user bumped verl past 0.7.1).
        _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
        logger.info("qwen3_vl PR #5881 patch: source already uses out-of-place add; nothing to patch.")
        return

    # Compile and exec into the module's globals so the rebound function shares
    # the module's namespace (torch, Optional, etc.) and so callers in the
    # same module (qwen3_vl_base_forward) pick up the patched version on the
    # next attribute lookup.
    exec(compile(new_src, mod.__file__, "exec"), mod.__dict__)

    _VERL_QWEN3_VL_DUMMY_INPLACE_PATCHED = True
    logger.info("Patched verl qwen3_vl._get_input_embeds: dummy visual path now uses out-of-place addition (backport of volcengine/verl#5881)")


# ---------------------------------------------------------------------------
# Verl tensordict utils: preserve _ragged_idx when rebuilding 3D NestedTensors
# (backport PR #6127)
# ---------------------------------------------------------------------------


def patch_verl_tensordict_jagged_layout() -> None:
    """Backport `volcengine/verl#6127` for ``verl.utils.tensordict_utils``.

    Fixes a bug in the rebuilding of 3D jagged NestedTensors after
    selection / chunking. ``torch.nested.as_nested_tensor(tensors,
    layout=torch.jagged)`` is ambiguous when all input tensors share the
    same last-dimension length — torch picks the wrong dimension as the
    jagged axis. For mRoPE ``position_ids`` with per-sample shape
    ``(num_heads, seq_len)``, this produces a rebuilt nested tensor whose
    ``_ragged_idx`` is 1 (heads dim) instead of 2 (seq dim), causing two
    downstream crashes:

    - ``index_select_tensor_dict`` → ``unbind()`` →
      ``torch.split(values, [batch_size], dim=ragged_idx-1)`` fails because
      ``values`` has total length = sum-of-row-lengths, not batch_size
      (hits the ``use_dynamic_bsz=True`` micro-batch partitioning path).
    - rmpad path's rotary cos/sin gets shape ``(B,)`` instead of ``(B, S)``
      and ``apply_rotary_pos_emb`` crashes with
      ``RuntimeError: The size of tensor a (<S>) must match the size of
      tensor b (<B>) at non-singleton dimension 2``.

    The fix introduces a ``nested_tensor_from_tensor_list(tensors,
    ragged_idx)`` helper that explicitly preserves the intended ragged
    dimension via ``torch.nested.nested_tensor_from_jagged`` plus an
    explicit ``_ragged_idx`` set, and uses it in ``concat_nested_tensors``,
    ``chunk_tensordict``, and ``index_select_tensor_dict``.

    Merged into verl main 2026-04-24; not in 0.7.1.
    """
    global _VERL_TENSORDICT_JAGGED_PATCHED
    if _VERL_TENSORDICT_JAGGED_PATCHED:
        return

    import torch
    from tensordict import TensorDict
    from verl.utils import tensordict_utils as tu

    if hasattr(tu, "nested_tensor_from_tensor_list"):
        # Upstream already provides the helper (e.g. user bumped verl past 0.7.1).
        _VERL_TENSORDICT_JAGGED_PATCHED = True
        logger.info("verl tensordict jagged-layout patch: upstream already provides nested_tensor_from_tensor_list; nothing to patch.")
        return

    def nested_tensor_from_tensor_list(tensors, ragged_idx=None):
        assert len(tensors) > 0, "Must provide at least one tensor"
        sample_dim = tensors[0].dim()
        if ragged_idx is None:
            ragged_idx = sample_dim
        assert ragged_idx == sample_dim, f"Only last-dimension ragged tensors are supported. Got {ragged_idx=} and {sample_dim=}"

        if sample_dim == 1:
            return torch.nested.as_nested_tensor(tensors, layout=torch.jagged)

        values = torch.cat(tensors, dim=-1)
        lengths = torch.tensor([t.shape[-1] for t in tensors], dtype=torch.long, device=values.device)
        offsets = torch.zeros(len(tensors) + 1, dtype=torch.long, device=values.device)
        torch.cumsum(lengths, dim=0, out=offsets[1:])

        nested_tensor = torch.nested.nested_tensor_from_jagged(values=values, offsets=offsets)
        nested_tensor._ragged_idx = ragged_idx
        return nested_tensor

    def concat_nested_tensors(tensors):
        for tensor in tensors:
            assert tensor.is_nested and tensor.is_contiguous()
        unbind_tensors = []
        for tensor in tensors:
            assert len(tensor.shape) >= 2, f"nested tensor must have 2 or more dimensions. Got {tensor.shape}"
            unbind_tensors.extend(list(tensor.unbind(0)))
        return nested_tensor_from_tensor_list(unbind_tensors, ragged_idx=tensors[0].dim() - 1)

    def chunk_tensordict(td, chunks):
        assert isinstance(td, TensorDict) and len(td) % chunks == 0, f"expecting td with length divisible by chunks, but got {len(td)} and {chunks}"
        chunk_size = len(td) // chunks
        nested_keys = {key for key, val in td.items() if isinstance(val, torch.Tensor) and val.is_nested}
        new_td = TensorDict({k: v for k, v in td.items() if k not in nested_keys}, batch_size=td.batch_size, device=td.device)
        tds = new_td.chunk(chunks=chunks)
        for key in nested_keys:
            nt = td[key]
            try:
                tensors = nt.unbind(dim=0)
            except RuntimeError:
                padded = nt.to_padded_tensor(0)
                padded_chunks = padded.chunk(chunks, dim=0)
                offsets = nt.offsets()
                lengths = offsets.diff().tolist()
                for i, chunk_td in enumerate(tds):
                    chunk_lengths = lengths[i * chunk_size : (i + 1) * chunk_size]
                    chunk_tensors = [padded_chunks[i][j, :seq_len] for j, seq_len in enumerate(chunk_lengths)]
                    chunk_td[key] = nested_tensor_from_tensor_list(chunk_tensors, ragged_idx=nt.dim() - 1)
                continue
            for i, chunk_td in enumerate(tds):
                chunk_td[key] = nested_tensor_from_tensor_list(list(tensors[i * chunk_size : (i + 1) * chunk_size]), ragged_idx=nt.dim() - 1)
        return tds

    def index_select_tensor_dict(batch, indices):
        if isinstance(indices, list):
            indices = torch.tensor(indices)
        assert indices.dim() == 1, "indices must be a 1D tensor"
        data_dict = {}
        batch_size = indices.shape[0]
        if batch is not None:
            for key, tensor in batch.items():
                if isinstance(tensor, torch.Tensor) and not tensor.is_nested:
                    data_dict[key] = tensor[indices]
                elif isinstance(tensor, torch.Tensor) and tensor.is_nested:
                    tensor_lst = tensor.unbind()
                    selected_tensors = [tensor_lst[idx] for idx in indices]
                    data_dict[key] = nested_tensor_from_tensor_list(selected_tensors, ragged_idx=tensor.dim() - 1)
                else:
                    if tensor.shape:
                        data_dict[key] = tensor[indices]
                    else:
                        data_dict[key] = tensor
            selected_batch = TensorDict(source=data_dict, batch_size=batch_size)
        else:
            selected_batch = None
        return selected_batch

    tu.nested_tensor_from_tensor_list = nested_tensor_from_tensor_list
    tu.concat_nested_tensors = concat_nested_tensors
    tu.chunk_tensordict = chunk_tensordict
    tu.index_select_tensor_dict = index_select_tensor_dict

    _VERL_TENSORDICT_JAGGED_PATCHED = True
    logger.info("Patched verl.utils.tensordict_utils: rebuild jagged NestedTensors with explicit _ragged_idx (backport of volcengine/verl#6127)")


# ---------------------------------------------------------------------------
# Worker-side entry point (used as Ray runtime_env worker_process_setup_hook)
# ---------------------------------------------------------------------------

_ALL_VERL_PATCHES = {
    "patch_verl_dynamic_batch_sync": patch_verl_dynamic_batch_sync,
    "patch_verl_qwen3_vl_dummy_inplace": patch_verl_qwen3_vl_dummy_inplace,
    "patch_verl_tensordict_jagged_layout": patch_verl_tensordict_jagged_layout,
}


def apply_all_verl_patches() -> None:
    """Apply every Verl patch that is safe to run unconditionally on workers.

    Designed to be wired in as ``runtime_env.worker_process_setup_hook =
    "rllm.experimental.verl.patch.apply_all_verl_patches"`` so that each Ray
    worker process applies the patches in its own interpreter (driver-side
    monkey-patches do not propagate to worker processes).

    Each patch below is lazy and idempotent, so it is safe to call this
    repeatedly and from any process.

    Optional extension hook: if the ``RLLM_EXTRA_WORKER_SETUP_HOOK``
    environment variable is set, this function will additionally invoke the
    callable it names. The value is ``"<absolute-file-path.py>:<func>"``;
    the function is loaded directly from the file via ``importlib.util`` so
    it does not need to live in a package on ``sys.path``. This is intended
    for environment-specific workarounds (e.g. disabling cuDNN on a host
    with a broken cuDNN install) that should not be baked into rLLM itself.

    Note: ``patch_vllm_for_sdk`` is NOT included here — it is gated behind
    the ``rllm.sdk.enable`` config flag and should only be applied when SDK
    instrumentation is requested.
    """
    for patch_name, patch_func in _ALL_VERL_PATCHES.items():
        try:
            patch_func()
        except Exception:  # pragma: no cover — patch is best-effort
            logger.exception(f"{patch_name} failed in worker setup hook")

    _run_extra_worker_setup_hook()


def _run_extra_worker_setup_hook() -> None:
    """Optionally invoke a user-supplied setup hook from RLLM_EXTRA_WORKER_SETUP_HOOK.

    Format: ``"<absolute path to .py file>:<function name>"``. The function
    is loaded via ``importlib.util.spec_from_file_location`` so it works
    even when the file is not on ``sys.path`` (e.g. lives under a
    gitignored ``tmp/`` directory). Failures are logged but never raised —
    this is a best-effort extension point and must not crash worker init.
    """
    import os

    spec_str = os.environ.get("RLLM_EXTRA_WORKER_SETUP_HOOK")
    if not spec_str:
        return

    path, _, func_name = spec_str.rpartition(":")
    if not path or not func_name:
        logger.warning(
            "RLLM_EXTRA_WORKER_SETUP_HOOK=%r is malformed; expected '<file.py>:<func>'",
            spec_str,
        )
        return

    try:
        import importlib.util

        mod_name = f"_rllm_extra_setup_hook_{os.getpid()}"
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            logger.warning("RLLM_EXTRA_WORKER_SETUP_HOOK: could not load spec for %s", path)
            return
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        getattr(mod, func_name)()
    except Exception:
        logger.exception("RLLM_EXTRA_WORKER_SETUP_HOOK=%r failed", spec_str)
