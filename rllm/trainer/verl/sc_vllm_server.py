"""vLLM server actor class with the score-centering sampler head.

Importing this module imports vLLM, so it is only imported by the process that
creates rollout replicas (``install_rollout_adapter``) and, through Ray's
by-reference class pickling, by the vLLM server actors themselves. No Ray
worker setup hook is needed: Ray ignores ``worker_process_setup_hook`` in
per-actor ``runtime_env`` options, so a hook cannot reliably reach the servers.
"""

from verl.workers.rollout.vllm_rollout import vllm_async_server as _upstream

from rllm.trainer.verl.sc_rollout_adapter import build_sc_generate

UpstreamvLLMHttpServer = getattr(_upstream.vLLMHttpServer, "_rllm_sc_upstream", _upstream.vLLMHttpServer)


class SCvLLMHttpServer(UpstreamvLLMHttpServer):
    """Upstream server whose ``generate`` also returns the sampler head on SC requests."""

    _rllm_sc_upstream = UpstreamvLLMHttpServer


def _install_generate():
    generate = build_sc_generate(UpstreamvLLMHttpServer, _upstream)
    # cloudpickle ships functions by (module, qualname) reference when that name resolves
    # to the same object. Name it after this class: in the TaskRunner,
    # ``vllm_async_server.vLLMHttpServer.generate`` *also* resolves here (the class is
    # swapped there), so the server process would otherwise get the unpatched upstream
    # method. Globals still come from the upstream module.
    generate.__module__ = __name__
    generate.__qualname__ = "SCvLLMHttpServer.generate"
    SCvLLMHttpServer.generate = generate


_install_generate()
