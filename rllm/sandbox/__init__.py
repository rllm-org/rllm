"""Ephemeral per-task sandboxes behind the :class:`rllm.sandbox.protocol.Sandbox` protocol.

Backends (local, docker, modal, daytona) are created via
:func:`rllm.sandbox.sandboxed_flow.create_sandbox`; per-task lifecycle is
managed by :class:`rllm.hooks.SandboxTaskHooks`, with optional cold-start
acceleration from :mod:`rllm.sandbox.snapshot` and prefetching from
:mod:`rllm.sandbox.warm_queue`.
"""
