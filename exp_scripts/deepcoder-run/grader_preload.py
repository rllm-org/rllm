"""Forkserver preload: cap the address space of every grading child.

livecodebench.run_test calls reliability_guard() with maximum_memory_bytes=None, so a
model-generated program may allocate without bound. On 2026-09-19 one such grading
child (a fork of the forkserver, cmdline "multiprocessing.forkserver main") reached
148.7 GB RSS, pushed .24 to 95% RAM and Ray's memory monitor killed the training
workers (run aborted at step 233). This module is imported by the forkserver server
process at startup (set_forkserver_preload); rlimits are inherited by every child it
forks, so each grading subprocess gets MemoryError instead of eating the host.
"""
import os
import resource

GRADER_MAX_AS_BYTES = int(os.environ.get("GRADER_MAX_AS_GB", "16")) * 1024**3

soft, hard = resource.getrlimit(resource.RLIMIT_AS)
if hard == resource.RLIM_INFINITY or hard > GRADER_MAX_AS_BYTES:
    resource.setrlimit(resource.RLIMIT_AS, (GRADER_MAX_AS_BYTES, GRADER_MAX_AS_BYTES))
