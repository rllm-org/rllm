"""Stirrup configured for Artificial Analysis' GDPval-AA v2 contract.

:class:`~rllm.harnesses.stirrup.StirrupHarness` is benchmark-agnostic: it knows
how to run Stirrup inside an rLLM sandbox but carries no prompt, no submission
contract and no provenance format. This subclass supplies AA's, so the generic
harness stays usable for other datasets.

Every value here is read from :mod:`rllm.data.gdpval_aa` rather than restated,
so the published spec remains the single source of truth.
"""

from __future__ import annotations

from rllm.data import gdpval_aa
from rllm.harnesses.stirrup import StirrupHarness


class GdpvalStirrupHarness(StirrupHarness):
    """Stirrup under AA's GDPval-AA v2 runtime contract."""

    #: Deliberately *not* renamed. ``name`` is the agent component of the
    #: submission path (``<model>__<name>/``), so changing it would file these
    #: results under a new directory and orphan every earlier run.
    name = "stirrup"

    system_prompt = gdpval_aa.AA_GDPVAL_SYSTEM_PROMPT
    methodology = "GDPval-AA v2"
    workdir = gdpval_aa.AA_WORKDIR
    submittable_roots = gdpval_aa.AA_SUBMITTABLE_ROOTS
    #: Written next to each task by ``gdpval_builder``; carries the dataset
    #: revision, sandbox image digest and reference-file list that the
    #: submission manifest reports.
    provenance_filename = "gdpval_aa.json"

    #: Kept at GDPval's established layout so submissions land where every
    #: earlier run put them; the generic default would be /tmp/stirrup.
    submission_root = "/tmp/gdpval-aa"
    agent_name = "gdpval-aa-solver"

    max_turns = gdpval_aa.AA_MAX_TURNS
    shell_timeout = gdpval_aa.AA_SHELL_TIMEOUT_SEC
    context_summarization_cutoff = gdpval_aa.AA_CONTEXT_SUMMARIZATION_CUTOFF
