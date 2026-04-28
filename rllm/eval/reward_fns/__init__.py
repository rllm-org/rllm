"""Built-in score functions used by `tests/evaluate.py` files and catalog datasets.

Each module exports a single ``evaluate(task: Task, episode: Episode) -> EvalOutput``
function. Catalog entries reference these via:

    [verifier]
    import_path = "rllm.eval.reward_fns.math:evaluate"

User-authored ``tests/evaluate.py`` files can also import from here to
reuse common scoring without rewriting boxed-answer extraction etc.

These are wrappers over the heavier reward infrastructure in
:mod:`rllm.rewards`; that module stays the source of truth for the
actual grading logic.
"""
