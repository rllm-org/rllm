"""Call the unchanged upstream Hydra entrypoint with a process-only evaluator."""
import os

if __name__ == "__main__":
    import train as upstream
    from compatible_grader import ForkserverEvaluator

    # DEEPCODER_DROP_TIMEOUTS=1 -> verifier timeouts become ERROR episodes that
    # rllm.compact_filtering drops (requires rllm.compact_filtering.enable=true).
    drop = os.environ.get("DEEPCODER_DROP_TIMEOUTS", "0") == "1"
    upstream.deepcoder_evaluator = ForkserverEvaluator(drop_timeouts=drop)
    print(f"[train_compatible] drop_timeouts={drop}", flush=True)
    upstream.main()
