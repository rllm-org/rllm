import os

# Set training mode BEFORE importing search_agent_strands (reads TRAIN env var at import time)
os.environ["TRAIN"] = "1"

# Force contextvar session backend BEFORE importing rllm.sdk modules
# (The env var RLLM_SESSION_BACKEND is ignored because rllm/sdk/session/__init__.py reads from config.yaml)
import rllm.sdk.session as _session_module
_session_module.SESSION_BACKEND = "contextvar"

import hydra

from rllm.data import DatasetRegistry
from rllm.trainer import AgentTrainer

from search_agent_strands import run_search_agent


# max_steps default=5 is the effective value. AgentSdkEngine does NOT inject
# rllm.agent.max_steps from config — that shell-script param is dead for the SDK path.
async def run_agent(question, ground_truth, max_steps=5, **kwargs):
    try:
        result = await run_search_agent(question, ground_truth, max_turns=max_steps)
    except Exception as e:
        print(f"[run_agent] Error: {e}")
        return 0.0
    import gc
    gc.collect()  # Clean up Agent objects after each rollout
    return result["reward"]


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(config):
    train_dataset = DatasetRegistry.load_dataset("hotpotqa", "train")
    val_dataset = DatasetRegistry.load_dataset("hotpotqa-small", "test")

    trainer = AgentTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        agent_run_func=run_agent,
    )

    from retrieve_tool import check_retrieval_server
    check_retrieval_server()  # Fail fast if port 9002 is down

    trainer.train()


if __name__ == "__main__":
    main()
