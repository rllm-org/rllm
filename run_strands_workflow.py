import asyncio

from rllm.engine.rollout_engine import RolloutEngine
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.workflows.strands_workflow import StrandsWorkflow
# from rllm.integrations.strands_adapter import strands_session_factory


async def main() -> None:
    rollout = RolloutEngine(config=None)

    async def session_factory(**kw):
        # Placeholder: wire your own session factory or adapter here
        # return await strands_session_factory(**kw)
        class _NoopSession:
            async def step(self, user_msg: str):
                if False:
                    yield {"type": "Stop", "final_text": ""}

            async def close(self):
                return None

        return _NoopSession()

    awe = AgentWorkflowEngine(
        workflow_cls=StrandsWorkflow,
        workflow_args={
            "strands_session_factory": session_factory,
            "system_prompt": "You are a helpful agent that uses tools when beneficial.",
            "tools": [],
            "max_steps": 8,
            "reward_fn": None,
        },
        rollout_engine=rollout,
        config=None,
        n_parallel_tasks=2,
        retry_limit=1,
    )

    tasks = ["Example question 1"]
    ids = ["ep-0"]
    episodes = await awe.execute_tasks(tasks, task_ids=ids, workflow_id="strands-example")
    print("done:", [ep.id for ep in episodes])


if __name__ == "__main__":
    asyncio.run(main())


