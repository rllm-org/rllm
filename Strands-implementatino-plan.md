Add one Workflow (StrandsWorkflow) that converts Strands’ event stream → Episode.

(Optional) Tiny adapter to stream Strands events (if your agent already streams MCP events, you don’t need this file).

One run snippet that wires AWE to the new Workflow.

1) New file: rllm/workflows/strands_workflow.py
# rllm/workflows/strands_workflow.py
from __future__ import annotations
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
from dataclasses import dataclass

from rllm.agents.agent import Episode, Trajectory, Step
from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.workflow import Workflow, TerminationReason

@dataclass
class StrandsEvent:  # standardize 4 minimal event types; if you already have this, reuse it.
    type: str                                # "TextDelta" | "ToolUse" | "ToolResult" | "Stop"
    text: Optional[str] = None               # for TextDelta
    name: Optional[str] = None               # for ToolUse
    input: Optional[Dict[str, Any]] = None   # for ToolUse
    toolUseId: Optional[str] = None          # for ToolUse/ToolResult
    content: Optional[Any] = None            # for ToolResult (usually text chunks)
    status: Optional[str] = None             # "success" | "error"
    final_text: Optional[str] = None         # for Stop
    logprob: Optional[float] = None          # optional: action span logprob for RL

class StrandsWorkflow(Workflow):
    """
    Minimal Workflow that:
      - Invokes a Strands Agent once (Strands orchestrates tool loop)
      - Consumes its event stream (TextDelta / ToolUse / ToolResult / Stop)
      - Packs rLLM Episode/Trajectory/Step so AgentWorkflowEngine can push to VERL
    """
    def __init__(
        self,
        rollout_engine: RolloutEngine,
        strands_session_factory,          # callable: (system_prompt, tools, **kw) -> session with .step(user_msg) -> AsyncIterator[StrandsEvent]
        system_prompt: str,
        tools: List[Dict[str, Any]],
        max_steps: int = 8,
        reward_fn=None,                   # callable: (episode) -> float OR None (AWE can still proceed with 0)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rollout_engine = rollout_engine
        self.session_factory = strands_session_factory
        self.system_prompt = system_prompt
        self.tools = tools
        self.max_steps = max_steps
        self.reward_fn = reward_fn

    async def __call__(self, task: str, uid: str) -> Episode:
        """
        Returns an rLLM Episode compatible with AgentWorkflowEngine._transform_results_for_verl:
          - episode.trajectories: List[(name, Trajectory)]
          - Trajectory.steps: List[Step]
          - Step.chat_completions: List[chat messages]  (the engine's chat_parser will tokenize these)
          - Trajectory.reward / Step.reward: float (optional; can set only final reward)
          - episode.is_correct / termination_reason / metrics: populated for VERL stats
        """
        # 1) Build initial chat and open a Strands session (Strands owns tool loop)
        chat: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task},
        ]
        session = await self.session_factory(system_prompt=self.system_prompt, tools=self.tools)

        trajectory = Trajectory(steps=[], reward=0.0)
        assistant_buffer = ""     # accumulate free-form assistant text between events
        num_steps = 0
        final_text = ""
        term_reason = TerminationReason.ENV_DONE

        try:
            async for ev in session.step(task):  # ← Strands stream: yield dicts or StrandsEvent
                et = ev["type"] if isinstance(ev, dict) else ev.type

                if et == "TextDelta":
                    txt = ev["text"] if isinstance(ev, dict) else ev.text
                    if txt:
                        assistant_buffer += txt

                elif et == "ToolUse":
                    # Flush buffered assistant text (if any) as an assistant message
                    if assistant_buffer:
                        chat.append({"role": "assistant", "content": assistant_buffer})
                        assistant_buffer = ""

                    # Add assistant message that indicates the tool call (MCP style)
                    name = ev["name"] if isinstance(ev, dict) else ev.name
                    args = ev.get("input") if isinstance(ev, dict) else ev.input
                    tuid = ev.get("toolUseId") if isinstance(ev, dict) else ev.toolUseId

                    chat.append({
                        "role": "assistant",
                        "content": [{"toolUse": {"name": name, "input": args or {}, "toolUseId": tuid or ""}}],
                    })

                    # OPTIONAL: you may want to record an intermediate step here for step-wise advantage.
                    # We'll wait for ToolResult so the step contains both action and observation.

                elif et == "ToolResult":
                    # Insert tool result as a user message (MCP convention)
                    tuid = ev.get("toolUseId") if isinstance(ev, dict) else ev.toolUseId
                    content = ev.get("content") if isinstance(ev, dict) else ev.content
                    status = ev.get("status") if isinstance(ev, dict) else ev.status or "success"

                    chat.append({
                        "role": "user",
                        "content": [{"toolResult": {"toolUseId": tuid or "", "status": status, "content": content}}],
                    })

                    # Now we have (… assistant toolUse) -> (… user toolResult). That’s one RL step.
                    step = Step(chat_completions=list(chat), reward=None)
                    trajectory.steps.append(step)
                    num_steps += 1
                    if num_steps >= self.max_steps:
                        term_reason = TerminationReason.MAX_STEPS
                        break

                elif et == "Stop":
                    final_text = ev.get("final_text") if isinstance(ev, dict) else ev.final_text or assistant_buffer
                    if final_text:
                        chat.append({"role": "assistant", "content": final_text})
                        assistant_buffer = ""
                    # Finalize a step if there were no tools at all or last step ended in text
                    if not trajectory.steps or trajectory.steps[-1].chat_completions != chat:
                        trajectory.steps.append(Step(chat_completions=list(chat), reward=None))
                    break

                # (optional) handle errors/timeouts here and set term_reason accordingly

        finally:
            # ensure session closed
            try:
                await session.close()
            except Exception:
                pass

        # 2) Fill Episode-level fields expected by AWE/VERL
        #    - If you have a task scorer, compute a terminal reward here.
        if self.reward_fn is not None:
            trajectory.reward = float(self.reward_fn(trajectory=trajectory, final_text=final_text, steps=num_steps))
        else:
            trajectory.reward = 0.0

        episode = Episode(
            id=uid,
            trajectories=[("solver", trajectory)],
            is_correct=None,                       # keep None/False if you don’t have correctness labels
            termination_reason=term_reason,
            metrics={"num_steps": num_steps},
        )
        return episode


Notes

This does not reimplement any tool loop. Strands orchestrates; we just mirror its events into chat_completions.

AgentWorkflowEngine._transform_results_for_verl(...) already knows how to read Step.chat_completions with your tokenizer/chat_parser and build VERL’s DataProto (step-wise or last-step). We reuse it as-is.

If you already have a streaming Strands agent (emitting MCP-style toolUse/toolResult), you can pass its session_factory straight in; no extra adapter needed.

2) (Optional) If you need a tiny adapter: rllm/integrations/strands_adapter.py

Use only if your current Strands agent returns raw SDK events and you want to normalize to the 4 minimal event types above. Keep it tiny:

# rllm/integrations/strands_adapter.py
from typing import Any, AsyncIterator, Dict

class StrandsSession:
    def __init__(self, agent, model_provider):
        self.agent = agent
        self.model_provider = model_provider

    async def step(self, user_msg: str) -> AsyncIterator[Dict[str, Any]]:
        # forward Strands events as standardized dicts
        async for e in self.agent.stream(user_msg, model=self.model_provider):
            # Map SDK-specific shapes → {"type": "...", ...}
            yield e  # if your agent already emits {"type": "TextDelta"/"ToolUse"/"ToolResult"/"Stop"}, no-op.

    async def close(self) -> None:
        await self.agent.aclose()

async def strands_session_factory(*, system_prompt: str, tools, **kw) -> StrandsSession:
    agent = kw["agent"]          # inject your existing Strands Agent (already configured with system+tools)
    model_provider = kw["model"] # your trainable policy; can use RolloutEngine internally
    return StrandsSession(agent, model_provider)


Again, if your agent already returns normalized events, you don’t need this file—just pass your own session_factory.

3) Use it with AWE (no changes to AWE)
# run_strands_workflow.py (or your existing runner)
import asyncio
from rllm.engine.agent_workflow_engine import AgentWorkflowEngine
from rllm.engine.rollout_engine import RolloutEngine
from rllm.workflows.strands_workflow import StrandsWorkflow
# from rllm.integrations.strands_adapter import strands_session_factory  # if you used the optional adapter

async def main():
    rollout = RolloutEngine(config=...)   # ← reuse your existing setup

    # Prepare a session factory.
    # If your Strands Agent already exposes a `stream()` that yields normalized events,
    # pass that as the factory. Otherwise, use the tiny adapter above.
    async def session_factory(**kw):
        # Provide your Strands agent + model provider here; no tool loop re-implementation.
        return await strands_session_factory(**kw)  # or your own factory

    workflow = StrandsWorkflow(
        rollout_engine=rollout,
        strands_session_factory=session_factory,
        system_prompt="You are a helpful agent that uses tools when beneficial.",
        tools=[...],              # reuse your existing tool specs
        max_steps=8,
        reward_fn=None,           # or a callable for terminal reward
    )

    awe = AgentWorkflowEngine(
        rollout_engine=rollout,
        workflow_cls=lambda: workflow,   # you can also pass workflow_args if you construct in AWE
        n_parallel_tasks=64,
        retry_limit=1,
        episode_store=None,
        config=...                       # your trainer/data config; unchanged
    )

    tasks = ["Question 1...", "Question 2..."]
    ids = [f"ep-{i}" for i in range(len(tasks))]
    episodes = await awe.execute_tasks(tasks, task_ids=ids, workflow_id="gaia-strands")
    print("done:", [ep.id for ep in episodes])

if __name__ == "__main__":
    asyncio.run(main())