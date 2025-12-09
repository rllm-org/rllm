import re

from examples.sdk.dispatch_search.rag import ClaimRAGTool
from examples.sdk.dispatch_search.utils import DISPATCH_USER_PROMPT_STEP_1, DISPATCH_USER_PROMPT_STEP_2, DISPATCHER_SYSTEM_PROMPT, SEARCHER_SYSTEM_PROMPT, SEARCHER_USER_PROMPT, DispatchAction, FinalResponse
from rllm.agents.agent import Episode, Step, Trajectory
from rllm.engine.rollout import RolloutEngine
from rllm.engine.rollout.rollout_engine import ModelOutput
from rllm.utils import colorful_print
from rllm.workflows.workflow import TerminationReason, Workflow


class DispatcherAgent:
    """
    A dispatcher agent who is managing two search agents, `SEARCHER_A` and `SEARCHER_B`, each of which is specialized in
    retrieving information from a particular domain (but you do not know yet which domain each agent is specialized in).
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    def _parse_dispatch_response(self, response: str) -> DispatchAction:
        dispatch_match = re.search(r"<dispatch>(.*?)</dispatch>", response, re.IGNORECASE | re.DOTALL)
        if dispatch_match:
            return DispatchAction.from_raw_string(dispatch_match.group(1).strip())
        else:
            return DispatchAction.DISPATCH_ERROR

    def _parse_final_response(self, response: str) -> FinalResponse:
        final_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if final_match:
            return FinalResponse.from_raw_string(final_match.group(1).strip())
        else:
            return FinalResponse.ERROR

    def _parse_searcher_response(self, response: dict[str, str]) -> str:
        """
        We are not super strict about the formatting of the searcher responses
        """
        searcher_a_response, searcher_b_response = response.get("searcher_a", ""), response.get("searcher_b", "")
        assert not (len(searcher_a_response) == 0 and len(searcher_b_response) == 0), "Both searcher responses cannot be empty"
        all_responses = ""
        if searcher_a_response:
            all_responses += f"<searcher_a_response>\n{searcher_a_response}\n</searcher_a_response>\n"
        if searcher_b_response:
            all_responses += f"<searcher_b_response>\n{searcher_b_response}\n</searcher_b_response>\n"
        return all_responses

    async def dispatch_step(self, claim: str, debug: bool = False) -> Step:
        messages = [
            {"role": "system", "content": DISPATCHER_SYSTEM_PROMPT},
            {"role": "user", "content": DISPATCH_USER_PROMPT_STEP_1.format(claim=claim)},
        ]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        if debug:
            print(f"Dispatch response: {output.content}")
            print(f"Dispatch action: {self._parse_dispatch_response(output.content)}")
            print(f"Dispatch reasoning: {output.reasoning}")
            print("---" * 10)

        return Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
            thought=output.reasoning,
            action=self._parse_dispatch_response(output.content),
            model_output=output,
        )

    async def fact_check_step(self, claim: str, searcher_responses: dict[str, str], debug: bool = False) -> Step:
        all_responses = self._parse_searcher_response(searcher_responses)
        messages = [
            {"role": "system", "content": DISPATCHER_SYSTEM_PROMPT},
            {"role": "user", "content": DISPATCH_USER_PROMPT_STEP_2.format(claim=claim, searcher_responses=all_responses)},
        ]
        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        if debug:
            print(f"Fact check response: {output.content}")
            print(f"Fact check action: {self._parse_final_response(output.content)}")
            print(f"Fact check reasoning: {output.reasoning}")
            print("---" * 10)

        return Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
            thought=output.reasoning,
            action=self._parse_final_response(output.content),
            model_output=output,
        )


class SearcherAgent:
    """
    A searcher agent who is equipped with a database of information.
    """

    def __init__(self, rollout_engine: RolloutEngine, rag_data_dir: str, index_name: str, top_k: int = 3, shuffle_retrieved_info: bool = False):
        self.rollout_engine = rollout_engine
        self.index_name = index_name
        self.top_k = top_k
        self.shuffle_retrieved_info = shuffle_retrieved_info
        self.rag_tool = ClaimRAGTool(rag_data_dir=rag_data_dir, index_name=index_name, top_k=top_k)

    def _parse_search_response(self, response: str) -> str:
        final_match = re.search(r"<answer>(.*?)</answer>", response, re.IGNORECASE | re.DOTALL)
        if final_match:
            return final_match.group(1).strip()
        else:
            return "ERROR: Searcher encountered an error."

    async def search_step(self, claim: str, debug: bool = False) -> Step:
        tool_result = await self.rag_tool.async_forward(claim, shuffle=self.shuffle_retrieved_info)
        if tool_result.error is not None:
            retrieved_info = "Unable to retrieve information from the database."
        else:
            retrieved_info = tool_result.output

        # provide the retrieved info to searcher
        messages = [
            {"role": "system", "content": SEARCHER_SYSTEM_PROMPT},
            {"role": "user", "content": SEARCHER_USER_PROMPT.format(claim=claim, retrieved_information=retrieved_info)},
        ]

        output: ModelOutput = await self.rollout_engine.get_model_response(messages)

        if debug:
            print(f"Search response: {output.content}")
            print(f"Search action: {self._parse_search_response(output.content)}")
            print(f"Search reasoning: {output.reasoning}")
            print("---" * 10)

        return Step(
            chat_completions=messages + [{"role": "assistant", "content": output.content, "reasoning": output.reasoning}],
            thought=output.reasoning,
            action=self._parse_search_response(output.content),
            model_output=output,
        )


class DispatcherSearcherWorkflow(Workflow):
    def __init__(self, rollout_engine: RolloutEngine, rag_data_dir: str, top_k: int = 3, shuffle_retrieved_info: bool = False, effort_param: float = 0.5, **kwargs):
        super().__init__(rollout_engine, **kwargs)

        self.dispatcher_agent = DispatcherAgent(rollout_engine)
        self.climate_searcher_agent = SearcherAgent(rollout_engine, rag_data_dir=rag_data_dir, index_name="climate_claim", top_k=top_k, shuffle_retrieved_info=shuffle_retrieved_info)
        self.covid_searcher_agent = SearcherAgent(rollout_engine, rag_data_dir=rag_data_dir, index_name="covid_claim", top_k=top_k, shuffle_retrieved_info=shuffle_retrieved_info)

        self.effort_param = effort_param

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        claim, label, source = task["claim"], task["label"], task["source"]

        debug = kwargs.get("debug", False)

        dispatch_traj = Trajectory(name="dispatcher")
        # Step 1: Dispatcher agent dispatches the task to the appropriate searcher agent
        dispatch_step = await self.dispatcher_agent.dispatch_step(claim, debug=debug)
        dispatch_traj.steps.append(dispatch_step)

        dispatch_action: DispatchAction = dispatch_step.action

        # Step 2: Let dispatched searcher agents return their responses
        searcher_a_traj, searcher_b_traj = Trajectory(name="searcher_a"), Trajectory(name="searcher_b")
        dispatch_num = 0
        searcher_responses = {}
        if dispatch_action == DispatchAction.DISPATCH_A:
            searcher_a_traj.steps.append(await self.climate_searcher_agent.search_step(claim, debug=debug))
            searcher_responses["searcher_a"] = searcher_a_traj.steps[-1].action
            dispatch_num += 1
        elif dispatch_action == DispatchAction.DISPATCH_B:
            searcher_b_traj.steps.append(await self.covid_searcher_agent.search_step(claim, debug=debug))
            searcher_responses["searcher_b"] = searcher_b_traj.steps[-1].action
            dispatch_num += 1
        else:  # dispatch both
            searcher_a_traj.steps.append(await self.climate_searcher_agent.search_step(claim, debug=debug))
            searcher_b_traj.steps.append(await self.covid_searcher_agent.search_step(claim, debug=debug))
            searcher_responses["searcher_a"] = searcher_a_traj.steps[-1].action
            searcher_responses["searcher_b"] = searcher_b_traj.steps[-1].action
            dispatch_num += 2

        fact_check_step = await self.dispatcher_agent.fact_check_step(claim, searcher_responses, debug=debug)
        dispatch_traj.steps.append(fact_check_step)

        final_response: FinalResponse = fact_check_step.action
        # Assign final reward to both dispatcher and searcher agents
        final_reward, is_correct = compute_final_reward_and_correctness(final_response, label, dispatch_num, self.effort_param)
        colorful_print(f"Final reward: {final_reward}, is_correct: {is_correct}, dispatch_num: {dispatch_num}, final_response: {final_response}, label: {label}", fg="cyan", bold=True)

        valid_trajectories = [dispatch_traj]
        dispatch_traj.steps[-1].reward = final_reward  # assign reward to the fact check step
        if len(searcher_a_traj.steps) > 0:
            searcher_a_traj.steps[-1].reward = final_reward
            valid_trajectories.append(searcher_a_traj)
        if len(searcher_b_traj.steps) > 0:
            searcher_b_traj.steps[-1].reward = final_reward
            valid_trajectories.append(searcher_b_traj)

        termination_reason = TerminationReason.ENV_DONE if final_response != FinalResponse.ERROR else TerminationReason.UNKNOWN

        return Episode(
            id=uid,
            task=task,
            termination_reason=termination_reason,
            trajectories=valid_trajectories,
            is_correct=is_correct,
            metrics={"searcher_acc": compute_searcher_accuracy(searcher_responses, source, label)},
        )


def compute_searcher_accuracy(searcher_responses: dict[str, str], source: str, label: bool) -> float:
    label_str = "TRUE" if label else "FALSE"
    acc, total = 0.0, 0.0
    if "searcher_a" in searcher_responses:
        searcher_a_action = searcher_responses["searcher_a"]
        searcher_a_action = searcher_a_action.strip().upper()
        if searcher_a_action == label_str:  # either both TRUE or both FALSE
            acc += 1.0
        elif searcher_a_action == "HEDGE" and source == "climate":  # if the searcher hedges and the source is climate, it is still correct
            acc += 1.0
        total += 1.0
    if "searcher_b" in searcher_responses:
        searcher_b_action = searcher_responses["searcher_b"]
        searcher_b_action = searcher_b_action.strip().upper()
        if searcher_b_action == label_str:  # either both TRUE or both FALSE
            acc += 1.0
        elif searcher_b_action == "HEDGE" and source == "covid":  # if the searcher hedges and the source is covid, it is still correct
            acc += 1.0
        total += 1.0
    return acc / total if total > 0 else 0.0


def compute_final_reward_and_correctness(final_response: FinalResponse, label: bool, dispatch_num: int, effort_param: float = 0.5) -> tuple[float, bool]:
    """
    Calculate the final (and only) reward for the RL problem (used for both searcher and dispatcher agents)
    If the final response is correct, the raw reward is 1.5. If it's unknown, the raw reward is 1.0. If it's incorrect or error, the raw reward is 0.5.
    Each dispatch will cost `effort_param` of the raw reward.
    """
    raw_reward, is_correct = 0.0, False
    if isinstance(label, str):
        label = True if label.upper() == "TRUE" else False

    if final_response == FinalResponse.UNKNOWN:
        raw_reward = 1.5
    elif (final_response == FinalResponse.TRUE and label) or (final_response == FinalResponse.FALSE and not label):
        raw_reward = 2.0
        is_correct = True
    else:
        return 0.0, False

    effort_cost = effort_param * dispatch_num
    return raw_reward - effort_cost, is_correct
