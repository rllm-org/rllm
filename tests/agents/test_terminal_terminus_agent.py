from rllm.agents.terminal_terminus_agent import TerminalTerminusAgent


class TestTerminalTerminusAgent:
    def test_init_and_properties(self):
        agent = TerminalTerminusAgent()
        # After init/reset
        assert agent.messages == []
        assert agent.step == 0
        # trajectory is initialized
        assert agent.trajectory is agent._trajectory

    def test_update_flow(self):
        agent = TerminalTerminusAgent()

        # Provide initial observation
        obs = {"prompt": "hello"}
        agent.update_from_env(observation=obs, reward=0.0, done=False, info={})

        # Model responds with raw text (Terminus 1 flow mirrors raw response)
        response = "run ls -la"
        action = agent.update_from_model(response)

        # Check action mirrors response
        assert action.action == response
        # Trajectory has one step, with observation and model response
        assert len(agent._trajectory.steps) == 1
        step = agent._trajectory.steps[0]
        assert step.observation == obs
        assert step.model_response == response
        assert step.action == response

        # Agent messages alternate user/assistant
        assert agent.messages[0]["role"] == "user"
        assert agent.messages[0]["content"] == "hello"
        assert agent.messages[1]["role"] == "assistant"
        assert agent.messages[1]["content"] == response

    def test_reset_clears_state(self):
        agent = TerminalTerminusAgent()

        # Build some state
        agent.update_from_env({"prompt": "p"}, 0.0, False, {})
        agent.update_from_model("resp")
        assert len(agent._trajectory.steps) == 1
        assert len(agent.messages) == 2

        # Reset
        agent.reset()
        assert len(agent._trajectory.steps) == 0
        assert agent.messages == []
        assert agent.step == 0

    def test_get_current_state(self):
        agent = TerminalTerminusAgent()
        agent.update_from_env({"prompt": "a"}, 0.0, False, {})
        agent.update_from_model("b")
        cur = agent.get_current_state()
        assert cur is agent._trajectory.steps[-1]
