from unittest.mock import Mock, patch

from terminal_bench.agents.terminus_1 import CommandBatchResponse

from rllm.environments.terminal.terminal_terminus import TerminalTerminusEnv


def _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal, run_tests_in_same_shell=None):
    # Trial handler task config
    mock_task = Mock()
    mock_task.parser_name = "pytest"
    mock_task.max_test_timeout_sec = 10
    if run_tests_in_same_shell is not None:
        mock_task.run_tests_in_same_shell = run_tests_in_same_shell

    mock_th_instance = Mock()
    mock_th_instance.task = mock_task
    # Create nested path mocks explicitly
    mock_th_instance.task_paths = Mock()
    mock_th_instance.trial_paths = Mock()
    mock_th_instance.task_paths.run_tests_path = Mock()
    mock_th_instance.task_paths.test_dir = Mock()
    mock_th_instance.trial_paths.sessions_path = Mock()
    mock_th_instance.trial_paths.agent_logging_dir = Mock()
    mock_th_instance.client_container_name = "client"
    mock_th_instance.client_image_name = "image"
    mock_th_instance.task_paths.docker_compose_path = Mock()
    mock_th_instance.docker_image_name_prefix = "prefix"
    mock_trial_handler.return_value = mock_th_instance

    # Terminal
    mock_term_instance = Mock()
    mock_term_instance.create_session = Mock(return_value=Mock())
    mock_terminal.return_value = mock_term_instance

    return mock_th_instance, mock_term_instance


def _make_env_and_reset():
    env = TerminalTerminusEnv(model_name="m", api_base=None, task_path="/tmp", instruction="do x", task_id="t1")
    obs, info = env.reset(uid="u1")
    return env, obs, info


def _cbr_json(is_done: bool):
    from terminal_bench.agents.terminus_1 import CommandBatchResponse

    return CommandBatchResponse(
        state_analysis="",
        explanation="",
        commands=[],
        is_task_complete=is_done,
    ).model_dump_json()


class TestTerminalTerminusEnv:
    # Patch where the symbols are looked up (in the module under test)
    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_reset_initializes_terminal_and_returns_observation(self, mock_get_parser, mock_trial_handler, mock_terminal):
        # Mocks
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        mock_get_parser.return_value = Mock()

        env, obs, info = _make_env_and_reset()

        assert isinstance(obs, dict)
        assert "prompt" in obs
        # We don't assert on info here since we didn't capture the actual return values

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_step_parses_and_executes_commands(self, mock_get_parser, mock_trial_handler, mock_terminal):
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        mock_get_parser.return_value = Mock()

        env, _, _ = _make_env_and_reset()
        sample = _cbr_json(is_done=False)

        with patch.object(CommandBatchResponse, "model_validate_json", wraps=CommandBatchResponse.model_validate_json) as spy_validate, patch.object(TerminalTerminusEnv, "_execute_commands", return_value=(False, "out")) as spy_exec:
            obs, reward, done, info = env.step(sample)

            spy_validate.assert_called()
            spy_exec.assert_called()
            assert done is False
            assert obs["prompt"] == "out"

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_parse_failure_sets_done_and_info(self, mock_get_parser, mock_trial_handler, mock_terminal):
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        mock_get_parser.return_value = Mock()

        env, _, _ = _make_env_and_reset()

        # Send invalid JSON to trigger parse failure branch
        obs, reward, done, info = env.step("{not-json}")
        assert done is True
        assert info.get("parse_error") is True

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_is_task_complete_runs_tests_once(self, mock_get_parser, mock_trial_handler, mock_terminal):
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        mock_get_parser.return_value = Mock()

        env, _, _ = _make_env_and_reset()
        sample_done = _cbr_json(is_done=True)

        # Spy on evaluation path
        with patch.object(TerminalTerminusEnv, "_evaluate_completion_sync", return_value=(1.0, True)) as spy_eval:
            obs, reward, done, info = env.step(sample_done)
            spy_eval.assert_called_once()
            assert done is True

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_run_tests_in_new_shell_when_flag_false(self, mock_get_parser, mock_trial_handler, mock_terminal):
        # Default run_tests_in_same_shell is False; expect a second session for tests
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal, run_tests_in_same_shell=False)
        mock_get_parser.return_value = Mock()

        # Configure created sessions to have send_keys and capture_pane
        agent_sess = mock_terminal.return_value.create_session.return_value
        agent_sess.send_keys = Mock()
        agent_sess.capture_pane = Mock(return_value="output")

        env, _, _ = _make_env_and_reset()

        with patch.object(TerminalTerminusEnv, "_execute_commands", return_value=(False, "out")):
            sample_done = _cbr_json(is_done=True)

            # After reset: one session created. On finalize tests, another session should be created
            initial_calls = mock_terminal.return_value.create_session.call_count
            _ = env.step(sample_done)
            assert mock_terminal.return_value.create_session.call_count == initial_calls + 1

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_copy_to_container_inputs(self, mock_get_parser, mock_trial_handler, mock_terminal):
        th, term = _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        # Pretend tests dir exists
        th.task_paths.test_dir.exists.return_value = True
        mock_get_parser.return_value = Mock()

        # Configure created sessions to have send_keys and capture_pane
        agent_sess = mock_terminal.return_value.create_session.return_value
        agent_sess.send_keys = Mock()
        agent_sess.capture_pane = Mock(return_value="output")

        env, _, _ = _make_env_and_reset()

        # Trigger test run path
        with patch.object(TerminalTerminusEnv, "_execute_commands", return_value=(False, "out")):
            sample_done = _cbr_json(is_done=True)
            _ = env.step(sample_done)

        # copy_to_container should be called with run-tests and tests dir
        assert term.copy_to_container.called

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_close_stops_terminal(self, mock_get_parser, mock_trial_handler, mock_terminal):
        # Set up mocks and reset to initialize terminal
        _, term = _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal)
        mock_get_parser.return_value = Mock()

        env, _, _ = _make_env_and_reset()

        # Terminal.stop should be invoked by close()
        assert not term.stop.called
        env.close()
        assert term.stop.called

    @patch("rllm.environments.terminal.terminal_terminus.Terminal")
    @patch("rllm.environments.terminal.terminal_terminus.TrialHandler")
    @patch("rllm.environments.terminal.terminal_terminus.ParserFactory.get_parser")
    def test_run_tests_in_same_shell_toggle(self, mock_get_parser, mock_trial_handler, mock_terminal):
        # Configure with same-shell flag
        _setup_trial_handler_and_terminal(mock_trial_handler, mock_terminal, run_tests_in_same_shell=True)
        mock_get_parser.return_value = Mock()

        env, _, _ = _make_env_and_reset()

        with patch.object(TerminalTerminusEnv, "_execute_commands", return_value=(False, "out")):
            with patch.object(TerminalTerminusEnv, "_evaluate_completion_sync", return_value=(1.0, True)):
                sample_done = _cbr_json(is_done=True)

                _ = env.step(sample_done)

                # Because run_tests_in_same_shell=True, workflow should reuse agent session
                # and not create a new one for tests
                # After reset there should be exactly one create_session call; no extra calls should be made
                mock_terminal.return_value.create_session.assert_called_once()
