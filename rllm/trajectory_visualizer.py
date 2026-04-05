import json
import re

import gradio as gr
import torch
from fire import Fire


def main(trajectory_file: str = "./trajectories/sample_trajectories/search_trajectories.pt", server_port: int = 12345):
    trajs_data = torch.load(trajectory_file, weights_only=False)
    all_trajs = list(filter(lambda x: hasattr(x, "steps") and len(x.steps) > 0, trajs_data))

    def filter_trajectories_by_reward(filter_option: str):
        """Filter trajectories based on reward"""
        if filter_option == "All Trajectories":
            return all_trajs
        elif filter_option == "Zero Reward (Failed)":
            return [t for t in all_trajs if float(t.reward) == 0.0]
        elif filter_option == "Nonzero Reward (Partial/Full Success)":
            return [t for t in all_trajs if float(t.reward) > 0.0]
        elif filter_option == "Perfect Score (Reward = 1)":
            return [t for t in all_trajs if float(t.reward) == 1.0]
        else:
            return all_trajs

    def extract_thinking_and_response(model_response: str) -> tuple[str, str]:
        """Extract thinking and final response from model output"""
        if not model_response:
            return "", ""

        think_match = re.search(r"<think>(.*?)</think>", model_response, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            response = model_response.split("</think>", 1)[-1].strip()
        else:
            thinking = ""
            response = model_response.strip()

        return thinking, response

    def format_tool_call_detailed(tool_call: dict) -> str:
        """Format tool calls with detailed information"""
        if isinstance(tool_call, str):
            return tool_call

        function = tool_call.get("function", {})
        name = function.get("name", "unknown")
        args = function.get("arguments", {})
        tool_id = tool_call.get("id", "no-id")

        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw_arguments": args}

        if not isinstance(args, dict):
            args = {"raw_arguments": str(args)}

        if name == "local_search":
            query = args.get("query", "No query")
            return f"🔍 **Search Query:** `{query}`\n*Tool ID: {tool_id}*"
        elif name == "finish":
            response = args.get("response", "No response")
            return f"✅ **Finish Action:**\n```\n{response}\n```\n*Tool ID: {tool_id}*"
        else:
            return f"🛠️ **Tool:** `{name}`\n**Arguments:**\n```json\n{json.dumps(args, indent=2)}\n```\n*Tool ID: {tool_id}*"

    def format_tool_outputs(tool_outputs: dict) -> str:
        """Format tool execution results"""
        if not tool_outputs:
            return "*No tool outputs*"

        formatted_outputs = []
        for tool_id, output in tool_outputs.items():
            display_output = str(output)
            if len(display_output) > 500:
                display_output = display_output[:500] + "... (truncated)"

            formatted_outputs.append(f"**Tool ID `{tool_id}`:**\n```\n{display_output}\n```")

        return "\n\n".join(formatted_outputs)

    def extract_boxed_answer(text: str) -> str:
        """Extract answer from \\boxed{} format"""
        if not text:
            return ""

        match = re.search(r"\\boxed\{([^}]*)\}", text)
        if match:
            return match.group(1)
        return ""

    def get_trajectory_metadata(trajectory) -> dict:
        """Extract metadata from trajectory - supports both observation and info fields"""
        if not trajectory.steps:
            return {}

        first_step = trajectory.steps[0]

        # Try multiple sources for metadata in order of preference
        if isinstance(first_step.observation, dict):
            return first_step.observation
        elif hasattr(first_step, "info") and isinstance(first_step.info, dict):
            return first_step.info
        elif hasattr(trajectory, "task") and isinstance(trajectory.task, dict):
            return trajectory.task
        else:
            return {}

    def detect_response_structure(step):
        """Detect if this step uses thinking structure or direct response structure - updated for new API"""
        # Check for thinking format (separate thought and action fields)
        has_thinking = hasattr(step, "thought") and step.thought and hasattr(step, "action") and step.action

        # Check for direct response format (model_response and action)
        has_direct_response = hasattr(step, "model_response") and step.model_response and hasattr(step, "action") and step.action

        # Updated logic for new API pattern
        uses_thinking_format = has_thinking and step.thought.strip()

        return {
            "has_thinking": has_thinking,
            "has_direct_response": has_direct_response,
            "uses_thinking_format": uses_thinking_format,
            "response_field": "thought" if uses_thinking_format else "model_response",
        }

    def get_task_type(metadata):
        """Detect task type from metadata"""
        if "data_source" in metadata:
            if metadata["data_source"] in ["hotpotqa", "bamboogle", "musique"]:
                return "search"
            elif metadata["data_source"] in ["deepcoder", "livecodebench", "code_generation"]:
                return "code"
            elif metadata["data_source"] in ["math", "gsm8k", "math_word_problems"]:
                return "math"

        if "question_type" in metadata or "level" in metadata:
            return "search"
        elif "difficulty" in metadata or "contest_id" in metadata or "platform" in metadata:
            return "code"
        elif "problem_type" in metadata or "solution_type" in metadata:
            return "math"

        return "unknown"

    def advance_step_or_trajectory(current_traj_idx_val, current_step_idx_val, direction, level, filtered_trajs):
        current_traj_idx_val = int(current_traj_idx_val)
        current_step_idx_val = int(current_step_idx_val)
        num_filtered_trajectories = len(filtered_trajs)

        if num_filtered_trajectories == 0:
            return 0, 0

        if level == "step":
            num_steps_current_traj = len(filtered_trajs[current_traj_idx_val].steps)
            if direction == "next":
                next_step_idx = current_step_idx_val + 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx >= num_steps_current_traj:
                    next_step_idx = 0
            else:  # prev
                next_step_idx = current_step_idx_val - 1
                next_traj_idx = current_traj_idx_val
                if next_step_idx < 0:
                    next_step_idx = num_steps_current_traj - 1 if num_steps_current_traj > 0 else 0
        else:  # trajectory
            if direction == "next":
                next_traj_idx = current_traj_idx_val + 1
                if next_traj_idx >= num_filtered_trajectories:
                    next_traj_idx = 0
            else:  # prev
                next_traj_idx = current_traj_idx_val - 1
                if next_traj_idx < 0:
                    next_traj_idx = num_filtered_trajectories - 1 if num_filtered_trajectories > 0 else 0
            next_step_idx = 0

        return next_traj_idx, next_step_idx

    def update_step_view(traj_idx: int, step_idx: int, filter_option: str):
        """Update the step view with detailed information - updated for new API pattern"""
        empty_content = "*No data available*"

        filtered_trajs = filter_trajectories_by_reward(filter_option)

        if traj_idx >= len(filtered_trajs):
            return (empty_content,) * 10

        trajectory = filtered_trajs[traj_idx]
        num_steps = len(trajectory.steps)

        position_text = f"**Trajectory {traj_idx + 1}/{len(filtered_trajs)}**  |  **Step {step_idx + 1}/{num_steps}**"
        metadata = get_trajectory_metadata(trajectory)
        task_type = get_task_type(metadata)

        if task_type == "search":
            question = metadata.get("question", "No question found")
            gt = metadata.get("ground_truth", "N/A")
            ground_truth = str(gt).lower() if isinstance(gt, str | int | float | bool) else str(gt)
        elif task_type == "code":
            question = metadata.get("question_content", metadata.get("question", metadata.get("problem", "No problem statement found")))
            test_cases = metadata.get("test_cases", [])
            if test_cases and len(test_cases) > 0:
                expected_output = test_cases[0].get("expected_output", "N/A")
                ground_truth = str(expected_output)
            else:
                ground_truth = "See test cases"
        elif task_type == "math":
            question = metadata.get("problem", metadata.get("question", "No problem found"))
            ground_truth = str(metadata.get("answer", metadata.get("solution", "N/A")))
        else:
            question = str(trajectory.steps[0].observation) if num_steps > 0 else "No question"
            ground_truth = "Unknown"

        task_icons = {"search": "🔍", "code": "💻", "math": "🧮", "unknown": "❓"}
        task_names = {"search": "Search", "code": "Code", "math": "Math", "unknown": "Unknown"}

        metadata_text = f"**Task Type:** {task_icons[task_type]} {task_names[task_type]}\n"
        metadata_text += f"**Data Source:** {metadata.get('data_source', 'N/A')}\n"

        if task_type == "search":
            metadata_text += f"**Question Type:** {metadata.get('question_type', 'N/A')}\n"
            metadata_text += f"**Level:** {metadata.get('level', 'N/A')}\n"
        elif task_type == "code":
            metadata_text += f"**Difficulty:** {metadata.get('difficulty', 'N/A')}\n"
            metadata_text += f"**Platform:** {metadata.get('platform', 'N/A')}\n"
            metadata_text += f"**Contest ID:** {metadata.get('contest_id', 'N/A')}\n"
        elif task_type == "math":
            metadata_text += f"**Problem Type:** {metadata.get('problem_type', 'N/A')}\n"
            metadata_text += f"**Difficulty:** {metadata.get('difficulty', 'N/A')}\n"

        metadata_text += f"**Split:** {metadata.get('split', 'N/A')}\n"
        metadata_text += f"**UID:** `{metadata.get('uid', metadata.get('question_id', 'N/A'))}`"

        perf_text = f"**Overall Reward:** {trajectory.reward:.3f}\n"
        perf_text += f"**Total Steps:** {num_steps}\n"
        perf_text += f"**Completed:** {'✅ Yes' if (num_steps > 0 and getattr(trajectory.steps[-1], 'done', False)) else '❌ No'}"

        question_text = f"**Question:**\n{question}\n\n**Ground Truth Answer:** `{ground_truth}`"

        if num_steps == 0:
            return (position_text, metadata_text, perf_text, question_text, empty_content, empty_content, empty_content, empty_content, empty_content, empty_content)

        step = trajectory.steps[step_idx]
        structure = detect_response_structure(step)

        if structure["uses_thinking_format"]:
            thinking_text = getattr(step, "thought", "") or "*No thinking recorded*"
            response_text = getattr(step, "action", "") or "*No response recorded*"
        else:
            thinking, response = extract_thinking_and_response(getattr(step, "model_response", ""))
            thinking_text = thinking if thinking else "*No thinking recorded*"
            response_text = response if response else (getattr(step, "action", "") or "*No response recorded*")

        # Safe field access for step performance
        step_reward = getattr(step, "reward", 0.0)
        step_mc_return = getattr(step, "mc_return", 0.0)
        step_done = getattr(step, "done", False)
        step_number = getattr(step, "step", step_idx)  # Fallback to step_idx if step field doesn't exist

        step_perf_text = f"**Step Reward:** {step_reward}\n"
        step_perf_text += f"**MC Return:** {step_mc_return:.3f}\n"
        step_perf_text += f"**Done:** {'✅ Yes' if step_done else '❌ No'}\n"
        step_perf_text += f"**Step Number:** {step_number}\n"
        step_perf_text += f"**Response Structure:** {'🧠 Thinking Format' if structure['uses_thinking_format'] else '📝 Direct Response'}"

        actions_text = empty_content
        step_action = getattr(step, "action", None)
        if step_action:
            if task_type == "search" and isinstance(step_action, list):
                actions_text = "\n\n".join([format_tool_call_detailed(tc) for tc in step_action])
            elif task_type in ["code", "math"] and isinstance(step_action, str):
                actions_text = f"**Generated {'Code' if task_type == 'code' else 'Solution'}:**\n```\n{step_action}\n```"
            else:
                actions_text = format_tool_call_detailed(step_action) if isinstance(step_action, dict) else str(step_action)

        outputs_text = empty_content
        # Handle outputs - be more flexible with field access
        if task_type == "search":
            # Try multiple ways to get tool outputs for search tasks
            current_obs = getattr(step, "observation", None)
            if current_obs and isinstance(current_obs, dict):
                tool_outputs = current_obs.get("tool_outputs", {})
                if tool_outputs:
                    outputs_text = format_tool_outputs(tool_outputs)
            elif hasattr(step, "next_observation"):
                next_obs = getattr(step, "next_observation", None)
                if next_obs and isinstance(next_obs, dict):
                    tool_outputs = next_obs.get("tool_outputs", {})
                    if tool_outputs:
                        outputs_text = format_tool_outputs(tool_outputs)
            # Also check if outputs are in the current step's info
            elif hasattr(step, "info") and isinstance(step.info, dict):
                tool_outputs = step.info.get("tool_outputs", {})
                if tool_outputs:
                    outputs_text = format_tool_outputs(tool_outputs)
        elif task_type in ["code", "math"]:
            step_observation = getattr(step, "observation", None)
            if step_idx > 0 and isinstance(step_observation, dict):
                if "test_results" in step_observation:
                    outputs_text = f"**Test Results:**\n{step_observation['test_results']}"
                elif "error" in step_observation:
                    outputs_text = f"**Error:**\n{step_observation['error']}"
                elif "feedback" in step_observation:
                    outputs_text = f"**Feedback:**\n{step_observation['feedback']}"

        predicted_answer = ""
        has_finish_action = False

        if task_type == "search" and step_action:
            actions_to_check = step_action if isinstance(step_action, list) else [step_action]
            for action in actions_to_check:
                if isinstance(action, dict) and action.get("function", {}).get("name") == "finish":
                    has_finish_action = True
                    try:
                        args_str = action["function"]["arguments"]
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        finish_response = args.get("response", "")
                        predicted_answer = extract_boxed_answer(finish_response)
                        break
                    except Exception:
                        pass
        elif task_type in ["code", "math"]:
            if step_done and step_action:
                predicted_answer = step_action
                has_finish_action = True

        if has_finish_action:
            if predicted_answer:
                if task_type == "search":
                    is_correct = predicted_answer.lower().strip() == ground_truth.lower().strip()
                    final_answer_text = "**🎯 Final Answer Provided:**\n"
                    final_answer_text += f"**Predicted:** `{predicted_answer}`\n"
                    final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                    final_answer_text += f"**Correct:** {'✅ Yes' if is_correct else '❌ No'}"
                else:
                    # Code/Math tasks: show the final solution with test results
                    final_answer_text = f"**{'💻' if task_type == 'code' else '🧮'} Final {'Code' if task_type == 'code' else 'Solution'} Submitted:**\n"
                    final_answer_text += f"**Length:** {len(predicted_answer)} characters\n"
                    final_answer_text += f"**Test Status:** {'✅ Passed' if step_reward > 0 else '❌ Failed'}\n"
                    final_answer_text += f"**Expected Output:** `{ground_truth}`"
            else:
                final_answer_text = "**⚠️ Finish action found but no answer extracted:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`"
        else:
            if step_idx == num_steps - 1:
                final_answer_text = "**❌ No finish action in final step:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += "**Status:** Trajectory ended without explicit final answer"
            else:
                final_answer_text = "**⏳ No final answer yet:**\n"
                final_answer_text += f"**Ground Truth:** `{ground_truth}`\n"
                final_answer_text += f"**Status:** Step {step_idx + 1}/{num_steps} - {'No finish action' if task_type == 'search' else 'Continuing...'}"

        return (position_text, metadata_text, perf_text, question_text, thinking_text, response_text, step_perf_text, actions_text, outputs_text, final_answer_text)

    custom_css = """
    .trajectory-container { margin-bottom: 20px !important; }

    .metadata-box { 
        background-color: #f8f9fa !important; 
        color: #2d3748 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #dee2e6 !important; 
    }
    .performance-box { 
        background-color: #e8f5e8 !important; 
        color: #2d5016 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #c3e6c3 !important; 
    }
    .thinking-box { 
        background-color: #fff3cd !important; 
        color: #856404 !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #ffeaa7 !important; 
    }
    .actions-box { 
        background-color: #e2e3f5 !important; 
        color: #3c366b !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b8bdf0 !important; 
    }
    .outputs-box { 
        background-color: #f0f8ff !important; 
        color: #1e3a8a !important; 
        padding: 15px !important; 
        border-radius: 8px !important; 
        border: 1px solid #b3d9ff !important; 
    }
    
    .dark .metadata-box { 
        background-color: #2d3748 !important; 
        color: #e2e8f0 !important; 
        border-color: #4a5568 !important; 
    }
    .dark .performance-box { 
        background-color: #2f4f2f !important; 
        color: #c6f6d5 !important; 
        border-color: #48bb78 !important; 
    }
    .dark .thinking-box { 
        background-color: #7c6f47 !important; 
        color: #fef5e7 !important; 
        border-color: #d69e2e !important; 
    }
    .dark .actions-box { 
        background-color: #3c366b !important; 
        color: #e9e7fd !important; 
        border-color: #805ad5 !important; 
    }
    .dark .outputs-box { 
        background-color: #1e3a8a !important; 
        color: #dbeafe !important; 
        border-color: #3b82f6 !important; 
    }
    
    .gr-textbox textarea { 
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace !important; 
        color: #2d3748 !important;
    }
    .dark .gr-textbox textarea { 
        color: #e2e8f0 !important; 
    }
    
    .step-display-box textarea { 
        text-align: center !important; 
        font-weight: bold !important; 
        font-size: 1.2em !important; 
        background-color: #ebf8ff !important; 
        color: #2b6cb0 !important; 
        border: 2px solid #bee3f8 !important; 
    }
    .dark .step-display-box textarea { 
        background-color: #1e3a8a !important; 
        color: #bfdbfe !important; 
        border-color: #3b82f6 !important; 
    }
    
    .nav-button { min-width: 120px !important; }
    .gr-button { 
        font-size: 1.1em !important; 
        padding: 0.5em 0.8em !important; 
        border-radius: 8px !important;
        background-color: #3182ce !important;
        color: white !important;
    }
    .gr-button:hover { 
        background-color: #2c5aa0 !important; 
    }
    
    .metadata-box p, .metadata-box h1, .metadata-box h2, .metadata-box h3, .metadata-box h4,
    .performance-box p, .performance-box h1, .performance-box h2, .performance-box h3, .performance-box h4,
    .actions-box p, .actions-box h1, .actions-box h2, .actions-box h3, .actions-box h4,
    .outputs-box p, .outputs-box h1, .outputs-box h2, .outputs-box h3, .outputs-box h4 {
        color: inherit !important;
    }
    
    .metadata-box code, .performance-box code, .actions-box code, .outputs-box code {
        background-color: rgba(0,0,0,0.1) !important;
        color: inherit !important;
        padding: 2px 4px !important;
        border-radius: 3px !important;
    }
    
    .dark .metadata-box code, .dark .performance-box code, .dark .actions-box code, .dark .outputs-box code {
        background-color: rgba(255,255,255,0.1) !important;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Agent Trajectory Visualizer") as interface:
        gr.Markdown("# 🔍 Agent Trajectory Visualizer")

        current_traj_idx_state = gr.State(0)
        current_step_idx_state = gr.State(0)

        with gr.Row():
            with gr.Column(scale=1):
                filter_dropdown = gr.Dropdown(
                    choices=[
                        "All Trajectories",
                        "Zero Reward (Failed)",
                        "Nonzero Reward (Partial/Full Success)",
                        "Perfect Score (Reward = 1)",
                    ],
                    value="All Trajectories",
                    label="🎯 Filter by Reward",
                    interactive=True,
                )
            with gr.Column(scale=1):
                zero_count = len([t for t in all_trajs if float(t.reward) == 0.0])
                nonzero_count = len([t for t in all_trajs if float(t.reward) > 0.0])
                perfect_count = len([t for t in all_trajs if float(t.reward) == 1.0])

                _ = gr.Markdown(
                    f"**Dataset Stats:**\n- Total: {len(all_trajs)} trajectories\n- Failed (0): {zero_count}\n- Partial/Full Success (>0): {nonzero_count}\n- Perfect Score (=1): {perfect_count}"
                )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    prev_traj_button = gr.Button("⬅️ Previous Trajectory", elem_classes=["nav-button"])
                    next_traj_button = gr.Button("Next Trajectory ➡️", elem_classes=["nav-button"])
                with gr.Row():
                    prev_step_button = gr.Button("⬅️ Previous Step", elem_classes=["nav-button"])
                    next_step_button = gr.Button("Next Step ➡️", elem_classes=["nav-button"])

            with gr.Column(scale=2):
                current_pos_display = gr.Textbox(label="Current Position", interactive=False, elem_classes=["step-display-box"])

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Accordion("📊 Trajectory Metadata", open=True):
                    metadata_output = gr.Markdown(elem_classes=["metadata-box"])

                with gr.Accordion("🎯 Performance", open=True):
                    performance_output = gr.Markdown(elem_classes=["performance-box"])

                with gr.Accordion("❓ Question & Answer", open=True):
                    question_output = gr.Markdown()
                    final_answer_output = gr.Markdown()

            with gr.Column(scale=2):
                with gr.Accordion("🧠 Agent Thinking", open=True):
                    thinking_output = gr.Textbox(label="Internal Reasoning", lines=6, interactive=False, elem_classes=["thinking-box"])

                with gr.Accordion("💬 Agent Response", open=True):
                    response_output = gr.Textbox(label="Final Response to User", lines=4, interactive=False)

                with gr.Accordion("📈 Step Performance", open=True):
                    step_perf_output = gr.Markdown(elem_classes=["performance-box"])

                with gr.Accordion("🛠️ Actions Taken", open=True):
                    actions_output = gr.Markdown(elem_classes=["actions-box"])

                with gr.Accordion("📋 Tool Results", open=True):
                    outputs_output = gr.Markdown(elem_classes=["outputs-box"])

        all_outputs = [
            current_pos_display,
            metadata_output,
            performance_output,
            question_output,
            thinking_output,
            response_output,
            step_perf_output,
            actions_output,
            outputs_output,
            final_answer_output,
        ]

        def reset_to_first_trajectory():
            return 0, 0

        prev_traj_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, "prev", "trajectory", filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state],
        )
        next_traj_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, "next", "trajectory", filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state],
        )
        prev_step_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, "prev", "step", filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state],
        )
        next_step_button.click(
            fn=lambda t, s, f: advance_step_or_trajectory(t, s, "next", "step", filter_trajectories_by_reward(f)),
            inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown],
            outputs=[current_traj_idx_state, current_step_idx_state],
        )

        filter_dropdown.change(fn=reset_to_first_trajectory, outputs=[current_traj_idx_state, current_step_idx_state])

        current_traj_idx_state.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)
        current_step_idx_state.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)
        filter_dropdown.change(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)

        interface.load(fn=update_step_view, inputs=[current_traj_idx_state, current_step_idx_state, filter_dropdown], outputs=all_outputs)

    interface.launch(share=True, server_port=server_port)


if __name__ == "__main__":
    Fire(main)
