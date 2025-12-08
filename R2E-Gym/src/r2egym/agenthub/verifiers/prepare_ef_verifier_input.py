import re
import json
import logging
from typing import List, Dict, Tuple, Optional

from transformers import AutoTokenizer

from r2egym.agenthub.trajectory.trajectory import Trajectory

logger = logging.getLogger(__name__)


def deepswe_condense_thoughts(
    input_str: str,
    max_tokens: int = 31000,
    tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct",
) -> str:
    """
    If the token count of input_str exceeds max_tokens, then starting with the second
    [ASSISTANT]...[/ASSISTANT] block (the oldest after the first), replace its inner content with
    a placeholder until the total token count is under the limit.

    The first [ASSISTANT] block is left intact.
    """
    placeholder = "<Thought condensed for saving context>"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Check initial token count
    if len(tokenizer.encode(input_str)) <= max_tokens:
        return input_str

    # Regex to match thoughts between [ASSISTANT] and <function
    pattern = re.compile(r"(\[ASSISTANT\])(.*?)(<function)", re.DOTALL)

    new_str = input_str
    # Continue condensing until token count is below the threshold or nothing changes.
    while len(tokenizer.encode(new_str)) > max_tokens:
        # Re-find all [ASSISTANT] blocks in the updated string
        matches = list(pattern.finditer(new_str))
        if len(matches) <= 1:
            # Nothing more to condense (either no [ASSISTANT] blocks or only one exists)
            break

        # Sort matches by content length (descending) - biggest first
        # Filter out already condensed blocks
        uncondensed_matches = [m for m in matches if m.group(2).strip() != placeholder]

        if not uncondensed_matches:
            # All blocks are already condensed
            break

        # Sort by content length (group(2) is the content)
        uncondensed_matches.sort(key=lambda m: len(m.group(2)), reverse=True)

        # Replace the longest uncondensed block
        m = uncondensed_matches[0]
        new_block = m.group(1) + placeholder + m.group(3)
        # Replace this block in the string using its current indices
        start, end = m.start(), m.end()
        new_str = new_str[:start] + new_block + new_str[end:]

        # print warning for removing
        print(
            f"Warning: Removing {len(tokenizer.encode(m.group(2)))} tokens from [ASSISTANT] block"
        )

    return new_str


def compute_total_tokens(
    training_data_entry, tokenizer_name="Qwen/Qwen2.5-Coder-32B-Instruct"
):
    """
    Compute the total number of tokens in the training data entry
       Args:
           training_data_entry: e.g.,
               [{'role': 'system', 'content': 'System prompt'},
                {'role': 'user', 'content': 'User prompt'},
                {'role': 'assistant', 'content': 'Assistant prompt'}]
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    combined_text = " ".join([x["content"] for x in training_data_entry])
    # Encode the text to get the token count
    # add_special_tokens=False so that e.g. GPT-2's <|endoftext|> is not counted
    tokens = tokenizer.encode(combined_text, add_special_tokens=False)
    return len(tokens)


def traj2verifier_data(
    json_entry: Dict,
    max_tokens: int = 65536,
) -> Tuple[List[Dict], bool]:
    """Convert a trajectory entry to verifier training data format."""
    try:
        # Extract trajectory data
        if "trajectory_steps" in json_entry:
            problem_statement = json_entry.get("problem_statement", "")
            traj = json_entry["trajectory_steps"]
            reward = json_entry.get("reward", 0)
        else:
            problem_statement = json_entry.get("problem_statement", "")
            traj = json_entry.get("trajectory", [])
            reward = json_entry.get("success", json_entry.get("reward", 0))

        # Create trajectory object for additional data with proper parsing
        try:
            import json

            trajclass = Trajectory.load_from_model_dump_json(json.dumps(json_entry))
            output_patch = trajclass.true_output_patch_only_existing_files
        except Exception as e:
            logger.warning(f"Could not get true_output_patch: {e}")
            try:
                trajclass = Trajectory.model_construct(**json_entry)
                output_patch = getattr(trajclass, "output_patch", "")
            except Exception:
                output_patch = json_entry.get("output_patch", "")

        # Default system prompt for verifier

        system_prompt = """You are an expert judge evaluating AI assistant interactions. Your task is to determine if the assistant successfully resolved the user's request.

Key evaluation criteria:
1. Did the assistant complete the main task requested by the user?
2. Did the assistant handle all edge cases and requirements specified?
3. Were there any errors or issues in the final solution?
4. Did the assistant verify the solution works as intended?

Respond only with "<judgement>YES</judgement>" or "<judgement>NO</judgement>"."""

        # Default instance prompt

        instance_prompt = """You are a software engineer working on a repository. A user has submitted an issue, and you need to resolve it.

Repository Issue:
{problem_statement}

Please analyze the issue and implement a solution."""

        # Build the training data entry
        data_entry = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]

        # Create the user message with interaction log
        user_message = "Please evaluate the following interaction between an AI assistant and a user:\n\n"
        user_message += "=== INTERACTION LOG ===\n\n"
        user_message += f"[SYSTEM]\n{system_prompt}\n[/SYSTEM]\n\n"
        user_message += f"[USER]\n{instance_prompt.format(problem_statement=problem_statement)}\n[/USER]"

        # Add trajectory steps
        for stepidx, entry in enumerate(traj):
            thought = entry.get("thought", "")
            action = entry.get("action", "")
            observation = entry.get("observation", "")

            assistant_response = f"{thought}\n\n{action}" if thought else action

            user_message += f"\n\n[STEP]\n{stepidx}\n[/STEP]"
            user_message += f"\n\n[ASSISTANT]\n{assistant_response}\n[/ASSISTANT]"
            user_message += f"\n\n[USER]\n{observation}\n[/USER]"

        # Add final patch and evaluation request
        user_message += "\n\n=== END INTERACTION ==="
        user_message += "\n\n=== FINAL PATCH ==="
        user_message += f"\n\n[PATCH]\n{output_patch}\n[/PATCH]"
        user_message += "\n\n=== END FINAL PATCH ==="
        user_message += "\n\nBased on the above interaction, did the assistant successfully resolve the user's initial request? Respond with YES or NO."

        data_entry.append({"role": "user", "content": user_message})
        data_entry.append(
            {
                "role": "assistant",
                "content": "<judgement>"
                + ("YES" if reward == 1 else "NO")
                + "</judgement>",
            }
        )

        # total_tokens = compute_total_tokens(data_entry)
        total_nonuser_tokens = compute_total_tokens([data_entry[0], data_entry[2]])
        data_entry[1]["content"] = deepswe_condense_thoughts(
            data_entry[1]["content"],
            max_tokens=max_tokens - total_nonuser_tokens - 500,  ## 500 is just a buffer
        )

        # if total_tokens > max_tokens:
        #     return [], False
        return data_entry, True

    except Exception as e:
        logger.error(f"Error processing trajectory entry: {e}")
        return [], False
