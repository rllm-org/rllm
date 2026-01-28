# Standard imports
import os
import sys
import json
import re
import time
from pathlib import Path
import httpx

# Third Party Imports
import openai
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
from rllm.rewards.reward_types import RewardOutput

# Relative Imports
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.append(BASE_DIR)

from src.constants import (
    FIN_QA_CORRECTNESS_SYSTEM_PROMPT_PATH,
    FIN_QA_MULTI_TABLE_CORRECTNESS_SYSTEM_PROMPT_PATH,
)

with open(FIN_QA_CORRECTNESS_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    CORRECTNESS_PROMPT = f.read()

with open(FIN_QA_MULTI_TABLE_CORRECTNESS_SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    MULTI_TABLE_CORRECTNESS_PROMPT = f.read()

PORTKEY_API_KEY = os.environ.get("PORTKEY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
JUDGE_MODEL = "gpt-5-nano"
MULTI_TABLE_JUDGE_MODEL = "gpt-5-mini"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
# LOG_PATH = RESULTS_DIR / "judge_usage_log_setting_7.jsonl"

# (Cache + Retry)
GATEWAY_CONFIG = {
    "retry": {"attempts": 5},
    "cache": {"mode": "simple", "max_age" : 1209600},  # Exact match, cache TTL 14 days
}

custom_http_client = httpx.Client(
    http2=True,
    limits=httpx.Limits(max_connections=5000, max_keepalive_connections=2000),
    timeout=75.0,
    trust_env=False,
)

try:
    JUDGE_CLIENT = openai.OpenAI(
        base_url=PORTKEY_GATEWAY_URL,
        api_key=OPENAI_API_KEY,
        http_client=custom_http_client,
        default_headers=createHeaders(api_key=PORTKEY_API_KEY, provider="openai",config=GATEWAY_CONFIG),
    )
    
except Exception as e:
    print(f"Warning: Failed to initialize global OpenAI client: {e}")
    JUDGE_CLIENT = None

_FINAL_ANSWER_CODE_BLOCK_RE = re.compile(r"```\s*FINAL ANSWER:\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_PARAGRAPH_RE = re.compile(r"FINAL ANSWER:\s*(.*?)(?=\n\s*\n)", re.DOTALL | re.IGNORECASE)
_FINAL_ANSWER_TAIL_RE = re.compile(r"FINAL ANSWER:\s*(.*)$", re.DOTALL | re.IGNORECASE)

# Weight configuration for multi-table scoring 
CORRECTNESS_WEIGHTS = {
    "primary_data_score": 0.30,      # High weight - core correctness
    "derived_metrics_score": 0.30,   # High weight - core correctness
    "reasoning_score": 0.15,         # Medium weight
    "consistency_score": 0.10,       # Medium weight
    "completeness_score": 0.10,      # Medium weight
    "structure_score": 0.05,         # Low weight
}


# def _log_judge_call(entry: dict) -> None:
#     """Append judge metadata to a local jsonl file."""
#     with LOG_PATH.open("a", encoding="utf-8") as log_file:
#         log_file.write(json.dumps(entry) + "\n")


# def _collect_usage(usage) -> dict | None:
#     if not usage:
#         return None
#     tokens = {}
#     for field in ("input_tokens", "output_tokens", "total_tokens"):
#         value = getattr(usage, field, None)
#         if value is not None:
#             tokens[field] = value
#     return tokens or None

    
def _call_judge(
    system_prompt: str,
    user_prompt: str,
    evaluation_type: str,
    is_multi_table: bool = False,
) -> tuple[bool | float, dict]:
    
    if JUDGE_CLIENT is None:
        return (False if not is_multi_table else 0.0), {}
    
    model = MULTI_TABLE_JUDGE_MODEL if is_multi_table else JUDGE_MODEL
    
  
    request_kwargs = {
        "model": model,
        "instructions": system_prompt, 
        "input": user_prompt,
        "max_output_tokens": 5000 if is_multi_table else 512,
    }

    if is_multi_table:
        # Structured Output Schema
        schema = {
            "type": "object",
            "properties": {
                "primary_data_score": {"type": "number"},
                "derived_metrics_score": {"type": "number"},
                "completeness_score": {"type": "number"},
                "structure_score": {"type": "number"},
                "reasoning_score": {"type": "number"},
                "consistency_score": {"type": "number"},
                "explanation": {"type": "string"},
            },
            "required": [
                "primary_data_score", "derived_metrics_score", 
                "completeness_score", "structure_score", 
                "reasoning_score", "consistency_score", "explanation"
            ],
            "additionalProperties": False,
        }
        request_kwargs["reasoning"] = {"effort": "medium"}
        request_kwargs["text"] = {
            "format": {
                "type": "json_schema",
                "name": "finqa_multi_table_rubric",
                "schema": schema,
                "strict": True,
            }
        }
    else:
        request_kwargs["reasoning"] = {"effort": "low"}
        request_kwargs["text"] = {"verbosity": "low"} 
    
    try:
        response = JUDGE_CLIENT.responses.create(**request_kwargs)        
        judge_output = getattr(response, "output_text", "")
        
        if is_multi_table:
            try:
                parsed = json.loads(judge_output)
            except json.JSONDecodeError:
                parsed = {}
            
            weighted_score = 0.0
            total_weight = 0.0
            for key, weight in CORRECTNESS_WEIGHTS.items():
                score = parsed.get(key)
                if isinstance(score, (int, float)):
                    normalized = float(score) / 100.0
                    weighted_score += normalized * weight
                    total_weight += weight
            
            overall = weighted_score / total_weight if total_weight > 0 else 0.0
            overall = max(0.0, min(1.0, overall))
            result = overall
        else:
            decision_text = judge_output.lower()
            decision = ("true" in decision_text) and ("false" not in decision_text)
            parsed = {}
            result = decision

        # log_entry = {
        #     "decision": result,
        #     "evaluation_type": evaluation_type,
        #     "input": {
        #         "instructions": system_prompt,
        #         "input": user_prompt
        #     },
        #     "judge_response": judge_output,
        # }
        
        # if is_multi_table:
        #     log_entry["parsed_rubric"] = parsed
        
        # tokens = _collect_usage(getattr(response, "usage", None))
        # if tokens:
        #     log_entry["tokens"] = tokens
        
        # _log_judge_call(log_entry)
        return result, parsed
        
    except Exception as exc:
        # _log_judge_call({
        #     "decision": False if not is_multi_table else 0.0,
        #     "evaluation_type": evaluation_type,
        #     "input": {
        #         "instructions": system_prompt,
        #         "input": user_prompt
        #     },
        #     "error": str(exc),
        # })
        
        return (False if not is_multi_table else 0.0), {}


def _check_right_table_accessed(accessed_tables: list[str], expected_table_names: str | list[str]) -> float:
    """Return fraction of required tables that were accessed at least once."""
    if not accessed_tables or not expected_table_names:
        return 0.0

    normalized_access = {
        table.lower().strip() for table in accessed_tables
        if isinstance(table, str) and table.strip()
    }

    if isinstance(expected_table_names, list):
        expected = [name.lower().strip() for name in expected_table_names if isinstance(name, str) and name.strip()]
    else:
        expected = [expected_table_names.lower().strip()] if isinstance(expected_table_names, str) else []

    if not expected:
        return 0.0

    hits = sum(1 for name in expected if name in normalized_access)
    return hits / len(expected)


def _extract_final_answer(action: str, *, prefer_tail: bool = False) -> str:
    """Extract FINAL ANSWER section from model response."""
    # First try: handle code block format (```FINAL ANSWER: ... ```)
    code_match = _FINAL_ANSWER_CODE_BLOCK_RE.search(action)
    if code_match:
        return code_match.group(1).strip()

    # For long, multi-paragraph templates we often want everything after FINAL ANSWER:
    # In that case, skip the paragraph heuristic and fall back directly to the tail match.
    if not prefer_tail:
        # Second try: find FINAL ANSWER: and extract content until double newline
        match = _FINAL_ANSWER_PARAGRAPH_RE.search(action)
        if match:
            return match.group(1).strip()

    # Third try: find FINAL ANSWER: and extract content until end of string
    match = _FINAL_ANSWER_TAIL_RE.search(action)
    if match:
        return match.group(1).strip()
    
    # Fallback: return entire action if no FINAL ANSWER found
    return action


def fin_qa_reward_function(task_info: dict, action: str) -> RewardOutput:
    """
    Calculate the reward for a financial question answering agent's action.

    Args:
        task_info: The task dictionary containing question, answer, and other metadata
        action: The agent's response/solution

    Returns:
        RewardOutput: The calculated reward value.
    """
    question = task_info.get("question")
    core_question = task_info.get("core_question") or question
    ground_truth = task_info.get("ground_truth")
    question_type = (task_info.get("question_type") or "").lower()

    if not action or not question or not ground_truth:
        return RewardOutput(
            reward=0.0,
            is_correct=False,
            metadata={"correctness_reward": 0.0, "right_table_access_reward": 0.0},
        )

    is_multi_table = question_type.startswith("multi_table")
    
    # Build correctness input
    if is_multi_table:
        correctness_input = (
            f"question : {core_question}\n"
            f"model response : {action}\n"
            f"label : {ground_truth}"
        )
        system_prompt = MULTI_TABLE_CORRECTNESS_PROMPT
    else:
        final_answer = _extract_final_answer(action)
        correctness_input = (
            f"question : {question}\n"
            f"model response : {final_answer}\n"
            f"label : {ground_truth}"
        )
        system_prompt = CORRECTNESS_PROMPT
    
    result, rubric = _call_judge(
        system_prompt,
        correctness_input,
        evaluation_type="correctness",
        is_multi_table=is_multi_table,
    )
    
    if is_multi_table:
        correctness_reward = float(result)
        is_correct = correctness_reward >= 0.9
    else:
        is_correct = bool(result)
        correctness_reward = 1.0 if is_correct else 0.0
    
    # Check table access
    accessed_tables = task_info.get("accessed_tables", [])
    expected_table_names = task_info.get("table_name", "")
    right_table_access_reward = _check_right_table_accessed(accessed_tables, expected_table_names)

    # Build metadata
    metadata = {
        "right_table_access_reward": right_table_access_reward,
    }

    if is_multi_table:
        # Add all rubric scores to metadata
        for key in CORRECTNESS_WEIGHTS.keys():
            score = rubric.get(key)
            if isinstance(score, (int, float)):
                metadata[f"multi_table_{key}"] = float(score)
        metadata["multi_table_overall_score"] = correctness_reward

    return RewardOutput(
        reward=correctness_reward,
        is_correct=is_correct,
        metadata=metadata,
    )
