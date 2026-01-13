"""
Data adapter for converting rLLM batch format to SkyRL format.

This module provides functions to adapt rLLM dataset batches to the format
expected by SkyRL's prepare_generator_input function.
"""

from typing import Any, Dict, List


# Priority order for prompt keys
PROMPT_KEYS = ["prompt", "question", "problem"]

# Reserved keys that should not be included in env_extras
RESERVED_KEYS = {"prompt", "question", "problem", "env_class", "uid", "unique_id", "data_source"}


def adapt_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Adapt a single rLLM dataset item to SkyRL format.

    This function:
    1. Finds the prompt in one of: prompt, question, problem keys
    2. Converts string prompts to chat format: [{"role": "user", "content": text}]
    3. Passes through prompts already in chat format (list)
    4. Extracts env_class from env_class or data_source fields
    5. Extracts uid from uid or unique_id fields
    6. Puts all other fields into env_extras

    Args:
        item: A single dataset item from rLLM format.

    Returns:
        A dictionary in SkyRL format with keys: prompt, env_class, uid, env_extras.

    Raises:
        ValueError: If no valid prompt key is found in the item.
    """
    # Find the prompt
    prompt = None
    prompt_key = None
    
    for key in PROMPT_KEYS:
        if key in item and item[key] is not None:
            prompt = item[key]
            prompt_key = key
            break
    
    if prompt is None:
        raise ValueError(
            f"Cannot find prompt in item. Expected one of {PROMPT_KEYS}, "
            f"but found keys: {list(item.keys())}"
        )
    
    # Convert prompt to chat format if it's a string
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, list):
        # Already in chat format, pass through
        # Validate it's a list of dicts with role/content
        if not all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in prompt):
            raise ValueError(
                f"Prompt in list format must be a list of dicts with 'role' and 'content' keys. "
                f"Got: {prompt}"
            )
    else:
        raise ValueError(
            f"Prompt must be either a string or a list of dicts. Got type: {type(prompt)}, value: {prompt}"
        )
    
    # Extract env_class from env_class or data_source
    env_class = item.get("env_class")
    if env_class is None:
        env_class = item.get("data_source")
    
    # Extract uid from uid or unique_id
    uid = item.get("uid")
    if uid is None:
        uid = item.get("unique_id")
    
    # Put all other fields into env_extras
    env_extras = {}
    for key, value in item.items():
        if key not in RESERVED_KEYS and key != prompt_key:
            env_extras[key] = value
    
    # Build the adapted item
    adapted_item = {
        "prompt": prompt,
        "env_extras": env_extras,
    }
    
    # Add env_class if available
    if env_class is not None:
        adapted_item["env_class"] = env_class
    
    # Add uid if available
    if uid is not None:
        adapted_item["uid"] = uid
    
    return adapted_item


def adapt_rllm_batch_to_skyrl(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adapt a batch of rLLM dataset items to SkyRL format.

    This function processes each item in the batch using adapt_single_item.

    Args:
        batch: A list of dataset items from rLLM format.

    Returns:
        A list of dictionaries in SkyRL format.

    Examples:
        >>> batch = [{"question": "What is 2+2?", "answer": "4", "data_source": "math"}]
        >>> result = adapt_rllm_batch_to_skyrl(batch)
        >>> result[0]["prompt"]
        [{"role": "user", "content": "What is 2+2?"}]
        >>> result[0]["env_class"]
        "math"
        >>> result[0]["env_extras"]["answer"]
        "4"
    """
    if not batch:
        return []
    
    return [adapt_single_item(item) for item in batch]

