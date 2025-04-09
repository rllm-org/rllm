import asyncio
import json
import inspect
import typing

def chat_completion_with_tool(
    client: "AsyncOpenAI",
    tool_caller: "ToolCaller",
    messages_list,
    model="gpt-4",
    max_round=20,
    batch_size=32,  # Added batch_size parameter
):
    from openai import AsyncOpenAI

    async def apply_tool(completion, messages, tool_caller, id=None):
        tool_calls = tool_caller.parse_tool_calls(completion.choices[0].message.content)

        if len(tool_calls) > 0:
            tool_call = tool_calls[0]
            if id is not None and isinstance(tool_call["parameters"], dict):
                tool_call["parameters"]["id"] = id
            tool_call_result = await tool_caller(tool_call["name"], tool_call["parameters"])
            print("tool_call_result", tool_call_result)
            messages.append(tool_call_result)
            return True

        return False

    async def tool_call_flow(example, request_id):
        try:
            messages = example["messages"]
            tool_infos = tool_caller.get_tool_infos()

            completion = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_infos,
                temperature=0.6,
                max_tokens=8192,
                top_p=0.95,
                stop=["```\n\n"],
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": completion.choices[0].message.content + "```\n\n",
                }
            )
            print("round: 0", completion.choices[0].message.content)
            curr_round = 0
            while curr_round < max_round:
                use_tools = await apply_tool(
                    completion, messages, tool_caller, id=request_id
                )
                if use_tools:
                    completion = await client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=tool_infos,
                        temperature=0.6,
                        max_tokens=8192,
                        top_p=0.95,
                        stop=["```\n\n"],
                    )
                    messages.append(
                        {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                            + "```\n\n",
                        }
                    )
                else:
                    break

                curr_round += 1
                print(f"round {curr_round}:", completion.choices[0].message.content)
        except Exception as e:
            print("Exception:", str(e))
            pass

        return example

    async def run_batch():
        # Initialize pool with first batch of requests
        active_requests = []
        results = []
        messages_iter = iter(messages_list)
        processed_count = 0
        request_id = 0

        # Fill initial pool
        for _ in range(batch_size):
            try:
                messages = next(messages_iter)
                task = asyncio.create_task(tool_call_flow(messages, request_id))
                active_requests.append((task, request_id))
                request_id += 1
            except StopIteration:
                break

        # Process requests and refill pool
        while active_requests:
            done, pending = await asyncio.wait(
                [task for task, _ in active_requests],
                return_when=asyncio.FIRST_COMPLETED,
            )
            # Update active_requests with pending tasks
            active_requests = [
                (task, id) for task, id in active_requests if task in pending
            ]

            for completed_task in done:
                # Find the ID for the completed task
                # task_id = next(id for task, id in active_requests if task == completed_task)
                final_messages = await completed_task
                # results.append({"id": task_id, "data": final_messages})
                results.append(final_messages)
                processed_count += 1

                # Save results checkpoint every 200 examples
                if processed_count % 600 == 0:
                    with open("messages_checkpoint.json", "w") as f:
                        json.dump(results, f, indent=2)

                # Try to add new request to maintain pool
                try:
                    messages = next(messages_iter)
                    new_task = asyncio.create_task(tool_call_flow(messages, request_id))
                    active_requests.append((new_task, request_id))
                    request_id += 1
                except StopIteration:
                    pass

            print("Active requests:", len(active_requests))

        return results

    return asyncio.run(run_batch())

def function_to_dict(func):
    """
    Converts a function into a dictionary representation suitable for JSON Schema.

    Parameters:
        func (function): The function to convert.

    Returns:
        dict: A dictionary representing the function in JSON Schema format.
    """
    # Get the function name
    func_name = func.__name__

    # Get the docstring
    docstring = func.__doc__ or ''

    # Get the function signature
    sig = inspect.signature(func)

    # Initialize the parameters dictionary
    params = {
        "type": "object",
        "properties": {},
        "required": []
    }

    # Map Python types to JSON Schema types
    type_mapping = {
        int: "integer",
        float: "number",
        bool: "boolean",
        str: "string",
        dict: "object",
        list: "array",
    }

    for param_name, param in sig.parameters.items():
        # Get the type annotation
        annotation = param.annotation

        param_type = "string"  # Default type
        param_description = ""

        # Determine the JSON Schema type and description
        if annotation != inspect.Parameter.empty:
            # Handle Annotated types
            origin = typing.get_origin(annotation)
            if origin is typing.Annotated:
                args = typing.get_args(annotation)
                base_type = args[0]
                metadata = args[1:]
                param_type = type_mapping.get(base_type, "string")
                # Assuming the first metadata argument is the description
                if metadata:
                    param_description = metadata[0]
            else:
                param_type = type_mapping.get(annotation, "string")
        # Add the parameter to properties
        param_schema = {"type": param_type}
        if param_description:
            param_schema["description"] = param_description
        params["properties"][param_name] = param_schema

        # Add to required if there's no default value
        if param.default == inspect.Parameter.empty:
            params["required"].append(param_name)

    # Build the final dictionary
    function_dict = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": docstring.strip().split('\n')[0],  # First line of docstring
            "parameters": params
        }
    }

    return function_dict
