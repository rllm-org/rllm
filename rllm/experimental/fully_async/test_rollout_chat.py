#!/usr/bin/env python3
"""
Test script for RolloutClientChat - uses /v1/chat/completions endpoint.
Returns structured responses with decoded text and parsed tool calls.
"""

import os
import sys
from pathlib import Path

# Add verl package to Python path
verl_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(verl_root))

import asyncio
import json

from rollout_client_chat import ChatResponse, Message, RolloutClientChat, ToolCall

# Define fake tools for testing
FAKE_TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a given location", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The temperature unit to use"}}, "required": ["location"]}}}, {"type": "function", "function": {"name": "calculate", "description": "Perform a mathematical calculation", "parameters": {"type": "object", "properties": {"expression": {"type": "string", "description": "The mathematical expression to evaluate, e.g. '2 + 2' or '10 * 5'"}}, "required": ["expression"]}}}, {"type": "function", "function": {"name": "search_database", "description": "Search for information in a database", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The search query"}, "limit": {"type": "integer", "description": "Maximum number of results to return", "default": 10}}, "required": ["query"]}}}]


async def test_simple_chat():
    """Test simple chat without tools."""
    print("\n" + "=" * 60)
    print("TEST 1: Simple Chat (No Tools) - Structured Response")
    print("=" * 60)

    # Initialize client (no tokenizer needed!)
    print("Initializing RolloutClientChat at localhost:4000...")
    client = RolloutClientChat(router_url="http://localhost:4000", max_concurrency=128)

    try:
        messages = [{"role": "user", "content": "What is the capital of France?"}]

        print(f"\nMessages: {json.dumps(messages, indent=2)}")
        print("\nGenerating response...")

        response = await client.generate_chat(messages=messages, temperature=0.7, max_tokens=128, top_p=0.95)

        print(f"\n--- Structured Response ---")
        print(f"Role: {response.message.role}")
        print(f"Content: {response.message.content}")
        print(f"Finish reason: {response.finish_reason}")
        print(f"Token usage:")
        print(f"  - Prompt: {response.prompt_tokens}")
        print(f"  - Completion: {response.completion_tokens}")
        print(f"  - Total: {response.total_tokens}")

    finally:
        await client.close()

    print("\n✓ Test 1 completed successfully!")


async def test_chat_with_tools():
    """Test chat with tool calling."""
    print("\n" + "=" * 60)
    print("TEST 2: Chat with Tool Calling - Parsed Tool Calls")
    print("=" * 60)

    # Initialize client
    print("Initializing RolloutClientChat at localhost:4000...")
    client = RolloutClientChat(router_url="http://localhost:4000", max_concurrency=128)

    try:
        messages = [{"role": "user", "content": "What's the weather like in San Francisco?"}]

        print(f"\nMessages: {json.dumps(messages, indent=2)}")
        print(f"\nTools provided: {len(FAKE_TOOLS)} tools")
        for tool in FAKE_TOOLS:
            print(f"  - {tool['function']['name']}: {tool['function']['description']}")

        print("\nGenerating response with tools...")

        response = await client.generate_chat(messages=messages, tools=FAKE_TOOLS, tool_choice="auto", temperature=0.7, max_tokens=256, top_p=0.95)

        print(f"\n--- Structured Response ---")
        print(f"Role: {response.message.role}")
        print(f"Content: {response.message.content}")
        print(f"Finish reason: {response.finish_reason}")

        if response.message.tool_calls:
            print(f"\nTool Calls Detected: {len(response.message.tool_calls)}")
            for i, tc in enumerate(response.message.tool_calls):
                print(f"\nTool Call {i + 1}:")
                print(f"  ID: {tc.id}")
                print(f"  Type: {tc.type}")
                print(f"  Function Name: {tc.function['name']}")
                print(f"  Function Arguments: {tc.function['arguments']}")

                # Parse arguments JSON
                try:
                    args = json.loads(tc.function["arguments"])
                    print(f"  Parsed Arguments: {json.dumps(args, indent=4)}")
                except:
                    print(f"  (Could not parse arguments as JSON)")
        else:
            print("\nNo tool calls in response")

        print(f"\nToken usage:")
        print(f"  - Prompt: {response.prompt_tokens}")
        print(f"  - Completion: {response.completion_tokens}")
        print(f"  - Total: {response.total_tokens}")

    finally:
        await client.close()

    print("\n✓ Test 2 completed successfully!")


async def test_multi_turn_conversation():
    """Test multi-turn conversation with tools."""
    print("\n" + "=" * 60)
    print("TEST 3: Multi-turn Conversation with Tools")
    print("=" * 60)

    # Initialize client
    print("Initializing RolloutClientChat at localhost:4000...")
    client = RolloutClientChat(router_url="http://localhost:4000", max_concurrency=128)

    try:
        messages = [{"role": "user", "content": "Calculate 15 * 23 for me"}, {"role": "assistant", "content": "I'll calculate that for you."}, {"role": "user", "content": "Now add 100 to the result"}]

        print(f"\nMessages: {json.dumps(messages, indent=2)}")
        print(f"\nTools provided: {len(FAKE_TOOLS)} tools")

        print("\nGenerating response...")

        response = await client.generate_chat(messages=messages, tools=FAKE_TOOLS, temperature=0.8, max_tokens=200, top_p=0.9)

        print(f"\n--- Structured Response ---")
        print(f"Role: {response.message.role}")
        print(f"Content: {response.message.content}")

        if response.message.tool_calls:
            print(f"\nTool Calls: {len(response.message.tool_calls)}")
            for tc in response.message.tool_calls:
                print(f"  - {tc.function['name']}: {tc.function['arguments']}")

        print(f"Finish reason: {response.finish_reason}")
        print(f"Tokens: {response.completion_tokens}")

    finally:
        await client.close()

    print("\n✓ Test 3 completed successfully!")


async def test_batch_requests():
    """Test multiple concurrent requests."""
    print("\n" + "=" * 60)
    print("TEST 4: Batch Concurrent Requests")
    print("=" * 60)

    # Initialize client
    print("Initializing RolloutClientChat at localhost:4000...")
    client = RolloutClientChat(router_url="http://localhost:4000", max_concurrency=128)

    try:
        requests = [
            [{"role": "user", "content": "What is 2+2?"}],
            [{"role": "user", "content": "What is the capital of Japan?"}],
            [{"role": "user", "content": "Tell me a short joke."}],
            [{"role": "user", "content": "What color is the sky?"}],
            [{"role": "user", "content": "Count from 1 to 5."}],
        ]

        print(f"\nSending {len(requests)} concurrent requests...")

        # Send all requests concurrently
        tasks = [client.generate_chat(messages=messages, temperature=0.7, max_tokens=100) for messages in requests]

        responses = await asyncio.gather(*tasks)

        print(f"\nReceived {len(responses)} responses:")
        for i, response in enumerate(responses):
            print(f"\nRequest {i + 1}:")
            print(f"  Query: {requests[i][0]['content']}")
            print(f"  Response: {response.message.content[:100]}...")
            print(f"  Tokens: {response.completion_tokens}")
            print(f"  Finish: {response.finish_reason}")

    finally:
        await client.close()

    print("\n✓ Test 4 completed successfully!")


async def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("RolloutClientChat Test Suite")
    print("Using /v1/chat/completions endpoint")
    print("=" * 60)
    print("Endpoint: http://localhost:4000")
    print("No tokenizer needed - responses are pre-decoded!")
    print("=" * 60)

    try:
        await test_simple_chat()
        await test_chat_with_tools()
        await test_multi_turn_conversation()
        await test_batch_requests()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nKey Benefits of /v1/chat/completions:")
        print("  ✓ Pre-decoded text (no manual decoding needed)")
        print("  ✓ Parsed tool calls (structured ToolCall objects)")
        print("  ✓ Token usage statistics included")
        print("  ✓ No tokenizer dependency")
        print("  ✓ OpenAI-compatible API")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
