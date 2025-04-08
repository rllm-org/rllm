"""
Sleep Tool module that provides both synchronous and asynchronous sleep functionality.

This module demonstrates how to implement a tool that can be used in both
synchronous and asynchronous contexts.
"""

from typing import Any, Dict
import asyncio
import time

from rllm.tools.tool_base import Tool


class SleepTool(Tool):
    """
    A tool that sleeps for a specified number of seconds.
    
    This tool demonstrates both synchronous and asynchronous implementations
    of the same functionality.
    """
    
    @property
    def json(self) -> Dict[str, Any]:
        """
        Return the tool's JSON representation for tool registration.
        
        Returns:
            Dict[str, Any]: A dictionary containing the tool's type and function name.
        """
        return {
            "type": "function", 
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "seconds": {
                            "type": "number",
                            "description": "Number of seconds to sleep"
                        }
                    },
                    "required": ["seconds"]
                }
            }
        }
    
    def forward(self, seconds: float) -> str:
        """
        Synchronously sleep for the specified number of seconds.
        
        Args:
            seconds (float): The number of seconds to sleep.
            
        Returns:
            str: A message indicating the sleep duration.
        """
        print(f"Starting sleep for {seconds} seconds for synchronous invocation.")
        time.sleep(seconds)  # Blocking operation
        print(f"Finished sleep after {seconds} seconds for synchronous invocation.")
        return f"Slept for {seconds} seconds"


async def main() -> None:
    """
    Demonstrate the usage of the SleepTool in various contexts.
    
    This function shows how to use the tool both synchronously and asynchronously,
    including running multiple asynchronous operations concurrently.
    """
    tool = SleepTool(name="sleep_tool", description="This tool sleeps for a given number of seconds.")
    
    # This doesn't execute the tool yet, just returns the coroutine
    coro = tool(3, use_async=True)
    print("Got coroutine, not yet executed")
    
    # Actual execution happens when we await
    result = await coro
    print(f"Result: {result}")
    
    # Test synchronous invocation.
    result = tool(3)
    print(f"Result: {result}")
    
    # We can also use in an expression that expects a coroutine
    tasks = [
        tool(1, use_async=True),
        tool(2, use_async=True)
    ]
    # These will run concurrently
    results = await asyncio.gather(*tasks)
    print(f"Multiple results: {results}")


if __name__ == "__main__":
    asyncio.run(main())