import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass

from rllm.tools.utils import function_to_dict

@dataclass
class ToolCall:
    name: str
    parameters: Dict[str, Any]

@dataclass
class ToolInputs:
    inputs: List[ToolCall]
    
@dataclass
class ToolOutput:
    name: str
    output: Union[str, list, dict] = None
    error: Optional[str] = None

@dataclass
class ToolOutputs:
    outputs: List[ToolOutput]

class Tool(ABC):
    """
    Abstract base class for all tools that provides a common interface.
    
    All tools should inherit from this class and implement either:
    - forward() for synchronous tools
    - async_forward() for asynchronous tools
    """
    
    def __init__(self, name: str=None, description: str = None, function: Callable[[Any], Any]=None):
        """
        Initialize the tool with a name.
        
        Args:
            name (str): The name of the tool, used for tool registry and calling.
            description (str): A description of the tool's purpose and functionality.
            function (Callable, optional): Function to convert to tool format.
        """
        self.name = name
        self.description = description
        self.function = function
        
        if function is not None:
            self._json = function_to_dict(function)
            self.name = self._json["function"]["name"]
            self.description = self._json["function"]["description"]
        else:
            assert name is not None, "Name is required for Tool class."
            assert description is not None, "Description is required for Tool class."
            # User must provide json
            self._json = self.json
            assert self._json is not None, "Json representation of the tool is required."
    
    @property
    def json(self) -> Dict[str, Any]:
        """
        Return the tool's information in a standardized format for tool registration.
        
        Returns:
            Dict[str, Any]: Tool information including name, description, and parameters.
                Expected format:
                {
                    "type": "function",
                    "function": {
                        "name": self.name,
                        "description": "Description of what the tool does",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                # Parameter definitions
                            },
                            "required": ["list", "of", "required", "parameters"]
                        }
                    }
                }
        """
        return self._json
    
    def forward(self, *args, **kwargs) -> ToolOutput:
        """
        Synchronous implementation of the tool functionality.
        Override this method for synchronous tools.
        
        Returns:
            Any: The result of the tool execution.
        """
        if self.function:
            try:
                output = self.function(*args, **kwargs)
                return ToolOutput(name=self.name, output=output)
            except Exception as e:
                return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")
        else:
            raise NotImplementedError(
                "Tool must implement either forward() or async_forward(). "
                "This tool has not implemented the synchronous forward() method."
            )

    async def async_forward(self, *args, **kwargs) -> ToolOutput:
        """
        Asynchronous implementation of the tool functionality.
        Override this method for asynchronous tools.
        
        Returns:
            Any: The result of the tool execution.
        """
        return self.forward(*args, **kwargs)

    def __call__(self, *args, use_async=False, **kwargs) -> ToolOutput:
        """
        Make the tool instance callable.
        - If use_async is True, delegates to async_forward
        - If use_async is False, delegates to forward
        - If use_async is None (default), uses async_forward if implemented, otherwise forward
        
        Args:
            *args: Positional arguments to pass to the implementation
            use_async: Whether to use the async implementation (if None, auto-detect)
            **kwargs: Keyword arguments to pass to the implementation
            
        Returns:
            Any: The result of the tool execution, which may be a coroutine if using async_forward
        """
        
        has_async = inspect.isfunction(self.__class__.async_forward) or self.__class__.async_forward is Tool.async_forward
        has_sync = not (inspect.isfunction(self.__class__.forward) and self.__class__.forward is Tool.forward)
        # Explicit routing based on use_async flag
        if use_async is True:
            if has_async:
                return self.async_forward(*args, **kwargs)
            else:
                raise NotImplementedError(
                    f"Tool {self.__class__.__name__} does not implement async_forward() but use_async=True was specified."
                )
        elif use_async is False:
            return self.forward(*args, **kwargs)
        
        # Auto-detect implementation if use_async is None
        if has_async:
            return self.async_forward(*args, **kwargs)
        elif has_sync:
            return self.forward(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"Tool {self.__class__.__name__} must implement either forward() or async_forward()."
            )
    
    def __del__(self):
        """
        Attempt to clean up resources when the tool instance is garbage collected.
        
        This is a fallback mechanism and explicit cleanup() calls are preferred.
        """
        pass


if __name__ == "__main__":
    
    def add(a: int, b: int) -> int:
        r"""Adds two numbers.

        Args:
            a (int): The first number to be added.
            b (int): The second number to be added.

        Returns:
            integer: The sum of the two numbers.
        """
        return a + b

    tool = Tool(function=add)
    print(tool.json)
    print(tool(1, 2))
    
    from camel.toolkits import SearchToolkit
    google_fn = SearchToolkit().search_google
    google_tool = Tool(function=google_fn)
    print(google_tool(query="What is the capital of France?", num_result_pages=5))
