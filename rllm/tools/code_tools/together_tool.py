from typing import Dict, Any, Optional
import os

from rllm.tools.code_tools.code_tool import CodeTool, CodeToolOutput


TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", None)

class TogetherCodeTool(CodeTool):
    """Tool for executing Python code using Together Code Interpreter (TCI).
    
    This tool integrates with Together's Code Interpreter API to provide
    a secure sandbox environment for executing Python code.
    """

    def __init__(
        self, 
        api_key: Optional[str] = TOGETHER_API_KEY,
    ):
        """Initialize the TogetherCodeTool.
        
        Args:
            api_key: Together API key (optional, will use environment variables if not provided)
            name: The name of the tool
            description: Description of what the tool does
        """
        self.api_key = api_key
        self.client = self._setup_client()
        self.session_id = None
        super().__init__(name='together_python', description='Execute Python code using Together Code Interpreter.')

    def _setup_client(self):
        """Set up the Together client for interacting with the API."""
        try:
            from together import Together
            return Together(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "The 'together' package is required to use TogetherCodeTool. "
                "Install it with 'pip install together'."
            )
    
    def forward(self, code: str, timeout: int = 12, session_id: Optional[str] = None, **kwargs) -> CodeToolOutput:
        """
        Execute Python code using Together Code Interpreter.
        
        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds (not directly used but retained for compatibility)
            session_id: Optional session ID to maintain state between runs
            **kwargs: Additional parameters to pass to Together's code_interpreter.run
            
        Returns:
            CodeToolOutput containing execution results, stdout, and stderr
        """
        self.client.timeout = timeout
        try:
            # If session_id is provided, use it to maintain state
            if session_id:
                self.session_id = session_id
            # Execute the code
            response = self.client.code_interpreter.run(
                code=code,
                language="python",
                **kwargs
            )
            
            # Save the session_id for potential future use
            self.session_id = response.data.session_id
            # Process the outputs
            stdout = ""
            stderr = ""
            output = ""
            
            for output_item in response.data.outputs:
                if output_item.type == "stdout":
                    stdout += output_item.data + "\n"
                elif output_item.type == "stderr":
                    stderr += output_item.data + "\n"
                else:
                    output += str(output_item.data) + "\n"
            
            # Handle errors
            error = None
            if response.data.errors:
                error = str(response.data.errors)
                stderr += error + "\n"
            
            # Return formatted output
            return CodeToolOutput(
                name=self.name,
                output=output.strip() if output else None,
                stdout=stdout.strip() if stdout else None,
                stderr=stderr.strip() if stderr else None,
                error=error
            )
            
        except Exception as e:
            return CodeToolOutput(
                name=self.name,
                error=f"{type(e).__name__} - {str(e)}",
                stderr=str(e)
            )

    def _init_sandbox(self):
        """Initialize a new sandbox session by resetting the session ID."""
        self.session_id = None

    def _kill_sandbox(self):
        """Clean up sandbox resources."""
        self.session_id = None

    def _restart_sandbox(self):
        """Restart the sandbox by creating a new session."""
        self.session_id = None

if __name__ == "__main__":
    tool = TogetherCodeTool()
    print(tool("print('Hello, world!')"))
