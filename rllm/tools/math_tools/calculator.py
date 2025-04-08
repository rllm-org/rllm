from rllm.tools.tool_base import Tool, ToolOutput


SYMBOL_REPLACEMENTS = {
    "^": "**",  # Power
    "×": "*",   # Multiplication 
    "÷": "/"    # Division
}

ALLOWED_CHARS = set("0123456789.+-*/() **")

class CalculatorTool(Tool):
    """A tool for evaluating mathematical expressions safely."""

    def __init__(self):
        """Initialize the Calculator tool."""
        super().__init__(
            name="calculator",
            description="Evaluate mathematical expressions, prefer using this instead of calculating it yourself."
        )

    @property
    def json(self):
        """Return the tool's information in a standardized format for tool registration."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The mathematical expression to evaluate",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    def forward(self, expression: str) -> ToolOutput:
        """
        Safely evaluate a mathematical expression.
        
        Args:
            expression: Mathematical expression as string
            
        Returns:
            Result as string, or error message if evaluation fails
        """
        try:
            # Replace mathematical symbols with Python operators
            for old, new in SYMBOL_REPLACEMENTS.items():
                expression = expression.replace(old, new)
            
            # Validate characters
            if not all(c in ALLOWED_CHARS for c in expression):
                return ToolOutput(name=self.name, error="Error: Invalid characters in expression")
            
            # Evaluate with empty namespace for safety
            result = eval(expression, {"__builtins__": {}}, {})
            return ToolOutput(name=self.name, output=str(result))
            
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Error evaluating expression: {str(e)}")

if __name__ == "__main__":
    # Test the calculator tool
    calculator = CalculatorTool()
    
    # Test some expressions
    test_expressions = [
        "2 + 2",
        "10 * 5",
        "2^8",  # Should use symbol replacement
        "sqrt(16)",
        "sin(0)",
        "invalid expression",
        "import os"  # Should fail due to security restrictions
    ]
    
    print("Testing Calculator Tool:")
    print("-----------------------")
    for expr in test_expressions:
        result = calculator(expr)
        print(f"Expression: {expr}")
        print(f"Result: {result}")
        print()
    
    # Test async version
    import asyncio
    
    async def test_async():
        print("Testing Async Calculator:")
        print("-----------------------")
        for expr in test_expressions:
            coro = calculator(expr, use_async=True)
            print(f"Coroutine: {coro}")
            result = await coro
            print(f"Expression: {expr}")
            print(f"Result: {result}")
            print()
    asyncio.run(test_async())

