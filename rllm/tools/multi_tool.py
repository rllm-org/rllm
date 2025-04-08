from typing import List

from rllm.tools.tool_base import Tool, ToolOutput
from rllm.tools import TOOL_REGISTRY

class  MultiTool(Tool):
    def __init__(self, tools: List[str]):
        # Check if all tools are in the registry
        assert all(tool in TOOL_REGISTRY for tool in tools), "All tools must be in the registry TOOL_REGISTRY"

        # Initialize the tool map
        self.tool_map = {tool: TOOL_REGISTRY[tool]() for tool in tools}

    @property
    def json(self):
        return [tool.json for tool in self.tool_map.values()]
    
    def forward(self, *args, tool_name: str, **kwargs) -> ToolOutput:
        assert tool_name in self.tool_map, f"Tool {tool_name} not found in tool map"
        tool = self.tool_map[tool_name]
        return tool(*args,**kwargs)

if __name__ == "__main__":
    multi_tool = MultiTool(["calculator", "firecrawl"])
    print(multi_tool.json)
    print(multi_tool("1 + 2*3", tool_name="calculator"))
    print(multi_tool("https://www.yahoo.com", tool_name="firecrawl"))
    
    # Create an async examples
    import asyncio
    async def test_async():
        tasks = [
            multi_tool("1 + 2*3", tool_name="calculator", use_async=True),
            multi_tool("https://www.google.com", tool_name="firecrawl", use_async=True)
        ]
        results = await asyncio.gather(*tasks)
        print(results)

    asyncio.run(test_async())