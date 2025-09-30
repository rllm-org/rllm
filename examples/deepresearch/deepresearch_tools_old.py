"""
DeepResearch Tools - Simplified implementations for rLLM integration

These are simplified versions of the original DeepResearch tools, adapted to work
with our rLLM workflow while maintaining the core functionality for research tasks.
"""

import asyncio
import os

import requests


class DeepResearchTool:
    """Base class for DeepResearch tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    async def call(self, **kwargs) -> str:
        """Call the tool with given arguments."""
        raise NotImplementedError("Subclasses must implement call method")


class SearchTool(DeepResearchTool):
    """Web search tool for finding current information."""

    def __init__(self):
        super().__init__(
            name="Search", description="Search the web for current information and news"
        )

    async def call(self, query: str, **kwargs) -> str:
        """
        Perform web search.

        Args:
            query: Search query string

        Returns:
            Search results as formatted string
        """
        try:
            return await self._search_with_serper(query)
        except Exception as e:
            return f"Search error: {e}. Please try with a different query."

    async def _search_with_serper(self, query: str) -> str:
        """Use Serper API for web search (adapted from original DeepResearch)."""

        # Check for API key
        serper_key = os.getenv("SERPER_KEY_ID") or os.getenv("SERPER_API_KEY")
        if not serper_key:
            return f"""Search results for "{query}":

[No Serper API key configured]
To enable real web search, set SERPER_KEY_ID or SERPER_API_KEY in your .env file.
Get your free API key from: https://serper.dev/

Basic information for query "{query}":
- This would normally return current web search results
- Configure the API key for actual search functionality"""

        def contains_chinese_basic(text: str) -> bool:
            return any("\u4e00" <= char <= "\u9fff" for char in text)

        # Prepare request payload
        if contains_chinese_basic(query):
            payload = {"q": query, "location": "China", "gl": "cn", "hl": "zh-cn"}
        else:
            payload = {"q": query, "location": "United States", "gl": "us", "hl": "en"}

        headers = {"X-API-KEY": serper_key, "Content-Type": "application/json"}

        # Use requests instead of http.client for easier async handling
        url = "https://google.serper.dev/search"

        # Retry logic
        for attempt in range(3):
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                results = response.json()
                break
            except Exception:
                if attempt == 2:
                    return f"Search timeout for '{query}'. Please try again later."
                await asyncio.sleep(1)  # Wait before retry
                continue

        try:
            if "organic" not in results:
                return (
                    f"No search results found for '{query}'. Try a more general query."
                )

            web_snippets = []
            idx = 0

            for page in results["organic"]:
                idx += 1
                date_published = ""
                if "date" in page:
                    date_published = "\nDate published: " + page["date"]

                source = ""
                if "source" in page:
                    source = "\nSource: " + page["source"]

                snippet = ""
                if "snippet" in page:
                    snippet = "\n" + page["snippet"]

                formatted_result = f"{idx}. [{page['title']}]({page['link']}){date_published}{source}\n{snippet}"
                formatted_result = formatted_result.replace(
                    "Your browser can't play this video.", ""
                )
                web_snippets.append(formatted_result)

            content = (
                f"A Google search for '{query}' found {len(web_snippets)} results:\n\n## Web Results\n"
                + "\n\n".join(web_snippets)
            )
            return content

        except Exception as e:
            return f"Error processing search results for '{query}': {e}"


class FileParserTool(DeepResearchTool):
    """Tool for parsing and analyzing files."""

    def __init__(self):
        super().__init__(
            name="FileParser",
            description="Parse and analyze files (PDF, DOCX, TXT, CSV, etc.)",
        )

    async def call(self, files: list, **kwargs) -> str:
        """
        Parse files and extract content.

        Args:
            files: List of file paths to parse

        Returns:
            Parsed content as string
        """
        try:
            results = []
            for file_path in files:
                if os.path.exists(file_path):
                    try:
                        # Simple text file reading - can be enhanced with specific parsers
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()[:5000]  # Limit content size
                            results.append(
                                f"File: {file_path}\nContent:\n{content}\n---"
                            )
                    except Exception as e:
                        results.append(f"File: {file_path}\nError: {e}\n---")
                else:
                    results.append(f"File: {file_path}\nError: File not found\n---")

            return "\n".join(results) if results else "No files processed"

        except Exception as e:
            return f"File parsing error: {e}"


class ScholarTool(DeepResearchTool):
    """Academic search tool for scholarly information."""

    def __init__(self):
        super().__init__(
            name="Scholar",
            description="Search for academic papers and scholarly information",
        )

    async def call(self, query: str, **kwargs) -> str:
        """
        Search for academic papers.

        Args:
            query: Academic search query

        Returns:
            Academic search results as string
        """
        try:
            return f"""Academic search results for "{query}":

[Placeholder academic search results]
1. Paper Title 1 - Authors et al. (2024)
   Abstract: Academic paper about {query}...

2. Paper Title 2 - Authors et al. (2023)
   Abstract: Research on {query}...

3. Paper Title 3 - Authors et al. (2022)
   Abstract: Study of {query}...

Note: This is a placeholder implementation. In production, this would connect to
academic databases like Google Scholar, arXiv, or DBLP for real results."""

        except Exception as e:
            return f"Scholar search error: {e}"


class VisitTool(DeepResearchTool):
    """Tool for visiting and analyzing web pages."""

    def __init__(self):
        super().__init__(name="Visit", description="Visit and analyze web pages")

    async def call(self, url: str, **kwargs) -> str:
        """
        Visit a URL and extract content.

        Args:
            url: URL to visit

        Returns:
            Page content as string
        """
        try:
            # Placeholder implementation - in production would use requests/selenium
            return f"""Visited: {url}

[Placeholder web page content]
Title: Sample Page Title
Content: This is placeholder content from the visited page {url}.
In a real implementation, this would fetch and parse the actual webpage content.

Key information extracted:
- Main topic: Related to the search query
- Important facts: Placeholder facts from the page
- Links: Placeholder related links"""

        except Exception as e:
            return f"Visit error: {e}"


class PythonInterpreterTool(DeepResearchTool):
    """Tool for executing Python code safely.

    Enhanced version inspired by Tongyi's PythonInterpreter with:
    - Better error handling
    - Timeout support
    - More comprehensive output capture
    """

    def __init__(self):
        super().__init__(
            name="PythonInterpreter",
            description="Execute Python code for calculations and data analysis",
        )
        self.timeout = 50  # Match Tongyi's default timeout

    async def call(self, code: str, timeout: int = None, **kwargs) -> str:
        """
        Execute Python code with enhanced safety and error handling.

        Inspired by Tongyi's implementation with improvements for:
        - Timeout handling
        - Better error messages
        - More comprehensive output capture

        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds (default: 50)

        Returns:
            Execution result as string
        """
        timeout = timeout or self.timeout

        try:
            # Enhanced safety check - reject dangerous operations
            dangerous_patterns = [
                "import os",
                "import subprocess",
                "import sys",
                "exec",
                "eval",
                "__import__",
                "open(",
                "file(",
                "input(",
                "raw_input(",
                "compile(",
                "globals(",
                "locals(",
                "vars(",
            ]

            code_lower = code.lower()
            for pattern in dangerous_patterns:
                if pattern in code_lower:
                    return f"[Security Error] Dangerous operation '{pattern}' not allowed for safety reasons."

            # Enhanced execution environment matching Tongyi's capabilities
            import io
            import sys
            from concurrent.futures import ThreadPoolExecutor, TimeoutError

            # More comprehensive allowed modules
            allowed_modules = {
                "math": __import__("math"),
                "datetime": __import__("datetime"),
                "json": __import__("json"),
                "random": __import__("random"),
                "re": __import__("re"),
                "collections": __import__("collections"),
                "itertools": __import__("itertools"),
                "statistics": __import__("statistics"),
            }

            # Try to add numpy and pandas if available (like Tongyi)
            try:
                import numpy as np

                allowed_modules["numpy"] = np
                allowed_modules["np"] = np
            except ImportError:
                pass

            try:
                import pandas as pd

                allowed_modules["pandas"] = pd
                allowed_modules["pd"] = pd
            except ImportError:
                pass

            # Enhanced restricted globals
            restricted_builtins = {
                "abs": abs,
                "all": all,
                "any": any,
                "bin": bin,
                "bool": bool,
                "chr": chr,
                "dict": dict,
                "enumerate": enumerate,
                "filter": filter,
                "float": float,
                "hex": hex,
                "int": int,
                "len": len,
                "list": list,
                "map": map,
                "max": max,
                "min": min,
                "oct": oct,
                "ord": ord,
                "pow": pow,
                "print": print,
                "range": range,
                "reversed": reversed,
                "round": round,
                "set": set,
                "slice": slice,
                "sorted": sorted,
                "str": str,
                "sum": sum,
                "tuple": tuple,
                "type": type,
                "zip": zip,
            }

            global_vars = {"__builtins__": restricted_builtins}
            global_vars.update(allowed_modules)

            local_vars = {}

            # Enhanced output capture
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            def execute_with_timeout():
                try:
                    sys.stdout = stdout_buffer
                    sys.stderr = stderr_buffer
                    exec(code, global_vars, local_vars)
                    return True
                except Exception as e:
                    stderr_buffer.write(f"Execution error: {e}")
                    return False
                finally:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

            # Execute with timeout (similar to Tongyi's approach)
            with ThreadPoolExecutor() as executor:
                try:
                    future = executor.submit(execute_with_timeout)
                    future.result(timeout=timeout)

                    stdout_content = stdout_buffer.getvalue()
                    stderr_content = stderr_buffer.getvalue()

                    # Format output like Tongyi
                    if stderr_content:
                        return f"[Execution Error]\n{stderr_content}"
                    elif stdout_content:
                        return f"[Execution Output]\n{stdout_content.rstrip()}"
                    elif local_vars:
                        # Show meaningful variables (filter out internals)
                        meaningful_vars = {
                            k: v
                            for k, v in local_vars.items()
                            if not k.startswith("_") and k not in allowed_modules
                        }
                        if meaningful_vars:
                            return f"[Variables]\n{meaningful_vars}"
                        else:
                            return "[Success] Code executed successfully (no output)"
                    else:
                        return "[Success] Code executed successfully (no output)"

                except TimeoutError:
                    return f"[Timeout Error] Code execution exceeded {timeout} seconds timeout"

        except Exception as e:
            return f"[System Error] Python execution failed: {e}"


# Tool registry for easy access
DEEPRESEARCH_TOOLS = {
    "Search": SearchTool(),
    "FileParser": FileParserTool(),
    "Scholar": ScholarTool(),
    "Visit": VisitTool(),
    "PythonInterpreter": PythonInterpreterTool(),
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()
