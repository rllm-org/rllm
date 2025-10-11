"""
DeepResearch Tools - Production-ready implementations

This module provides tool implementations for the DeepResearch agent, with real
functionality ported from Tongyi's original implementations where possible.

Now supports both:
- ReAct text format (for gpt-4o, Claude, etc.)
- OpenAI native function calling (for o3, o3-mini, etc.)
"""

import http.client
import json
import os
from abc import ABC, abstractmethod

from rllm.tools.tool_base import Tool as RLLMTool


class DeepResearchTool(RLLMTool, ABC):
    """
    Base class for all DeepResearch tools.

    Inherits from rLLM's Tool to support OpenAI native function calling,
    while maintaining compatibility with ReAct text format.
    """

    def __init__(self, name: str, description: str, parameters: dict | None = None):
        """
        Initialize DeepResearch tool with OpenAI function calling support.

        Args:
            name: Tool name
            description: Tool description
            parameters: OpenAI-style parameter schema (optional)
        """
        # Set _json BEFORE calling super().__init__
        # because the parent's __init__ may access self.json
        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters or {"type": "object", "properties": {}, "required": []},
            },
        }

        super().__init__(name=name, description=description)

    @abstractmethod
    async def call(self, **kwargs) -> str:
        """Execute the tool with given arguments."""
        pass

    async def async_forward(self, **kwargs):
        """rLLM Tool interface - delegates to call()"""
        from rllm.tools.tool_base import ToolOutput

        try:
            result = await self.call(**kwargs)
            return ToolOutput(name=self.name, output=result)
        except Exception as e:
            return ToolOutput(name=self.name, error=f"{type(e).__name__} - {str(e)}")


class SearchTool(DeepResearchTool):
    """Web search tool using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="Search",
            description="Performs web searches using Google via Serper API",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string",
                    }
                },
                "required": ["query"],
            },
        )

    def contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _google_search_fallback(self, query: str | list) -> str:
        """Use Google Custom Search API as fallback."""
        try:
            import requests

            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            queries = [query] if isinstance(query, str) else query
            all_results = []

            for q in queries:
                params = {"key": google_key, "cx": engine_id, "q": q, "num": 10}

                response = requests.get(
                    "https://customsearch.googleapis.com/customsearch/v1",
                    params=params,
                    timeout=5,
                )

                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])

                    web_snippets = []
                    for idx, item in enumerate(items[:10], 1):
                        title = item.get("title", "")
                        link = item.get("link", "")
                        snippet = item.get("snippet", "")
                        entry = f"{idx}. [{title}]({link})\n   {snippet}"
                        web_snippets.append(entry)

                    result = f"Google search for '{q}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                    all_results.append(result)
                else:
                    all_results.append(f"Google search error for '{q}': {response.status_code}")

            return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]

        except Exception as e:
            return f"Google search fallback error: {e}"

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search the web using Serper API or Google Custom Search.

        Args:
            query: Search query string or list of queries

        Returns:
            Formatted search results
        """
        api_key = os.getenv("SERPER_API_KEY")

        # Try Google Custom Search as fallback if no Serper key
        if not api_key:
            google_key = os.getenv("GOOGLE_SEARCH_SECRET_KEY")
            google_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")

            if google_key and google_engine_id:
                return self._google_search_fallback(query)

            return f"""[Search - API Key Required]

To enable real web search, use one of these options:

Option 1 - Serper (Recommended, simpler):
1. Get a free API key from https://serper.dev (2500 searches/month free)
2. Add to .env: SERPER_API_KEY=your_key_here

Option 2 - Google Custom Search:
1. Set up at https://developers.google.com/custom-search
2. Add to .env:
   GOOGLE_SEARCH_SECRET_KEY=your_key
   GOOGLE_SEARCH_ENGINE_ID=your_engine_id

Placeholder results for '{query}'..."""

        # Handle single query or list
        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")

                # Localize for Chinese queries
                if self.contains_chinese(q):
                    payload = json.dumps({"q": q, "location": "China", "gl": "cn", "hl": "zh-cn"})
                else:
                    payload = json.dumps({"q": q, "location": "United States", "gl": "us", "hl": "en"})

                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                # Retry logic
                for i in range(5):
                    try:
                        conn.request("POST", "/search", payload, headers)
                        res = conn.getresponse()
                        break
                    except Exception:
                        if i == 4:
                            all_results.append(f"Google search timeout for '{q}'")
                            continue

                data = res.read()
                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    all_results.append(f"No results found for '{q}'")
                    continue

                web_snippets = []
                for idx, page in enumerate(results.get("organic", [])[:10], 1):
                    date_published = f"\nDate: {page['date']}" if "date" in page else ""
                    source = f"\nSource: {page['source']}" if "source" in page else ""
                    snippet = f"\n{page['snippet']}" if "snippet" in page else ""

                    entry = f"{idx}. [{page.get('title', 'Untitled')}]({page.get('link', '')}){date_published}{source}{snippet}"
                    web_snippets.append(entry)

                content = f"Google search for '{q}' found {len(web_snippets)} results:\n\n" + "\n\n".join(web_snippets)
                all_results.append(content)

            except Exception as e:
                all_results.append(f"Search error for '{q}': {e}")

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class ScholarTool(DeepResearchTool):
    """Google Scholar search using Serper API (ported from Tongyi)."""

    def __init__(self):
        super().__init__(
            name="Scholar",
            description="Search Google Scholar for academic papers",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The academic search query",
                    }
                },
                "required": ["query"],
            },
        )

    async def call(self, query: str | list, **kwargs) -> str:
        """
        Search Google Scholar using Serper API.

        Args:
            query: Search query string or list of queries

        Returns:
            Academic search results
        """
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return """[Scholar - API Key Required]

To enable Google Scholar search, configure SERPER_API_KEY in your .env file."""

        queries = [query] if isinstance(query, str) else query
        all_results = []

        for q in queries:
            try:
                conn = http.client.HTTPSConnection("google.serper.dev")
                payload = json.dumps({"q": q, "type": "scholar", "num": 10})
                headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

                conn.request("POST", "/scholar", payload, headers)
                res = conn.getresponse()
                data = res.read()
                results = json.loads(data.decode("utf-8"))

                if "organic" not in results:
                    all_results.append(f"No scholar results for '{q}'")
                    continue

                papers = []
                for idx, paper in enumerate(results.get("organic", [])[:10], 1):
                    title = paper.get("title", "Untitled")
                    link = paper.get("link", "")
                    snippet = paper.get("snippet", "")
                    publication = paper.get("publication", "")
                    year = paper.get("year", "")
                    cited_by = paper.get("citedBy", {}).get("value", 0)

                    entry = f"{idx}. [{title}]({link})"
                    if publication:
                        entry += f"\n   Publication: {publication}"
                    if year:
                        entry += f" ({year})"
                    if cited_by:
                        entry += f"\n   Cited by: {cited_by}"
                    if snippet:
                        entry += f"\n   {snippet}"

                    papers.append(entry)

                result_text = f"Google Scholar search for '{q}':\n\n" + "\n\n".join(papers)
                all_results.append(result_text)

            except Exception as e:
                all_results.append(f"Scholar search error for '{q}': {e}")

        return "\n=======\n".join(all_results) if len(all_results) > 1 else all_results[0]


class VisitTool(DeepResearchTool):
    """Web page visiting with content extraction."""

    def __init__(self):
        super().__init__(
            name="Visit",
            description="Visit and extract content from web pages",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to visit"},
                    "goal": {
                        "type": "string",
                        "description": "Optional goal for the visit",
                    },
                },
                "required": ["url"],
            },
        )

    async def call(self, url: str | list, goal: str = "", **kwargs) -> str:
        """
        Visit web pages and extract content.

        Args:
            url: URL string or list of URLs
            goal: Optional goal for the visit

        Returns:
            Extracted webpage content
        """
        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            return """[Visit Tool - Dependencies Required]

To enable webpage visiting:
pip install requests beautifulsoup4

Then the tool will fetch and parse webpage content."""

        import re
        from urllib.parse import urlparse

        urls = [url] if isinstance(url, str) else url
        all_results = []

        for target_url in urls[:5]:  # Limit to 5 URLs
            try:
                # Validate and normalize URL
                parsed = urlparse(target_url)
                if not parsed.scheme:
                    target_url = f"https://{target_url}"

                # Fetch webpage
                headers = {"User-Agent": "Mozilla/5.0 (compatible; DeepResearch/1.0)"}
                response = requests.get(target_url, headers=headers, timeout=10)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove unwanted elements
                for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                    element.decompose()

                # Extract title
                title = soup.title.string if soup.title else "No Title"

                # Extract main content
                content = ""
                for selector in ["main", "article", ".content", "#content", ".post"]:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text(separator="\n", strip=True)
                        break

                if not content:
                    body = soup.find("body")
                    if body:
                        content = body.get_text(separator="\n", strip=True)

                # Clean up text
                content = re.sub(r"\n{3,}", "\n\n", content)
                content = re.sub(r" {2,}", " ", content)

                # Limit length
                if len(content) > 5000:
                    content = content[:5000] + "\n[Content truncated...]"

                # Format result
                result = f"[Webpage: {target_url}]\nTitle: {title}"
                if goal:
                    result += f"\nGoal: {goal}"
                result += f"\n\nContent:\n{content}"

                all_results.append(result)

            except Exception as e:
                all_results.append(f"[Error visiting {target_url}]: {e}")

        return "\n\n=======\n\n".join(all_results)


class FileParserTool(DeepResearchTool):
    """Enhanced file parsing for multiple formats."""

    def __init__(self):
        super().__init__(
            name="FileParser",
            description="Parse files: TXT, JSON, CSV, PDF, DOCX, etc.",
            parameters={
                "type": "object",
                "properties": {
                    "files": {
                        "type": "string",
                        "description": "File path or list of file paths to parse",
                    }
                },
                "required": ["files"],
            },
        )

    async def call(self, files: str | list, **kwargs) -> str:
        """
        Parse files and extract content.

        Args:
            files: File path string or list of paths

        Returns:
            Extracted file content
        """
        import csv
        from pathlib import Path

        file_paths = [files] if isinstance(files, str) else files
        all_results = []

        for file_path in file_paths[:10]:  # Limit to 10 files
            if not os.path.exists(file_path):
                all_results.append(f"Error: File not found at {file_path}")
                continue

            try:
                file_ext = Path(file_path).suffix.lower()
                file_name = os.path.basename(file_path)
                file_size = os.path.getsize(file_path)

                content = ""

                # Text files
                if file_ext in [
                    ".txt",
                    ".md",
                    ".log",
                    ".py",
                    ".js",
                    ".java",
                    ".cpp",
                    ".c",
                    ".h",
                ]:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                # JSON files
                elif file_ext == ".json":
                    with open(file_path, encoding="utf-8") as f:
                        data = json.load(f)
                        content = json.dumps(data, indent=2, ensure_ascii=False)

                # CSV files
                elif file_ext == ".csv":
                    rows = []
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        reader = csv.reader(f)
                        for i, row in enumerate(reader):
                            if i >= 100:
                                rows.append("[... truncated ...]")
                                break
                            rows.append(", ".join(row))
                    content = "\n".join(rows)

                # PDF files
                elif file_ext == ".pdf":
                    try:
                        import PyPDF2

                        with open(file_path, "rb") as f:
                            pdf_reader = PyPDF2.PdfReader(f)
                            pages = []
                            for i in range(min(len(pdf_reader.pages), 10)):
                                page = pdf_reader.pages[i]
                                pages.append(f"Page {i + 1}:\n{page.extract_text()}")
                            content = "\n\n".join(pages)
                    except ImportError:
                        content = "[PDF parsing requires: pip install PyPDF2]"

                # Word documents
                elif file_ext in [".docx", ".doc"]:
                    try:
                        from docx import Document

                        doc = Document(file_path)
                        paragraphs = []
                        for i, para in enumerate(doc.paragraphs):
                            if i >= 100:
                                paragraphs.append("[... truncated ...]")
                                break
                            if para.text.strip():
                                paragraphs.append(para.text)
                        content = "\n\n".join(paragraphs)
                    except ImportError:
                        content = "[DOCX parsing requires: pip install python-docx]"

                # Default: try as text
                else:
                    try:
                        with open(file_path, encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                    except Exception:
                        content = f"[Cannot parse file type: {file_ext}]"

                # Limit content
                if len(content) > 10000:
                    content = content[:10000] + "\n[Content truncated...]"

                result = f"[File: {file_name}]\nType: {file_ext}\nSize: {file_size:,} bytes\n\nContent:\n{content}"
                all_results.append(result)

            except Exception as e:
                all_results.append(f"Error parsing {file_path}: {e}")

        return "\n\n=======\n\n".join(all_results)


class PythonInterpreterTool(DeepResearchTool):
    """Safe Python code execution (from existing implementation)."""

    def __init__(self):
        super().__init__(
            name="PythonInterpreter",
            description="Execute Python code for calculations and analysis",
            parameters={
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python code to execute"}},
                "required": ["code"],
            },
        )
        self.timeout = 50

    async def call(self, code: str, timeout: int = None, **kwargs) -> str:
        """Execute Python code safely with timeout."""
        timeout = timeout or self.timeout

        # Security checks - check for dangerous imports/operations
        dangerous_patterns = [
            "import os",
            "import subprocess",
            "import sys",
            "from os import",
            "from subprocess import",
            "from sys import",
            "exec(",
            "eval(",
            "compile(",
            "open(",
            "file(",
        ]

        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                return f"[Security Error] '{pattern}' not allowed for safety reasons"

        import io
        import sys
        from concurrent.futures import ThreadPoolExecutor, TimeoutError

        # Setup safe environment
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

        # Add numpy/pandas if available
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

        # Restricted builtins with safe import capability
        def safe_import(name, *args, **kwargs):
            """Allow importing only safe modules."""
            safe_modules = [
                "math",
                "datetime",
                "json",
                "random",
                "re",
                "collections",
                "itertools",
                "statistics",
                "numpy",
                "pandas",
                "scipy",
                "scipy.linalg",  # Add scipy submodules
                "scipy.optimize",
                "scipy.signal",
                "scipy.special",
                "matplotlib",
                "matplotlib.pyplot",
            ]
            # Check if the module or its parent is allowed
            if name in safe_modules or any(name.startswith(m + ".") for m in safe_modules):
                return __import__(name, *args, **kwargs)
            else:
                raise ImportError(f"Module '{name}' is not allowed for safety reasons")

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
            "__import__": safe_import,  # Allow safe imports
            # Add exception classes for proper error handling
            "Exception": Exception,
            "ImportError": ImportError,
            "ValueError": ValueError,
            "TypeError": TypeError,
            "KeyError": KeyError,
            "IndexError": IndexError,
            "AttributeError": AttributeError,
        }

        global_vars = {"__builtins__": restricted_builtins}
        global_vars.update(allowed_modules)
        local_vars = {}

        # Capture output
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

        # Execute with timeout
        with ThreadPoolExecutor() as executor:
            try:
                future = executor.submit(execute_with_timeout)
                future.result(timeout=timeout)

                stdout_content = stdout_buffer.getvalue()
                stderr_content = stderr_buffer.getvalue()

                if stderr_content:
                    return f"[Error]\n{stderr_content}"
                elif stdout_content:
                    return f"[Output]\n{stdout_content.rstrip()}"
                else:
                    meaningful_vars = {k: v for k, v in local_vars.items() if not k.startswith("_") and k not in allowed_modules}
                    if meaningful_vars:
                        return f"[Variables]\n{meaningful_vars}"
                    else:
                        return "[Success] Code executed (no output)"

            except TimeoutError:
                return f"[Timeout] Execution exceeded {timeout}s"

        return "[Error] Unexpected execution error"


# Tool registry
DEEPRESEARCH_TOOLS = {
    "Search": SearchTool(),
    "Scholar": ScholarTool(),
    "Visit": VisitTool(),
    "FileParser": FileParserTool(),
    "PythonInterpreter": PythonInterpreterTool(),
}


def get_tool(name: str) -> DeepResearchTool:
    """Get a tool by name."""
    return DEEPRESEARCH_TOOLS.get(name)


def get_all_tools() -> dict[str, DeepResearchTool]:
    """Get all available tools."""
    return DEEPRESEARCH_TOOLS.copy()
