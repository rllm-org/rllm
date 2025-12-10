#!/usr/bin/env python3

import logging
import os
from typing import Any

import httpx

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)


class LocalRetrievalTool(Tool):
    """
    A tool for dense search using the local retrieval server.

    This tool connects to a locally running dense retrieval server (launched via retrieval_launch.sh)
    and performs dense retrieval using E5 embeddings on the indexed Wikipedia corpus.
    """

    NAME = "local_search"
    DESCRIPTION = "Search for information using a dense retrieval server with Wikipedia corpus"

    def __init__(
        self,
        name: str = NAME,
        description: str = DESCRIPTION,
        server_url: str = None,
        timeout: float = 30.0,
        max_results: int = 10,
    ):
        """
        Initialize the Local Retrieval Tool.

        Args:
            name: Tool name
            description: Tool description
            server_url: URL of the local retrieval server (if None, checks RETRIEVAL_SERVER_URL env var)
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
        """
        # Use environment variable if server_url not provided
        if server_url is None:
            server_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")

        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.max_results = max_results
        self.client = httpx.Client(timeout=timeout)

        super().__init__(name=name, description=description)

        # Test server connection
        self._test_connection()

    def _test_connection(self):
        """Test connection to the retrieval server."""
        try:
            response = self.client.get(f"{self.server_url}/health")
            if response.status_code == 200:
                logger.info(f"Successfully connected to retrieval server at {self.server_url}")
            else:
                logger.warning(f"Retrieval server returned status code {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not connect to retrieval server: {e}")

    @property
    def json(self):
        """Return tool JSON schema for LLM function calling."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query to retrieve relevant documents"},
                        "top_k": {"type": "integer", "description": f"Number of results to return (default: {self.max_results})", "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
            },
        }

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."

        formatted_results = []
        for i, result in enumerate(results[: self.max_results], 1):
            # Extract key information
            doc_id = result.get("id", f"doc_{i}")
            content = result.get("content", "")  # Fixed: use "content" not "contents"
            score = result.get("score", 0.0)

            # Truncate content if too long (keep first 300 characters)
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_result = f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    def forward(self, query: str, top_k: int | None = None) -> ToolOutput:
        """
        Execute a search query using the dense retrieval server.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            ToolOutput: Search results or error message
        """
        try:
            # Use provided parameters or defaults
            top_k = top_k or self.max_results

            # Prepare request payload
            payload = {
                "query": query,
                "top_k": min(top_k, 50),  # Cap at 50 results
            }

            # Make request to retrieval server
            response = self.client.post(f"{self.server_url}/retrieve", json=payload)

            if not response.is_success:
                error_msg = f"Retrieval server error: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', 'Unknown error')}"
                    except Exception:
                        error_msg += f" - {response.text}"

                return ToolOutput(name=self.name, error=error_msg)

            # Parse response
            response_data = response.json()
            results = response_data.get("results", [])

            if not results:
                return ToolOutput(name=self.name, output="No relevant documents found for the query.")

            # Format results
            formatted_output = self._format_search_results(results)

            # Create metadata for potential downstream use
            metadata = {"query": query, "num_results": len(results), "retriever_type": "dense", "server_url": self.server_url}

            return ToolOutput(name=self.name, output=formatted_output, metadata=metadata)

        except httpx.TimeoutException:
            return ToolOutput(name=self.name, error=f"Request timeout after {self.timeout} seconds. Please check if the retrieval server is running.")
        except httpx.ConnectError:
            return ToolOutput(name=self.name, error=f"Could not connect to retrieval server at {self.server_url}. Please ensure the server is running.")
        except Exception as e:
            return ToolOutput(name=self.name, error=f"Unexpected error: {str(e)}")

    def __del__(self):
        """Clean up HTTP client."""
        try:
            if hasattr(self, "client"):
                self.client.close()
        except Exception:
            pass


# Convenience function for tool registry
def create_local_retrieval_tool(server_url: str = "http://127.0.0.1:8000", max_results: int = 10) -> LocalRetrievalTool:
    """
    Create a LocalRetrievalTool instance with specified configuration.

    Args:
        server_url: URL of the dense retrieval server
        max_results: Maximum number of results to return

    Returns:
        LocalRetrievalTool instance
    """
    return LocalRetrievalTool(server_url=server_url, max_results=max_results)


def to_langchain_tool(server_url: str = "http://127.0.0.1:8000", max_results: int = 10, timeout: float = 30.0):
    """
    Create a LangChain StructuredTool for retrieval server without RLLM Tool dependency.

    This function creates a standalone LangChain tool that directly calls the retrieval server,
    making it compatible with LangGraph's ToolNode and agent workflows.

    Args:
        server_url: URL of the dense retrieval server (default: http://127.0.0.1:8000)
        max_results: Maximum number of results to return (default: 10)
        timeout: Request timeout in seconds (default: 30.0)

    Returns:
        A LangChain StructuredTool instance that can be used with LangGraph agents

    Example:
        >>> from langgraph.prebuilt import ToolNode
        >>>
        >>> # Create LangChain-compatible tool
        >>> retriever_tool = to_langchain_tool(server_url="http://127.0.0.1:9002", max_results=10)
        >>>
        >>> # Use with LangGraph
        >>> tool_node = ToolNode([retriever_tool])
        >>> response_model.bind_tools([retriever_tool])
    """
    try:
        from langchain_core.tools import StructuredTool
    except ImportError:
        raise ImportError("langchain_core is required to use to_langchain_tool(). Install it with: pip install langchain-core") from None

    # Create HTTP client for the retrieval server
    client = httpx.Client(timeout=timeout)
    server_url = server_url.rstrip("/")

    def retrieve_from_server(query: str, top_k: int = max_results) -> str:
        """
        Search for information using a dense retrieval server with Wikipedia corpus.

        Args:
            query: Search query to retrieve relevant documents
            top_k: Number of results to return

        Returns:
            A string containing the retrieved document content with scores
        """
        try:
            # Prepare request payload
            payload = {
                "query": query,
                "top_k": min(top_k, 50),  # Cap at 50 results
            }

            # Make request to retrieval server
            response = client.post(f"{server_url}/retrieve", json=payload)

            if not response.is_success:
                error_msg = f"Retrieval server error: {response.status_code}"
                if response.content:
                    try:
                        error_data = response.json()
                        error_msg += f" - {error_data.get('error', 'Unknown error')}"
                    except Exception:
                        error_msg += f" - {response.text}"
                return f"Error: {error_msg}"

            # Parse response
            response_data = response.json()
            results = response_data.get("results", [])

            if not results:
                return "No relevant documents found for the query."

            # Format results for LLM consumption
            formatted_results = []
            for i, result in enumerate(results[:max_results], 1):
                doc_id = result.get("id", f"doc_{i}")
                content = result.get("content", "")
                score = result.get("score", 0.0)

                # Truncate content if too long (keep first 300 characters)
                if len(content) > 300:
                    content = content[:300] + "..."

                formatted_result = f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n"
                formatted_results.append(formatted_result)

            return "\n".join(formatted_results)

        except httpx.TimeoutException:
            return f"Error: Request timeout after {timeout} seconds. Please check if the retrieval server is running."
        except httpx.ConnectError:
            return f"Error: Could not connect to retrieval server at {server_url}. Please ensure the server is running."
        except Exception as e:
            return f"Error: Unexpected error - {str(e)}"

    # Convert to LangChain StructuredTool
    langchain_tool = StructuredTool.from_function(
        func=retrieve_from_server,
        name="local_search",
        description="Search for information using a dense retrieval server with Wikipedia corpus",
    )

    return langchain_tool
