#!/usr/bin/env python3
"""Simple async local retrieval tool with client-side load balancing."""

import json
import os
import time
from typing import Any, Literal

import httpx


class LocalRetrievalTool:
    """Simple async tool for dense search with client-side load balancing across multiple servers."""

    def __init__(
        self,
        server_url: str = None,
        server_urls: list[str] = None,
        url_file: str = "/home/user/ram_rl_memory/.deepresearch/rag_url",
        timeout: float = 30.0,
        max_results: int = 10,
        format_style: Literal["original", "concise"] = "concise",
        max_content_length: int = 300,
        max_connections: int = 2000,
        max_retries: int = 2,
    ):
        """
        Args:
            server_url: URL of a single retrieval server (for backward compatibility)
            server_urls: List of server URLs for load balancing (takes precedence over server_url)
            url_file: Path to file containing URLs (one per line, auto-refreshes every 5 min)
            timeout: Request timeout in seconds
            max_results: Maximum number of results to return
            format_style: Output format style
                - "original": Verbose format with ID and score: [Document 1] (ID: doc_1, Score: 0.834)\n{raw_content}
                - "concise": Clean numbered format with title extraction: [1] Title\nContent
            max_content_length: Maximum content length before truncation
            max_connections: Maximum concurrent HTTP connections (default 2000)
            max_retries: Number of retries on different servers before failing
        """
        # Build list of server URLs
        self.server_urls: list[str] = []
        self._url_file = url_file
        self._last_file_read = 0.0
        self._file_refresh_interval = 300.0  # 5 minutes

        if url_file:
            self._load_urls_from_file()
        elif server_urls:
            self.server_urls = [url.rstrip("/") for url in server_urls]
        elif server_url:
            self.server_urls = [server_url.rstrip("/")]
        else:
            # Check environment variable - supports comma-separated list
            env_url = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:8000")
            self.server_urls = [url.strip().rstrip("/") for url in env_url.split(",") if url.strip()]

        if not self.server_urls:
            self.server_urls = ["http://127.0.0.1:8000"]

        # For backward compatibility
        self.server_url = self.server_urls[0]

        self.timeout = timeout
        self.max_results = max_results
        self.max_retries = min(max_retries, len(self.server_urls))
        self.name = "local_search"
        self.description = "Search for information using a dense retrieval server with Wikipedia corpus"

        self.format_style = format_style
        self.max_content_length = max_content_length
        self.max_connections = max_connections

        # Lazy init: client created on first use (avoids pickle issues with Ray)
        self._client: httpx.AsyncClient | None = None

        # Simple round-robin counter for load balancing
        self._server_index = 0

    def _load_urls_from_file(self):
        """Load server URLs from file (one URL per line)."""
        if not self._url_file:
            return

        try:
            with open(self._url_file) as f:
                urls = [line.strip().rstrip("/") for line in f if line.strip() and not line.startswith("#")]

            if urls:
                self.server_urls = urls
                self.server_url = urls[0]
                self.max_retries = min(self.max_retries, len(urls))
                self._last_file_read = time.time()
        except Exception:
            # Keep existing URLs if file read fails
            pass

    def _maybe_refresh_urls(self):
        """Check if URL file needs refreshing and reload if necessary."""
        if not self._url_file:
            return

        if time.time() - self._last_file_read >= self._file_refresh_interval:
            self._load_urls_from_file()

    def _ensure_client(self):
        """Lazily create the HTTP client on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=self.max_connections, max_keepalive_connections=self.max_connections // 10),
            )

    def _get_next_server(self, exclude: set[str] = None) -> str | None:
        """Get the next server URL using round-robin, optionally excluding failed servers."""
        self._maybe_refresh_urls()  # Check if URLs need refreshing

        exclude = exclude or set()
        available = [url for url in self.server_urls if url not in exclude]

        if not available:
            return None

        # Simple round-robin
        idx = self._server_index % len(available)
        self._server_index = (self._server_index + 1) % len(self.server_urls)
        return available[idx]

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
                    },
                    "required": ["query"],
                },
            },
        }

    def _extract_raw_content(self, result: dict[str, Any]) -> str:
        """Extract raw content from a search result, handling nested structures.

        Returns:
            str: The raw content text (no title extraction or quote cleanup)
        """
        content = result.get("content", "")

        # Handle case where content is a dict with 'contents' key
        if isinstance(content, dict):
            return content.get("contents", content.get("content", str(content)))
        elif isinstance(content, str):
            # Try to parse as JSON if it looks like a dict
            if content.startswith("{") and "contents" in content:
                try:
                    parsed = json.loads(content.replace("'", '"'))
                    return parsed.get("contents", content)
                except Exception:
                    return content
            else:
                return content
        else:
            return str(content)

    def _extract_content(self, result: dict[str, Any]) -> tuple[str, str]:
        """Extract title and content from a search result, handling nested structures.

        Returns:
            tuple: (title, content) where title may be empty string
        """
        text = self._extract_raw_content(result)

        # Extract title (first quoted text at start)
        title = ""
        if text.startswith('"'):
            end_quote = text.find('"', 1)
            if end_quote > 0:
                title = text[1:end_quote]
                text = text[end_quote + 1 :].lstrip('\n "')

        # Clean up the text - replace double quotes with single
        text = text.replace('""', '"').strip()

        return title, text

    def _truncate_content(self, content: str) -> str:
        """Truncate content to max length."""
        if len(content) <= self.max_content_length:
            return content

        return content[: self.max_content_length] + "..."

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Format search results for LLM consumption."""
        if not results:
            return "No relevant documents found."

        formatted_results = []
        for i, result in enumerate(results[: self.max_results], 1):
            doc_id = result.get("id", f"doc_{i}")
            score = result.get("score", 0.0)

            if self.format_style == "original":
                # Original verbose format - extract raw content and truncate
                content = self._extract_raw_content(result)
                if len(content) > self.max_content_length:
                    content = content[: self.max_content_length] + "..."
                formatted_result = f"[Document {i}]\n{content}\n"
            else:
                # Concise XML-like format with title extraction
                title, content = self._extract_content(result)
                content = self._truncate_content(content)
                if title:
                    formatted_result = f'<result id="{doc_id}" score="{score:.3f}">\n<title>{title}</title>\n{content}\n</result>'
                else:
                    formatted_result = f'<result id="{doc_id}" score="{score:.3f}">\n{content}\n</result>'

            formatted_results.append(formatted_result)

        separator = "\n" if self.format_style == "original" else "\n\n"
        return separator.join(formatted_results)

    async def run(self, query: str, top_k: int | None = None) -> str:
        """
        Execute a search query asynchronously with automatic failover.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            str: Formatted search results or error message
        """
        success = False
        top_k = top_k or self.max_results
        payload = {"query": query, "top_k": min(top_k, 50)}

        self._ensure_client()

        tried_servers: set[str] = set()
        last_error: str = ""
        result: str = ""

        # Try servers with automatic failover
        for attempt in range(self.max_retries):
            server_url = self._get_next_server(exclude=tried_servers)
            if server_url is None:
                break

            tried_servers.add(server_url)

            try:
                response = await self._client.post(f"{server_url}/retrieve", json=payload)

                if not response.is_success:
                    last_error = f"Server {server_url} returned {response.status_code}"
                    continue  # Try next server

                response_data = response.json()
                results = response_data.get("results", [])

                success = True
                if not results:
                    result = "No relevant documents found for the query."
                else:
                    result = self._format_search_results(results)
                break  # Success, exit loop

            except httpx.TimeoutException:
                last_error = f"Timeout on {server_url}"
                continue  # Try next server
            except httpx.ConnectError:
                last_error = f"Connection failed to {server_url}"
                continue  # Try next server
            except Exception as e:
                last_error = f"Error on {server_url}: {str(e)}"
                continue  # Try next server

        # Handle failure case
        if not success:
            if last_error:
                result = f"Error: All servers failed. Last error: {last_error}"
            else:
                result = "Error: No retrieval servers available."

        return result
