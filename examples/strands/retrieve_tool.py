"""
Strands Tool: Local Search (RAG Retrieval)

Uses the @tool decorator format expected by Strands SDK.
Connects to the RAG server running on the cluster.
"""

import os
import time
import httpx
from strands import tool


# RAG server configuration
RETRIEVAL_SERVER_URL = os.environ.get("RETRIEVAL_SERVER_URL", "http://127.0.0.1:9002")

# HTTP client — 30s timeout (matches LangGraph/fully_async), 2000 connections.
# Server-side batching handles concurrency; no client-side throttling needed.
_client = httpx.AsyncClient(
    timeout=httpx.Timeout(30.0, connect=5.0),
    limits=httpx.Limits(max_connections=2000, max_keepalive_connections=200),
)


@tool
async def local_search(query: str, top_k: int = 5) -> str:
    """
    Search for information using a dense retrieval server with Wikipedia corpus.

    Args:
        query: Search query to retrieve relevant documents
        top_k: Number of results to return (default: 5, max: 50)

    Returns:
        Formatted search results with document IDs and scores
    """
    # Validate top_k
    top_k = min(max(1, top_k), 50)

    try:
        t0 = time.perf_counter()
        response = await _client.post(
            f"{RETRIEVAL_SERVER_URL}/retrieve",
            json={"query": query, "top_k": top_k},
        )
        response.raise_for_status()
        rag_latency_ms = (time.perf_counter() - t0) * 1000
        print(f"[RAG_REQ] latency={rag_latency_ms:.0f}ms query={query[:60]}")

        # Parse response
        results = response.json().get("results", [])

        if not results:
            return "No relevant documents found for the query."

        # Format results for LLM consumption
        formatted_results = []
        for i, result in enumerate(results[:top_k], 1):
            doc_id = result.get("id", f"doc_{i}")
            # Handle nested content structure: {"content": {"contents": "...", "id": "..."}}
            content_field = result.get("content", "")
            if isinstance(content_field, dict):
                content = content_field.get("contents", "")
            else:
                content = content_field
            score = result.get("score", 0.0)

            # Truncate content if too long (keep first 300 characters)
            if len(content) > 300:
                content = content[:300] + "..."

            formatted_result = f"[Document {i}] (ID: {doc_id}, Score: {score:.3f})\n{content}\n"
            formatted_results.append(formatted_result)

        return "\n".join(formatted_results)

    except httpx.TimeoutException:
        return f"Error: Request timeout after 30 seconds. The retrieval server at {RETRIEVAL_SERVER_URL} may be down or overloaded."
    except httpx.ConnectError:
        return f"Error: Could not connect to retrieval server at {RETRIEVAL_SERVER_URL}. Please ensure the server is running."
    except Exception as e:
        return f"Error: Unexpected error - {str(e)}"


def check_retrieval_server(url: str = None, timeout: float = 5.0) -> bool:
    """
    Synchronous health check for the retrieval server.
    Call this before training to verify port 9002 is alive.
    Returns True if healthy, raises RuntimeError if not.
    """
    url = url or RETRIEVAL_SERVER_URL
    import httpx as _httpx
    try:
        with _httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/health")
            resp.raise_for_status()
            data = resp.json()
            print(f"[retrieve_tool] Health check passed: {data.get('status', 'unknown')}, "
                  f"corpus={data.get('corpus_size', '?')}, "
                  f"index={data.get('index_type', '?')}")
            return True
    except _httpx.ConnectError:
        raise RuntimeError(
            f"Retrieval server at {url} is not running. "
            f"Start it with: cd examples/strands/rag && bash launch_rag.sh /path/to/prebuilt_indices 9002"
        )
    except _httpx.TimeoutException:
        raise RuntimeError(
            f"Retrieval server at {url} is not responding (timeout={timeout}s). "
            f"It may be stuck or OOM. Check the server process."
        )
    except Exception as e:
        raise RuntimeError(f"Retrieval server health check failed: {e}")


# For testing
if __name__ == "__main__":
    # Test the tool directly
    print("Testing local_search tool...")
    print("-" * 50)

    # Health check first
    try:
        check_retrieval_server()
    except RuntimeError as e:
        print(f"FAILED: {e}")
        exit(1)

    print("-" * 50)
    query = "capital of France"
    print(f"Query: {query}")
    print("-" * 50)

    import asyncio
    result = asyncio.run(local_search(query=query, top_k=3))
    print(result)
