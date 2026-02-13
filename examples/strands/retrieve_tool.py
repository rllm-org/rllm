"""Retrieve tool for Strands Agent."""

import os

import httpx

from strands import tool


@tool
def retrieve(query: str, top_k: int = 5) -> str:
    """
    Search the knowledge base for relevant information.

    Args:
        query: Search query
        top_k: Number of documents to retrieve (default: 5)

    Returns:
        Relevant passages with source IDs and scores.
    """
    server_url = os.getenv("RETRIEVAL_SERVER_URL", "http://127.0.0.1:9002")

    try:
        response = httpx.post(
            f"{server_url}/retrieve",
            json={"query": query, "top_k": top_k},
            timeout=30.0,
        )

        if not response.is_success:
            return f"Error: Retrieval server returned status {response.status_code}"

        results = response.json().get("results", [])

        if not results:
            return "No relevant documents found for this query."

        output = []
        for i, r in enumerate(results, 1):
            doc_id = r.get("id", f"doc_{i}")
            content = r.get("content", "")
            score = r.get("score", 0.0)

            if len(content) > 400:
                content = content[:400] + "..."

            output.append(f"[{i}] (ID: {doc_id}, Score: {score:.3f})\n{content}")

        return "\n\n".join(output)

    except httpx.ConnectError:
        return f"Error: Cannot connect to retrieval server at {server_url}"
    except httpx.TimeoutException:
        return "Error: Request timed out"
    except Exception as e:
        return f"Error: {type(e).__name__} - {str(e)}"


if __name__ == "__main__":
    print("Testing retrieve tool...")
    result = retrieve("Who founded Stripe?")
    print(result)
