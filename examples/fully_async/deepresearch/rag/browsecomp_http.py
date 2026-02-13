#!/usr/bin/env python3
"""
Simple HTTP server for the retrieval searcher.

Usage:
  # BM25
  python http_search_server.py --searcher-type bm25 --index-path indexes/bm25/ --host 0.0.0.0 --port 8000

  # FAISS
  python http_search_server.py --searcher-type faiss \
      --index-path "indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
      --model-name "Qwen/Qwen3-Embedding-8B" \
      --host 0.0.0.0 --port 8000

Then query with:
  curl "http://localhost:8000/search?query=Who+invented+the+telephone&k=5"
  curl "http://localhost:8000/document/doc123"
  curl -X POST http://localhost:8000/search -H "Content-Type: application/json" -d '{"query": "capital of France", "k": 5}'
"""

import argparse
import os
import sys

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import uvicorn
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from searcher.searchers import SearcherType


# Request/Response models
class SearchRequest(BaseModel):
    query: str
    k: int = 5


class RetrieveRequest(BaseModel):
    """Request model for /retrieve endpoint (compatible with tool.py)."""

    query: str
    top_k: int = 10
    k: int | None = None


class SearchResult(BaseModel):
    docid: str
    score: float | None = None
    text: str
    snippet: str | None = None


class SearchResponse(BaseModel):
    query: str
    k: int
    results: list[SearchResult]


class DocumentResponse(BaseModel):
    docid: str
    text: str


# Global searcher instance
searcher = None
snippet_max_tokens = 512
tokenizer = None


def truncate_text(text: str, max_tokens: int) -> str:
    """Truncate text to max_tokens using the tokenizer."""
    global tokenizer
    if tokenizer is None or max_tokens <= 0:
        return text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=True)
    return text


# Create FastAPI app
app = FastAPI(
    title="BrowseComp-Plus Search Server",
    description="HTTP API for BM25/FAISS retrieval",
    version="1.0.0",
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "searcher_type": searcher.search_type if searcher else None,
    }


@app.get("/health")
def health():
    """Health check."""
    return {"status": "healthy"}


@app.get("/search", response_model=SearchResponse)
def search_get(
    query: str = Query(..., description="Search query"),
    k: int = Query(5, description="Number of results to return"),
):
    """Search via GET request."""
    return do_search(query, k)


@app.post("/search", response_model=SearchResponse)
def search_post(request: SearchRequest):
    """Search via POST request with JSON body."""
    return do_search(request.query, request.k)


def do_search(query: str, k: int) -> dict:
    """Perform search and return results."""
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        results = searcher.search(query, k=k)

        # Add snippets (truncated text)
        formatted_results = []
        for r in results:
            formatted_results.append(
                {
                    "docid": r.get("docid"),
                    "score": r.get("score"),
                    "text": r.get("text", ""),
                    "snippet": truncate_text(r.get("text", ""), snippet_max_tokens),
                }
            )

        return {
            "query": query,
            "k": k,
            "results": formatted_results,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/document/{docid}", response_model=DocumentResponse)
def get_document(docid: str):
    """Get full document by ID."""
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        doc = searcher.get_document(docid)
        if doc is None:
            raise HTTPException(status_code=404, detail=f"Document {docid} not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/retrieve")
def retrieve(request: RetrieveRequest):
    """
    Retrieve endpoint compatible with tool.py and rag_server_v2.py.

    Accepts: {"query": "...", "top_k": 10} or {"query": "...", "k": 10}
    Returns: {"query": "...", "results": [{"id": "...", "content": {...}, "score": ...}, ...]}

    Note: Scores are returned as-is from the searcher. Use --normalize flag on FAISS
    searcher to get proper cosine similarity scores in [0, 1] range.
    """
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized")

    try:
        k = request.k if request.k is not None else request.top_k
        results = searcher.search(request.query, k=k)

        # Format results to match rag_server_v2.py output (compatible with tool.py)
        # RAG server v2 returns: {"id": "doc_N", "content": {"id": docid, "contents": text}, "score": 0.xx}
        formatted_results = []

        for i, r in enumerate(results, 1):
            docid = r.get("docid", f"doc_{i}")
            text = r.get("text", "")
            score = r.get("score", 0.0)

            formatted_results.append(
                {
                    "id": f"doc_{i}",
                    "content": {
                        "id": docid,
                        "contents": text,
                    },
                    "score": score,
                }
            )

        return {
            "query": request.query,
            "method": f"{searcher.search_type.lower()}_search",
            "results": formatted_results,
            "num_results": len(formatted_results),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    global searcher, snippet_max_tokens, tokenizer

    parser = argparse.ArgumentParser(description="HTTP Search Server")
    parser.add_argument(
        "--searcher-type",
        choices=SearcherType.get_choices(),
        required=True,
        help="Type of searcher to use",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind (default: 127.0.0.1, use 0.0.0.0 for all interfaces)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind (default: 8000)",
    )
    parser.add_argument(
        "--snippet-max-tokens",
        type=int,
        default=512,
        help="Max tokens for snippets (default: 512, -1 to disable)",
    )

    # Parse known args first to get searcher type
    temp_args, _ = parser.parse_known_args()
    searcher_class = SearcherType.get_searcher_class(temp_args.searcher_type)
    searcher_class.parse_args(parser)

    args = parser.parse_args()

    # Initialize tokenizer for snippets
    snippet_max_tokens = args.snippet_max_tokens
    if snippet_max_tokens > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

    # Initialize searcher
    print(f"[*] Initializing {args.searcher_type} searcher...")
    searcher = searcher_class(args)
    print(f"[*] Searcher type: {searcher.search_type}")
    print(f"[*] Starting HTTP server on {args.host}:{args.port}")
    print("\n[*] Endpoints:")
    print(f"    GET  http://{args.host}:{args.port}/search?query=...&k=5")
    print(f"    POST http://{args.host}:{args.port}/search  (JSON body)")
    print(f"    GET  http://{args.host}:{args.port}/document/<docid>")
    print(f"    GET  http://{args.host}:{args.port}/health")
    print()

    # Run server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

"""
# FAISS with Qwen3-Embedding-8B
python browsecomp_http.py --searcher-type faiss \
    --index-path "/path/to/BrowseComp-Plus/indexes/qwen3-embedding-8b/corpus.shard*.pkl" \
    --model-name "Qwen/Qwen3-Embedding-8B" \
    --normalize \
    --host 0.0.0.0 --port 8000

# BM25
python browsecomp_http.py --searcher-type bm25 \
    --index-path "/path/to/BrowseComp-Plus/indexes/bm25/" \
    --host 0.0.0.0 --port 8000

# Or use the launch script:
bash launch_browsecomp.sh faiss 8000
bash launch_browsecomp.sh bm25 8000
"""
