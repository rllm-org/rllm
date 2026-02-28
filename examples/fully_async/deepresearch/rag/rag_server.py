#!/usr/bin/env python3
"""
Multi-GPU sharded retrieval server with AUTO-BATCHING for maximum throughput.

This version collects incoming requests and processes them in batches for
significantly higher throughput. Both embedding and FAISS search are batched.

For a 60GB index with 8 GPUs: ~7.5GB per GPU instead of 60GB on one GPU.

Usage:
    # Default: batch every 100ms, max batch size 64
    python rag_server.py --data_dir ./search_data/prebuilt_indices --port 9002

    # Custom batching: 50ms window, max 128 queries per batch
    python rag_server.py --data_dir ./data --port 9002 --batch_timeout 0.05 --max_batch_size 128

    # With uvicorn:
    uvicorn rag_server:app --host 0.0.0.0 --port 9002 --workers 1
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# Global retriever instance
retriever = None


@dataclass
class PendingQuery:
    """A query waiting to be processed in a batch."""

    query: str
    k: int
    future: asyncio.Future
    submit_time: float


class BatchingRetriever:
    """
    Multi-GPU sharded retrieval system with automatic request batching.

    Collects incoming queries and processes them in batches for higher throughput.
    Both embedding and FAISS search operations are batched.
    """

    def __init__(
        self,
        data_dir: str,
        use_gpu: bool = True,
        ngpus: int = None,
        embedding_device: str = "cuda",
        embedding_gpu: int = None,
        faiss_temp_mem_mb: int = 2048,
        faiss_use_float16: bool = False,
        faiss_query_batch_size: int = 16,
        batch_timeout: float = 0.1,  # 100ms default
        max_batch_size: int = 64,
    ):
        """
        Initialize batching retriever.

        Args:
            data_dir: Directory containing corpus and index files
            use_gpu: Whether to use GPU acceleration for index
            ngpus: Number of GPUs to use for index (None = use all available)
            embedding_device: Device for embedding model ('cpu' or 'cuda')
            embedding_gpu: Specific GPU ID for embedding model
            faiss_temp_mem_mb: Per-GPU FAISS temporary memory (MB) for GEMM/workspace
            faiss_use_float16: Use float16 storage/compute in FAISS GPU (lower memory)
            faiss_query_batch_size: Max queries per FAISS search() call (micro-batching)
            batch_timeout: Max time to wait for batch to fill (seconds)
            max_batch_size: Maximum queries per batch
        """
        self.data_dir = Path(data_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.ngpus = ngpus
        self.embedding_device = embedding_device
        self.corpus = []
        self.dense_index = None
        self.encoder = None

        # FAISS GPU resource tuning
        self.faiss_temp_mem_mb = faiss_temp_mem_mb
        self.faiss_use_float16 = faiss_use_float16
        self.faiss_query_batch_size = max(1, int(faiss_query_batch_size))

        # Batching parameters
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size

        # Request queue and processing
        self.pending_queries: list[PendingQuery] = []
        self.queue_lock = asyncio.Lock()
        self.batch_event = asyncio.Event()
        self.processing_task = None
        self.processing_lock = asyncio.Lock()  # Ensure only one batch processes at a time

        # Stats
        self.total_batches = 0
        self.total_queries = 0
        self.total_batch_time = 0.0

        # Disable tokenizer parallelism
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.set_num_threads(4)

        # Initialize encoder
        self._init_encoder(embedding_device, embedding_gpu)

        # Load data
        self._load_data()

    def _init_encoder(self, embedding_device: str, embedding_gpu: int):
        """Initialize the encoder model."""
        if embedding_device == "cuda" and torch.cuda.is_available():
            if embedding_gpu is not None:
                device = f"cuda:{embedding_gpu}"
            else:
                device = "cuda:0"
        else:
            device = "cpu"

        print(f"Initializing encoder on {device}...")
        self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=device)
        self.embedding_device = device
        print(f"✓ Encoder initialized on {device}")

    def _load_data(self):
        """Load corpus and dense index, shard across multiple GPUs."""
        print(f"Loading data from {self.data_dir}")

        # Load corpus (JSONL format)
        corpus_file = self.data_dir / "corpus.json"
        with open(corpus_file) as f:
            self.corpus = [json.loads(line.strip()) for line in f if line.strip()]
        print(f"Loaded corpus with {len(self.corpus):,} documents")

        # Load dense index
        dense_index_file = self.data_dir / "e5_Flat.index"
        print(f"Loading index from {dense_index_file.name}...")
        index_cpu = faiss.read_index(str(dense_index_file))
        print(f"Loaded dense index with {index_cpu.ntotal:,} vectors")

        # Move to GPU(s) if requested
        if self.use_gpu:
            available_gpus = faiss.get_num_gpus()
            if not available_gpus:
                print("⚠ Warning: No GPUs detected by FAISS, falling back to CPU")
                self.dense_index = index_cpu
                self.use_gpu = False
            else:
                if self.ngpus is None:
                    self.ngpus = available_gpus
                else:
                    self.ngpus = min(self.ngpus, available_gpus)

                print(f"\n{'=' * 70}")
                print("Multi-GPU Configuration")
                print(f"{'=' * 70}")
                print(f"Available GPUs: {available_gpus}")
                print(f"Using GPUs: {self.ngpus}")

                index_size_gb = dense_index_file.stat().st_size / 1e9
                per_gpu_gb = index_size_gb / self.ngpus
                print(f"Total index size: {index_size_gb:.1f} GB")
                print(f"Per GPU (sharded): ~{per_gpu_gb:.1f} GB")

                try:
                    print(f"Sharding index across {self.ngpus} GPUs...")
                    print(f"FAISS temp memory per GPU: {self.faiss_temp_mem_mb} MB")
                    print(f"FAISS useFloat16: {self.faiss_use_float16}")

                    # Create explicit GPU resources and increase temp memory.
                    # This avoids cuBLAS execution failures caused by insufficient/fragmented workspace.
                    resources = [faiss.StandardGpuResources() for _ in range(self.ngpus)]
                    temp_bytes = int(self.faiss_temp_mem_mb) * 1024 * 1024
                    for r in resources:
                        r.setTempMemory(temp_bytes)

                    vres = faiss.GpuResourcesVector()
                    vdev = faiss.IntVector()
                    for gpu_id, r in enumerate(resources):
                        vres.push_back(r)
                        vdev.push_back(gpu_id)

                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True
                    co.useFloat16 = bool(self.faiss_use_float16)

                    # Use the explicit resources vector so each GPU has enough workspace.
                    self.dense_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index_cpu, co)
                    print(f"✓ Index successfully sharded across {self.ngpus} GPUs")
                except Exception as e:
                    print(f"⚠ Failed to shard index on GPU: {e}")
                    print("  Falling back to CPU index")
                    self.dense_index = index_cpu
                    self.use_gpu = False
        else:
            self.dense_index = index_cpu
            print("Using CPU index")

    async def start_batch_processor(self):
        """Start the background batch processing task."""
        if self.processing_task is None:
            self.processing_task = asyncio.create_task(self._batch_processor_loop())
            print(f"✓ Batch processor started (timeout={self.batch_timeout}s, max_batch={self.max_batch_size})")

    async def stop_batch_processor(self):
        """Stop the background batch processing task."""
        if self.processing_task:
            self.processing_task.cancel()
            try:
                await self.processing_task
            except asyncio.CancelledError:
                pass
            self.processing_task = None

    async def _batch_processor_loop(self):
        """Background loop that processes batches of queries."""
        while True:
            try:
                # Wait for either timeout or batch_event (triggered when queue is full)
                try:
                    await asyncio.wait_for(self.batch_event.wait(), timeout=self.batch_timeout)
                except asyncio.TimeoutError:
                    pass

                # Reset event
                self.batch_event.clear()

                # Get pending queries
                async with self.queue_lock:
                    if not self.pending_queries:
                        continue

                    # Take up to max_batch_size queries
                    batch = self.pending_queries[: self.max_batch_size]
                    self.pending_queries = self.pending_queries[self.max_batch_size :]

                # Process the batch
                if batch:
                    await self._process_batch(batch)

            except asyncio.CancelledError:
                # Process remaining queries before shutting down
                async with self.queue_lock:
                    if self.pending_queries:
                        await self._process_batch(self.pending_queries)
                        self.pending_queries = []
                raise
            except Exception as e:
                print(f"Error in batch processor: {e}")
                import traceback

                traceback.print_exc()

    async def _process_batch(self, batch: list[PendingQuery]):
        """Process a batch of queries. Uses lock to prevent concurrent GPU access."""
        if not batch:
            return

        # Ensure only one batch is processed at a time (FAISS GPU is not thread-safe)
        async with self.processing_lock:
            batch_start = time.perf_counter()
            batch_size = len(batch)

            try:
                # Extract queries and find max k
                queries = [f"query: {pq.query}" for pq in batch]
                max_k = max(pq.k for pq in batch)

                # Batch encode all queries - run synchronously to avoid GPU conflicts
                # SentenceTransformer.encode() is already efficient on GPU
                query_vectors = self.encoder.encode(queries, show_progress_bar=False)
                query_vectors = np.ascontiguousarray(query_vectors, dtype="float32")

                # FAISS GPU batched search can trigger cuBLAS execution failures at larger
                # n_queries (e.g., 32/64) depending on workspace/algorithm selection.
                # Micro-batch the FAISS search to keep behavior closer to v1.
                step = batch_size
                if self.use_gpu and batch_size > self.faiss_query_batch_size:
                    step = self.faiss_query_batch_size

                if step == batch_size:
                    scores_batch, indices_batch = self.dense_index.search(query_vectors, max_k)
                else:
                    scores_batch = np.empty((batch_size, max_k), dtype=np.float32)
                    indices_batch = np.empty((batch_size, max_k), dtype=np.int64)
                    for start in range(0, batch_size, step):
                        end = min(start + step, batch_size)
                        s, idx = self.dense_index.search(query_vectors[start:end], max_k)
                        scores_batch[start:end] = s
                        indices_batch[start:end] = idx

                # Distribute results to each query
                for i, pq in enumerate(batch):
                    try:
                        k = pq.k
                        scores = scores_batch[i][:k]
                        indices = indices_batch[i][:k]

                        results = [{"content": self.corpus[idx], "score": float(score)} for score, idx in zip(scores, indices, strict=False) if idx < len(self.corpus)]

                        pq.future.set_result(results)
                    except Exception as e:
                        pq.future.set_exception(e)

                # Update stats
                batch_time = time.perf_counter() - batch_start
                self.total_batches += 1
                self.total_queries += batch_size
                self.total_batch_time += batch_time

                avg_latency = batch_time / batch_size * 1000  # ms per query
                throughput = batch_size / batch_time  # queries per second

                print(f"Batch {self.total_batches}: {batch_size} queries in {batch_time * 1000:.1f}ms ({avg_latency:.1f}ms/query, {throughput:.1f} q/s)")

            except Exception as e:
                # Set exception for all pending queries
                for pq in batch:
                    if not pq.future.done():
                        pq.future.set_exception(e)

    async def search_async(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Submit a query for batched processing.

        The query is added to a queue and processed with other queries
        in the next batch for higher throughput.
        """
        # Create future for this query's result
        loop = asyncio.get_event_loop()
        future = loop.create_future()

        pending = PendingQuery(query=query, k=k, future=future, submit_time=time.perf_counter())

        # Add to queue
        async with self.queue_lock:
            self.pending_queries.append(pending)

            # Trigger immediate processing if batch is full
            if len(self.pending_queries) >= self.max_batch_size:
                self.batch_event.set()

        # Wait for result
        return await future

    def get_stats(self) -> dict:
        """Get batching statistics."""
        avg_batch_size = self.total_queries / self.total_batches if self.total_batches > 0 else 0
        avg_throughput = self.total_queries / self.total_batch_time if self.total_batch_time > 0 else 0

        return {
            "total_batches": self.total_batches,
            "total_queries": self.total_queries,
            "avg_batch_size": avg_batch_size,
            "avg_throughput_qps": avg_throughput,
            "batch_timeout_ms": self.batch_timeout * 1000,
            "max_batch_size": self.max_batch_size,
            "faiss_query_batch_size": self.faiss_query_batch_size,
        }


# Pydantic models for request/response
class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 10
    k: int | None = None


# FastAPI app
app = FastAPI(title="Multi-GPU RAG Server v2 (Auto-Batching)")


@app.on_event("startup")
async def startup_event():
    """Start the batch processor on app startup."""
    if retriever is not None:
        await retriever.start_batch_processor()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the batch processor on app shutdown."""
    if retriever is not None:
        await retriever.stop_batch_processor()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    stats = retriever.get_stats()

    return {
        "status": "healthy",
        "version": "v2-batching",
        "corpus_size": len(retriever.corpus),
        "index_type": f"dense_gpu_sharded_{retriever.ngpus}" if retriever.use_gpu else "dense_cpu",
        "index_loaded": retriever.dense_index is not None,
        "gpu_available": retriever.use_gpu,
        "num_gpus": retriever.ngpus if retriever.use_gpu else 0,
        "embedding_device": retriever.embedding_device,
        "batching": {
            "timeout_ms": retriever.batch_timeout * 1000,
            "max_batch_size": retriever.max_batch_size,
        },
        "stats": stats,
        "pid": os.getpid(),
    }


@app.get("/stats")
async def get_stats():
    """Get detailed batching statistics."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    return retriever.get_stats()


@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    """Main retrieval endpoint with auto-batching."""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialized")

    try:
        query = request.query
        k = request.k if request.k is not None else request.top_k

        # Submit to batch processor and wait for result
        results = await retriever.search_async(query=query, k=k)

        formatted_results = [{"id": f"doc_{i}", "content": result["content"], "score": result["score"]} for i, result in enumerate(results, 1)]

        method = f"dense_gpu_sharded_{retriever.ngpus}_batched" if retriever.use_gpu else "dense_batched"

        return {"query": query, "method": method, "results": formatted_results, "num_results": len(formatted_results)}

    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=500, detail=str(e)) from e


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU sharded retrieval server with auto-batching")
    parser.add_argument("--data_dir", default="./search_data/prebuilt_indices", help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind to")
    parser.add_argument("--ngpus", type=int, default=None, help="Number of GPUs to use for index (default: all available)")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration for index")
    parser.add_argument("--embedding_device", default="cuda", choices=["cpu", "cuda"], help="Device for embedding model")
    parser.add_argument("--embedding_gpu", type=int, default=0, help="GPU ID for embedding model")
    parser.add_argument("--faiss_temp_mem_mb", type=int, default=2048, help="FAISS temp memory per GPU in MB (default: 2048)")
    parser.add_argument("--faiss_use_float16", action="store_true", help="Use float16 in FAISS GPU (reduces memory, may affect accuracy)")
    parser.add_argument("--faiss_query_batch_size", type=int, default=8, help="Max queries per FAISS search() call (default: 8)")
    parser.add_argument("--batch_timeout", type=float, default=0.1, help="Batch timeout in seconds (default: 0.1)")
    parser.add_argument("--max_batch_size", type=int, default=64, help="Maximum batch size (default: 64)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-GPU SHARDED RETRIEVAL SERVER v2 (AUTO-BATCHING)")
    print("=" * 70)

    # Initialize retriever
    global retriever
    try:
        retriever = BatchingRetriever(
            args.data_dir,
            use_gpu=not args.no_gpu,
            ngpus=args.ngpus,
            embedding_device=args.embedding_device,
            embedding_gpu=args.embedding_gpu,
            faiss_temp_mem_mb=args.faiss_temp_mem_mb,
            faiss_use_float16=args.faiss_use_float16,
            faiss_query_batch_size=args.faiss_query_batch_size,
            batch_timeout=args.batch_timeout,
            max_batch_size=args.max_batch_size,
        )
        print("\n✓ Batching retrieval server initialized")
        print(f"  Corpus: {len(retriever.corpus):,} documents")
        if retriever.use_gpu:
            print(f"  Index GPUs: {retriever.ngpus} (sharded)")
        else:
            print("  Index Mode: CPU")
        print(f"  Embedding Device: {retriever.embedding_device}")
        print(f"  Batch Timeout: {args.batch_timeout * 1000:.0f}ms")
        print(f"  Max Batch Size: {args.max_batch_size}")
    except Exception as e:
        print(f"\n✗ Failed to initialize retriever: {e}")
        import traceback

        traceback.print_exc()
        return

    # Start server
    print(f"\n{'=' * 70}")
    print(f"Starting server on {args.host}:{args.port}")
    print("Auto-batching mode for maximum throughput")
    print(f"{'=' * 70}\n")

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
