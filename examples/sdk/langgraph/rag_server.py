#!/usr/bin/env python3
"""
Multi-GPU sharded retrieval server for large-scale search.
Distributes the 60GB FAISS index across all available GPUs.

For a 60GB index with 8 GPUs: ~7.5GB per GPU instead of 60GB on one GPU.

Usage:
    python server_gpu_sharded.py --data_dir ./search_data/prebuilt_indices --port 9002
"""

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import torch
from flask import Flask, jsonify, request
from sentence_transformers import SentenceTransformer


class MultiGPURetriever:
    """Multi-GPU sharded retrieval system using FAISS."""

    def __init__(self, data_dir: str, use_gpu: bool = True, ngpus: int = None, embedding_device: str = "cpu", embedding_gpu: int = None):
        """
        Initialize multi-GPU retriever.

        Args:
            data_dir: Directory containing corpus and index files
            use_gpu: Whether to use GPU acceleration for index
            ngpus: Number of GPUs to use for index (None = use all available)
            embedding_device: Device for embedding model ('cpu' or 'cuda')
            embedding_gpu: Specific GPU ID for embedding model (only if embedding_device='cuda')
        """
        self.data_dir = Path(data_dir)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.ngpus = ngpus
        self.embedding_device = None  # Will be set below
        self.corpus = []
        self.dense_index = None

        # Disable tokenizer parallelism to prevent thread explosion with high concurrency
        import os

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Limit torch threads to prevent CPU thread explosion
        torch.set_num_threads(2)

        # Initialize encoder on specified device
        if embedding_device == "cuda" and torch.cuda.is_available():
            if embedding_gpu is not None:
                device = f"cuda:{embedding_gpu}"
            else:
                device = "cuda"
            print(f"Initializing encoder on {device}...")
        else:
            device = "cpu"
            print("Initializing encoder on CPU (thread-limited for concurrency)...")

        self.embedding_device = device
        self.encoder = SentenceTransformer("intfloat/e5-base-v2", device=device)

        self._load_data()

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
                # Determine number of GPUs to use
                if self.ngpus is None:
                    self.ngpus = available_gpus
                else:
                    self.ngpus = min(self.ngpus, available_gpus)

                print(f"\n{'=' * 70}")
                print("Multi-GPU Configuration")
                print(f"{'=' * 70}")
                print(f"Available GPUs: {available_gpus}")
                print(f"Using GPUs: {self.ngpus}")

                # Calculate memory requirements
                index_size_gb = dense_index_file.stat().st_size / 1e9
                per_gpu_gb = index_size_gb / self.ngpus

                print(f"Total index size: {index_size_gb:.1f} GB")
                print(f"Per GPU (sharded): ~{per_gpu_gb:.1f} GB")
                print(f"{'=' * 70}\n")

                # Check GPU memory for each GPU
                print("GPU Memory Status:")
                for gpu_id in range(self.ngpus):
                    gpu_mem = torch.cuda.get_device_properties(gpu_id).total_memory
                    gpu_mem_gb = gpu_mem / 1e9
                    gpu_name = torch.cuda.get_device_name(gpu_id)
                    print(f"  GPU {gpu_id} ({gpu_name}): {gpu_mem_gb:.1f} GB total")
                    if per_gpu_gb > gpu_mem_gb * 0.8:
                        print(f"    ⚠ Warning: {per_gpu_gb:.1f} GB may not fit")

                print()

                try:
                    print(f"Sharding index across {self.ngpus} GPUs...")

                    # Create GPU cloner options
                    co = faiss.GpuMultipleClonerOptions()
                    co.shard = True  # Shard the index (split vectors across GPUs)
                    co.useFloat16 = False  # Use full precision

                    # Shard index across GPUs
                    self.dense_index = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=self.ngpus)

                    print(f"✓ Index successfully sharded across {self.ngpus} GPUs")
                    print(f"  Each GPU holds ~{per_gpu_gb:.1f} GB of the index")
                    print("  Search queries will be distributed across all GPUs in parallel")

                except Exception as e:
                    print(f"⚠ Failed to shard index across GPUs: {e}")
                    print("  Falling back to CPU")
                    self.dense_index = index_cpu
                    self.use_gpu = False
        else:
            self.dense_index = index_cpu
            print("Using CPU index")

    def search(self, query: str, k: int = 10) -> list[dict[str, Any]]:
        """Dense retrieval using multi-GPU sharded FAISS."""
        query_vector = self.encoder.encode([f"query: {query}"]).astype("float32")
        scores, indices = self.dense_index.search(query_vector, k)

        return [{"content": self.corpus[idx], "score": float(score)} for score, idx in zip(scores[0], indices[0], strict=False) if idx < len(self.corpus)]


# Flask app
app = Flask(__name__)
retriever = None


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "corpus_size": len(retriever.corpus),
            "index_type": f"dense_gpu_sharded_{retriever.ngpus}" if retriever.use_gpu else "dense_cpu",
            "index_loaded": retriever.dense_index is not None,
            "gpu_available": retriever.use_gpu,
            "num_gpus": retriever.ngpus if retriever.use_gpu else 0,
            "sharding_enabled": retriever.use_gpu,
            "embedding_device": retriever.embedding_device,
        }
    )


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Main retrieval endpoint."""
    try:
        data = request.get_json()
        print(data)
        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400

        query = data["query"]
        k = data.get("top_k", data.get("k", 10))

        results = retriever.search(query=query, k=k)

        formatted_results = [{"id": f"doc_{i}", "content": result["content"], "score": result["score"]} for i, result in enumerate(results, 1)]

        method = f"dense_gpu_sharded_{retriever.ngpus}" if retriever.use_gpu else "dense"

        return jsonify({"query": query, "method": method, "results": formatted_results, "num_results": len(formatted_results)})

    except Exception as e:
        print("Error: ", e)
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU sharded retrieval server")
    parser.add_argument("--data_dir", default="./search_data/prebuilt_indices", help="Directory containing corpus and dense index")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9002, help="Port to bind to")
    parser.add_argument("--ngpus", type=int, default=None, help="Number of GPUs to use for index (default: all available)")
    parser.add_argument("--no_gpu", action="store_true", help="Disable GPU acceleration for index (use CPU only)")
    parser.add_argument("--embedding_device", default="cpu", choices=["cpu", "cuda"], help="Device for embedding model (default: cpu)")
    parser.add_argument("--embedding_gpu", type=int, default=None, help="Specific GPU ID for embedding model (only if --embedding_device=cuda)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-GPU SHARDED RETRIEVAL SERVER")
    print("=" * 70)

    # Initialize retriever
    global retriever
    try:
        retriever = MultiGPURetriever(args.data_dir, use_gpu=not args.no_gpu, ngpus=args.ngpus, embedding_device=args.embedding_device, embedding_gpu=args.embedding_gpu)
        print("\n✓ Multi-GPU retrieval server initialized")
        print(f"  Corpus: {len(retriever.corpus):,} documents")
        if retriever.use_gpu:
            print(f"  Index GPUs: {retriever.ngpus} (sharded)")
        else:
            print("  Index Mode: CPU")
        print(f"  Embedding Device: {retriever.embedding_device}")
    except Exception as e:
        print(f"\n✗ Failed to initialize retriever: {e}")
        import traceback

        traceback.print_exc()
        return

    # Start server
    print(f"\n{'=' * 70}")
    print(f"Starting server on {args.host}:{args.port}")
    print("Single-threaded mode to prevent CPU thread explosion")
    print(f"{'=' * 70}\n")

    # Use threaded=False to handle one request at a time, preventing thread explosion
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False)


if __name__ == "__main__":
    main()
