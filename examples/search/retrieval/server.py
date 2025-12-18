#!/usr/bin/env python3
"""
Dense retrieval server for Search training.
Based on Search-R1 retrieval_server.py pattern.
Provides E5 embeddings + FAISS dense indexing with FastAPI/uvicorn.

Environment setup:
    conda create -n retriever python=3.10
    conda activate retriever
    conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
    pip install transformers datasets pyserini
    conda install -c pytorch -c nvidia faiss-gpu=1.8.0
    pip install uvicorn fastapi

Usage:
    python server.py --index_path ./search_data/prebuilt_indices/e5_Flat.index \
                     --corpus_path ./search_data/wikipedia/wiki-18.jsonl \
                     --topk 10 --faiss_gpu
"""

import argparse
import json
import logging
import warnings

import datasets
import faiss
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_corpus(corpus_path: str):
    """Load corpus from JSONL file using datasets library for efficiency."""
    corpus = datasets.load_dataset("json", data_files=corpus_path, split="train", num_proc=4)
    return corpus


def load_docs(corpus, doc_idxs):
    """Load documents by indices from corpus."""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results


def load_model(model_path: str, use_fp16: bool = False, device: torch.device | None = None):
    """Load transformer model and tokenizer."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.to(device)

    if use_fp16 and device.type == "cuda":
        model = model.half()
    elif use_fp16 and device.type != "cuda":
        logger.warning("FP16 requested but CUDA is not available; running in FP32 on CPU.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer


def pooling(pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"):
    """Apply pooling to get sentence embeddings."""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    """Encoder for query/document embeddings using transformer models."""

    def __init__(self, model_name: str, model_path: str, pooling_method: str, max_length: int, use_fp16: bool, gpu_id: int = 0):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.gpu_id = gpu_id

        # Set the device
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=self.device)
        self.model.eval()

        logger.info(f"Encoder initialized: model={model_name}, device={self.device}, fp16={use_fp16}")

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query: bool = True) -> np.ndarray:
        """Encode queries or passages into embeddings."""
        if isinstance(query_list, str):
            query_list = [query_list]

        # Add prefixes for E5 models
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        # Add prefixes for BGE models
        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list, max_length=self.max_length, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros((inputs["input_ids"].shape[0], 1), dtype=torch.long).to(inputs["input_ids"].device)
            output = self.model(**inputs, decoder_input_ids=decoder_input_ids, return_dict=True)
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output, output.last_hidden_state, inputs["attention_mask"], self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return query_emb


class Config:
    """Configuration class for retrieval server."""

    def __init__(self, retrieval_method: str = "e5", retrieval_topk: int = 10, index_path: str = "./index/e5_Flat.index", corpus_path: str = "./data/corpus.jsonl", faiss_gpu: bool = True, gpu_id: int = 0, retrieval_model_path: str = "intfloat/e5-base-v2", retrieval_pooling_method: str = "mean", retrieval_query_max_length: int = 256, retrieval_use_fp16: bool = True, retrieval_batch_size: int = 512):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.gpu_id = gpu_id
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class BaseRetriever:
    """Base class for retrievers."""

    def __init__(self, config: Config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)

    def batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)


class BM25Retriever(BaseRetriever):
    """BM25 sparse retriever using Pyserini."""

    def __init__(self, config: Config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher

        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
        logger.info(f"BM25Retriever initialized: index={self.index_path}")

    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        hits = self.searcher.search(query, num)
        if len(hits) < 1:
            if return_score:
                return [], []
            else:
                return []
        scores = [hit.score for hit in hits]
        if len(hits) < num:
            warnings.warn("Not enough documents retrieved!", stacklevel=2)
        else:
            hits = hits[:num]

        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())["contents"] for hit in hits]
            results = [{"title": content.split("\n")[0].strip('"'), "text": "\n".join(content.split("\n")[1:]), "contents": content} for content in all_contents]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in hits])

        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results


class DenseRetriever(BaseRetriever):
    """Dense retriever using FAISS and transformer encoder."""

    def __init__(self, config: Config):
        super().__init__(config)

        # Load FAISS index
        logger.info(f"Loading FAISS index from {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

        # Move to GPU if requested
        if config.faiss_gpu:
            try:
                num_gpus = faiss.get_num_gpus()
                if num_gpus > 0:
                    co = faiss.GpuMultipleClonerOptions()
                    co.useFloat16 = True
                    co.shard = True
                    self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)
                    logger.info(f"FAISS index moved to {num_gpus} GPU(s)")
                else:
                    logger.warning("faiss-gpu not available, using CPU")
            except Exception as e:
                logger.warning(f"Could not move FAISS to GPU: {e}")

        # Load corpus
        logger.info(f"Loading corpus from {self.corpus_path}")
        self.corpus = load_corpus(self.corpus_path)
        logger.info(f"Loaded corpus with {len(self.corpus)} documents")

        # Initialize encoder
        self.encoder = Encoder(model_name=self.retrieval_method, model_path=config.retrieval_model_path, pooling_method=config.retrieval_pooling_method, max_length=config.retrieval_query_max_length, use_fp16=config.retrieval_use_fp16, gpu_id=config.gpu_id)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=num)
        idxs = idxs[0]
        scores = scores[0]
        results = load_docs(self.corpus, idxs)
        if return_score:
            return results, scores.tolist()
        else:
            return results

    def _batch_search(self, query_list: list[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc="Retrieval process"):
            query_batch = query_list[start_idx : start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            batch_scores, batch_idxs = self.index.search(batch_emb, k=num)
            batch_scores = batch_scores.tolist()
            batch_idxs = batch_idxs.tolist()

            # load_docs is not vectorized, but is a python list approach
            flat_idxs = sum(batch_idxs, [])
            batch_results = load_docs(self.corpus, flat_idxs)
            # chunk them back
            batch_results = [batch_results[i * num : (i + 1) * num] for i in range(len(batch_idxs))]

            results.extend(batch_results)
            scores.extend(batch_scores)

            del batch_emb, batch_scores, batch_idxs, query_batch, flat_idxs, batch_results
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config: Config):
    """Factory function to create appropriate retriever based on config."""
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
# FastAPI server
#####################################


class QueryRequest(BaseModel):
    """Request model for retrieval endpoint."""

    queries: list[str] | None = None
    query: str | None = None
    topk: int | None = None
    top_k: int | None = None
    k: int | None = None
    return_scores: bool = True


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    corpus_size: int
    index_type: str
    index_loaded: bool
    retrieval_method: str
    faiss_gpu: bool
    batch_size: int


app = FastAPI(title="Dense Retrieval Server", version="1.0.0")

# Global retriever instance
retriever = None
config = None


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "corpus_size": len(retriever.corpus), "index_type": "dense" if not config.retrieval_method == "bm25" else "bm25", "index_loaded": retriever.index is not None if hasattr(retriever, "index") else True, "retrieval_method": config.retrieval_method, "faiss_gpu": config.faiss_gpu, "batch_size": config.retrieval_batch_size, "topk": config.retrieval_topk, "model_path": config.retrieval_model_path, "use_fp16": config.retrieval_use_fp16}


@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    Main retrieval endpoint.

    Supports both single query and batch queries:
    - Single: {"query": "What is Python?", "topk": 3}
    - Batch: {"queries": ["What is Python?", "Tell me about AI"], "topk": 3}

    Returns search results with scores.
    """
    # Determine topk value
    topk = request.topk or request.top_k or request.k or config.retrieval_topk

    # Handle single query
    if request.query is not None:
        queries = [request.query]
    elif request.queries is not None:
        queries = request.queries
    else:
        return {"error": "Missing 'query' or 'queries' in request"}

    if not queries:
        return {"error": "'queries' list cannot be empty"}

    logger.info(f"Processing {len(queries)} queries with topk={topk}")

    # Perform batch retrieval
    results, scores = retriever.batch_search(query_list=queries, num=topk, return_score=True)

    # Format response
    resp = []
    for i, (single_result, single_scores) in enumerate(zip(results, scores, strict=False)):
        combined = []
        for doc, score in zip(single_result, single_scores, strict=False):
            combined.append({"document": doc, "score": float(score) if not isinstance(score, float) else score})
        resp.append(combined)

    # Return appropriate format based on single vs batch
    if request.query is not None:
        # Single query - return flat results
        return {"query": request.query, "method": config.retrieval_method, "results": resp[0] if resp else [], "num_results": len(resp[0]) if resp else 0}
    else:
        # Batch queries
        return {"method": config.retrieval_method, "batch_size": len(queries), "result": resp}


def main():
    """Main entry point for the server."""
    global retriever, config

    parser = argparse.ArgumentParser(description="Dense Retrieval Server (Search-R1 style)")
    parser.add_argument("--index_path", type=str, default="./search_data/prebuilt_indices/e5_Flat.index", help="Path to FAISS index file")
    parser.add_argument("--corpus_path", type=str, default="./search_data/wikipedia/wiki-18.jsonl", help="Path to corpus JSONL file")
    parser.add_argument("--topk", type=int, default=10, help="Number of retrieved passages per query")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model (e5, bge, bm25, etc.)")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="HuggingFace model path for the encoder")
    parser.add_argument("--faiss_gpu", action="store_true", help="Move FAISS index to GPU for faster search")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use for encoding")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for encoding")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum query length for tokenization")
    parser.add_argument("--use_fp16", action="store_true", default=True, help="Use FP16 for model inference")
    parser.add_argument("--no_fp16", dest="use_fp16", action="store_false", help="Disable FP16")
    parser.add_argument("--pooling_method", type=str, default="mean", choices=["mean", "cls", "pooler"], help="Pooling method for embeddings")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind server to")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    args = parser.parse_args()

    # Set logging level
    log_level = getattr(logging, args.log_level.upper())
    logger.setLevel(log_level)
    logging.getLogger().setLevel(log_level)

    # Build config
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        gpu_id=args.gpu_id,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=args.pooling_method,
        retrieval_query_max_length=args.max_length,
        retrieval_use_fp16=args.use_fp16,
        retrieval_batch_size=args.batch_size,
    )

    # Print configuration
    print("=" * 60)
    print("Dense Retrieval Server Configuration")
    print("=" * 60)
    print(f"  Index Path: {args.index_path}")
    print(f"  Corpus Path: {args.corpus_path}")
    print(f"  Retriever: {args.retriever_name}")
    print(f"  Model: {args.retriever_model}")
    print(f"  Top-K: {args.topk}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  FAISS GPU: {args.faiss_gpu}")
    print(f"  GPU ID: {args.gpu_id}")
    print(f"  FP16: {args.use_fp16}")
    print(f"  Pooling: {args.pooling_method}")
    print(f"  Max Length: {args.max_length}")
    print("=" * 60)

    # Initialize retriever
    import time

    start_time = time.time()

    try:
        retriever = get_retriever(config)
        load_time = time.time() - start_time
        print(f"\n✓ Retriever initialized in {load_time:.2f} seconds")
        print(f"  Corpus size: {len(retriever.corpus)} documents")
        if hasattr(retriever, "index"):
            print(f"  Index size: {retriever.index.ntotal} vectors")
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        import traceback

        traceback.print_exc()
        return

    # Print GPU status
    if torch.cuda.is_available():
        print(f"\n✓ GPU Available: {torch.cuda.get_device_name(args.gpu_id)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        if args.faiss_gpu:
            try:
                num_gpus = faiss.get_num_gpus()
                print(f"  FAISS GPUs: {num_gpus}")
            except Exception:
                print("  FAISS GPUs: N/A")
    else:
        print("\n⚠ Running on CPU (no GPU available)")

    print(f"\n🚀 Starting server on http://{args.host}:{args.port}")
    print("=" * 60)

    # Launch server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
