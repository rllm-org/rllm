"""
Simplified RAG utilities for claim+label corpora (self-play fact-check).

Design goals
------------
- Keep infra tiny: single-file module, pure-Python dependencies.
- Support BOTH dense and sparse retrieval; gracefully degrade if some deps aren't installed.
- Treat each example as a "claim document", with diverse templated rewrites to enrich retrieval.
- Focus on core functionality: build index, search, save/load.

Intended usage
--------------
from claim_rag import ClaimDoc, build_corpus, RagIndexer

# 1) Build docs from your processed datasets
docs = build_corpus(dataset)  # HuggingFace Dataset with 'claim' and 'label' columns

# 2) Index
indexer = RagIndexer(method="auto")   # "auto" = prefer dense; fallback to sparse
indexer.build(docs)
indexer.save("artifacts/")

# 3) Retrieve
hits = indexer.search(query="Masks prevent COVID transmission", top_k=5)
# Returns: [{"id": ..., "score": ..., "label": ..., "text": ..., "rank": ...}, ...]

# 4) Load and search
indexer = RagIndexer.load("artifacts/")
hits = indexer.search(query="...", top_k=5)
"""

import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any

from rllm.tools.tool_base import Tool, ToolOutput

logger = logging.getLogger(__name__)

# Optional deps
_ST_AVAILABLE = True
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:
    _ST_AVAILABLE = False
    import numpy as np  # still use numpy

# Sparse options: BM25 preferred, else TF-IDF
_BM25_AVAILABLE = True
try:
    from rank_bm25 import BM25Okapi
except Exception:
    _BM25_AVAILABLE = False


@dataclass
class ClaimDoc:
    id: str
    claim: str
    label: bool
    text: str  # the retrieval text (claim + label + template)
    meta: dict[str, Any] | None = None


# Diverse templates for creating retrieval text from (claim, label) pairs
DIVERSE_TEMPLATES = [
    # Template 1: Fact-check verdict style
    lambda claim, label: f"Fact-check verdict: The claim '{claim}' has been verified as {('accurate and true' if label else 'false and misleading')}.",
    # Template 2: Scientific assessment style
    lambda claim, label: f"Scientific assessment: '{claim}' — This statement is {('supported by evidence' if label else 'refuted by evidence')}.",
    # Template 3: Direct verification style
    lambda claim, label: f"Verification result: {claim} — {('TRUE' if label else 'FALSE')}",
    # Template 4: Expert review style
    lambda claim, label: f"Expert review confirms that the assertion '{claim}' is {('factually correct' if label else 'factually incorrect')}.",
    # Template 5: Evidence-based conclusion style
    lambda claim, label: f"Based on available evidence, '{claim}' can be classified as {('true' if label else 'false')}.",
    # Template 6: Journalistic style
    lambda claim, label: f"Investigation reveals: The statement that {claim} is {('accurate' if label else 'inaccurate')}.",
    # Template 7: Academic citation style
    lambda claim, label: f"Claim: {claim} | Status: {('VERIFIED' if label else 'DEBUNKED')} by fact-checkers.",
    # Template 8: Plain assertion style
    lambda claim, label: f"'{claim}' — This is {('true' if label else 'false')}.",
]


def create_document_text(claim: str, label: bool, template_idx: int | None = None) -> str:
    """Create retrieval text using one of the diverse templates."""
    if template_idx is None:
        template = random.choice(DIVERSE_TEMPLATES)
    else:
        template = DIVERSE_TEMPLATES[template_idx % len(DIVERSE_TEMPLATES)]
    return template(claim, label)


def build_corpus(
    dataset: list[dict[str, Any]] | Any,
    claim_col: str = "claim",
    label_col: str = "label",
    id_prefix: str = "doc",
    shuffle_templates: bool = True,
) -> list[ClaimDoc]:
    """
    Turn dataset into ClaimDoc list. Works with HuggingFace Dataset or list of dicts.
    Each row needs: claim and label (bool) columns.

    Args:
        dataset: HuggingFace Dataset or list of dicts with claim and label fields
        claim_col: name of claim column (default: "claim")
        label_col: name of label column (default: "label")
        id_prefix: prefix for auto-generated IDs (default: "doc")
        shuffle_templates: randomly select templates for diversity (default: True)

    Returns:
        List of ClaimDoc objects ready for indexing
    """
    docs: list[ClaimDoc] = []

    # Handle HuggingFace Dataset
    if hasattr(dataset, "__iter__") and not isinstance(dataset, list | dict):
        rows = list(dataset)
    else:
        rows = dataset

    for idx, row in enumerate(rows):
        doc_id = row.get("id", f"{id_prefix}_{idx}")
        claim = str(row[claim_col]).strip()
        label = bool(row[label_col])

        # Create retrieval text using diverse templates
        text = create_document_text(claim, label, template_idx=None if shuffle_templates else idx)

        docs.append(ClaimDoc(id=str(doc_id), claim=claim, label=label, text=text, meta={"index": idx}))

    return docs


class RagIndexer:
    """
    Minimal dual-mode indexer:
    - Dense: SentenceTransformer embeddings + FAISS-like numpy search (brute force cosine)
    - Sparse: BM25 (if available) or TF-IDF cosine

    Artifacts:
    - config.json
    - ids.json
    - labels.npy
    - dense.npy (if dense)
    - tfidf.npz + vocab.json (if tfidf)
    - bm25.pkl (if bm25)
    """

    def __init__(self, method: str = "auto", st_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.method = method  # "auto", "dense", "sparse"
        self.st_model_name = st_model
        self._dense_model = None
        self._dense_mat = None  # (N, D) float32
        self._tfidf = None  # sklearn vectorizer
        self._tfidf_mat = None  # (N, V) sparse
        self._bm25 = None  # BM25Okapi
        self._texts: list[str] = []
        self._ids: list[str] = []
        self._labels: np.ndarray | None = None

    def build(self, docs: list[ClaimDoc], use_dense: bool | None = None):
        self._texts = [d.text for d in docs]
        self._ids = [d.id for d in docs]
        self._labels = np.array([int(d.label) for d in docs], dtype=np.int8)

        if use_dense is None:
            if self.method == "dense":
                use_dense = True
            elif self.method == "sparse":
                use_dense = False
            else:
                use_dense = _ST_AVAILABLE  # auto

        if use_dense and not _ST_AVAILABLE:
            print("[claim_rag] SentenceTransformers not available; falling back to sparse.")
            use_dense = False

        if use_dense:
            self._build_dense()
        else:
            self._build_sparse()

    def _build_dense(self):
        self._dense_model = SentenceTransformer(self.st_model_name)
        emb = self._dense_model.encode(self._texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        self._dense_mat = np.asarray(emb, dtype=np.float32)

    def _build_sparse(self):
        if _BM25_AVAILABLE:
            tokenized = [t.lower().split() for t in self._texts]
            self._bm25 = BM25Okapi(tokenized)
        else:
            raise ValueError("BM25 is not available")

        # Always build TF-IDF as a fallback + for scoring
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._tfidf = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.95)
        self._tfidf_mat = self._tfidf.fit_transform(self._texts)

    def save(self, outdir: str):
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "config.json"), "w") as f:
            json.dump({"method": self.method, "st_model": self.st_model_name}, f, indent=2)
        with open(os.path.join(outdir, "ids.json"), "w") as f:
            json.dump(self._ids, f)
        np.save(os.path.join(outdir, "labels.npy"), self._labels)

        if self._dense_mat is not None:
            np.save(os.path.join(outdir, "dense.npy"), self._dense_mat)

        # Save TF-IDF
        if self._tfidf is not None and self._tfidf_mat is not None:
            import joblib

            joblib.dump(self._tfidf, os.path.join(outdir, "tfidf.vectorizer.pkl"))
            joblib.dump(self._tfidf_mat, os.path.join(outdir, "tfidf.matrix.pkl"))

        # Save BM25
        if self._bm25 is not None:
            import joblib

            joblib.dump(self._bm25, os.path.join(outdir, "bm25.pkl"))

        # Save texts
        with open(os.path.join(outdir, "texts.jsonl"), "w") as f:
            for t in self._texts:
                f.write(json.dumps({"text": t}) + "\n")

    @classmethod
    def load(cls, outdir: str) -> "RagIndexer":
        with open(os.path.join(outdir, "config.json")) as f:
            cfg = json.load(f)
        self = cls(method=cfg.get("method", "auto"), st_model=cfg.get("st_model", "sentence-transformers/all-MiniLM-L6-v2"))
        with open(os.path.join(outdir, "ids.json")) as f:
            self._ids = json.load(f)
        self._labels = np.load(os.path.join(outdir, "labels.npy"))
        # Load texts
        texts = []
        with open(os.path.join(outdir, "texts.jsonl")) as f:
            for line in f:
                texts.append(json.loads(line)["text"])
        self._texts = texts

        # Dense
        dense_path = os.path.join(outdir, "dense.npy")
        if os.path.exists(dense_path):
            self._dense_mat = np.load(dense_path)

        # Sparse
        bm25_path = os.path.join(outdir, "bm25.pkl")
        if os.path.exists(bm25_path):
            import joblib

            self._bm25 = joblib.load(bm25_path)
            tfidf_vec_path = os.path.join(outdir, "tfidf.vectorizer.pkl")
            tfidf_mat_path = os.path.join(outdir, "tfidf.matrix.pkl")
            assert os.path.exists(tfidf_vec_path) and os.path.exists(tfidf_mat_path), "TF-IDF files are missing for sparse mode"
            self._tfidf = joblib.load(tfidf_vec_path)
            self._tfidf_mat = joblib.load(tfidf_mat_path)

        return self

    def search(self, query: str, top_k: int = 5, exclude_ids: set | None = None) -> list[dict[str, Any]]:
        """Return list of {id, score, label, text, rank} sorted by score desc."""
        assert self._labels is not None, "Labels are not loaded"
        exclude_ids = exclude_ids or set()
        N = len(self._ids)
        if self._dense_mat is not None:
            if self._dense_model is None:
                self._dense_model = SentenceTransformer(self.st_model_name)
            q = self._dense_model.encode([query], normalize_embeddings=True)[0].astype(np.float32)
            sims = self._dense_mat @ q  # cosine
            # mask excluded
            mask = np.ones(N, dtype=bool)
            if exclude_ids:
                id2idx = {i: idx for idx, i in enumerate(self._ids)}
                for _id in exclude_ids:
                    if _id in id2idx:
                        mask[id2idx[_id]] = False
            sims_masked = np.where(mask, sims, -1e9)
            idx = np.argpartition(-sims_masked, kth=min(top_k, N - 1))[:top_k]
            idx = idx[np.argsort(-sims_masked[idx])]
            return [dict(id=self._ids[i], score=float(sims[i]), label=bool(self._labels[i]), text=self._texts[i], rank=r + 1) for r, i in enumerate(idx)]
        else:
            # Sparse route: BM25 if available, else TF-IDF cosine
            if self._bm25 is not None:
                toks = query.lower().split()
                scores = np.array(self._bm25.get_scores(toks))
                # mask
                if exclude_ids:
                    id2idx = {i: idx for idx, i in enumerate(self._ids)}
                    for _id in exclude_ids:
                        if _id in id2idx:
                            scores[id2idx[_id]] = -1e9
                idx = np.argpartition(-scores, kth=min(top_k, N - 1))[:top_k]
                idx = idx[np.argsort(-scores[idx])]
                return [dict(id=self._ids[i], score=float(scores[i]), label=bool(self._labels[i]), text=self._texts[i], rank=r + 1) for r, i in enumerate(idx)]
            else:
                # TF-IDF cosine
                vec = self._tfidf.transform([query])
                scores = (self._tfidf_mat @ vec.T).toarray().ravel()
                if exclude_ids:
                    id2idx = {i: idx for idx, i in enumerate(self._ids)}
                    for _id in exclude_ids:
                        if _id in id2idx:
                            scores[id2idx[_id]] = -1e9
                idx = np.argpartition(-scores, kth=min(top_k, N - 1))[:top_k]
                idx = idx[np.argsort(-scores[idx])]
                return [dict(id=self._ids[i], score=float(scores[i]), label=bool(self._labels[i]), text=self._texts[i], rank=r + 1) for r, i in enumerate(idx)]

    def id_to_label(self, _id: str) -> bool:
        assert self._labels is not None, "Labels are not loaded"
        try:
            i = self._ids.index(_id)
            return bool(self._labels[i])
        except ValueError as e:
            raise KeyError(_id) from e

    def size(self) -> int:
        return len(self._ids)


"""
The actual tool class to be used by a dispatcher agent.
"""
_indexer_registry: dict[str, RagIndexer] = {}


class ClaimRAGTool(Tool):
    def __init__(self, rag_data_dir: str, top_k: int = 3, index_name: str = "climate_claim"):
        name = index_name + "_rag_tool"
        description = f"A tool that can search the {index_name} RAG index for relevant fact-check information about {index_name.split('_')[0]}."

        if index_name not in _indexer_registry:
            load_dir = os.path.join(rag_data_dir, index_name)
            logger.info(f"Loading RAG index from {load_dir}")

            _indexer_registry[index_name] = RagIndexer.load(load_dir)

        self.indexer = _indexer_registry[index_name]
        self.top_k = top_k

        self._json = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The query/claim to search the RAG index for."}},
                    "required": ["query"],
                },
            },
        }

        super().__init__(name=name, description=description)

    def _format_rag_search_hits(self, hits: list[dict], shuffle: bool = False) -> str:
        """
        Format the RAG search hits into a string.
        """
        if shuffle:
            random.shuffle(hits)

        retrieved_lines = []
        for i, hit in enumerate(hits):
            retrieved_lines.append(f"[Document {i}, Similarity Score: {hit['score']:.4f}]: {hit['text']}")
        retrieved_results = "\n".join(retrieved_lines)
        return retrieved_results

    async def async_forward(self, query: str, shuffle: bool = False) -> ToolOutput:
        try:
            hits = self.indexer.search(query, top_k=self.top_k)
            return ToolOutput(name=self.name or "unknown", output=self._format_rag_search_hits(hits, shuffle=shuffle))
        except Exception as e:
            logger.error(f"Error searching RAG index: {e}")
            return ToolOutput(name=self.name or "unknown", error=f"Error searching RAG index: {e}", output="")
