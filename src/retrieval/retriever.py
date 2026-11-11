"""向量检索 + BM25 混合召回"""
from __future__ import annotations
import re
from typing import Optional

from loguru import logger

from ..models import Chunk
from ..config import get_config


# ── 嵌入模型 ──────────────────────────────────────────────────────────

class EmbeddingModel:
    _instance = None

    def __init__(self):
        cfg = get_config()["embedding"]
        self.provider = cfg.get("provider", "local")
        self.batch_size = cfg.get("batch_size", 32)

        if self.provider in ("vllm", "openai"):
            # 远程 API 模式（vLLM / OpenAI 兼容）
            self.base_url = cfg["base_url"].rstrip("/")
            self.model_name = cfg.get("model")
            logger.info(f"Using remote embedding API: {self.base_url} (model={self.model_name})")
        else:
            # 本地 SentenceTransformer 模式
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {cfg['model']}")
            self._local_model = SentenceTransformer(cfg["model"], device=cfg.get("device", "cpu"))

    @classmethod
    def get(cls) -> "EmbeddingModel":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def encode(self, texts: list[str]):
        import numpy as np
        if self.provider in ("vllm", "openai"):
            return self._encode_remote(texts)
        return self._encode_local(texts)

    def _encode_local(self, texts: list[str]) -> np.ndarray:
        return self._local_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def _encode_remote(self, texts: list[str]) -> np.ndarray:
        import httpx
        import numpy as np

        url = f"{self.base_url}/v1/embeddings"
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {
                "model": self.model_name,
                "input": batch,
            }
            headers = {}
            try:
                resp = httpx.post(url, json=payload, headers=headers, timeout=60.0)
                resp.raise_for_status()
                data = resp.json()
                emb = [item["embedding"] for item in data["data"]]
                all_embeddings.extend(emb)
            except Exception as e:
                logger.error(f"Remote embedding API call failed: {e}")
                raise RuntimeError(f"Embedding API call failed: {e}") from e

        arr = np.array(all_embeddings, dtype=np.float32)
        # L2 归一化（与本地 SentenceTransformer 行为一致）
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        return arr


# ── ChromaDB 向量库封装 ───────────────────────────────────────────────

class VectorStore:
    def __init__(self, cfg: dict = None):
        if cfg is None:
            cfg = get_config()["vectordb"]
        import chromadb
        self.client = chromadb.PersistentClient(path=cfg.get("persist_dir", "./data/chroma_db"))
        self.collection = self.client.get_or_create_collection(
            name=cfg.get("collection", "private_docs"),
            metadata={"hnsw:space": "cosine"},
        )
        self._chunks_cache: dict[str, Chunk] = {}

    def add_chunks(self, chunks: list[Chunk]):
        emb_model = EmbeddingModel.get()
        texts = [c.content for c in chunks]
        embeddings = emb_model.encode(texts).tolist()

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page or 0,
                    **{k: str(v) for k, v in c.metadata.items()},
                }
                for c in chunks
            ],
        )
        for c in chunks:
            self._chunks_cache[c.chunk_id] = c

    def search(self, query: str, top_k: int = 5, where: dict = None) -> list[Chunk]:
        emb_model = EmbeddingModel.get()
        query_emb = emb_model.encode([query]).tolist()

        kwargs = {
            "query_embeddings": query_emb,
            "n_results": min(top_k, max(1, self.collection.count())),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)
        chunks = []
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            cid = results["ids"][0][i]
            chunk = Chunk(
                chunk_id=cid,
                content=doc,
                source=meta.get("source", ""),
                page=int(meta.get("page", 0)) or None,
                metadata={k: v for k, v in meta.items() if k not in ("source", "page")},
                score=1.0 - dist,  # cosine 距离转相似度
            )
            chunks.append(chunk)
        return chunks

    def count(self) -> int:
        return self.collection.count()


# ── BM25 检索 ─────────────────────────────────────────────────────────

class BM25Index:
    """基于 rank_bm25 的关键词检索（离线，无需外部服务）"""

    def __init__(self):
        self._corpus: list[Chunk] = []
        self._bm25 = None

    def _tokenize(self, text: str) -> list[str]:
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            return list(text)

    def build(self, chunks: list[Chunk]):
        self._corpus = chunks
        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(c.content) for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[Chunk]:
        if not self._bm25 or not self._corpus:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        results = []
        for idx in top_idx:
            if scores[idx] > 0:
                c = self._corpus[idx]
                c.score = float(scores[idx])
                results.append(c)
        return results


# ── 混合检索器 ────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.bm25: Optional[BM25Index] = None
        cfg = get_config()["retrieval"]
        self.top_k = cfg.get("top_k", 5)
        self.expand_k = cfg.get("expand_k", 10)
        self.use_bm25 = cfg.get("use_bm25", True)

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        version_filter: dict = None,
        expand: bool = False,
    ) -> list[Chunk]:
        k = (self.expand_k if expand else (top_k or self.top_k))

        # 向量检索
        vector_results = self.vector_store.search(query, top_k=k, where=version_filter)

        if not self.use_bm25 or self.bm25 is None:
            return vector_results

        # BM25 检索
        bm25_results = self.bm25.search(query, top_k=k)

        # RRF 融合 (Reciprocal Rank Fusion)
        scores: dict[str, float] = {}
        id_to_chunk: dict[str, Chunk] = {}

        for rank, chunk in enumerate(vector_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (rank + 60)
            id_to_chunk[chunk.chunk_id] = chunk

        for rank, chunk in enumerate(bm25_results):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + 1.0 / (rank + 60)
            id_to_chunk[chunk.chunk_id] = chunk

        ranked = sorted(scores.items(), key=lambda x: -x[1])[:k]
        merged = []
        for cid, score in ranked:
            chunk = id_to_chunk[cid]
            chunk.score = score
            merged.append(chunk)
        return merged

    def rewrite_query(self, original_query: str, failed_sentence: str) -> str:
        """查询改写：把失败句子的关键词融入查询（无需 LLM）"""
        import jieba
        keywords = list(jieba.cut_for_search(failed_sentence))
        # 取 TF 较高的非停用词
        stopwords = {"的", "了", "在", "是", "和", "有", "与", "或", "及", "对"}
        keywords = [k for k in keywords if len(k) > 1 and k not in stopwords][:5]
        if keywords:
            return f"{original_query} {' '.join(keywords)}"
        return original_query
