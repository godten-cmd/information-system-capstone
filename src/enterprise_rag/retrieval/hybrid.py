"""Hybrid retrieval: BM25 + Dense fused via Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from enterprise_rag.retrieval.bm25 import BM25Index
from enterprise_rag.retrieval.dense import DenseIndex
from enterprise_rag.retrieval.fusion import reciprocal_rank_fusion
from enterprise_rag.retrieval.result_schema import RetrievalRun, RetrievedChunk


@dataclass
class HybridConfig:
    """Configuration for hybrid BM25 + Dense retrieval."""

    rrf_k: int = 60
    # Candidates retrieved from each method before fusion.
    # Should be ≥ max evaluation K; larger means better recall ceiling.
    candidate_k: int = 100
    max_k: int = 10


class HybridIndex:
    """Combines a BM25Index and DenseIndex via RRF fusion.

    Does not own the indexes — they must be built/loaded before passing in.
    This lets the runner reuse the same models across multiple rrf_k sweeps.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        dense_index: DenseIndex,
        config: HybridConfig | None = None,
    ) -> None:
        self._bm25 = bm25_index
        self._dense = dense_index
        self._config = config or HybridConfig()
        # Build chunk_id → metadata lookup from BM25 index (always present)
        self._chunk_meta: dict[str, dict[str, Any]] = self._build_meta_lookup()

    def _build_meta_lookup(self) -> dict[str, dict[str, Any]]:
        meta = self._bm25._meta
        return {
            cid: {
                "document_id": meta.document_ids[i],
                "category": meta.categories[i],
                "section_path": meta.section_paths[i],
                "text": meta.texts[i],
            }
            for i, cid in enumerate(meta.chunk_ids)
        }

    def with_rrf_k(self, rrf_k: int) -> "HybridIndex":
        """Return a new HybridIndex with a different rrf_k (cheap: no data copy)."""
        new_config = HybridConfig(
            rrf_k=rrf_k,
            candidate_k=self._config.candidate_k,
            max_k=self._config.max_k,
        )
        return HybridIndex(self._bm25, self._dense, config=new_config)

    # ── Retrieve ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        query_id: str,
        k: int = 10,
    ) -> list[RetrievedChunk]:
        """Retrieve top-k chunks by fusing BM25 and Dense results via RRF."""
        candidate_k = max(k, self._config.candidate_k)
        bm25_results = self._bm25.retrieve(query_text, query_id, k=candidate_k)
        dense_results = self._dense.retrieve(query_text, query_id, k=candidate_k)

        bm25_ids = [r.chunk_id for r in bm25_results]
        dense_ids = [r.chunk_id for r in dense_results]

        fused = reciprocal_rank_fusion(
            [bm25_ids, dense_ids],
            rrf_k=self._config.rrf_k,
            top_n=k,
        )

        results: list[RetrievedChunk] = []
        for rank, (chunk_id, score) in enumerate(fused, start=1):
            m = self._chunk_meta.get(chunk_id, {})
            results.append(
                RetrievedChunk(
                    query_id=query_id,
                    chunk_id=chunk_id,
                    document_id=m.get("document_id", ""),
                    rank=rank,
                    score=round(score, 6),
                    category=m.get("category", ""),
                    section_path=m.get("section_path", []),
                    text=m.get("text", "")[:200],
                )
            )
        return results

    def run(
        self,
        queries: list[dict],
        k: int = 10,
        method: str = "hybrid",
    ) -> RetrievalRun:
        """Run hybrid retrieval for all queries."""
        run = RetrievalRun(
            method=method,
            k=k,
            config={
                "rrf_k": self._config.rrf_k,
                "candidate_k": self._config.candidate_k,
                "bm25_config": asdict(self._bm25._config),
                "dense_config": asdict(self._dense._config),
            },
        )
        for q in queries:
            query_id = q["query_id"]
            query_text = q.get("query", q.get("query_text", ""))
            results = self.retrieve(query_text, query_id, k=k)
            run.results.extend(results)
        return run


def build_hybrid_index(
    bm25_index: BM25Index,
    dense_index: DenseIndex,
    config: HybridConfig | None = None,
) -> HybridIndex:
    """Construct a HybridIndex from existing BM25 and Dense indexes."""
    return HybridIndex(bm25_index, dense_index, config=config)
