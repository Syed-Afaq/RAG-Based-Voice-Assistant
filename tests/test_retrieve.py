from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeEmbedder
from voice_rag.index import build_index
from voice_rag.retrieve import retrieve, search


def test_search_fuses_cosine_and_bm25(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    question = "delivery to Islamabad working days"
    ranked = search(index, question, embedder.embed_query(question))
    assert len(ranked) == len(index.chunks)
    assert ranked[0].chunk.source == "delivery.txt"
    assert "Islamabad" in ranked[0].chunk.text
    assert ranked[0].bm25_rank == 1
    assert ranked[0].fused >= ranked[1].fused >= ranked[-1].fused
    # A chunk sharing no token with the question has no BM25 rank.
    assert any(r.bm25_rank is None for r in ranked)


def test_retrieve_limits_to_top_k_when_confident(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    result = retrieve(index, embedder, "damaged items reported hours", top_k=3, min_similarity=0.0)
    assert len(result.hits) == 3
    assert result.seconds >= 0.0
    assert not result.low_confidence
    assert result.hits[0].chunk.source == "returns.txt"
    assert "48 hours" in result.hits[0].chunk.text


def test_low_confidence_falls_back_to_all_chunks(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    result = retrieve(index, embedder, "zebra quantum tuesday", top_k=3, min_similarity=0.5)
    assert result.low_confidence
    assert result.best_cosine < 0.5
    assert len(result.hits) == result.total_chunks == len(index.chunks)


def test_anchor_terms_force_product_passages_in(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    result = retrieve(
        index,
        embedder,
        "total with delivery?",
        top_k=1,
        min_similarity=0.0,
        anchor_terms=["L1", "S1"],
    )
    texts = " ".join(h.chunk.text for h in result.hits)
    assert "L1" in texts and "S1" in texts
    assert len(result.hits) == 3
