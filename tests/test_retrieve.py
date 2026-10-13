from __future__ import annotations

from pathlib import Path

from tests.fakes import FakeEmbedder
from voice_rag.index import build_index
from voice_rag.retrieve import retrieve, search


def test_search_orders_by_similarity_and_limits_k(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    hits = search(index, embedder.embed_query("delivery to Islamabad working days"), top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score
    assert hits[0].chunk.source == "delivery.txt"
    assert "Islamabad" in hits[0].chunk.text


def test_retrieve_returns_timing_and_filenames(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    hits, seconds = retrieve(index, embedder, "damaged items reported hours", top_k=3)
    assert seconds >= 0.0
    assert hits[0].chunk.source == "returns.txt"
    assert "48 hours" in hits[0].chunk.text


def test_top_k_larger_than_index_is_safe(kb_dir: Path) -> None:
    embedder = FakeEmbedder()
    index = build_index(kb_dir, embedder)
    hits = search(index, embedder.embed_query("anything"), top_k=50)
    assert len(hits) == len(index.chunks)
