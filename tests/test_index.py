from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.fakes import FakeEmbedder
from voice_rag.index import (
    build_index,
    chunk_file,
    file_hashes,
    load_index,
    load_or_build,
    save_index,
    split_sentences,
    stale_reasons,
)


def test_split_sentences_keeps_semicolon_clauses_together() -> None:
    text = "A costs PKR 2,500. To X takes 2–3 days; Y takes 3–5 days. Done!"
    assert split_sentences(text) == [
        "A costs PKR 2,500.",
        "To X takes 2–3 days; Y takes 3–5 days.",
        "Done!",
    ]


def test_split_sentences_ignores_blank_input() -> None:
    assert split_sentences("  \n ") == []


def test_chunk_file_tags_source_and_position(kb_dir: Path) -> None:
    chunks = chunk_file(kb_dir / "delivery.txt")
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert {c.source for c in chunks} == {"delivery.txt"}
    assert len(chunks) == 3


def test_build_index_covers_every_file_and_normalises(kb_dir: Path) -> None:
    index = build_index(kb_dir, FakeEmbedder())
    assert {c.source for c in index.chunks} == {"delivery.txt", "products.txt", "returns.txt"}
    assert index.vectors.shape == (len(index.chunks), 64)
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert set(index.hashes) == {"delivery.txt", "products.txt", "returns.txt"}


def test_save_and_load_round_trip(kb_dir: Path, index_dir: Path) -> None:
    index = build_index(kb_dir, FakeEmbedder())
    save_index(index, index_dir)
    loaded = load_index(index_dir)
    assert loaded is not None
    assert loaded.chunks == index.chunks
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.embed_model == "fake-embed"


def test_load_index_missing_returns_none(index_dir: Path) -> None:
    assert load_index(index_dir) is None


def test_load_or_build_reuses_saved_index(kb_dir: Path, index_dir: Path) -> None:
    embedder = FakeEmbedder()
    _, first = load_or_build(kb_dir, index_dir, embedder)
    _, second = load_or_build(kb_dir, index_dir, embedder)
    assert first.rebuilt and first.reasons == ["no saved index"]
    assert not second.rebuilt
    assert embedder.document_calls == 1


def test_changed_file_triggers_rebuild_with_new_chunk(kb_dir: Path, index_dir: Path) -> None:
    embedder = FakeEmbedder()
    load_or_build(kb_dir, index_dir, embedder)
    delivery = kb_dir / "delivery.txt"
    delivery.write_text(delivery.read_text().replace("PKR 200", "PKR 350"))

    index, report = load_or_build(kb_dir, index_dir, embedder)

    assert report.rebuilt and report.reasons == ["delivery.txt changed"]
    assert any("PKR 350" in c.text for c in index.chunks)
    assert not any("PKR 200" in c.text for c in index.chunks)
    assert embedder.document_calls == 2


def test_added_and_removed_files_are_reported(kb_dir: Path, index_dir: Path) -> None:
    index, _ = load_or_build(kb_dir, index_dir, FakeEmbedder())
    (kb_dir / "faq.txt").write_text("Support is open on weekdays.")
    (kb_dir / "returns.txt").unlink()
    assert stale_reasons(index, kb_dir, "fake-embed") == ["faq.txt added", "returns.txt removed"]


def test_model_change_triggers_rebuild(kb_dir: Path, index_dir: Path) -> None:
    load_or_build(kb_dir, index_dir, FakeEmbedder("model-a"))
    _, report = load_or_build(kb_dir, index_dir, FakeEmbedder("model-b"))
    assert report.rebuilt
    assert report.reasons == ["embedding model changed (model-a -> model-b)"]


def test_force_rebuild(kb_dir: Path, index_dir: Path) -> None:
    load_or_build(kb_dir, index_dir, FakeEmbedder())
    _, report = load_or_build(kb_dir, index_dir, FakeEmbedder(), force=True)
    assert report.rebuilt and report.reasons == ["forced"]


def test_file_hashes_change_with_content(kb_dir: Path) -> None:
    before = file_hashes(kb_dir)
    (kb_dir / "products.txt").write_text("The L1 desk lamp costs PKR 9,999.")
    after = file_hashes(kb_dir)
    assert before["products.txt"] != after["products.txt"]
    assert before["delivery.txt"] == after["delivery.txt"]
