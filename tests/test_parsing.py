"""Chunking is vendored from production precisely so demo behaviour predicts real
behaviour. These tests pin the properties whose breakage is silent: chunk identity
(duplicate chunks on re-ingest rather than an upsert), table/image isolation (a table
fused into prose stops being individually citable), and the tuned German parameters
themselves (a chunker built with the wrong caps still produces plausible chunks)."""

from pathlib import Path

from graph_rag.config import ChunkingSettings
from graph_rag.parsing import document_id_for, make_chunker
from graph_rag.vendored.chunking import TitleChunker
from graph_rag.vendored.models import Document, DocumentMetadata, Element

GERMAN_DEFAULTS = ChunkingSettings(
    strategy="by_title",
    max_characters=2000,
    new_after_n_chars=1600,
    combine_under_n_chars=500,
    overlap=250,
)


def _document(*elements: Element) -> Document:
    return Document(
        id="doc-1", metadata=DocumentMetadata(source="test.pdf"), elements=list(elements)
    )


def _prose(element_id: str, text: str, page: int = 1) -> Element:
    return Element(element_id=element_id, type="NarrativeText", text=text, page_number=page)


def test_the_configured_chunker_carries_the_tuned_german_defaults():
    chunker = make_chunker(GERMAN_DEFAULTS)
    assert isinstance(chunker, TitleChunker)
    assert (chunker._max, chunker._soft, chunker._combine, chunker._overlap) == (
        2000,
        1600,
        500,
        250,
    )


def test_chunk_ids_are_deterministic_from_document_identity_and_index():
    """Re-ingestion must replace a document's chunks, not accumulate duplicates."""
    document = _document(_prose("e1", "Erster Absatz."), _prose("e2", "Zweiter Absatz."))
    first = make_chunker(GERMAN_DEFAULTS).chunk(document)
    second = make_chunker(GERMAN_DEFAULTS).chunk(document)

    assert [c.id for c in first] == [c.id for c in second]
    assert [c.index for c in first] == list(range(len(first)))


def test_two_documents_do_not_share_chunk_ids():
    a = _document(_prose("e1", "Gleicher Text."))
    b = Document(
        id="doc-2", metadata=DocumentMetadata(source="other.pdf"), elements=a.elements
    )
    chunker = make_chunker(GERMAN_DEFAULTS)
    assert chunker.chunk(a)[0].id != chunker.chunk(b)[0].id


def test_identical_files_under_different_folders_are_different_documents():
    """Near-identical reports are the point of this corpus; identity is the path, so
    two byte-identical quarterly reports under two clients stay separate documents."""
    assert document_id_for(Path("a/2023/Projekt Atlas/report.pdf")) != document_id_for(
        Path("b/2023/Projekt Atlas/report.pdf")
    )
    assert document_id_for(Path("a/report.pdf")) == document_id_for(Path("a/report.pdf"))


def test_tables_are_isolated_into_their_own_chunks():
    document = _document(
        _prose("e1", "Vor der Tabelle."),
        Element(
            element_id="t1",
            type="Table",
            text="Teilprojekt Status",
            text_as_html="<table><tr><td>Teilprojekt</td><td>Status</td></tr></table>",
            page_number=1,
        ),
        _prose("e2", "Nach der Tabelle."),
    )
    chunks = make_chunker(GERMAN_DEFAULTS).chunk(document)

    kinds = [c.metadata["kind"] for c in chunks]
    assert "table" in kinds
    table_chunk = next(c for c in chunks if c.metadata["kind"] == "table")
    assert "Vor der Tabelle." not in table_chunk.text
    assert "Nach der Tabelle." not in table_chunk.text
    assert table_chunk.metadata["text_as_html"]


def test_images_are_isolated_into_their_own_chunks():
    document = _document(
        _prose("e1", "Absatz."),
        Element(element_id="i1", type="Image", text="Abbildung 1", page_number=2),
    )
    chunks = make_chunker(GERMAN_DEFAULTS).chunk(document)
    assert [c.metadata["kind"] for c in chunks] == ["text", "image"]


def test_no_chunk_exceeds_the_hard_character_cap():
    long_text = "Projektfortschritt. " * 400  # ~8000 characters
    chunks = make_chunker(GERMAN_DEFAULTS).chunk(_document(_prose("e1", long_text)))
    assert len(chunks) > 1
    assert all(len(c.text) <= GERMAN_DEFAULTS.max_characters for c in chunks)


def test_an_unknown_strategy_is_rejected():
    settings = ChunkingSettings(
        strategy="semantic", max_characters=2000, new_after_n_chars=1600,
        combine_under_n_chars=500, overlap=250,
    )
    try:
        make_chunker(settings)
    except ValueError as error:
        assert "semantic" in str(error)
    else:
        raise AssertionError("expected ValueError")
