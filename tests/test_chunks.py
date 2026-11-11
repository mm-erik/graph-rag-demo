"""Two silent failures guard this file.

The first is *misalignment*: embeddings come back from a batched API and are zipped onto
chunks by position. A batcher that reorders, drops or duplicates an input produces a
graph where every chunk carries a neighbour's vector — no error, no exception, just
retrieval that quietly answers from the wrong passage.

The second is a *too-large batch*. The embedding API rejects a request whose inputs
exceed a total token budget, so the batcher must respect a character budget as well as a
count; a batch that only honours the count fails the whole ingest on long chunks.
"""

import pytest

from graph_rag.chunks import ChunkAlignmentError, page_of, to_chunk_records
from graph_rag.embedding import batch_spans
from graph_rag.vendored.models import Chunk


def _texts(count: int, length: int = 10) -> list[str]:
    return [f"{index:0{length}d}" for index in range(count)]


def _batched(texts, max_count, max_characters):
    return [texts[start:stop] for start, stop in batch_spans(texts, max_count, max_characters)]


def test_batching_covers_every_text_exactly_once_and_in_order():
    texts = _texts(37)

    assert [text for batch in _batched(texts, 8, 10_000) for text in batch] == texts


def test_no_batch_exceeds_either_cap():
    texts = _texts(37, length=30)

    batches = _batched(texts, 8, 100)
    assert all(len(batch) <= 8 for batch in batches)
    assert all(sum(len(text) for text in batch) <= 100 for batch in batches)


def test_a_single_text_over_the_character_budget_is_sent_on_its_own():
    """Dropping it would silently lose a chunk; packing it with others would fail the
    request and take the rest of the batch down with it."""
    texts = ["short", "x" * 500, "short"]

    assert _batched(texts, 8, 100) == [["short"], ["x" * 500], ["short"]]


def test_no_empty_batches():
    assert all(_batched(_texts(5), 2, 10_000))
    assert _batched([], 8, 100) == []


def _chunk(index: int, text: str = "Text", pages: list[int] | None = None) -> Chunk:
    return Chunk(
        id=f"chunk-{index}",
        document_id="document",
        text=text,
        index=index,
        metadata={"pages": pages if pages is not None else [index + 1]},
    )


def test_each_chunk_keeps_its_own_embedding():
    chunks = [_chunk(0), _chunk(1), _chunk(2)]
    embeddings = [[0.0], [1.0], [2.0]]

    records = to_chunk_records(chunks, embeddings)

    assert [(record.id, record.index, record.embedding) for record in records] == [
        ("chunk-0", 0, [0.0]),
        ("chunk-1", 1, [1.0]),
        ("chunk-2", 2, [2.0]),
    ]


def test_a_count_mismatch_is_refused_rather_than_zipped_short():
    """`zip` would silently truncate to the shorter list, writing a document's first
    chunks and dropping the rest with nothing to show for it."""
    with pytest.raises(ChunkAlignmentError):
        to_chunk_records([_chunk(0), _chunk(1)], [[0.0]])


def test_a_chunk_spanning_a_page_break_records_the_page_it_starts_on():
    assert page_of(_chunk(0, pages=[4, 5])) == 4


def test_a_chunk_the_parser_gave_no_page_records_none():
    assert page_of(_chunk(0, pages=[])) is None
