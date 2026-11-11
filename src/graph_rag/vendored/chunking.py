"""Chunking — split a Document's ordered elements into retrieval-sized Chunks.

Chunking here is *element-aware*, mirroring Unstructured: chunks are built by
combining whole :class:`~models.document.Element` units in reading order (not
by splitting raw text on ``"\\n\\n"``). Tables and images are isolated into their
own chunks so each stays a citable unit; an individual element that exceeds the
hard size cap is split — prose on word boundaries, tables on row boundaries.

Behaviour and defaults track Unstructured's chunking parameters so swapping
between local chunking and the vendor's own stays predictable:

- ``max_characters`` — hard cap; no chunk exceeds it.
- ``new_after_n_chars`` — soft cap; a chunk past this is not extended.
- ``combine_under_n_chars`` — floor for ``by_title`` only; sequential small
  sections are combined (across titles) up to the window. ``0`` disables it.
- ``overlap`` — characters carried over when splitting an oversized element or
  table row.

Two strategies share the engine and differ only in where a chunk breaks:

- ``basic`` (:class:`BasicChunker`) — packs consecutive blocks up to the cap,
  ignoring structure. Use when there's no reliable heading structure.
- ``by_title`` (:class:`TitleChunker`) — like ``basic`` but also breaks at every
  ``Title`` so chunks align to sections, then merges tiny sections via
  ``combine_under_n_chars``. Use when headings carry meaning.

---

VENDORED — not demo-authored code.

Origin: ``apps/ai/ingestion/chunking.py`` in the sibling ``monorepo`` checkout (the production ingestion
service). Copied wholesale, with that system's settings/DI framework stripped; every
deviation from the original is marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that behaviour here matches production, so the
demo predicts the real thing; edits silently decouple the two.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from html.parser import HTMLParser
from uuid import NAMESPACE_URL, uuid5

from graph_rag.vendored.models import Chunk, Document, Element

_SEP = "\n\n"
_SEP_LEN = len(_SEP)
_ROW_SEP = "\n"
_CELL_SEP = " | "
_SPLIT_SEPARATORS = ("\n", " ")


def _chunk_id(document_id: str, index: int) -> str:
    """Deterministic, Qdrant-valid point ID for a chunk.

    Derived via ``uuid5`` from ``(document_id, index)`` so a reprocessing
    run's :class:`~ingestion.indexing_stage.IndexingStage` upsert replaces the
    prior chunks instead of accumulating duplicates.
    """
    return str(uuid5(NAMESPACE_URL, f"urn:chunk:{document_id}:{index}"))


class _TableHTMLParser(HTMLParser):
    """Extract rows of cell text from an HTML ``<table>`` using the stdlib."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: object) -> None:
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th"):
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def parse_table_html(html: str) -> list[list[str]]:
    """Parse an HTML table body into a list of rows of cell text."""
    if not html:
        return []
    parser = _TableHTMLParser()
    parser.feed(html)
    return parser.rows


class Chunker(ABC):
    """Split a :class:`Document` into a list of :class:`Chunk` objects."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks produced from ``document``."""
        raise NotImplementedError


def _text_of(element: Element) -> str | None:
    """Searchable text for a prose/equation element (parsed text + summary).

    For equations, ``text_as_latex`` is preferred as the faithful
    representation; ``text`` is the fallback.
    """
    if element.modality == "equation":
        base = element.text_as_latex or element.text
    else:
        base = element.text
    parts = [p for p in (base, element.summary) if p]
    return "\n".join(parts) or None


def _image_text(element: Element) -> str:
    """Render an isolated image element to chunk text (caption + description)."""
    parts = [element.text, element.summary]
    return "\n".join(p for p in parts if p) or (element.type or "image")


def _table_lines(element: Element) -> list[str]:
    """Render searchable lines for a table (structured cells + summary).

    Prefers cells parsed from ``text_as_html`` (row-linearised as
    ``cell | cell``), falling back to flattened ``text`` only when there is
    no HTML. Raw ``text_as_html`` is kept on chunk metadata for display and
    citation, not embedded here.
    """
    lines: list[str] = []
    rows = parse_table_html(element.text_as_html or "")
    if rows:
        lines.extend(_CELL_SEP.join(row) for row in rows)
    elif element.text:
        lines.append(element.text)
    if element.summary:
        lines.append(element.summary)
    return lines


def _rightmost_separator(text: str, start: int, end: int) -> tuple[int, int]:
    """Rightmost split point in ``text[start:end]`` and its separator length.

    Returns ``(-1, 0)`` when the window holds no preferred separator.
    """
    for sep in _SPLIT_SEPARATORS:
        idx = text.rfind(sep, start + 1, end)
        if idx != -1:
            return idx, len(sep)
    return -1, 0


def _split_oversized(text: str, size: int, overlap: int) -> Iterator[str]:
    """Split an oversized string into ``<= size`` pieces at the rightmost
    separator, falling back to a hard cut when no separator is present.
    ``overlap`` characters are carried into the next window.
    """
    if size <= 0:
        yield text
        return
    n = len(text)
    start = 0
    while start < n:
        end = start + size
        if end >= n:
            piece = text[start:]
            if piece:
                yield piece
            break
        cut, sep_len = _rightmost_separator(text, start, end)
        if cut == -1:
            cut, sep_len = end, 0
        piece = text[start:cut]
        if piece:
            yield piece
        advance = cut + sep_len
        if 0 < overlap < size:
            advance = max(advance - overlap, start + 1)
        start = advance


def _split_table(element: Element, size: int, overlap: int) -> Iterator[str]:
    """Split a table into ``<= size`` chunks at row boundaries; a row larger
    than ``size`` is character-split via :func:`_split_oversized`.
    """
    lines = _table_lines(element)
    if not lines:
        yield "table"
        return
    buffer: list[str] = []

    def buffered_len() -> int:
        return sum(len(x) for x in buffer) + max(len(buffer) - 1, 0)

    for line in lines:
        if size > 0 and len(line) > size:
            if buffer:
                yield _ROW_SEP.join(buffer)
                buffer = []
            yield from _split_oversized(line, size, overlap)
            continue
        projected = buffered_len() + (1 if buffer else 0) + len(line)
        if buffer and size > 0 and projected > size:
            yield _ROW_SEP.join(buffer)
            buffer = []
        buffer.append(line)
    if buffer:
        yield _ROW_SEP.join(buffer)


def _box(element: Element) -> dict | None:
    """One element's position as ``{"page": int, "bbox": [l, t, r, b]}``.

    ``None`` when the parser gave no coordinates. Each box carries its own
    page rather than relying on the chunk's broader ``pages`` list, so a
    chunk whose elements span several pages highlights correctly on each.
    """
    bbox = element.bbox
    if bbox is None:
        return None
    return {"page": bbox.page, "bbox": [bbox.left, bbox.top, bbox.right, bbox.bottom]}


def _is_title(element: Element) -> bool:
    return (element.type or "").lower() == "title"


def _make_chunk(
    document_id: str,
    index: int,
    text: str,
    *,
    kind: str,
    block_ids: list[str],
    pages: list[int | None],
    types: list[str | None],
    parents: list[str | None] | None = None,
    section_title: str | None = None,
    boxes: list[dict | None] | None = None,
) -> Chunk:
    metadata: dict = {
        "kind": kind,
        "block_ids": list(block_ids),
        "pages": sorted({p for p in pages if p is not None}),
        "types": [t for t in types if t is not None],
        # One box per source element that had coordinates, for highlighting
        # the cited passage (see :func:`retrieval.citations._highlights`).
        # Empty when the parser gave none, same as an old pre-change chunk.
        "boxes": [box for box in boxes if box is not None] if boxes else [],
    }
    if parents:
        # Section/parent element of the chunk's first source element, enabling
        # section-scoped retrieval expansion.
        parent = next((p for p in parents if p is not None), None)
        if parent is not None:
            metadata["parent_id"] = parent
    if section_title:
        metadata["section_title"] = section_title
    return Chunk(
        id=_chunk_id(document_id, index),
        document_id=document_id,
        text=text,
        index=index,
        metadata=metadata,
    )


def _combine_small_sections(
    chunks: list[Chunk], combine_under: int, max_characters: int
) -> list[Chunk]:
    """Merge sequential small *text* sections up to ``max_characters``.

    Never merges isolated image/table chunks.
    """
    merged: list[Chunk] = []
    for chunk in chunks:
        if merged:
            prev = merged[-1]
            if (
                prev.metadata.get("kind") == "text"
                and chunk.metadata.get("kind") == "text"
                and len(prev.text) < combine_under
                and len(prev.text) + _SEP_LEN + len(chunk.text) <= max_characters
            ):
                prev.text = f"{prev.text}{_SEP}{chunk.text}"
                prev.metadata["block_ids"] += chunk.metadata.get("block_ids", [])
                prev.metadata["pages"] = sorted(
                    set(prev.metadata.get("pages", [])) | set(chunk.metadata.get("pages", []))
                )
                prev.metadata["types"] += chunk.metadata.get("types", [])
                prev.metadata["boxes"] += chunk.metadata.get("boxes", [])
                continue
        merged.append(chunk)

    # The id is derived from the position, so renumbering must recompute it:
    # re-indexing relies on ``chunk.id == _chunk_id(document_id, chunk.index)``
    # to overwrite the previous version's chunks and trim the tail.
    for position, chunk in enumerate(merged):
        chunk.index = position
        chunk.id = _chunk_id(chunk.document_id, position)
    return merged


class _ElementChunker(Chunker):
    """Shared element-accumulation engine for the concrete chunkers."""

    def __init__(
        self,
        *,
        max_characters: int,
        new_after_n_chars: int,
        overlap: int = 0,
        combine_under_n_chars: int = 0,
        split_on_title: bool = False,
    ) -> None:
        self._max = max_characters
        self._soft = new_after_n_chars
        self._overlap = overlap
        self._combine = combine_under_n_chars
        self._split_on_title = split_on_title

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        texts: list[str] = []
        ids: list[str] = []
        pages: list[int | None] = []
        types: list[str | None] = []
        parents: list[str | None] = []
        boxes: list[dict | None] = []
        section_title: str | None = None

        def buffered_len() -> int:
            return sum(len(t) for t in texts) + _SEP_LEN * max(len(texts) - 1, 0)

        def flush() -> None:
            nonlocal texts, ids, pages, types, parents, boxes
            if not texts:
                return
            chunks.append(
                _make_chunk(
                    document.id,
                    len(chunks),
                    _SEP.join(texts),
                    kind="text",
                    block_ids=ids,
                    pages=pages,
                    types=types,
                    parents=parents,
                    section_title=section_title,
                    boxes=boxes,
                )
            )
            texts, ids, pages, types, parents, boxes = [], [], [], [], [], []

        for index, element in enumerate(document.elements):
            if element.modality == "image":
                flush()
                box = _box(element)
                image_chunk = _make_chunk(
                    document.id,
                    len(chunks),
                    _image_text(element),
                    kind="image",
                    block_ids=[element.element_id],
                    pages=[element.page_number],
                    types=[element.type],
                    parents=[element.parent_id],
                    boxes=[box],
                )
                if element.image_path:
                    image_chunk.metadata["image_path"] = element.image_path
                if element.summary:
                    # Generated description kept as a distinct field so the
                    # embedding stage can embed it instead of the caption/text.
                    image_chunk.metadata["summary"] = element.summary
                if box is not None:
                    image_chunk.metadata["bbox"] = box["bbox"]
                chunks.append(image_chunk)
                continue

            if element.modality == "table":
                flush()
                box = _box(element)
                pieces = list(_split_table(element, self._max, self._overlap))
                for part, piece in enumerate(pieces):
                    table_chunk = _make_chunk(
                        document.id,
                        len(chunks),
                        piece,
                        kind="table",
                        block_ids=[element.element_id],
                        pages=[element.page_number],
                        types=[element.type],
                        parents=[element.parent_id],
                        boxes=[box],
                    )
                    if box is not None:
                        table_chunk.metadata["bbox"] = box["bbox"]
                    if element.summary:
                        # Generated description kept as a distinct field so the
                        # embedding stage can embed it instead of the table rows.
                        table_chunk.metadata["summary"] = element.summary
                    if element.text_as_html:
                        # Keep the original table markup as a dual representation:
                        # the embedded text is row-linearised for search, while
                        # this preserves full fidelity for display/citation and
                        # for the LLM to read at answer time.
                        table_chunk.metadata["text_as_html"] = element.text_as_html
                    if len(pieces) > 1:
                        # Same table split for size; ``block_ids`` groups the
                        # pieces and ``part``/``parts`` orders them so the whole
                        # table can be reassembled downstream.
                        table_chunk.metadata["part"] = part
                        table_chunk.metadata["parts"] = len(pieces)
                    chunks.append(table_chunk)
                continue

            text = _text_of(element)
            if text is None:
                continue

            if self._split_on_title and _is_title(element):
                flush()
                section_title = text

            element_id = element.element_id or f"{document.id}-blk-{index}"

            if len(text) > self._max:
                flush()
                pieces = list(_split_oversized(text, self._max, self._overlap))
                for part, piece in enumerate(pieces):
                    text_chunk = _make_chunk(
                        document.id,
                        len(chunks),
                        piece,
                        kind="text",
                        block_ids=[element_id],
                        pages=[element.page_number],
                        types=[element.type],
                        parents=[element.parent_id],
                        section_title=section_title,
                        boxes=[_box(element)],
                    )
                    if len(pieces) > 1:
                        # One oversized element split for size; ``block_ids``
                        # groups the pieces and ``part``/``parts`` order them.
                        text_chunk.metadata["part"] = part
                        text_chunk.metadata["parts"] = len(pieces)
                    chunks.append(text_chunk)
                continue

            projected = buffered_len() + (_SEP_LEN if texts else 0) + len(text)
            if texts and (projected > self._max or buffered_len() >= self._soft):
                flush()

            texts.append(text)
            ids.append(element_id)
            pages.append(element.page_number)
            types.append(element.type)
            parents.append(element.parent_id)
            boxes.append(_box(element))

        flush()

        if self._split_on_title and self._combine > 0:
            chunks = _combine_small_sections(chunks, self._combine, self._max)
        return chunks


class BasicChunker(_ElementChunker):
    """Greedily fill each chunk to ``max_characters`` (Unstructured ``basic``).

    Tables/images are isolated; oversized blocks are text-split. No section
    floor.
    """

    def __init__(
        self,
        *,
        max_characters: int = 500,
        new_after_n_chars: int | None = None,
        overlap: int = 0,
    ) -> None:
        super().__init__(
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars or max_characters,
            overlap=overlap,
            combine_under_n_chars=0,
            split_on_title=False,
        )


class TitleChunker(_ElementChunker):
    """Start a new chunk at each ``Title`` (Unstructured ``by_title``).

    Via ``combine_under_n_chars``, combines sequential small sections up to
    the window (mirrors ``combine_text_under_n_chars``).
    """

    def __init__(
        self,
        *,
        max_characters: int = 500,
        new_after_n_chars: int | None = None,
        overlap: int = 0,
        combine_under_n_chars: int | None = None,
    ) -> None:
        super().__init__(
            max_characters=max_characters,
            new_after_n_chars=new_after_n_chars or max_characters,
            overlap=overlap,
            combine_under_n_chars=(
                max_characters if combine_under_n_chars is None else combine_under_n_chars
            ),
            split_on_title=True,
        )

