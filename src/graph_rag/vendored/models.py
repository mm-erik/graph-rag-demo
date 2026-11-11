"""VENDORED — not demo-authored code.

Origin: ``apps/ai/models/document.py`` and ``apps/ai/models/chunk.py`` in the sibling
``monorepo`` checkout (the production ingestion service).

Reduced to the parts the vendored parsing and chunking path actually uses: the document
record, ingestion-status and versioning machinery of the original belong to that system's
lifecycle, not to this demo. Deviations from the original are marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that chunk boundaries here match production, so
demo behaviour predicts real behaviour; edits silently decouple the two.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal


@dataclass
class DocumentMetadata:
    source: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    extra: dict = field(default_factory=dict)


@dataclass
class RawDocument:
    """Unprocessed document as received from an external source.

    LOCAL CHANGE: the origin's ``reference`` field (backend-owned object
    storage) is dropped — here the bytes always come from the local filesystem.
    """

    content: bytes
    filename: str
    mime_type: str
    metadata: DocumentMetadata | None = None


@dataclass
class BoundingBox:
    """Position of an element on a page.

    Coordinates are normalised to ``[0, 1]`` relative to the page size so they
    stay comparable across parsing backends.
    """

    page: int
    left: float
    top: float
    right: float
    bottom: float


Modality = Literal["text", "image", "table", "equation"]


_IMAGE_CATEGORIES = frozenset({"image", "figure"})


_TABLE_CATEGORIES = frozenset({"table"})


_EQUATION_CATEGORIES = frozenset({"formula", "equation"})


def modality_of(element_type: str | None) -> Modality:
    """Map a parser element ``type`` to a coarse modality.

    Handles both Unstructured (``Image``/``Table``/``Formula``) and MinerU
    (``image``/``table``/``equation``) vocabularies, case-insensitively.
    Anything else is treated as text.
    """
    value = (element_type or "").lower()
    if value in _IMAGE_CATEGORIES:
        return "image"
    if value in _TABLE_CATEGORIES:
        return "table"
    if value in _EQUATION_CATEGORIES:
        return "equation"
    return "text"


@dataclass
class Element:
    """A single document element in a backend-agnostic shape.

    One logical unit as emitted by the parser (paragraph, title, list item,
    table, image, formula). Field names follow Unstructured's vocabulary so
    :func:`normalize` mostly renames rather than invents. The parser's own
    label is kept verbatim in ``type`` and the coarse ``modality`` is derived
    from it.

    Which content fields apply depends on modality: a table fills
    ``text_as_html``, an image fills ``image_path``/``image_bytes``, a formula
    fills ``text_as_latex``. ``summary`` is *generated* (VLM/LLM description);
    every other field is *parsed*, so callers can always tell parsed content
    from generated.
    """

    element_id: str
    type: str | None = None
    """Parser element type, verbatim (``Title``, ``Image``, ``Table``, …).
    Drives :attr:`modality`."""
    text: str | None = None
    """Parsed text: paragraph/title/list body, or an image/table caption."""
    text_as_html: str | None = None
    """Original table HTML, preserved for downstream fidelity and rendering."""
    text_as_latex: str | None = None
    """LaTeX of a formula element, when available."""
    image_path: str | None = None
    """Reference to an image (relative path or stored URL), when available."""
    image_bytes: bytes | None = None
    """Decoded inline bytes of an image (or scanned-table crop). Unstructured
    delivers these base64-encoded; they are decoded to raw bytes here."""
    bbox: BoundingBox | None = None
    page_number: int | None = None
    parent_id: str | None = None
    """``element_id`` of the section/parent element, when the parser exposes
    a hierarchy. ``None`` when the backend has no hierarchy."""
    summary: str | None = None
    """Generated description/summary from an enrichment stage (not parsed).
    Used as the embedding source for image/table chunks."""

    @property
    def modality(self) -> Modality:
        """Coarse modality (``text``/``image``/``table``/``equation``)."""
        return modality_of(self.type)


@dataclass
class Document:
    """Structured document after extraction.

    Content is an ordered list of :class:`Element` objects, preserving reading
    order and the position of figures/tables relative to text.
    """

    id: str
    metadata: DocumentMetadata
    elements: list[Element] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Concatenated parsed text of all text and equation elements, in order."""
        parts: list[str] = []
        for element in self.elements:
            if element.modality == "equation":
                value = element.text or element.text_as_latex
                if value:
                    parts.append(value)
            elif element.modality == "text" and element.text:
                parts.append(element.text)
        return "\n\n".join(parts)

    @property
    def images(self) -> list[Element]:
        return [e for e in self.elements if e.modality == "image"]

    @property
    def tables(self) -> list[Element]:
        return [e for e in self.elements if e.modality == "table"]

    @property
    def equations(self) -> list[Element]:
        return [e for e in self.elements if e.modality == "equation"]


@dataclass
class Chunk:
    """A text segment produced by splitting a Document.

    LOCAL CHANGE: the origin's vector-store fields (``sparse_embedding``,
    ``score_source``) are dropped; this project has no sparse index and stores
    embeddings in Neo4j.
    """

    id: str
    document_id: str
    text: str
    index: int
    score: float | None = None
    embedding: list[float] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
