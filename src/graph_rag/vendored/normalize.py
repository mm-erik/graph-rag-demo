"""Normalisation of parsing-backend output into the shared Document model.

Parsing backends (Unstructured today, others later) emit their own layout
formats. The functions here translate a list of *elements* into the
backend-agnostic :class:`Document` shape so the rest of the ingestion pipeline
never sees a vendor-specific schema.

---

VENDORED — not demo-authored code.

Origin: ``apps/ai/providers/parsing/normalize.py`` in the sibling ``monorepo`` checkout (the production ingestion
service). Copied wholesale, with that system's settings/DI framework stripped; every
deviation from the original is marked ``LOCAL CHANGE``.

Do not improve this file. Its value is that behaviour here matches production, so the
demo predicts the real thing; edits silently decouple the two.
"""
import base64

from graph_rag.vendored.models import (
    BoundingBox,
    Document,
    DocumentMetadata,
    Element,
    modality_of,
)


def _clean(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip() or None
    return None


def _decode_image(value: object) -> bytes | None:
    """Unstructured returns inline images as base64 in ``image_base64``."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return base64.b64decode(value)
    except (ValueError, TypeError):
        return None


def _bbox(metadata: dict) -> BoundingBox | None:
    """Normalise Unstructured ``coordinates`` into a ``[0, 1]`` bounding box.

    Dividing by the ``coordinates`` object's own ``layout_width``/``layout_height``
    makes the result scale-invariant across element sources: ``hi_res`` always
    detects element regions on a rendered page image, so a scanned PDF/image
    (OCR'd for text) and a born-digital PDF (text from its layer) report
    coordinates in different underlying units, but each is paired with the
    ``layout_width``/``layout_height`` it was measured against, so the ``[0, 1]``
    fraction this produces is identical either way. The practical difference is
    precision, not shape: an OCR'd region is bounded by what the layout model
    detected, which can be looser than a born-digital PDF's exact text-layer box.
    """
    coords = metadata.get("coordinates")
    if not isinstance(coords, dict):
        return None
    points = coords.get("points")
    if not isinstance(points, (list, tuple)) or not points:
        return None
    try:
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
    except (TypeError, ValueError, IndexError):
        return None

    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    width = coords.get("layout_width")
    if isinstance(width, (int, float)) and width:
        left, right = left / width, right / width
    height = coords.get("layout_height")
    if isinstance(height, (int, float)) and height:
        top, bottom = top / height, bottom / height

    page = metadata.get("page_number", 1)
    return BoundingBox(
        page=int(page) if isinstance(page, (int, float)) else 1,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
    )


def _page(metadata: dict) -> int | None:
    page = metadata.get("page_number")
    return int(page) if isinstance(page, (int, float)) else None


def normalize_elements(
    elements: list[dict],
    *,
    document_id: str,
    metadata: DocumentMetadata,
) -> Document:
    """Convert a list of Unstructured elements into a normalised Document.

    Produces an *ordered* list of :class:`Element` objects, preserving reading
    order. Each element keeps the backend type verbatim on ``type``
    (``Image``/``Figure``, ``Table``, ``Formula``, ``Title``, ``NarrativeText``,
    …) and the coarse :attr:`Element.modality` is derived from it, so section-
    aware chunking and modality routing (VLM/LLM enrichment) stay possible.
    Image/table crops are decoded into ``image_bytes``; table HTML is kept on
    ``text_as_html``; formulas keep their ``text_as_latex``. Elements without a
    recognised shape degrade gracefully (skipped or treated as plain text) so
    new element types don't break parsing.
    """
    result: list[Element] = []

    for index, element in enumerate(elements):
        if not isinstance(element, dict):
            continue
        element_type = element.get("type")
        if not isinstance(element_type, str):
            element_type = None
        element_metadata = element.get("metadata")
        if not isinstance(element_metadata, dict):
            element_metadata = {}
        page = _page(element_metadata)
        parent_id = _clean(element_metadata.get("parent_id"))
        text = _clean(element.get("text"))
        modality = modality_of(element_type)

        if modality == "image":
            result.append(
                Element(
                    element_id=element.get("element_id") or f"{document_id}-img-{index}",
                    type=element_type,
                    text=text,
                    image_path=_clean(element_metadata.get("image_path"))
                    or _clean(element_metadata.get("image_url")),
                    image_bytes=_decode_image(element_metadata.get("image_base64")),
                    bbox=_bbox(element_metadata),
                    page_number=page,
                    parent_id=parent_id,
                )
            )
        elif modality == "table":
            html = element_metadata.get("text_as_html")
            result.append(
                Element(
                    element_id=element.get("element_id") or f"{document_id}-tbl-{index}",
                    type=element_type,
                    text=text,
                    text_as_html=html if isinstance(html, str) else None,
                    image_bytes=_decode_image(element_metadata.get("image_base64")),
                    bbox=_bbox(element_metadata),
                    page_number=page,
                    parent_id=parent_id,
                )
            )
        elif modality == "equation":
            result.append(
                Element(
                    element_id=element.get("element_id") or f"{document_id}-eqn-{index}",
                    type=element_type,
                    text=text,
                    text_as_latex=_clean(element_metadata.get("text_as_latex")),
                    bbox=_bbox(element_metadata),
                    page_number=page,
                    parent_id=parent_id,
                )
            )
        elif text:
            result.append(
                Element(
                    element_id=element.get("element_id") or f"{document_id}-txt-{index}",
                    type=element_type,
                    text=text,
                    bbox=_bbox(element_metadata),
                    page_number=page,
                    parent_id=parent_id,
                )
            )

    return Document(id=document_id, metadata=metadata, elements=result)
