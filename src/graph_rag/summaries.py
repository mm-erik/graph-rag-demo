"""A generated summary per document, written at ingestion time.

Folder-level and document-level questions ("worum geht es in diesem Bericht?", "was
haben wir 2023 für diesen Kunden gemacht?") cannot be answered from chunks: answering
them that way means pulling every chunk of every document in a branch, which exhausts
the context budget and retrieves nothing useful. A summary per document answers them
from one property each.

The corpus is engineered out of near-identical boilerplate, which sets the bar for this
module: a summary is only worth storing if it distinguishes its document from the
sibling report next to it. Two decisions follow from that and are load-bearing —
the excerpt is spread across the whole document rather than taken from its front, and
the prompt is given the document's place in the hierarchy.

The model is injected as a protocol; nothing here constructs one.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from graph_rag.ancestry import Ancestor, ancestry_lines


@dataclass(frozen=True)
class DocumentState:
    """What the graph already knows about a document's summary.

    Read without its chunk text: deciding whether a document needs summarising must not
    cost a corpus-wide read of everything a cached pass is about to skip.
    """

    path: str
    #: SHA-256 of the file's bytes, written by the structural pass.
    content_hash: str
    summary: str | None
    #: The content hash the stored summary was written from. Differs from
    #: `content_hash` exactly when the file changed since.
    summary_hash: str | None


def needs_summary(state: DocumentState, refresh: bool) -> bool:
    """Whether this document must be summarised on this run.

    The cache key is the document's content hash, which is already on the node — so
    re-ingesting an unchanged corpus costs no chat calls at all. `refresh` is the
    operator's `--no-cache`: a changed *prompt* is invisible to any content hash, and
    without an override the corpus would keep the summaries of the previous prompt
    forever.
    """
    if refresh:
        return True
    if not (state.summary or "").strip():
        return True
    return state.summary_hash != state.content_hash


def select_excerpt(
    chunks: Sequence[str], max_chunks: int, max_characters: int
) -> list[str]:
    """Chunks spread evenly across the document, in reading order.

    *Spread*, not the first few: every report in this corpus opens with the same cover
    page and the same preamble, so an excerpt taken from the front describes the
    template rather than the document. The first and last chunks are always included —
    a report's subject is stated at the top and its conclusions at the bottom.
    """
    texts = [text.strip() for text in chunks if text.strip()]
    if not texts or max_chunks < 1 or max_characters < 1:
        return []
    picked = _spread(texts, max_chunks)
    if sum(len(text) for text in picked) <= max_characters:
        return picked
    # Shared out rather than spent front to back: a long opening section would
    # otherwise swallow the budget and leave the excerpt describing the cover page
    # again, which is the one thing this function exists to avoid.
    share = max(1, max_characters // len(picked))
    return [_truncate(text, share) for text in picked]


def _spread(texts: list[str], count: int) -> list[str]:
    """`count` texts drawn evenly from `texts`, first and last included."""
    if len(texts) <= count:
        return texts
    if count == 1:
        return texts[:1]
    last = len(texts) - 1
    positions = sorted({round(step * last / (count - 1)) for step in range(count)})
    return [texts[position] for position in positions]


#: Marks a section the excerpt cut short, so the model is not told a truncated sentence
#: is where the document ends.
ELLIPSIS = " […]"


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - len(ELLIPSIS))].rstrip() + ELLIPSIS


#: Preambles the chat model reaches for despite being told not to. Stripped rather than
#: re-prompted: one retry per document doubles the cost of the pass for a defect that
#: is purely cosmetic.
_PREFIXES = ("zusammenfassung:", "summary:", "kurzfassung:")


def clean_summary(reply: str, max_characters: int) -> str:
    """A model reply as the property to store, or `""` when it is unusable.

    Total, like `parse_inference`: a reply that cannot be cleaned into a summary yields
    no summary rather than an exception, because one odd answer must not abort a pass
    over a hundred documents. An empty result is left unwritten — a `summary` property
    holding an apology reads, at query time, exactly like a real one.
    """
    text = (reply or "").strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.startswith("```")]
        text = "\n".join(lines).strip()
    text = text.strip("\"'").strip()
    for prefix in _PREFIXES:
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    text = " ".join(text.split())
    if not text:
        return ""
    return _cap(text, max_characters)


def _cap(text: str, limit: int) -> str:
    """Trim to the last whole sentence that fits, so a capped summary still reads as
    one rather than stopping mid-clause."""
    if limit < 1 or len(text) <= limit:
        return text
    head = text[:limit]
    stop = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    if stop > 0:
        return head[: stop + 1]
    return head.rstrip() + ELLIPSIS


@dataclass(frozen=True)
class DocumentSource:
    """Everything one summary is written from."""

    path: str
    filename: str
    page_count: int
    #: Outermost first, excluding the corpus root itself.
    ancestry: list[Ancestor]
    #: The document's chunks in reading order.
    chunks: list[str]


# German, per the project convention: the corpus is German and these summaries are read
# back to the audience in German answers.
#
# The instruction to avoid boilerplate is the point of the prompt rather than a
# nicety. Asked to "summarise this report", a model handed a quarterly status report
# produces the summary of a quarterly status report — true of all hundred of them, and
# therefore useless for telling them apart, which is the only job these summaries have.
SYSTEM_PROMPT = (
    "Du fasst Dokumente aus dem Projektarchiv einer Unternehmensberatung zusammen. "
    "Schreibe drei bis fünf Sätze Fliesstext auf Deutsch. "
    "Nenne konkret, worum es in genau diesem Dokument geht: Auftraggeber, Projekt, "
    "Zeitraum, Thema, die wichtigsten Ergebnisse, Entscheidungen, Zahlen und Namen. "
    "Das Archiv enthält viele fast gleichlautende Berichte. Schreibe deshalb nur, was "
    "dieses Dokument von einem Bericht zu einem anderen Projekt oder einem anderen "
    "Kunden unterscheidet, und vermeide allgemeine Formulierungen, die in jedem Bericht "
    "stehen könnten. "
    "Antworte ausschliesslich mit der Zusammenfassung — ohne Einleitung, ohne "
    "Überschrift, ohne Aufzählung."
)


def build_user_prompt(source: DocumentSource, excerpt: Sequence[str]) -> str:
    lines = ["Ablage:", *(f"- {line}" for line in ancestry_lines(source.ancestry))]
    lines.append(f"Dateiname: {source.filename}")
    if source.page_count:
        lines.append(f"Seiten: {source.page_count}")
    lines.append("")
    lines.append(
        "Auszüge aus dem Dokument, in Lesereihenfolge über das ganze Dokument verteilt:"
    )
    for position, text in enumerate(excerpt, start=1):
        lines.append(f"--- Auszug {position} von {len(excerpt)} ---")
        lines.append(text)
    return "\n".join(lines)


class ChatModel(Protocol):
    """The chat-tier model, in prose mode. One call per document, not per chunk."""

    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class DocumentSummary:
    path: str
    summary: str
    #: True when the stored summary already described this version of the file and no
    #: call was made.
    cached: bool
    #: Why this document has no summary, when it has none. Reported, not raised: a rate
    #: limit on document ninety must not discard the eighty-nine summaries already
    #: paid for.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class SummaryReport:
    documents: list[DocumentSummary]

    @property
    def written(self) -> int:
        return len([d for d in self.documents if d.ok and not d.cached])

    @property
    def cached(self) -> int:
        return len([d for d in self.documents if d.ok and d.cached])

    @property
    def failures(self) -> list[DocumentSummary]:
        return [document for document in self.documents if not document.ok]


def summarise_document(
    source: DocumentSource,
    model: ChatModel,
    *,
    max_chunks: int,
    max_characters: int,
    max_summary_characters: int,
) -> str:
    """One document's summary, or `""` when the model gave nothing usable."""
    excerpt = select_excerpt(source.chunks, max_chunks, max_characters)
    if not excerpt:
        return ""
    reply = model.complete(SYSTEM_PROMPT, build_user_prompt(source, excerpt))
    return clean_summary(reply, max_summary_characters)


class EmptyDocument(RuntimeError):
    """The document has no chunks to summarise — it failed to parse, or parsed to
    nothing. Reported against that document rather than raised out of the pass."""


class UnusableSummary(RuntimeError):
    """The model answered with nothing that could be stored as a summary.

    Distinct from `EmptyDocument` because the remedies differ and neither is visible
    from the other: one is a parsing problem, the other a model one.
    """


class SummaryStore(Protocol):
    """Where a document's text comes from and where its summary goes.

    A protocol rather than a driver, for the same reason the model is one: this module
    holds the pass's control flow — what to skip, what to call, what to do when one
    document fails — and that logic is the part worth keeping independent of Neo4j.
    `graph.summaries.GraphSummaryStore` is the implementation; nothing here constructs
    it.
    """

    def read_source(self, path: str) -> DocumentSource: ...

    def write(self, path: str, summary: str, content_hash: str) -> None: ...


async def summarise_documents(
    states: Sequence[DocumentState],
    *,
    model: ChatModel,
    store: SummaryStore,
    refresh: bool,
    concurrency: int,
    max_chunks: int,
    max_characters: int,
    max_summary_characters: int,
    on_document: Callable[[DocumentSummary], None] | None = None,
) -> SummaryReport:
    """Summarise every document that needs it, a bounded number at a time.

    One chat call per document and none at all for a document whose bytes have not
    changed since its summary was written. Results come back in the order `states` were
    given, so two runs over an unchanged corpus report the same thing twice.
    """
    limit = asyncio.Semaphore(concurrency)

    async def one(state: DocumentState) -> DocumentSummary:
        async with limit:
            if not needs_summary(state, refresh):
                result = DocumentSummary(
                    path=state.path, summary=state.summary or "", cached=True
                )
            else:
                try:
                    result = await asyncio.to_thread(_summarise, state)
                # Deliberately broad, as in `ingest_contents`: one document that
                # cannot be summarised is a missing property, not a failed ingest.
                except Exception as error:  # noqa: BLE001
                    result = DocumentSummary(
                        path=state.path, summary="", cached=False, error=str(error)
                    )
            if on_document is not None:
                on_document(result)
            return result

    def _summarise(state: DocumentState) -> DocumentSummary:
        """The blocking half: a graph read, a chat call and a graph write.

        Run in a thread because all three clients are synchronous — on the event loop,
        one document's chat call would stall every other document's.
        """
        source = store.read_source(state.path)
        if not source.chunks:
            raise EmptyDocument(
                "the document has no chunks — it failed to parse, or parsed to nothing"
            )
        summary = summarise_document(
            source,
            model,
            max_chunks=max_chunks,
            max_characters=max_characters,
            max_summary_characters=max_summary_characters,
        )
        if not summary:
            raise UnusableSummary("the model returned nothing that reads as a summary")
        store.write(state.path, summary, state.content_hash)
        return DocumentSummary(path=state.path, summary=summary, cached=False)

    documents = await asyncio.gather(*(one(state) for state in states))
    return SummaryReport(documents=list(documents))
