"""The two retrieval legs joined into one context, and an answer generated from it.

This is the first complete user-facing path: a question in, a cited German answer out.
Everything below it — both legs, the folder ancestry, the entity walk — already exists;
what this module adds is the join, the cap, and the generation.

Three decisions shape it.

**The legs are fused on rank, not merged by score.** The lexical leg scores in RRF
units and the entity leg in hop weights, and no arithmetic over the two numbers means
anything. The same `reciprocal_rank_fusion` that fuses vector against full text inside
the first leg fuses the two legs here, one level up, for the same reason.

**The cap is applied after the fusion, never before.** Capping each leg first would
decide the context budget before either leg knew what the other found, and on a
cross-client question that silently spends the whole budget on the leg that answers
fastest. Both legs return everything they ranked; the cap takes the best of the merged
order.

**The result is a structure, not a string.** Tickets 14 and 15 draw the subgraph behind
the answer, and that subgraph is the demo's entire burden of proof. The chunks with
their leg and rank, the matched entities, the walked edges and every chunk's folder
ancestry are therefore the product of this module as much as the prose is — the CLI
renders a view of it rather than being the only place it exists.

The module is pure. The index, the entity graph, the embedder and both model tiers are
injected; nothing here constructs a client.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from graph_rag.ancestry import Ancestor, ancestry_lines, branch
from graph_rag.retrieval import (
    DEFAULT_TOP_K,
    Chunk,
    ChunkIndex,
    Embedder,
    LegResult,
    Mode,
    reciprocal_rank_fusion,
    search_chunks,
)
from graph_rag.traversal import (
    DEFAULT_HOPS,
    EntityGraph,
    MatchedEntity,
    QuestionEntity,
    QuestionModel,
    ReachedEntity,
    TraversalResult,
    WalkedEdge,
    traverse_entities,
)

#: Chunks handed to the model. Roughly a dozen is what the ticket asks for, and the
#: reasoning behind "roughly": the corpus chunks at 2000 characters, so a dozen is a
#: context the chat tier reads comfortably and an operator can still read off a screen
#: when checking that an answer cited the chunk it should have. It is a call-time
#: parameter because it is expected to be turned up on stage — a question spanning two
#: clients needs room for both.
DEFAULT_MAX_CHUNKS = 12


class Source(StrEnum):
    """Which leg found a chunk.

    Named after the subcommands that run each leg alone — `just search`, `just traverse`
    — so that a chunk labelled `traversal` here is a chunk the operator can go and see
    the whole walk for, with the same question, in one other command.
    """

    SEARCH = "search"
    TRAVERSAL = "traversal"


class Legs(StrEnum):
    """Which legs run. Isolation survives into this layer because showing the same
    question answered by one leg, the other, and both is how the audience sees what the
    graph contributes — the ticket's requirement, not a debugging aid."""

    BOTH = "both"
    SEARCH = "search"
    TRAVERSAL = "traversal"

    @property
    def sources(self) -> tuple[Source, ...]:
        """The legs this setting runs, **in the order that breaks a tie**.

        As in `Mode.legs`, the order is load-bearing rather than incidental: a chunk
        ranked first by one leg and absent from the other scores exactly what the
        chunk ranked first by the other leg scores, and the formula has nothing further
        to say about them. The traversal leg goes first because its claim on a chunk is
        the stronger one — it is there because the question *named* an entity the chunk
        mentions, while a vector hit on this corpus is a 0.93 similarity against a field
        of 0.92s. On a tie, the context should open with the chunk whose connection to
        the question can be pointed at.
        """
        if self is Legs.BOTH:
            return (Source.TRAVERSAL, Source.SEARCH)
        return (Source(self.value),)

    def runs(self, source: Source) -> bool:
        return source in self.sources


@dataclass(frozen=True)
class Parameters:
    """The knobs, all of them adjustable at call time.

    Gathered into one object rather than passed as five arguments because ticket 14
    receives exactly this set as query parameters and ticket 15 renders what was used:
    a parameter added here reaches both without either having to be taught about it.

    The entity matching thresholds are deliberately absent — they are environment
    settings (see `TraversalSettings`), not stage knobs, and arrive as arguments.
    """

    legs: Legs = Legs.BOTH
    #: Which searches the lexical leg runs. `Legs.SEARCH` with `Mode.VECTOR` is the
    #: nearest thing this project has to the flat-RAG baseline that was cut from scope.
    mode: Mode = Mode.HYBRID
    #: Results per retrieval leg, before the merge.
    top_k: int = DEFAULT_TOP_K
    #: Relation edges walked outward from the matched entities.
    hops: int = DEFAULT_HOPS
    #: Chunks in the merged context handed to the model.
    max_chunks: int = DEFAULT_MAX_CHUNKS


#: The defaults as one value, so `answer_question` can name them without constructing
#: a `Parameters` in its own signature — and so ticket 14 has something to fall back to
#: for a request that sets only one of the five.
DEFAULT_PARAMETERS = Parameters()


@dataclass(frozen=True)
class ContextChunk:
    """One chunk of the merged context, carrying everything that put it there.

    The leg, the rank within that leg, and — when the entity leg reached it — which
    entities it was reached through. A merged context of bare chunks would answer the
    question and lose the only evidence that the graph contributed anything.
    """

    chunk: Chunk
    #: Reciprocal-rank-fusion score over the legs, best first.
    score: float
    #: 1-based rank per leg, holding only the legs that returned this chunk at all.
    ranks: Mapping[Source, int]
    #: Every reached entity this chunk names, nearest hop first. Empty unless the
    #: traversal leg returned it.
    via: tuple[ReachedEntity, ...] = ()

    def rank_in(self, source: Source) -> int | None:
        """Where `source` placed this chunk, or None if that leg never returned it."""
        return self.ranks.get(source)

    def found_by(self, source: Source) -> bool:
        return source in self.ranks

    @property
    def ancestry(self) -> list[Ancestor]:
        return self.chunk.ancestry

    @property
    def location(self) -> str:
        """`Kunde: Nordlicht Energie AG › Jahr: 2023 › Projekt: Projekt Atlas`.

        What the answer cites instead of a path — the whole reason ancestry is carried
        through both legs rather than reconstructed from `document_path`.
        """
        return " › ".join(ancestry_lines(self.chunk.ancestry)) or self.chunk.filename


def merge_context(
    *,
    search: LegResult | None,
    traversal: TraversalResult | None,
    legs: Legs = Legs.BOTH,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> list[ContextChunk]:
    """Both legs' chunks as one ranked, capped, deduplicated context.

    A chunk both legs found appears **once**, carrying both ranks. That is not only
    tidiness: a duplicate would spend two of a dozen context slots on one passage and,
    worse, read to the model as corroboration where there is only one document.

    Ranking is by reciprocal rank fusion over the legs' own orders, so a chunk both
    legs ranked highly outranks one leg's lone favourite — the same property that makes
    fusion worth doing inside the first leg.
    """
    results = {
        Source.SEARCH: [found.chunk for found in search.results] if search else [],
        Source.TRAVERSAL: (
            [found.chunk for found in traversal.results] if traversal else []
        ),
    }
    # Built in the tie-breaking order, which is what `reciprocal_rank_fusion` reads it
    # as. A leg that did not run contributes no list rather than an empty one.
    rankings = {
        source: [chunk.id for chunk in results[source]]
        for source in legs.sources
        if results[source]
    }
    chunks = {chunk.id: chunk for found in results.values() for chunk in found}
    via = (
        {found.chunk.id: found.via for found in traversal.results} if traversal else {}
    )
    return [
        ContextChunk(
            chunk=chunks[fused.key],
            score=fused.score,
            ranks=fused.ranks,
            via=via.get(fused.key, ()),
        )
        for fused in reciprocal_rank_fusion(rankings)
    ][: max(max_chunks, 0)]


class ChatModel(Protocol):
    """The chat tier, narrowed to the one call an answer makes. Prose, not JSON."""

    def complete(self, system: str, user: str) -> str: ...


# German, per the project convention, and specific about this corpus rather than
# generically careful.
#
# Rule 4 is the one that earns its place. Every report in the archive is built from the
# same boilerplate, so a context of a dozen chunks routinely holds two clients' versions
# of the same paragraph; a model that is not told to keep them apart will answer about
# "das Projekt Atlas" as though there were one of them. That is precisely the failure
# the whole project exists to disprove, and it would be committed in the last step.
SYSTEM_PROMPT = (
    "Du beantwortest Fragen zum Projektarchiv einer Unternehmensberatung.\n"
    "\n"
    "Regeln:\n"
    "1. Antworte ausschliesslich auf Grundlage der unten aufgeführten Quellen. "
    "Verwende kein Vorwissen und erfinde nichts.\n"
    "2. Belege jede Aussage mit der Nummer ihrer Quelle in eckigen Klammern, "
    "zum Beispiel [Quelle 3]. Mehrere Belege: [Quelle 1][Quelle 4].\n"
    "3. Sage zu jeder Aussage, aus welcher Ablage sie stammt — Kunde, Jahr und "
    "Projekt, so wie es bei der Quelle steht. Nenne niemals Dateipfade.\n"
    "4. Das Archiv enthält zu mehreren Kunden fast gleichlautende Berichte, "
    "teilweise mit gleichen Projektnamen. Führe Aussagen verschiedener Kunden "
    "niemals zusammen und schreibe eine Aussage niemals dem falschen Kunden zu.\n"
    "5. Beantworten die Quellen die Frage nicht oder nur teilweise, sage das "
    "ausdrücklich. Eine unvollständige Antwort mit klarer Lücke ist richtig, "
    "eine vollständig klingende erfundene ist falsch.\n"
    "6. Antworte auf Deutsch, in zusammenhängenden Sätzen, höchstens acht davon. "
    "Keine Überschrift, keine Aufzählung der Quellen am Ende."
)


def build_user_prompt(question: str, context: Sequence[ContextChunk]) -> str:
    """The question and the numbered sources it must be answered from.

    The sources carry their folder ancestry and nothing about *how* they were
    retrieved. Which leg found a chunk is provenance for the operator and for the
    subgraph the audience sees — telling the model would only invite it to narrate the
    retrieval mechanism in an answer that should be about the documents.

    The numbering is the citation scheme: it is 1-based and matches the order of
    `context`, so `[Quelle 3]` in the reply resolves to `context[2]` and the CLI can
    print what was actually cited.
    """
    lines: list[str] = ["Quellen:"]
    for position, found in enumerate(context, start=1):
        lines.append(f"--- Quelle {position} ---")
        lines.append(f"Ablage: {found.location}")
        where = f"Dokument: {found.chunk.filename}"
        if found.chunk.page is not None:
            where += f", Seite {found.chunk.page}"
        lines.append(where)
        lines.append(found.chunk.text)
        lines.append("")
    lines.append(f"Frage: {question}")
    return "\n".join(lines)


#: What is answered when retrieval returned nothing. A question the corpus cannot
#: answer must not reach the model at all: handed an empty source list and a rule
#: saying "answer only from the sources", the chat tier reliably produces a fluent
#: paragraph from its own knowledge of consulting reports — an answer about a client
#: that does not exist, in front of an audience, with no source to check it against.
NO_CONTEXT_ANSWER = (
    "Zu dieser Frage wurde im Archiv nichts gefunden — weder über die Suche noch "
    "über den Entitätsgraphen. Es wird daher nichts beantwortet."
)

#: `[Quelle 3]`, however the model spaces it.
_CITATION = re.compile(r"\[\s*Quelle\s*(\d+)\s*\]", re.IGNORECASE)


def cited_numbers(text: str) -> list[int]:
    """The source numbers an answer cites, in order of first appearance."""
    found: list[int] = []
    for match in _CITATION.finditer(text or ""):
        number = int(match.group(1))
        if number not in found:
            found.append(number)
    return found


@dataclass(frozen=True)
class Answer:
    """Everything the answer path did — the ticket's structured result.

    Both legs' full results are kept rather than only the merged context, because the
    case the demo makes is comparative: that this chunk was first for the traversal leg
    and nowhere at all for the other one. Ticket 14 serialises this; ticket 15 draws it.
    """

    question: str
    #: The generated prose, or `NO_CONTEXT_ANSWER` when nothing was retrieved.
    text: str
    #: The merged, capped context in the order the model was given it. Source number
    #: *n* in the answer's citations is `context[n - 1]`.
    context: list[ContextChunk]
    parameters: Parameters
    #: Each leg's own result, or None when that leg was not run. None rather than an
    #: empty result: "found nothing" and "was not asked" are different answers, and the
    #: UI has to be able to say which.
    search: LegResult | None = None
    traversal: TraversalResult | None = None

    @property
    def grounded(self) -> bool:
        """Whether an answer was generated from sources at all."""
        return bool(self.context)

    @property
    def entities(self) -> list[QuestionEntity]:
        """What the question was read as naming. Empty when the leg did not run."""
        return self.traversal.entities if self.traversal else []

    @property
    def matches(self) -> list[MatchedEntity]:
        """Where those names landed in the graph — the walk's starting points."""
        return self.traversal.matches if self.traversal else []

    @property
    def reached(self) -> list[ReachedEntity]:
        """Every entity the walk arrived at, matched entities at hop 0."""
        return self.traversal.reached if self.traversal else []

    @property
    def edges(self) -> list[WalkedEdge]:
        """Every relation edge the walk crossed, with the hop that crossed it."""
        return self.traversal.edges if self.traversal else []

    @property
    def branches(self) -> list[str]:
        """The distinct top-level folders — clients — the context came from.

        More than one is the project's whole claim, re-derived per question rather than
        measured once by hand.
        """
        return sorted(
            {
                name
                for name in (branch(found.ancestry) for found in self.context)
                if name
            }
        )

    @property
    def citations(self) -> list[int]:
        return cited_numbers(self.text)

    @property
    def uncited(self) -> list[int]:
        """Cited numbers that point at no source.

        A model that cites `[Quelle 9]` against seven sources has stopped quoting the
        context and started inventing, and the citation is what makes that visible —
        the one check on grounding this path can perform without a human reading every
        chunk.
        """
        return [
            number
            for number in self.citations
            if not 1 <= number <= len(self.context)
        ]


def generate_answer(
    question: str, context: Sequence[ContextChunk], model: ChatModel
) -> str:
    """The prose, from a context that has already been merged and capped."""
    if not context:
        return NO_CONTEXT_ANSWER
    return (model.complete(SYSTEM_PROMPT, build_user_prompt(question, context))).strip()


def answer_question(
    *,
    question: str,
    index: ChunkIndex,
    graph: EntityGraph,
    embedder: Embedder,
    chat: ChatModel,
    extraction: QuestionModel,
    threshold: float,
    candidates: int,
    parameters: Parameters = DEFAULT_PARAMETERS,
) -> Answer:
    """Answer a question from both retrieval legs, merged and cited.

    The legs run concurrently because they share no state and each is several round
    trips — one of them an embedding call and a model call — with an audience watching.
    Both are entirely synchronous underneath, so this is two threads rather than an
    async rewrite of two legs that have no other reason to be async.

    A leg the parameters exclude is not run at all, rather than run and discarded: the
    traversal leg costs a model call, and `--legs search` is also how the demo is given
    when the entity graph is the thing being compared against.
    """
    legs = parameters.legs

    def run_search() -> LegResult:
        return search_chunks(
            question=question,
            index=index,
            embedder=embedder,
            top_k=parameters.top_k,
            mode=parameters.mode,
        )

    def run_traversal() -> TraversalResult:
        return traverse_entities(
            question=question,
            graph=graph,
            embedder=embedder,
            model=extraction,
            hops=parameters.hops,
            top_k=parameters.top_k,
            threshold=threshold,
            candidates=candidates,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {
            source: pool.submit(run)
            for source, run in (
                (Source.SEARCH, run_search),
                (Source.TRAVERSAL, run_traversal),
            )
            if legs.runs(source)
        }
        # Resolved inside the pool's scope so an exception in one leg propagates here
        # rather than at the end of the `with`, where it would be raised after the
        # other leg had already been waited on for nothing.
        results = {source: future.result() for source, future in pending.items()}

    search = results.get(Source.SEARCH)
    traversal = results.get(Source.TRAVERSAL)
    context = merge_context(
        search=search,
        traversal=traversal,
        legs=legs,
        max_chunks=parameters.max_chunks,
    )
    return Answer(
        question=question,
        text=generate_answer(question, context, chat),
        context=context,
        parameters=parameters,
        search=search,
        traversal=traversal,
    )
