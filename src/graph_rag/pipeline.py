"""The answer path, wired up once, for every front end that has one.

Two front ends ask the same question of the same code: the CLI's `query` and `chat`,
and the HTTP service. What they share is not the *logic* — that is `answer.py` and
`conversation.py`, and it was already shared — but the **wiring**: a Neo4j driver, the
two graph adapters, the embedder and the two model tiers behind the cache, all built in
one order and torn down together. Built twice, the two front ends drift: the service
answers with an uncached extraction tier, or against a second driver opened per
request, and the only symptom is that the demo is slower from the browser than from the
terminal.

So this module is the composition root for answering, and `cli.py` keeps only the part
that is genuinely its own — turning a failure into a message and an exit code rather
than into an HTTP status.

It constructs clients, which nothing below the pipeline boundary is allowed to do. That
is the point of it: the seam the README describes is the *pipeline function*, and this
is the layer immediately above that seam where the three injected dependencies are
finally made real.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from neo4j import Driver

from graph_rag.answer import DEFAULT_PARAMETERS, Answer, Parameters, answer_question
from graph_rag.cache import CachingChatModel, CachingEmbeddingModel, DiskCache, embedding_signature
from graph_rag.clarification import Clarification, Folder, clarify
from graph_rag.chat import make_chat_model, make_extraction_model
from graph_rag.config import Settings
from graph_rag.conversation import Conversation, Rewrite, Turn, rewrite_question
from graph_rag.embedding import make_embedding_model
from graph_rag.graph.driver import graph_driver
from graph_rag.graph.hierarchy import read_folder_index
from graph_rag.graph.retrieval import GraphChunkIndex
from graph_rag.graph.traversal import GraphEntityGraph
from graph_rag.retrieval import ChunkIndex, Embedder
from graph_rag.traversal import EntityGraph


def embedder_for(settings: Settings, cache: DiskCache) -> CachingEmbeddingModel:
    """The embedding model, behind the cache.

    One factory for every call site — the chunks of an ingest, the names resolution
    embeds, and the question both retrieval legs embed — so they share a cache and
    none of them can be built uncached by accident.
    """
    return CachingEmbeddingModel(
        make_embedding_model(settings.mistral),
        cache,
        signature=embedding_signature(settings.mistral),
    )


def chat_model_for(settings: Settings, cache: DiskCache) -> CachingChatModel:
    """The chat tier, cached.

    Cached like every other model call: at temperature 0 the same prompt is the same
    reply, so a demo that is re-run costs nothing and cannot wander mid-stage. Shared
    by the two calls a turn makes — rewriting the follow-up and generating the answer —
    which are the same tier asked two different things.
    """
    return CachingChatModel(
        make_chat_model(settings.mistral),
        cache,
        signature=settings.mistral.chat_model,
    )


def extraction_model_for(settings: Settings, cache: DiskCache) -> CachingChatModel:
    """The extraction tier, cached. One call per question, to read the names in it."""
    return CachingChatModel(
        make_extraction_model(settings.mistral),
        cache,
        signature=settings.mistral.extraction_model,
    )


@dataclass(frozen=True)
class Pipeline:
    """Everything an answer needs, already built — and the three ways to ask for one.

    Held as a value rather than reached for through a module global, because the whole
    project's seam rests on the dependencies being passed: a handler that could import
    a driver would eventually import one.
    """

    settings: Settings
    driver: Driver
    index: ChunkIndex
    graph: EntityGraph
    embedder: Embedder
    chat: CachingChatModel
    extraction: CachingChatModel
    #: Cosine similarity floor for matching a question entity by embedding. Resolved
    #: here from the per-call override or the environment, so neither front end has to
    #: remember which of the two it is holding.
    threshold: float
    #: The corpus's folder tree, read once when the pipeline opens. The clarifier needs
    #: it on every question and it changes only on ingest, so re-reading it per question
    #: would add a round trip to the one check that must stay free.
    folders: tuple[Folder, ...] = ()

    def clarify(self, question: str) -> Clarification | None:
        """The question to put back, when this one names a folder the corpus cannot
        resolve. None when it resolves — which is the common case, and costs no model
        call and no query."""
        return clarify(question, self.folders)

    def ask(self, question: str, parameters: Parameters = DEFAULT_PARAMETERS) -> Answer:
        """One standalone question, answered from both legs."""
        return answer_question(
            question=question,
            index=self.index,
            graph=self.graph,
            embedder=self.embedder,
            chat=self.chat,
            extraction=self.extraction,
            threshold=self.threshold,
            candidates=self.settings.traversal.candidates,
            parameters=parameters,
        )

    def rewrite(self, question: str, conversation: Conversation) -> Rewrite:
        """A follow-up resolved against the conversation, before retrieval runs."""
        return rewrite_question(question, conversation, self.chat)

    def turn(
        self,
        question: str,
        conversation: Conversation,
        parameters: Parameters = DEFAULT_PARAMETERS,
    ) -> Turn:
        """One turn of a conversation: resolve the question, then answer it.

        The order is the design of ticket 13 and is kept here rather than repeated in
        each front end — the rewrite happens *before* the pipeline, and the pipeline is
        handed a question it cannot tell apart from a first one.
        """
        rewrite = self.rewrite(question, conversation)
        return Turn(rewrite=rewrite, answer=self.ask(rewrite.standalone, parameters))


@contextmanager
def open_pipeline(
    settings: Settings, cache: DiskCache, threshold: float | None = None
) -> Iterator[Pipeline]:
    """The answer path, open for as long as it is asked.

    A context manager rather than a function because questions come in batches against
    one driver and one set of clients: a conversation asks several, and the service
    answers for as long as it is up. Building them per question would reconnect to
    Neo4j between a question and its follow-up, with an audience watching.

    `threshold` overrides `TRAVERSAL_MATCH_THRESHOLD` for everything asked through this
    pipeline; `None` means the environment's value.
    """
    with graph_driver(settings.neo4j) as driver:
        driver.verify_connectivity()
        yield Pipeline(
            settings=settings,
            driver=driver,
            index=GraphChunkIndex(driver, settings.neo4j.database),
            graph=GraphEntityGraph(driver, settings.neo4j.database),
            embedder=embedder_for(settings, cache),
            chat=chat_model_for(settings, cache),
            extraction=extraction_model_for(settings, cache),
            threshold=(
                settings.traversal.match_threshold if threshold is None else threshold
            ),
            folders=tuple(read_folder_index(driver, settings.neo4j.database)),
        )
