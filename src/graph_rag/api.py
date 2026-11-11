"""The HTTP service: the same pipeline the CLI runs, with a drawable answer.

A thin layer, deliberately. Retrieval and generation are `pipeline.Pipeline`, which is
what `just query` and `just chat` run too — there is no second code path here, and the
only thing this module adds is the shape of the payload: an answer, the sources behind
it, and the subgraph that produced it with its cross-branch edges already flagged.

Three things are worth knowing before changing it.

**The pipeline is opened once, for the life of the process.** Not per request: a
request that opened a Neo4j driver would pay a connection handshake per question with
an audience watching, and the cache would be rebuilt each time. The lifespan owns it.

**Every tuning parameter is per request.** Hop depth in particular is turned live on
stage from the UI of ticket 15, so it arrives in the body rather than from the
environment. What was actually used comes back in the response — a demo in which the
operator cannot see which knob produced which picture is a demo that cannot be
debugged on stage.

**Conversation history is accepted, not held.** The client sends the turns it has, the
service resolves the follow-up against them and returns the standalone question. No
session state, so two browser tabs cannot interfere and a reload loses nothing the
client still holds.

No authentication, no deployment concerns, no production error handling — all out of
scope per the ticket. The one error path that is handled is a graph or model failure,
which becomes a 503 with the reason in it rather than a traceback in the console.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager, contextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from neo4j.exceptions import DriverError, Neo4jError
from pydantic import BaseModel, Field

from graph_rag.ancestry import Ancestor, ancestry_lines
from graph_rag.answer import (
    Answer,
    ContextChunk,
    Legs,
    Parameters,
)
from graph_rag.cache import DiskCache
from graph_rag.config import AnswerSettings, Settings
from graph_rag.conversation import Conversation, Exchange, Rewrite
from graph_rag.embedding import EmbeddingError
from graph_rag.graph.entities import count_bridges, entity_coverage
from graph_rag.graph.health import graph_status
from graph_rag.graph.levels import read_levels
from graph_rag.graph.summaries import summary_coverage
from graph_rag.pipeline import Pipeline, open_pipeline
from graph_rag.retrieval import Mode
from graph_rag.entities import EntityIdentity
from graph_rag.subgraph import (
    NodeKind,
    Subgraph,
    build_subgraph,
    chunk_detail,
    chunk_id,
    entity_id,
    leg_balance,
)

# --- Request -----------------------------------------------------------------


class TuningRequest(BaseModel):
    """The knobs of ticket 12, settable per request.

    Every field is optional and falls back to the same default the CLI uses, so a
    client that sends only a question gets the same answer `just query` gives — and a
    client that sends only `hops` changes only that.
    """

    legs: Legs | None = None
    mode: Mode | None = None
    top_k: int | None = Field(default=None, ge=1, le=100)
    hops: int | None = Field(default=None, ge=0, le=5)
    max_chunks: int | None = Field(default=None, ge=1, le=100)

    def parameters(self, defaults: AnswerSettings) -> Parameters:
        """This request's knobs, each falling back to the service's configured floor.

        `None` rather than a literal default per field, because the two are not the
        same thing: a service started with `ANSWER_LEGS=traversal` must answer a bare
        question from the graph alone, while still honouring a client that asks for
        both legs on one question. A class-level default would silently overrule the
        configuration on every request that omitted the field.
        """
        return Parameters(
            legs=self.legs if self.legs is not None else Legs(defaults.legs),
            mode=self.mode if self.mode is not None else Mode(defaults.mode),
            top_k=self.top_k if self.top_k is not None else defaults.top_k,
            hops=self.hops if self.hops is not None else defaults.hops,
            max_chunks=(
                self.max_chunks if self.max_chunks is not None else defaults.max_chunks
            ),
        )


class ExchangeRequest(BaseModel):
    """One past turn, as the client holds it. Plain strings, per ticket 13."""

    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    tuning: TuningRequest = TuningRequest()
    #: The conversation so far, oldest first, **with the standalone form of each
    #: question**. The client gets that back from `rewrite.standalone` on every
    #: response; sending the shorthand instead makes the history degrade one turn at a
    #: time, which looks like the model getting worse rather than like a bug.
    history: list[ExchangeRequest] = []
    #: Turns a follow-up is resolved against. 0 turns the memory off, which is how the
    #: demo shows the same follow-up failing.
    window: int = Field(default=4, ge=0, le=20)

    def conversation(self) -> Conversation:
        return Conversation(
            exchanges=tuple(
                Exchange(question=turn.question, answer=turn.answer)
                for turn in self.history
            ),
            window=self.window,
        )


# --- Response ----------------------------------------------------------------


def _ancestry(ancestry: tuple[Ancestor, ...] | list[Ancestor]) -> list[dict[str, Any]]:
    return [
        {"name": ancestor.name, "level": ancestor.level} for ancestor in ancestry
    ]


def _source(
    position: int, found: ContextChunk, cited: set[int], hop: int | None
) -> dict[str, Any]:
    """One numbered source, as the chat panel lists it beside the picture.

    The same payload the chunk's node carries, plus where it sits: `[Quelle 3]` in
    the answer is the source with number 3 here and the node named by `node`, which
    is how ticket 15 joins a citation to the picture.
    """
    chunk = found.chunk
    return {
        **chunk_detail(position, found, position in cited, hop),
        "node": chunk_id(chunk),
        "ancestry": _ancestry(chunk.ancestry),
        "location": found.location,
        "via": [
            {
                "entity": reached.entity.name,
                "type": reached.entity.identity.type,
                "hop": reached.hop,
            }
            for reached in found.via
        ],
    }


def _subgraph(subgraph: Subgraph) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind.value,
                "label": node.label,
                "branches": list(node.branches),
                "bridge": node.bridge,
                "ancestry": _ancestry(node.ancestry),
                "ancestry_lines": ancestry_lines(node.ancestry),
                **node.detail,
            }
            for node in subgraph.nodes
        ],
        "edges": [
            {
                "id": edge.id,
                "kind": edge.kind.value,
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
                # The flag ticket 15 colours by. Computed server-side; do not
                # recompute it in the browser.
                "crosses_branches": edge.crosses_branches,
                "branches": list(edge.branches),
                **edge.detail,
            }
            for edge in subgraph.edges
        ],
        "branches": subgraph.branches,
        # A count of edges, not of connections: one crossing is usually several
        # edges — the mention into the bridge and the relation out of it.
        "crossing_edges": len(subgraph.crossing),
    }


def _rewrite(rewrite: Rewrite) -> dict[str, Any]:
    return {
        "asked": rewrite.asked,
        "standalone": rewrite.standalone,
        "changed": rewrite.changed,
        "note": rewrite.note,
    }


def _traversal(answer: Answer) -> dict[str, Any] | None:
    """What the walk did, for the panel that explains the picture.

    None when the leg did not run — "found nothing" and "was not asked" are different
    answers, and the UI has to be able to say which.
    """
    if answer.traversal is None:
        return None
    return {
        "hops": answer.traversal.hops,
        "entities": [
            {"surface": entity.surface, "type": entity.type}
            for entity in answer.entities
        ],
        "matches": [
            {
                "surface": match.question.surface,
                "entity": match.entity.name,
                "node": entity_id(match.entity.identity),
                "scope": match.entity.identity.scope,
                "how": [way.value for way in match.how],
                "similarity": match.similarity,
            }
            for match in answer.matches
        ],
        "reached": len(answer.reached),
        "edges": len(answer.edges),
    }


def _mentions(
    pipeline: Pipeline, answer: Answer
) -> dict[str, list[EntityIdentity]]:
    """Which reached entities each drawn chunk names, as the graph knows it.

    One extra read per question, and it buys the picture its most important edges:
    a chunk the lexical leg found alone is reached through no entity, so without this
    it hangs in the subgraph unconnected and its client never counts towards the
    bridge it names. The read is the same `mentions_of` the traversal leg makes, over
    the entities already walked, so no new Cypher and nothing the graph was not asked
    for a moment ago.
    """
    reached = [found.entity.identity for found in answer.reached]
    drawn = {found.chunk.id for found in answer.context}
    if not reached or not drawn:
        return {}
    named: dict[str, list[EntityIdentity]] = {}
    for mention in pipeline.graph.mentions_of(reached):
        if mention.chunk_id in drawn:
            named.setdefault(mention.chunk_id, []).append(mention.entity)
    return named


def _answer_payload(
    answer: Answer, rewrite: Rewrite, subgraph: Subgraph
) -> dict[str, Any]:
    cited = set(answer.citations)
    # The hop each chunk sits at, computed once while the subgraph was built.
    hops = {
        node.id: node.detail.get("hop")
        for node in subgraph.nodes
        if node.kind is NodeKind.CHUNK
    }
    return {
        "question": {"asked": rewrite.asked, "answered": answer.question},
        "rewrite": _rewrite(rewrite),
        "answer": answer.text,
        "grounded": answer.grounded,
        "citations": answer.citations,
        #: Citations pointing at no source. A model that cites `[Quelle 9]` against
        #: seven sources has stopped quoting the context — surfaced rather than hidden.
        "uncited": answer.uncited,
        "parameters": {
            "legs": answer.parameters.legs.value,
            "mode": answer.parameters.mode.value,
            "top_k": answer.parameters.top_k,
            "hops": answer.parameters.hops,
            "max_chunks": answer.parameters.max_chunks,
        },
        "sources": [
            _source(position, found, cited, hops.get(chunk_id(found.chunk)))
            for position, found in enumerate(answer.context, start=1)
        ],
        "branches": answer.branches,
        "legs": leg_balance(answer),
        "traversal": _traversal(answer),
        "subgraph": _subgraph(subgraph),
    }


@contextmanager
def _reported():
    """Infrastructure failures as a 503 with the reason, not as a stack trace.

    The three that actually happen during a demo: Neo4j not running, the embedding
    call failing, and a Lucene parse error from a question. Each is recoverable by the
    operator and each needs to say *which* it was — the frontend prints this string.
    """
    try:
        yield
    except EmbeddingError as error:
        raise HTTPException(
            status_code=503, detail=f"Embedding failed: {error}"
        ) from error
    except (Neo4jError, DriverError) as error:
        raise HTTPException(status_code=503, detail=f"Neo4j: {error}") from error


# --- Application --------------------------------------------------------------


def pipeline_of(request: Request) -> Pipeline:
    """The one pipeline, from application state.

    A dependency rather than a module global so that a test — or a second app in one
    process — can hold its own, which is the same reason nothing below this file
    constructs a client.
    """
    pipeline: Pipeline | None = getattr(request.app.state, "pipeline", None)
    if pipeline is None:  # pragma: no cover - only reachable if the lifespan is skipped
        raise HTTPException(status_code=503, detail="The pipeline is not open.")
    return pipeline


PipelineDep = Annotated[Pipeline, Depends(pipeline_of)]


def create_app(settings: Settings) -> FastAPI:
    """The service over one set of settings, opening its own pipeline when it starts."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        cache = DiskCache(settings.cache.directory)
        # An ExitStack because `open_pipeline` is a synchronous context manager and
        # the lifespan is asynchronous: the driver is closed on shutdown either way.
        with ExitStack() as stack:
            app.state.pipeline = stack.enter_context(open_pipeline(settings, cache))
            yield

    app = FastAPI(
        title="Graph RAG demo",
        summary=(
            "Question answering over a nested document corpus, with the subgraph "
            "behind every answer."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.service.origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/ask")
    def ask(request: AskRequest, pipeline: PipelineDep) -> dict[str, Any]:
        """Answer a question, and describe the subgraph that answered it.

        One call does both halves of a turn: the follow-up is resolved against the
        history the client sent — ticket 13's rewrite, skipped when there is no
        history — and the standalone question goes through the same
        `answer_question` the CLI calls.
        """
        with _reported():
            turn = pipeline.turn(
                request.question,
                request.conversation(),
                request.tuning.parameters(settings.answer),
            )
            subgraph = build_subgraph(turn.answer, _mentions(pipeline, turn.answer))
        return _answer_payload(turn.answer, turn.rewrite, subgraph)

    @app.get("/api/status")
    def status(pipeline: PipelineDep) -> dict[str, Any]:
        """What has been ingested, and what each level of the hierarchy means.

        The operator's view: whether the corpus is in the graph, whether every index
        is online, whether every chunk carries a vector the index can see, and the
        inferred level names from ticket 05 — which are what an answer cites, so a
        wrong one is worth catching before going on stage rather than after.
        """
        driver = pipeline.driver
        database = pipeline.settings.neo4j.database
        with _reported():
            report = graph_status(
                driver,
                database=database,
                dimensions=pipeline.settings.mistral.embedding_dimensions,
            )
            entities = entity_coverage(driver, database)
            summaries = summary_coverage(driver, database)
            levels = read_levels(driver, database)
            bridges = count_bridges(driver, database)
        return {
            "server": report.server,
            "database": report.database,
            "ready": report.ready,
            # Not ready, but waiting is the remedy. Only meaningful when not ready —
            # a graph that is both would read as "still starting up" forever.
            "populating": not report.ready and report.only_populating,
            "nodes": report.node_counts,
            "indexes": [
                {
                    "name": index.name,
                    "state": index.state,
                    "detail": index.detail,
                    "ok": index.ok,
                }
                for index in report.indexes
            ],
            "missing_constraints": report.missing_constraints,
            "embeddings": {
                "total": report.embeddings.total,
                "embedded": report.embeddings.embedded,
                "widths": report.embeddings.widths,
                "ok": report.embeddings_ok,
                "dimensions": report.dimensions,
            },
            "summaries": {
                "total": summaries.total,
                "summarised": summaries.summarised,
                "stale": summaries.stale,
            },
            "entities": {
                "total": entities.total,
                "by_type": entities.by_type,
                "mentions": entities.mentions,
                "relations": entities.relations,
                "chunks_with_entities": entities.chunks_with_entities,
                "chunks": entities.chunks,
                # Must be zero: entities carry no folder edge, which is what makes a
                # cross-client path real structure rather than a query-time join.
                "folder_owned": entities.folder_owned,
                "bridges": bridges,
            },
            "levels": [
                {
                    "depth": level.depth,
                    "name": level.name,
                    "confidence": level.confidence,
                    "folders": level.folders,
                }
                for level in levels
            ],
            # How an ingest is triggered. Deliberately not an endpoint: an ingest is a
            # multi-minute pipeline with a progress display built for a terminal, and
            # a POST that returns in twenty minutes is worse than the command.
            "ingest_command": "just ingest corpus --levels \"Kunde,Jahr,Projekt\"",
        }

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """Is the service up. Says nothing about the graph — that is `/api/status`."""
        return {"ok": True}

    _mount_frontend(app, settings.service.frontend)
    return app


def _mount_frontend(app: FastAPI, dist: Path) -> None:
    """Serve ticket 15's built bundle at `/`, when there is one.

    Last, and deliberately: a mount at `/` matches everything, so it is added after
    the three `/api` routes, which FastAPI therefore still resolves first.

    `html=True` makes a path the bundle does not contain fall back to `index.html`.
    Silent when `dist` is missing — during development Vite serves the UI on its own
    port and proxies to here, and the service is usable with no frontend at all.
    """
    if not (dist / "index.html").is_file():
        return
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")

