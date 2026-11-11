"""The one test that catches wiring, and the only one that touches a client.

Everything else under `tests/` is a pure function: folder construction, entity merging,
rank fusion. None of them opens a driver, calls a model or speaks HTTP — which means
every seam *between* those functions is unverified until this file exists. A mistyped
model identifier, a parser pointed at a dead port, an embedding written at a width the
index cannot see, a Cypher statement that merges on the wrong property: none of them
would be caught by the fast suite, and all of them present as a demo that answers the
wrong thing in front of an audience.

So this test runs the real path: the real `ingest` command, against real Neo4j, the real
Mistral API and the real self-hosted parsing service. No fakes anywhere. That is the
deliberate alternative to building fake implementations at the pipeline seam, and it is
cheap precisely because all three are running anyway while the demo is being worked on.

It is deselected by default (`addopts = -m 'not smoke'`); `just smoke` runs it. If any of
the three dependencies is down it skips with the reason and the command that fixes it —
see RUNBOOK.md, which documents the same failures in the same order.

**It writes into the same graph the demo uses.** Neo4j Community has one database, so
isolation comes from the corpus root instead: the tiny tree lives under its own path, is
asserted on by that path, and is deleted again afterwards. The one thing that is not
path-scoped is entity resolution — it runs over every chunk in the graph by design — so
the test refuses to run against a graph that already holds another corpus unless
`SMOKE_ALLOW_POPULATED=1`, in which case it restores the entity set on the way out.
"""

from __future__ import annotations

import os
import shutil
import warnings
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from neo4j import Driver
from neo4j.exceptions import DriverError, Neo4jError
from typer.testing import CliRunner

from graph_rag.answer import DEFAULT_PARAMETERS, Answer
from graph_rag.cache import DiskCache
from graph_rag.cli import app
from graph_rag.config import Settings, load_settings
from graph_rag.corpus.generate import generate_corpus
from graph_rag.graph.driver import graph_driver
from graph_rag.graph.health import graph_status
from graph_rag.pipeline import open_pipeline

pytestmark = pytest.mark.smoke

#: Where the tiny tree is built. Stable rather than a `tmp_path`, so the chunk ids —
#: `uuid5` of the document path — are the same on every run and a second run reuses the
#: cache instead of re-parsing four PDFs.
SMOKE_ROOT = Path(".cache/smoke-corpus")

#: The generator's default seed, which is what `just corpus` uses. Same seed, same
#: bytes, so the four PDFs below hash to what the demo corpus already parsed and a run
#: on a warm cache makes no call to the parsing service at all.
CORPUS_SEED = 20241101

LEVELS = "Kunde,Jahr,Projekt"

#: Four documents, chosen for what they prove rather than to be a sample. Both clients
#: run a `Projekt Atlas` in 2023 with entirely different work behind it, and the only
#: thing joining them is Dr. Katharina Vogt and the Atlas-Referenzarchitektur she
#: carried across — so a tree this small still contains the bridge the whole demo rests
#: on, and the near-identical status reports that make the two clients confusable.
NORDLICHT = "Nordlicht Energie AG"
BRANDT = "Brandt und Soehne Maschinenbau"
PROJECT = "Projekt Atlas"
KNOWN_DOCUMENT = "projektbericht-2023.pdf"
DOCUMENTS = (
    f"{NORDLICHT}/2023/{PROJECT}/{KNOWN_DOCUMENT}",
    f"{NORDLICHT}/2023/{PROJECT}/statusbericht-2023-Q2.pdf",
    f"{BRANDT}/2023/{PROJECT}/{KNOWN_DOCUMENT}",
    f"{BRANDT}/2023/{PROJECT}/statusbericht-2023-Q2.pdf",
)

#: Answerable from exactly one of the four documents. "Netzleittechnik" is Nordlicht's
#: subject and appears nowhere in Brandt's product-data project, so a retrieval path
#: that is wired up reaches the Nordlicht report — and one that has quietly become
#: dense-only over near-identical boilerplate reaches whichever client it likes.
KNOWN_QUESTION = "Welches Zielarchitektur-Dokument Netzleittechnik wurde abgenommen und von wem?"


@pytest.fixture(scope="module")
def settings() -> Settings:
    return load_settings()


@pytest.fixture(scope="module")
def driver(settings: Settings) -> Iterator[Driver]:
    with graph_driver(settings.neo4j) as driver:
        yield driver


@pytest.fixture(scope="module")
def live(settings: Settings, driver: Driver) -> None:
    """Skip, loudly and specifically, rather than fail on infrastructure.

    The three dependencies are checked in the order RUNBOOK.md starts them, and each
    skip names the command that fixes it: a red test on a machine where Docker is not
    running says nothing about the code, and the one failure mode this project was
    specified against is exactly that machine.
    """
    if not settings.mistral.api_key:
        pytest.skip("MISTRAL_API_KEY is not set — see RUNBOOK.md, 'Cold start'")

    try:
        status = graph_status(
            driver,
            database=settings.neo4j.database,
            dimensions=settings.mistral.embedding_dimensions,
        )
    except (Neo4jError, DriverError) as error:
        pytest.skip(f"Neo4j is not reachable at {settings.neo4j.uri} ({error}) — run `just up`")

    missing = [index.name for index in status.indexes if not index.ok]
    if missing or status.missing_constraints:
        pytest.skip(
            "the graph schema is not ready "
            f"({', '.join(missing + status.missing_constraints)}) — run `just init`"
        )

    health = f"{settings.parsing.url.rstrip('/')}/healthcheck"
    try:
        httpx.get(health, timeout=10.0).raise_for_status()
    except httpx.HTTPError as error:
        pytest.skip(
            f"the parsing service is not answering at {health} ({error}) — "
            "see RUNBOOK.md, 'The parsing service'"
        )


@pytest.fixture(scope="module")
def corpus_root() -> Path:
    """The tiny tree, built from the generator rather than from `corpus/`.

    Generating all 120 PDFs takes half a second and no API call, and the output is
    byte-identical to what `just corpus` wrote — so the test depends on the generator,
    which is in the repository, rather than on a corpus directory that is gitignored and
    may not exist on the machine this is run on.
    """
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
    source = SMOKE_ROOT.parent / "smoke-corpus-source"
    generate_corpus(source, seed=CORPUS_SEED, force=True)
    for relative in DOCUMENTS:
        destination = SMOKE_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    shutil.rmtree(source)
    return SMOKE_ROOT.resolve()


@pytest.fixture(scope="module")
def ingested(live: None, settings: Settings, driver: Driver, corpus_root: Path) -> Iterator[Path]:
    """Run the real ingest, and take the tiny tree back out of the graph afterwards.

    The removal is in a `finally` that starts *before* the ingest, not after it: an
    ingest that dies halfway has still written folders, documents and chunks, and those
    are exactly the ones nothing else will ever clean up.
    """
    _refuse_a_populated_graph(driver, settings, corpus_root)
    try:
        result = CliRunner().invoke(
            app, ["ingest", str(corpus_root), "--levels", LEVELS], catch_exceptions=False
        )
        assert result.exit_code == 0, result.output
        yield corpus_root
    finally:
        _remove(driver, settings, corpus_root)


def _documents_outside(driver: Driver, settings: Settings, root: Path) -> int:
    records, _, _ = driver.execute_query(
        "MATCH (d:Document) WHERE NOT d.path STARTS WITH $prefix RETURN count(d) AS count",
        {"prefix": f"{root}/"},
        database_=settings.neo4j.database,
    )
    return records[0]["count"]


def _refuse_a_populated_graph(driver: Driver, settings: Settings, root: Path) -> None:
    """Entity resolution is the one stage that is not scoped to a corpus root.

    It reads every chunk in the graph on purpose — a name is judged against the whole
    archive, not against the branch that happens to be being ingested — so ingesting
    four more documents into a graph that already holds the demo corpus re-resolves the
    demo corpus too, with four extra documents' mentions on the scale. Nothing breaks,
    but the entity set the demo is about has been nudged by a test, which is not a thing
    to discover on stage. Run this before `just ingest`, or opt in and pay for the
    restore on the way out.
    """
    outside = _documents_outside(driver, settings, root)
    if outside and not _allow_populated():
        pytest.skip(
            f"the graph already holds {outside} document(s) from another corpus. The "
            "smoke test expects a cold graph — run it before `just ingest`, or set "
            "SMOKE_ALLOW_POPULATED=1 to run anyway and re-resolve entities afterwards."
        )


def _allow_populated() -> bool:
    return (os.getenv("SMOKE_ALLOW_POPULATED") or "").strip().casefold() in (
        "1",
        "true",
        "yes",
        "on",
    )


_CHUNK_IDS = """
MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE d.path STARTS WITH $prefix
RETURN collect(c.id) AS ids
"""

#: A relation carries the chunk it was read from, and that chunk is about to be deleted.
#: Nothing else would remove the edge: an entity surviving on the strength of the demo
#: corpus's mentions would keep a `RELATES` the evidence for is gone — exactly the kind
#: of edge that is drawn in gold on stage and cannot be justified when someone asks.
_REMOVE_RELATIONS = """
MATCH (:Entity)-[r:RELATES]->(:Entity) WHERE r.source_chunk IN $ids DELETE r
"""

_REMOVE = """
MATCH (f:Folder) WHERE f.path = $root OR f.path STARTS WITH $prefix
OPTIONAL MATCH (f)-[:CONTAINS]->(d:Document)
OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
DETACH DELETE c, d, f
"""

#: Entities are global and are written from the chunks that mention them, so the ones
#: this run introduced are exactly those left with no mention at all once its chunks are
#: gone. Scoping this by path is impossible by design — there is deliberately no edge
#: from a folder to an entity — and counting mentions is the honest equivalent.
_REMOVE_ORPHANS = """
MATCH (e:Entity) WHERE NOT (:Chunk)-[:MENTIONS]->(e) DETACH DELETE e
"""


def _remove(driver: Driver, settings: Settings, root: Path) -> None:
    database = settings.neo4j.database
    scope = {"root": str(root), "prefix": f"{root}/"}
    ids = _rows(driver, settings, _CHUNK_IDS, root)[0]["ids"]
    driver.execute_query(_REMOVE_RELATIONS, {"ids": ids}, database_=database)
    driver.execute_query(_REMOVE, scope, database_=database)
    driver.execute_query(_REMOVE_ORPHANS, database_=database)
    if _documents_outside(driver, settings, root):
        # Opted in above: the surviving corpus's entities were resolved with this run's
        # documents on the scale, so they are recomputed from what is left. Every call
        # it makes is cached by then, so this costs seconds and no API spend.
        restored = CliRunner().invoke(app, ["extract"], catch_exceptions=False)
        if restored.exit_code != 0:
            # Warned rather than raised: this runs in teardown, and failing here would
            # replace whatever the test was actually saying with a cleanup error.
            warnings.warn(
                "the smoke test could not re-resolve entities after removing its corpus"
                f" — run `just extract`.\n{restored.output}",
                stacklevel=2,
            )


def _rows(driver: Driver, settings: Settings, cypher: str, root: Path) -> list[dict]:
    records, _, _ = driver.execute_query(
        cypher,
        {"root": str(root), "prefix": f"{root}/"},
        database_=settings.neo4j.database,
    )
    return [record.data() for record in records]


def _first(found: list[tuple[str, list[str]]], matches) -> int | None:
    """Where the first chunk `matches` landed in the context, 0-based, or None."""
    return next(
        (position for position, (f, a) in enumerate(found) if matches(f, a)), None
    )


def _ask(settings: Settings) -> Answer:
    with open_pipeline(settings, DiskCache(settings.cache.directory)) as pipeline:
        return pipeline.ask(KNOWN_QUESTION, DEFAULT_PARAMETERS)


def test_a_tiny_corpus_ingests_and_answers_a_question_about_one_client(
    ingested: Path, settings: Settings, driver: Driver
) -> None:
    root = ingested

    # The hierarchy, with what each depth means. Four documents under two clients, one
    # year, one project name shared between them — the collision the corpus is built on,
    # which only survives ingestion if the folder path is preserved rather than flattened.
    folders = _rows(
        driver,
        settings,
        """
        MATCH (f:Folder) WHERE f.path = $root OR f.path STARTS WITH $prefix
        RETURN f.name AS name, f.depth AS depth,
               f.inferred_level_name AS level ORDER BY f.depth, f.name
        """,
        root,
    )
    assert [(f["name"], f["depth"], f["level"]) for f in folders] == [
        (root.name, 0, None),
        (BRANDT, 1, "Kunde"),
        (NORDLICHT, 1, "Kunde"),
        ("2023", 2, "Jahr"),
        ("2023", 2, "Jahr"),
        (PROJECT, 3, "Projekt"),
        (PROJECT, 3, "Projekt"),
    ]

    documents = _rows(
        driver,
        settings,
        """
        MATCH (f:Folder)-[:CONTAINS]->(d:Document) WHERE d.path STARTS WITH $prefix
        RETURN d.filename AS filename, d.page_count AS pages,
               d.content_hash AS hash, d.summary AS summary ORDER BY d.path
        """,
        root,
    )
    assert len(documents) == len(DOCUMENTS)
    assert all(d["pages"] and d["hash"] and d["summary"] for d in documents)

    # Chunks, in reading order, each carrying a vector the index can actually see. A
    # width the index was not built at is the project's quietest failure: vector search
    # returns nothing and the full-text leg covers for it, so the demo still answers.
    chunks = _rows(
        driver,
        settings,
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE d.path STARTS WITH $prefix
        RETURN count(c) AS chunks, count(c.embedding) AS embedded,
               collect(DISTINCT size(c.embedding)) AS widths
        """,
        root,
    )[0]
    assert chunks["chunks"] > len(DOCUMENTS)
    assert chunks["embedded"] == chunks["chunks"]
    assert chunks["widths"] == [settings.mistral.embedding_dimensions]

    # Reading order, per document: a chain, not a bag. One `NEXT` edge fewer than there
    # are chunks, and never an edge that leaves the document it belongs to.
    reading_order = _rows(
        driver,
        settings,
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE d.path STARTS WITH $prefix
        WITH d, d.path AS path, count(c) AS chunks
        MATCH (d)-[:HAS_CHUNK]->(:Chunk)-[next:NEXT]->(following:Chunk)
        WHERE (d)-[:HAS_CHUNK]->(following)
        RETURN path, chunks, count(next) AS edges ORDER BY path
        """,
        root,
    )
    assert len(reading_order) == len(DOCUMENTS)
    assert all(row["edges"] == row["chunks"] - 1 for row in reading_order), reading_order

    # The knowledge graph on top of it: entities named by those chunks, with evidence,
    # and relations between them.
    entities = _rows(
        driver,
        settings,
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(:Chunk)-[m:MENTIONS]->(e:Entity)
        WHERE d.path STARTS WITH $prefix
        RETURN count(DISTINCT e) AS entities, count(m) AS mentions,
               count(m.evidence) AS evidence
        """,
        root,
    )[0]
    assert entities["entities"] > 0
    assert entities["evidence"] == entities["mentions"]

    # Scoped by the chunk a relation was read from rather than by the entities it joins:
    # an entity is global and may well be joined to something by a document outside this
    # tree, and counting those would make the assertion pass on the demo corpus's work.
    relations = _rows(
        driver,
        settings,
        """
        MATCH (d:Document)-[:HAS_CHUNK]->(c:Chunk) WHERE d.path STARTS WITH $prefix
        WITH collect(c.id) AS ids
        MATCH (:Entity)-[r:RELATES]->(:Entity) WHERE r.source_chunk IN ids
        RETURN count(r) AS relations, count(r.evidence) AS evidence
        """,
        root,
    )[0]
    assert relations["relations"] > 0
    assert relations["evidence"] == relations["relations"]

    # And the one node the whole demo exists to show: an entity named under both
    # clients, which is a cross-branch edge in the graph rather than a join computed
    # when someone asks for it.
    bridges = _rows(
        driver,
        settings,
        """
        MATCH (branch:Folder {depth: 1})-[:CONTAINS*]->(:Document)-[:HAS_CHUNK]->(c:Chunk)
        WHERE branch.path STARTS WITH $prefix
        MATCH (c)-[:MENTIONS]->(e:Entity)
        WITH e, collect(DISTINCT branch.name) AS branches
        WHERE size(branches) > 1
        RETURN e.name AS name ORDER BY e.name
        """,
        root,
    )
    assert [bridge["name"] for bridge in bridges], "no entity joins the two clients"

    # Finally the other half of the system, over the same graph: a question only one of
    # these two clients can answer, asked through the pipeline both front ends use.
    answer = _ask(settings)
    assert answer.grounded
    assert answer.text
    assert not answer.uncited

    # Asserted on the document's place in the hierarchy rather than on its path, because
    # a graph that also holds the demo corpus holds the same report at a different path
    # and either copy answers this question correctly.
    #
    # And asserted on *rank* rather than on the other client being absent: on a corpus
    # of four documents the context is bigger than the corpus, so Brandt's report is in
    # it either way and excluding it would only be asserting that the corpus is small.
    # The claim worth making is the one the project is about — the right client's report
    # outranks the near-identical one from the other client, on a question only one of
    # them can answer.
    found = [(chunk.chunk.filename, [a.name for a in chunk.ancestry]) for chunk in answer.context]
    known = _first(found, lambda f, a: f == KNOWN_DOCUMENT and NORDLICHT in a)
    other = _first(found, lambda _, ancestry: BRANDT in ancestry)
    assert known is not None, found
    assert other is None or known < other, found
