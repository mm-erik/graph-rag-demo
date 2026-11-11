"""Command line entry point.

This module is the composition root: it loads settings, constructs the Neo4j driver and
the Mistral client, and injects them into functions that never construct their own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import httpx
import typer
from mistralai.client import Mistral
from neo4j.exceptions import AuthError, ClientError, DriverError, Neo4jError
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from graph_rag.ancestry import ancestry_lines, branch
from graph_rag.answer import (
    DEFAULT_MAX_CHUNKS,
    Answer,
    Legs,
    Parameters,
    Source,
)
from graph_rag.cache import (
    COMPLETION,
    EMBED,
    PARSE,
    CachingChatModel,
    DiskCache,
)
from graph_rag.chat import make_chat_model, make_extraction_model
from graph_rag.config import Settings, load_settings
from graph_rag.conversation import (
    DEFAULT_WINDOW,
    NOTHING_TO_RESOLVE,
    Conversation,
    Rewrite,
    Turn,
)
from graph_rag.clarification import Clarification
from graph_rag.corpus.generate import CorpusExists, generate_corpus
from graph_rag.embedding import EmbeddingError
from graph_rag.entities import (
    ENTITY_TYPES,
    ChunkResult,
    ExtractionReport,
    ambiguous_names,
    candidate_entities,
    canonical_edges,
    extract_chunks,
)
from graph_rag.graph.chunks import embedding_coverage
from graph_rag.graph.driver import graph_driver
from graph_rag.graph.entities import (
    BridgeEntity,
    EntityWriteReport,
    StoredEntity,
    count_bridges,
    entity_coverage,
    read_branch_folders,
    read_bridges,
    read_chunk_sources,
    read_entities,
    write_entities,
)
from graph_rag.graph.health import graph_status
from graph_rag.graph.hierarchy import HierarchyWriteReport, write_hierarchy
from graph_rag.graph.levels import read_level_samples, write_levels
from graph_rag.graph.retrieval import GraphChunkIndex
from graph_rag.graph.schema import (
    CHUNK_VECTOR_INDEX,
    SchemaApplicationError,
    apply_schema,
    list_indexes,
    vector_index_dimensions,
)
from graph_rag.graph.summaries import (
    GraphSummaryStore,
    StoredSummary,
    read_document_states,
    read_summaries,
    summary_coverage,
)
from graph_rag.graph.traversal import GraphEntityGraph
from graph_rag.hierarchy import Hierarchy, build_hierarchy, walk_documents
from graph_rag.ingestion import ContentReport, DocumentContent, ingest_contents
from graph_rag.levels import Level, LevelSample, infer_levels, parse_level_names
from graph_rag.mistral_check import verify_models
from graph_rag.parsing import ParsedDocument, make_chunker, make_parser, parse_file
from graph_rag.pipeline import Pipeline, embedder_for, open_pipeline
from graph_rag.resolution import (
    Candidate,
    Resolution,
    Thresholds,
    model_adjudicator,
    resolve_entities,
)
from graph_rag.retrieval import (
    DEFAULT_TOP_K,
    Chunk,
    Leg,
    LegResult,
    Mode,
    lucene_query,
    search_chunks,
)
from graph_rag.summaries import (
    DocumentSummary,
    SummaryReport,
    needs_summary,
    summarise_documents,
)
from graph_rag.traversal import (
    DEFAULT_HOPS,
    MatchedEntity,
    TraversalResult,
    traverse_entities,
)
from graph_rag.vendored.errors import ProviderError

app = typer.Typer(
    help="Graph RAG demo — ingest a document hierarchy into Neo4j and query it.",
    no_args_is_help=True,
)
console = Console()

#: Below this, an inferred level name is flagged so the operator knows to override it
#: *before* going on stage rather than discovering it in front of an audience. Tuned
#: against what `mistral-large-latest` actually emits: legible corpora come back at
#: 0.95-0.98 and opaque ones ("XK-01", "a7") at 0.70-0.80, so 0.9 separates them. It is
#: deliberately not lower — a threshold under the opaque band flags nothing at all.
LOW_CONFIDENCE = 0.9


#: The stages of a full ingest, in order, so the progress display can say *which* of
#: them is running and how many are left. A multi-minute run in front of an audience
#: that reports only "working…" is indistinguishable from one that has hung — which is
#: the specific failure this numbering exists to prevent.
INGEST_STAGES: tuple[str, ...] = (
    "Walking the corpus",
    "Writing the structure",
    "Inferring what each level means",
    "Parsing and embedding documents",
    "Summarising documents",
    "Extracting entities",
)
# Unpacked rather than indexed by name: a mistyped stage label would otherwise be a
# crash mid-ingest, and this is the one feature whose whole purpose is to be reassuring.
(
    WALK_STAGE,
    STRUCTURE_STAGE,
    LEVELS_STAGE,
    CONTENT_STAGE,
    SUMMARY_STAGE,
    ENTITY_STAGE,
) = INGEST_STAGES


def _stage_label(name: str) -> str:
    """`[4/6] Parsing and embedding documents` — where the run is, and how far it goes."""
    return f"[{INGEST_STAGES.index(name) + 1}/{len(INGEST_STAGES)}] {name}"


def _progress() -> Progress:
    """The one progress display shape every stage of an ingest uses."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    )


#: What each namespace holds, in the operator's terms rather than the cache's.
_CACHE_LABELS = {
    PARSE: "parsed documents",
    EMBED: "embeddings",
    COMPLETION: "model replies",
}


def _open_cache(
    settings: Settings, *, discard: bool, namespaces: tuple[str, ...] = ()
) -> DiskCache:
    """The ingestion cache, emptied first when the operator asked for a clean run.

    `discard` is `--no-cache`, and it *deletes* rather than bypasses. A flag that only
    skipped reads would leave the old entries in place underneath the new ones, so the
    next run without the flag would answer from whichever of the two it happened to
    find — which is a cache that cannot be cleared, and the one thing the ticket for
    this feature asks not to build.

    `namespaces` narrows what it deletes to what the command could actually re-do.
    `extract` re-runs model calls, so it discards replies; discarding the parses too
    would bill the next ingest twenty minutes of layout analysis for a flag that was
    about prompts.
    """
    cache = DiskCache(settings.cache.directory)
    if discard:
        cache.clear(*namespaces)
        discarded = ", ".join(_CACHE_LABELS[name] for name in namespaces) or "everything"
        console.print(f"[dim]Cache {cache.root}: {discarded} discarded.[/dim]")
    return cache


def _print_cache(cache: DiskCache) -> None:
    """What the cache saved this run.

    Printed every time, because a cache that has quietly stopped hitting — a changed
    prompt, a moved corpus, a cleared directory — is otherwise indistinguishable from
    one that is working until the ingest takes ten minutes and the bill arrives.
    """
    stats = cache.stats()
    if not stats:
        return
    parts = [
        f"{label}: {stats[namespace].hits} reused / {stats[namespace].misses} new"
        for namespace, label in _CACHE_LABELS.items()
        if namespace in stats
    ]
    console.print(f"[dim]Cache {cache.root} — {'; '.join(parts)}.[/dim]")
    console.print("[dim]Force a clean re-run with --no-cache.[/dim]")


def _settings() -> Settings:
    """Misconfiguration should read as a message, not a traceback — these are the
    commands an operator reaches for when the environment is wrong."""
    try:
        return load_settings()
    except ValueError as error:
        console.print(f"[bold red]Bad configuration[/bold red]: {error}")
        raise typer.Exit(code=1) from error


def _graph_failure(settings: Settings, error: Exception) -> typer.Exit:
    """Say what actually went wrong. A healthy container with the wrong password is not
    "unreachable", and telling the operator to run `just up` would waste their time —
    the README's own note says a changed password makes `just up` hang."""
    neo4j = settings.neo4j
    if isinstance(error, AuthError):
        console.print(
            f"[bold red]Authentication failed[/bold red] for user '{neo4j.user}' "
            f"at {neo4j.uri}: {error}"
        )
        console.print(
            "Check NEO4J_PASSWORD. Note that Neo4j keeps the password from the first "
            "start of its data volume, so changing it in .env is not enough."
        )
    elif isinstance(error, ClientError) and "DatabaseNotFound" in (error.code or ""):
        console.print(
            f"[bold red]Database '{neo4j.database}' does not exist[/bold red] "
            f"at {neo4j.uri}. Check NEO4J_DATABASE."
        )
    elif isinstance(error, Neo4jError):
        console.print(f"[bold red]Neo4j rejected the request[/bold red]: {error}")
    else:
        console.print(f"[bold red]Neo4j unreachable[/bold red] at {neo4j.uri}: {error}")
        console.print("Is it running? Try `just up`.")
    return typer.Exit(code=1)


@app.command("init-graph")
def init_graph() -> None:
    """Create every constraint and index. Safe to run repeatedly."""
    settings = _settings()
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            report = apply_schema(
                driver,
                database=settings.neo4j.database,
                dimensions=settings.mistral.embedding_dimensions,
            )
    except SchemaApplicationError as error:
        _print_drops(error.dropped)
        console.print(f"[bold red]Schema creation failed[/bold red]: {error}")
        if error.dropped:
            console.print(
                "[bold red]The indexes listed above were dropped and not recreated.[/bold red]"
            )
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    _print_drops(report.dropped)
    for applied in report.applied:
        mark = "[green]✓[/green]" if applied.created else "[dim]·[/dim]"
        suffix = "" if applied.created else " [dim](already present)[/dim]"
        console.print(
            f"{mark} {applied.statement.kind}: {applied.statement.name}{suffix}"
        )
    console.print(f"[bold green]Schema applied[/bold green] to '{settings.neo4j.database}'.")


def _print_drops(dropped) -> None:
    for drop in dropped:
        console.print(f"[yellow]dropped {drop.name}: {drop.reason}[/yellow]")
        if drop.needs_reingest:
            console.print(
                "[yellow]  embeddings already written at the old width are now invisible "
                "to vector search — re-ingest the corpus.[/yellow]"
            )


@app.command()
def status() -> None:
    """Report graph connectivity and which indexes exist."""
    settings = _settings()
    try:
        with graph_driver(settings.neo4j) as driver:
            report = graph_status(
                driver,
                database=settings.neo4j.database,
                dimensions=settings.mistral.embedding_dimensions,
            )
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    console.print(f"[bold]Connected[/bold] to {report.server} at {settings.neo4j.uri}")
    console.print(f"Database: {report.database}")

    index_table = Table(title="Indexes")
    index_table.add_column("name")
    index_table.add_column("state")
    index_table.add_column("detail")
    for index in report.indexes:
        mark = "[green]✓[/green]" if index.ok else "[red]✗[/red]"
        index_table.add_row(f"{mark} {index.name}", index.state or "MISSING", index.detail)
    console.print(index_table)

    console.print("Constraints: " + (", ".join(report.constraints) or "none"))
    if report.missing_constraints:
        console.print(
            "[red]Missing constraints: "
            + ", ".join(report.missing_constraints)
            + "[/red]"
        )
    counts = ", ".join(f"{label}={count}" for label, count in report.node_counts.items())
    console.print(f"Nodes: {counts}")
    _print_coverage(report.embeddings, report.dimensions)

    if report.ready:
        console.print("[bold green]Graph is ready.[/bold green]")
        return
    if report.only_populating:
        console.print(
            "[bold yellow]Indexes are still populating — wait and re-run `just status`."
            "[/bold yellow]"
        )
    elif not report.embeddings_ok:
        console.print(
            "[bold red]Chunks are missing an embedding, or carry one the vector index "
            "cannot see.[/bold red] Re-ingest the corpus."
        )
    else:
        console.print("[bold red]Graph is not ready — run `just init`.[/bold red]")
    raise typer.Exit(code=1)


@app.command("verify-models")
def verify_models_command() -> None:
    """Check the configured Mistral model identifiers against the live API."""
    settings = _settings()
    if not settings.mistral.api_key:
        console.print("[bold red]MISTRAL_API_KEY is not set[/bold red] — cannot verify models.")
        raise typer.Exit(code=1)

    client = Mistral(api_key=settings.mistral.api_key)
    report = verify_models(client, settings.mistral)

    if report.listing_error:
        console.print(f"[bold red]Could not list models[/bold red]: {report.listing_error}")

    table = Table(title="Mistral models")
    table.add_column("role")
    table.add_column("identifier")
    table.add_column("status")
    for check in report.checks:
        mark = "[green]available[/green]" if check.available else "[red]NOT FOUND[/red]"
        table.add_row(check.role, check.identifier, f"{mark} {check.note or ''}".strip())
    console.print(table)

    console.print(
        f"Embedding width: measured={report.embedding_dimensions}, "
        f"configured={report.configured_dimensions}"
    )
    if not report.ok:
        console.print(
            "[bold red]Discrepancy found[/bold red] — update .env and re-run `just init`."
        )
        raise typer.Exit(code=1)
    console.print("[bold green]Model identifiers verified.[/bold green]")


@app.command("generate-corpus")
def generate_corpus_command(
    output: Annotated[
        Path, typer.Argument(help="Directory to write the corpus tree into")
    ] = Path("corpus"),
    seed: Annotated[int, typer.Option(help="Seed — same seed, same corpus")] = 20241101,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing corpus in that directory")
    ] = False,
) -> None:
    """Generate the synthetic adversarial German corpus."""
    try:
        report = generate_corpus(output, seed=seed, force=force)
    except CorpusExists as error:
        console.print(f"[bold red]{error}[/bold red]")
        raise typer.Exit(code=1) from error
    console.print(
        f"[bold green]{len(report.all_documents)} PDFs[/bold green] under {report.root}: "
        f"{len(report.documents)} in {report.project_count} project folders, "
        f"{len(report.machine_documents)} in {report.model_count} machine model folders"
    )
    console.print(
        f"Text layer verified on {report.verified_file.name}: "
        f"{report.verified_characters} characters extracted"
    )
    console.print(f"Planted collisions documented in {report.note}")


async def _parse_one(
    pdf: Path, settings: Settings, cache: DiskCache
) -> ParsedDocument:
    """Own the HTTP client for the duration of the parse.

    The vendored transport falls back to constructing (and never closing) its own client
    per request, so the client is built here, once, and injected — otherwise an ingest
    over the whole corpus would abandon one connection pool per document.

    It shares the ingest's cache, so the parse behind a repeated call is paid for once.
    """
    async with httpx.AsyncClient(timeout=settings.parsing.timeout_seconds) as http_client:
        parser = make_parser(settings.parsing, http_client, cache)
        return await parse_file(pdf, parser, make_chunker(settings.chunking))


@app.command()
def parse(
    pdf: Annotated[Path, typer.Argument(help="PDF to parse")],
    text: Annotated[
        bool, typer.Option("--text", help="Print each chunk's full text")
    ] = False,
) -> None:
    """Parse one PDF and print the chunks it would contribute to the graph.

    Cached like an ingest, which is what makes it usable for the thing it is for:
    chunking re-runs on every call while the parse behind it does not, so the chunk
    parameters can be tuned against a real document without the parsing service being
    asked anything twice. A changed parsing *strategy* re-parses on its own, because it
    is part of the key; there is deliberately no flag here, since a diagnostic over one
    file should not be able to discard an ingest's worth of parses.
    """
    settings = _settings()
    if not pdf.is_file():
        console.print(f"[bold red]No such file[/bold red]: {pdf}")
        raise typer.Exit(code=1)

    try:
        parsed = asyncio.run(
            _parse_one(pdf, settings, _open_cache(settings, discard=False))
        )
    except ProviderError as error:
        console.print(f"[bold red]Parsing failed[/bold red]: {error}")
        console.print(
            f"Is the self-hosted Unstructured service running at {settings.parsing.url}? "
            "It is started by the monorepo's process-compose."
        )
        raise typer.Exit(code=1) from error

    console.print(
        f"[bold]{pdf.name}[/bold]: {len(parsed.document.elements)} elements -> "
        f"{len(parsed.chunks)} chunks (document id {parsed.document.id})"
    )
    table = Table(title="Chunks")
    table.add_column("#", justify="right")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("pages")
    table.add_column("chars", justify="right")
    table.add_column("preview" if not text else "text")
    for chunk in parsed.chunks:
        pages = ",".join(str(page) for page in chunk.metadata.get("pages", [])) or "-"
        body = chunk.text if text else chunk.text[:80].replace("\n", " ")
        table.add_row(
            str(chunk.index),
            chunk.id,
            chunk.metadata.get("kind", "-"),
            pages,
            str(len(chunk.text)),
            body,
        )
    console.print(table)


@app.command()
def ingest(
    directory: Annotated[Path, typer.Argument(help="Root of the document hierarchy")],
    levels: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated level names applied from depth 1 down, e.g. "
            "Kunde,Jahr,Projekt. Bypasses inference for the depths it covers."
        ),
    ] = None,
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Discard the ingestion cache first and re-do everything from scratch",
        ),
    ] = False,
) -> None:
    """Walk a directory and ingest it into the graph. (Tickets 04–09.)

    The folder and document structure, what each level of it means, every document's
    chunks with their embeddings, a generated summary per document, and the entities
    and relations those chunks name.

    Every external call it makes — the parse, each embedding, each model reply — is
    cached on disk under `CACHE_DIR`, keyed by the file's content hash and by the
    configuration the call depended on. A second run over an unchanged corpus therefore
    makes no API calls at all and finishes in seconds; a document whose bytes changed
    is the only one reprocessed. `--no-cache` empties the cache first, for a genuinely
    clean run.
    """
    settings = _settings()
    if not directory.is_dir():
        console.print(f"[bold red]No such directory[/bold red]: {directory}")
        raise typer.Exit(code=1)

    level_names = _level_names(levels)
    # After the arguments are known to be good: `--no-cache` deletes, and a mistyped
    # directory should not cost an hour of re-parsing.
    cache = _open_cache(settings, discard=no_cache)

    root = directory.resolve()
    with console.status(f"{_stage_label(WALK_STAGE)}: {root}…"):
        files = walk_documents(root)
    if not files:
        console.print(f"[bold red]No PDFs found[/bold red] under {root}.")
        console.print("Generate the demo corpus with `just corpus` if you have none.")
        raise typer.Exit(code=1)

    hierarchy = build_hierarchy(root, files)
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            console.print(f"[dim]{_stage_label(STRUCTURE_STAGE)}…[/dim]")
            report = write_hierarchy(
                driver, database=settings.neo4j.database, hierarchy=hierarchy
            )
            _print_hierarchy(hierarchy, report, root)
            samples = read_level_samples(
                driver, database=settings.neo4j.database, root=str(root)
            )
            # Constructed here at the composition root and passed down: everything
            # below sees only the `ChatModel` protocol, never the Mistral SDK and never
            # the cache wrapped around it. One client serves all three callers that
            # need it — level inference, the summary pass and the resolution tiebreak —
            # rather than one each.
            chat = (
                CachingChatModel(
                    make_chat_model(settings.mistral),
                    cache,
                    signature=settings.mistral.chat_model,
                )
                if settings.mistral.api_key
                else None
            )
            model = chat if _needs_model(samples, level_names, settings) else None
            with console.status(f"{_stage_label(LEVELS_STAGE)}…"):
                levels = infer_levels(samples, model, level_names)
            labelled = write_levels(
                driver, database=settings.neo4j.database, root=str(root), levels=levels
            )
            _print_levels(samples, levels, labelled)

            _check_embedding_width(driver, settings)
            content = asyncio.run(_ingest_contents(hierarchy, settings, driver, cache))
            _print_content(content, settings)
            # After the chunks, necessarily: a summary is written from them. `chat` is
            # non-None by here — `_check_embedding_width` has already refused to get
            # this far without an API key.
            summaries = asyncio.run(
                _summarise(
                    hierarchy,
                    settings,
                    driver,
                    chat,
                    refresh=no_cache,
                    skip={str(failure.path) for failure in content.failures},
                )
            )
            _print_summaries(summaries)
            # Last, and necessarily so: entities are extracted from the chunks the
            # content pass just wrote, and resolution needs the whole corpus in hand
            # before it can tell a shared node from a coincidence of spelling.
            entity_outcome = asyncio.run(
                _extract_entities(settings, driver, chat, cache, _stage_label(ENTITY_STAGE))
            )
            _print_entity_outcome(entity_outcome, settings)
            _print_cache(cache)
    except EmbeddingError as error:
        console.print(f"[bold red]Embedding failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    # Both passes report their failures per document and carry on, so the exit code is
    # decided once, here, after everything that could still succeed has.
    if content.failures or summaries.failures or entity_outcome.extraction.failures:
        raise typer.Exit(code=1)


def _level_names(levels: str | None) -> list[str]:
    """The operator's explicit level names, as a message rather than a traceback."""
    try:
        return parse_level_names(levels)
    except ValueError as error:
        console.print(f"[bold red]Bad --levels[/bold red]: {error}")
        raise typer.Exit(code=1) from error


#: Placeholder names for the "you need a key" hint, so the suggestion has the right
#: *shape* for this corpus even though nothing has been inferred yet.
_HINT_NAMES = ("Kunde", "Jahr", "Projekt", "Phase", "Revision")


def _print_hierarchy(
    hierarchy: Hierarchy, report: HierarchyWriteReport, root: Path
) -> None:
    """Show the shape that was written, per depth.

    Depth is printed rather than summarised because arbitrary depth is the feature: an
    operator pointing this at an unfamiliar corpus should see how deep it actually went.
    """
    table = Table(title=f"Hierarchy under {root}")
    table.add_column("depth", justify="right")
    table.add_column("folders", justify="right")
    table.add_column("example")
    for depth in range(hierarchy.max_depth + 1):
        names = sorted(folder.name for folder in hierarchy.folders_at(depth))
        table.add_row(str(depth), str(len(names)), names[0] if names else "")
    console.print(table)

    console.print(
        f"[bold green]{len(hierarchy.folders)} folders[/bold green] and "
        f"[bold green]{len(hierarchy.documents)} documents[/bold green] written to the "
        f"graph ({report.nodes_created} nodes and {report.relationships_created} "
        "relationships created this run"
        + ("; the structure was already in the graph)" if report.unchanged else ")")
    )
    if report.depths_corrected:
        console.print(
            f"[yellow]{report.depths_corrected} folder depths corrected[/yellow] — part "
            "of this tree had most likely been ingested from a different root."
        )
    console.print(
        "Browse it: [dim]MATCH p=(:Folder)-[:CONTAINS*]->(n) RETURN p LIMIT 200[/dim]"
    )


def _levels_hint(count: int) -> str:
    return ",".join(
        _HINT_NAMES[index] if index < len(_HINT_NAMES) else f"Ebene{index + 1}"
        for index in range(count)
    )


def _needs_model(
    samples: list[LevelSample], overrides: list[str], settings: Settings
) -> bool:
    """Whether any level is left for the model — and whether we can actually call it."""
    if len(samples) <= len(overrides):
        return False
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — cannot infer level names "
            f"for {len(samples) - len(overrides)} of {len(samples)} levels."
        )
        console.print(
            "The folder structure above is written; only the level names are missing. "
            "Set the key and re-run, or name every level explicitly — this corpus has "
            f'{len(samples)}: [dim]--levels "{_levels_hint(len(samples))}"[/dim]'
        )
        raise typer.Exit(code=1)
    return True


def _print_levels(
    samples: list[LevelSample], levels: list[Level], labelled: int
) -> None:
    """Report what each level was named, how sure the model was, and what it cost."""
    by_depth = {sample.depth: sample for sample in samples}
    table = Table(title="Hierarchy levels")
    table.add_column("depth", justify="right")
    table.add_column("level")
    table.add_column("confidence", justify="right")
    table.add_column("source")
    table.add_column("example folders")
    for level in levels:
        shaky = level.source == "inferred" and level.confidence < LOW_CONFIDENCE
        colour = "yellow" if shaky else "green"
        table.add_row(
            str(level.depth),
            f"[{colour}]{level.name}[/{colour}]",
            f"{level.confidence:.2f}",
            level.source,
            ", ".join(by_depth[level.depth].names[:3]),
        )
    console.print(table)
    console.print(f"{labelled} folder nodes labelled.")

    inferred = [level for level in levels if level.source == "inferred"]
    console.print(
        f"[dim]{len(inferred)} model call(s) — one per level, not per folder.[/dim]"
    )
    for level in levels:
        if level.note:
            console.print(
                f"[red]Level {level.depth} could not be named[/red]: {level.note}"
            )
    if any(level.confidence < LOW_CONFIDENCE for level in inferred):
        console.print(
            "[yellow]Some levels were inferred with low confidence.[/yellow] Override "
            "them before demoing, correcting whichever is wrong: "
            f'[dim]--levels "{",".join(level.name for level in levels)}"[/dim]'
        )


async def _ingest_contents(
    hierarchy: Hierarchy, settings: Settings, driver, cache: DiskCache
) -> ContentReport:
    """Parse, embed and write every discovered document.

    The HTTP client, the parser, the chunker and the embedding model are all built here
    — once, at the composition root — and injected. The client in particular must
    outlive the whole run: built per document, a hundred connection pools would be
    opened and abandoned.

    The parser and the embedding model are wrapped in the cache on the way down, so a
    document whose bytes have not changed is neither re-parsed nor re-embedded and this
    pass costs nothing on a re-run. Everything between the two — chunking — is pure and
    runs again every time, which is what lets the chunk parameters be changed without
    invalidating a single parse.
    """
    paths = [Path(document.path) for document in hierarchy.documents]
    async with httpx.AsyncClient(timeout=settings.parsing.timeout_seconds) as http_client:
        with _progress() as progress:
            task = progress.add_task(f"{_stage_label(CONTENT_STAGE)}…", total=len(paths))

            def advance(document: DocumentContent) -> None:
                progress.advance(task)
                if not document.ok:
                    progress.console.print(
                        f"[red]failed[/red] {document.path.name}: {document.error}"
                    )

            return await ingest_contents(
                paths,
                parser=make_parser(settings.parsing, http_client, cache),
                chunker=make_chunker(settings.chunking),
                embedder=embedder_for(settings, cache),
                driver=driver,
                database=settings.neo4j.database,
                concurrency=settings.parsing.concurrency,
                on_document=advance,
            )


def _check_embedding_width(driver, settings: Settings) -> None:
    """Refuse to embed a corpus the vector index cannot hold.

    Neo4j does not reject a vector of the wrong width — it accepts the write and then
    omits the node from every search. Catching the mismatch here costs one query;
    catching it later costs a demo.
    """
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — cannot embed chunks."
        )
        console.print("The folder structure above is written; set the key and re-run.")
        raise typer.Exit(code=1)

    configured = settings.mistral.embedding_dimensions
    index = next(
        (
            index
            for index in list_indexes(driver, settings.neo4j.database)
            if index["name"] == CHUNK_VECTOR_INDEX
        ),
        None,
    )
    if index is None:
        console.print(
            f"[bold red]No {CHUNK_VECTOR_INDEX} index[/bold red] — run `just init` first."
        )
        raise typer.Exit(code=1)
    width = vector_index_dimensions(index)
    if width != configured:
        console.print(
            f"[bold red]Vector index is {width}-dim but EMBEDDING_DIMENSIONS is "
            f"{configured}[/bold red]. Run `just init` to rebuild the index."
        )
        raise typer.Exit(code=1)


def _print_content(report: ContentReport, settings: Settings) -> None:
    """Report what was written, and prove the two indexes hold it."""
    written = len(report.documents) - len(report.failures)
    console.print(
        f"[bold green]{report.chunks} chunks[/bold green] from {written} of "
        f"{len(report.documents)} documents, embedded with "
        f"{settings.mistral.embedding_model}."
    )
    if report.removed:
        console.print(
            f"[dim]{report.removed} chunk(s) from a previous, longer version of a "
            "document removed.[/dim]"
        )
    for failure in report.failures:
        console.print(f"[red]no content for {failure.path}[/red]: {failure.error}")
    console.print(
        "Browse it: [dim]MATCH p=(:Document)-[:HAS_CHUNK]->(:Chunk)-[:NEXT*0..2]->() "
        "RETURN p LIMIT 100[/dim]"
    )


async def _summarise(
    hierarchy: Hierarchy,
    settings: Settings,
    driver,
    chat,
    *,
    refresh: bool,
    skip: set[str],
) -> SummaryReport:
    """Generate the missing document summaries.

    The cheap state read comes first and decides what the pass will actually cost: on a
    re-ingest of an unchanged corpus every document is already summarised at its current
    content hash, and the pass makes no model calls at all.

    `skip` holds the documents whose content pass failed this run. Their chunks are
    whatever the *previous* ingest left behind, while the structural pass has already
    written the new content hash — summarising them would describe the old text and
    then stamp the new hash on it, which is a stale summary that claims to be current.
    They are reported by the content pass and left alone here.
    """
    paths = [
        document.path for document in hierarchy.documents if document.path not in skip
    ]
    states = read_document_states(driver, settings.neo4j.database, paths)
    pending = sum(1 for state in states if needs_summary(state, refresh))
    if not pending:
        console.print(
            f"[dim]{len(states)} document summaries already current — no model calls.[/dim]"
        )
        return SummaryReport(documents=[])

    with _progress() as progress:
        task = progress.add_task(f"{_stage_label(SUMMARY_STAGE)}…", total=len(states))

        def advance(document: DocumentSummary) -> None:
            progress.advance(task)
            if not document.ok:
                progress.console.print(
                    f"[red]no summary[/red] {Path(document.path).name}: {document.error}"
                )

        return await summarise_documents(
            states,
            model=chat,
            store=GraphSummaryStore(driver, settings.neo4j.database),
            refresh=refresh,
            concurrency=settings.summaries.concurrency,
            max_chunks=settings.summaries.excerpt_chunks,
            max_characters=settings.summaries.excerpt_characters,
            max_summary_characters=settings.summaries.max_characters,
            on_document=advance,
        )


def _print_summaries(report: SummaryReport) -> None:
    """Report what the summary pass cost and what it could not do."""
    if not report.documents:
        return
    console.print(
        f"[bold green]{report.written} document summaries[/bold green] generated"
        + (f", {report.cached} already current" if report.cached else "")
        + " — one chat call per document, not per chunk."
    )
    for failure in report.failures:
        console.print(f"[red]no summary for {failure.path}[/red]: {failure.error}")
    console.print(
        "Compare two near-identical reports: "
        "[dim]just summaries --contains projektbericht-2023[/dim]"
    )


@app.command()
def summaries(
    contains: Annotated[
        str, typer.Option(help="Only documents whose path contains this text")
    ] = "",
    limit: Annotated[int, typer.Option(help="Documents to print")] = 10,
) -> None:
    """Print the stored document summaries.

    The acceptance test for this feature is a human reading two near-identical reports
    from different clients side by side and finding them distinguishable, so that
    comparison is one command rather than two hand-written Cypher queries.
    """
    settings = _settings()
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            coverage = summary_coverage(driver, settings.neo4j.database)
            stored = read_summaries(
                driver, settings.neo4j.database, contains=contains, limit=limit
            )
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    colour = "green" if coverage.summarised == coverage.total else "yellow"
    console.print(
        f"[{colour}]{coverage.summarised}/{coverage.total} documents summarised[/{colour}]"
        + (f", [yellow]{coverage.stale} out of date[/yellow]" if coverage.stale else "")
    )
    _print_stored_summaries(stored)


def _print_stored_summaries(stored: list[StoredSummary]) -> None:
    for document in stored:
        # The ancestry is what the summaries have to tell apart, so the path is printed
        # rather than the filename — two of these are called projektbericht-2023.pdf.
        console.print()
        console.print(f"[bold]{document.path}[/bold]")
        if document.stale:
            console.print("[yellow]  (describes an older version of this file)[/yellow]")
        console.print(f"  {document.summary or '[dim]no summary[/dim]'}")
    if not stored:
        console.print("[yellow]No documents matched.[/yellow]")


@dataclass(frozen=True)
class EntityOutcome:
    """What one extraction pass did, from the model call to the graph write."""

    extraction: ExtractionReport
    resolution: Resolution
    write: EntityWriteReport


async def _extract_entities(
    settings: Settings,
    driver,
    chat,
    cache: DiskCache,
    description: str = ENTITY_STAGE,
) -> EntityOutcome:
    """Turn the chunks already in the graph into a knowledge graph.

    Four steps and three of them cheap: one extraction-tier call per chunk, one
    embedding of each distinct name, resolution, and a single graph write. The
    ambiguity rule is read from the graph rather than from this run's walk, so a name
    is judged ambiguous against the whole archive even when the run covered one branch
    of it.

    The call per chunk is the dominant cost of an ingest, so the extraction model goes
    behind the cache here — keyed on the prompt as well as the chunk, which means an
    edited extraction prompt re-extracts on its own and an unedited one never does.
    Everything the reply is then turned into — the taxonomy filter, the scoping rule,
    the evidence spans — is pure and re-runs on every cache hit.
    """
    database = settings.neo4j.database
    sources = read_chunk_sources(driver, database)
    if not sources:
        return EntityOutcome(
            extraction=ExtractionReport(chunks=[]),
            resolution=Resolution(entities=[], canonical={}),
            write=EntityWriteReport(entities=0, mentions=0, relations=0, removed=0),
        )
    ambiguous = ambiguous_names(read_branch_folders(driver, database))

    with _progress() as progress:
        task = progress.add_task(f"{description}…", total=len(sources))

        def advance(chunk: ChunkResult) -> None:
            progress.advance(task)
            if not chunk.ok:
                progress.console.print(
                    f"[red]no entities[/red] {chunk.chunk_id}: {chunk.error}"
                )

        extraction = await extract_chunks(
            sources,
            model=CachingChatModel(
                make_extraction_model(settings.mistral),
                cache,
                signature=settings.mistral.extraction_model,
            ),
            ambiguous=ambiguous,
            concurrency=settings.entities.concurrency,
            evidence_limit=settings.entities.evidence_characters,
            on_chunk=advance,
        )

    with console.status("Resolving entities…"):
        resolution = await asyncio.to_thread(
            _resolve, extraction, settings, chat, embedder_for(settings, cache)
        )
    mentions, relations = canonical_edges(
        extraction.mentions, extraction.relations, resolution.canonical
    )
    write = await asyncio.to_thread(
        write_entities,
        driver,
        database,
        resolution.entities,
        mentions,
        relations,
        [source.id for source in sources],
    )
    return EntityOutcome(extraction=extraction, resolution=resolution, write=write)


def _resolve(
    extraction: ExtractionReport, settings: Settings, chat, embedder
) -> Resolution:
    """Embed each distinct name once, then run the three resolution stages.

    The blocking half: an embedding call and, for the ambiguous band alone, a chat
    call per pair. Run in a thread because both clients are synchronous.

    Both models arrive cached, so a re-run over an unchanged corpus re-embeds no name
    and re-adjudicates no pair — and because the thresholds are *not* part of either
    key, moving them re-reads the same vectors and re-decides the band for free, which
    is the whole point of being able to tune them.
    """
    distinct = candidate_entities(extraction.mentions)
    if not distinct:
        return Resolution(entities=[], canonical={})
    vectors = embedder.embed([named.surface for named in distinct])
    if len(vectors) != len(distinct):
        # Checked rather than zipped, as in `to_chunk_records`: `zip` would truncate and
        # leave the remaining entities carrying each other's vectors, which merges the
        # wrong things and reports success.
        raise EmbeddingError(
            f"{len(distinct)} entity names but {len(vectors)} embeddings — refusing to "
            "resolve entities that may carry each other's vectors"
        )
    candidates = [
        Candidate(
            identity=named.identity,
            surface=named.surface,
            embedding=vector,
            mentions=named.mentions,
        )
        for named, vector in zip(distinct, vectors, strict=True)
    ]
    return resolve_entities(
        candidates,
        Thresholds(
            merge=settings.entities.merge_threshold,
            review=settings.entities.review_threshold,
        ),
        adjudicate=model_adjudicator(chat) if settings.entities.adjudicate else None,
    )


def _print_entity_outcome(outcome: EntityOutcome, settings: Settings) -> None:
    """Report what the pass cost and — more importantly — what it merged.

    The merge counts are the operator's only window onto the riskiest decision in the
    project. Printed every run, because a threshold that has started collapsing
    distinct entities shows up here first and nowhere else until the demo.
    """
    extraction = outcome.extraction
    if not extraction.chunks:
        return
    resolution = outcome.resolution
    console.print(
        f"[bold green]{len(resolution.entities)} entities[/bold green] and "
        f"[bold green]{outcome.write.relations} relations[/bold green] from "
        f"{len(extraction.chunks) - len(extraction.failures)} of "
        f"{len(extraction.chunks)} chunks "
        f"({outcome.write.mentions} mentions, {settings.mistral.extraction_model})."
    )
    console.print(
        f"[dim]resolution: {resolution.exact} mention(s) folded in by name, "
        f"{resolution.merged} by similarity ≥ {settings.entities.merge_threshold}; "
        f"{resolution.adjudications} pair(s) put to {settings.mistral.chat_model} in "
        f"the {settings.entities.review_threshold}–{settings.entities.merge_threshold} "
        f"band, of which {resolution.adjudicated_merges} merged.[/dim]"
    )
    if outcome.write.removed:
        console.print(
            f"[dim]{outcome.write.removed} entity node(s) from a previous run no longer "
            "mentioned by any chunk, removed.[/dim]"
        )
    for failure in extraction.failures[:5]:
        console.print(f"[red]no entities for chunk {failure.chunk_id}[/red]: {failure.error}")
    if len(extraction.failures) > 5:
        console.print(f"[red]… and {len(extraction.failures) - 5} more chunks[/red]")
    console.print(
        "See the cross-client nodes: [dim]just entities --bridges[/dim]"
    )


@app.command()
def extract(
    no_cache: Annotated[
        bool,
        typer.Option(
            "--no-cache",
            help="Discard the ingestion cache first and re-extract from scratch",
        ),
    ] = False,
) -> None:
    """Extract entities and relations from the chunks already in the graph.

    The same pass `ingest` runs last, on its own. Re-running `ingest` to change an
    extraction prompt or a resolution threshold would re-parse and re-embed the whole
    corpus first — minutes of layout analysis to redo work that has not changed — and
    this stage is the one worth iterating on.

    An edited prompt re-extracts by itself, since the prompt is part of the cache key.
    `--no-cache` is for everything a key cannot see: a model whose answers have drifted
    under the same identifier, or an entry written by a version of this code that
    parsed replies differently. It discards the cached *replies* only — the parses this
    command does not even use are left alone, since refilling them costs an ingest.
    """
    settings = _settings()
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — cannot extract entities."
        )
        raise typer.Exit(code=1)
    cache = _open_cache(settings, discard=no_cache, namespaces=(COMPLETION,))
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            chat = CachingChatModel(
                make_chat_model(settings.mistral),
                cache,
                signature=settings.mistral.chat_model,
            )
            outcome = asyncio.run(_extract_entities(settings, driver, chat, cache))
    except EmbeddingError as error:
        console.print(f"[bold red]Embedding failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    if not outcome.extraction.chunks:
        console.print(
            "[bold red]No chunks in the graph[/bold red] — run `just ingest corpus` first."
        )
        raise typer.Exit(code=1)
    _print_entity_outcome(outcome, settings)
    _print_cache(cache)
    if outcome.extraction.failures:
        raise typer.Exit(code=1)


@app.command()
def entities(
    contains: Annotated[
        str, typer.Option(help="Only entities whose name contains this text")
    ] = "",
    entity_type: Annotated[
        str, typer.Option("--type", help=f"One of: {', '.join(ENTITY_TYPES)}")
    ] = "",
    bridges: Annotated[
        bool,
        typer.Option(
            "--bridges",
            help="Only entities named under more than one top-level folder",
        ),
    ] = False,
    limit: Annotated[int, typer.Option(help="Rows to print")] = 20,
) -> None:
    """Print the extracted entities, or just the ones joining two branches.

    `--bridges` is the acceptance test for this whole stage: a consultant who worked
    for two clients must appear once, joining both subtrees. Two rows for one person
    means resolution is too strict; one row covering two different clients' projects
    means it is too loose.
    """
    settings = _settings()
    if entity_type and entity_type not in ENTITY_TYPES:
        console.print(
            f"[bold red]Unknown entity type[/bold red] '{entity_type}'. The taxonomy is "
            f"fixed: {', '.join(ENTITY_TYPES)}."
        )
        raise typer.Exit(code=1)
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            coverage = entity_coverage(driver, settings.neo4j.database)
            total_bridges = count_bridges(driver, settings.neo4j.database)
            found = read_bridges(
                driver, settings.neo4j.database, contains=contains, limit=limit
            )
            stored = (
                []
                if bridges
                else read_entities(
                    driver,
                    settings.neo4j.database,
                    contains=contains,
                    entity_type=entity_type,
                    limit=limit,
                )
            )
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    _print_entity_coverage(coverage)
    if bridges:
        _print_bridges(found)
    else:
        _print_stored_entities(stored)
        console.print(
            f"[dim]{total_bridges} entities are named under more than one top-level "
            "folder — see `just entities --bridges`.[/dim]"
        )


def _print_entity_coverage(coverage) -> None:
    colour = "green" if coverage.chunks_with_entities else "yellow"
    types = ", ".join(
        f"{count} {name}" for name, count in sorted(coverage.by_type.items())
    )
    console.print(
        f"[{colour}]{coverage.total} entities[/{colour}] "
        f"({types or 'none'}), {coverage.relations} relations, named by "
        f"{coverage.chunks_with_entities}/{coverage.chunks} chunks."
    )
    # The project's single most important modelling rule, asserted on every run rather
    # than trusted: an entity owned by a folder would make the cross-client path a
    # query-time join rather than real structure, and nothing on screen would show it.
    if coverage.folder_owned:
        console.print(
            f"[bold red]{coverage.folder_owned} entities carry a folder edge[/bold red] "
            "— entities are global and must never be attached to a folder."
        )


def _print_bridges(bridges: list[BridgeEntity]) -> None:
    table = Table(title="Entities joining more than one top-level folder")
    table.add_column("entity")
    table.add_column("type")
    table.add_column("mentions", justify="right")
    table.add_column("branches")
    for bridge in bridges:
        table.add_row(
            bridge.name,
            bridge.type,
            str(bridge.mentions),
            ", ".join(bridge.branches),
        )
    console.print(table)
    if not bridges:
        console.print(
            "[yellow]No entity joins two branches.[/yellow] Either nothing has been "
            "extracted yet, or resolution is too strict — the corpus plants a "
            "consultant and an artefact that cross from one client to another."
        )


def _print_stored_entities(stored: list[StoredEntity]) -> None:
    table = Table(title="Entities")
    table.add_column("entity")
    table.add_column("type")
    table.add_column("scope")
    table.add_column("mentions", justify="right")
    table.add_column("edges", justify="right")
    table.add_column("merged spellings")
    for entity in stored:
        table.add_row(
            entity.name,
            entity.type,
            entity.scope or "[dim]global[/dim]",
            str(entity.mentions),
            str(entity.relations),
            ", ".join(entity.aliases[:3]) or "[dim]-[/dim]",
        )
    console.print(table)
    if not stored:
        console.print("[yellow]No entities matched.[/yellow]")


@app.command()
def search(
    question: Annotated[str, typer.Argument(help="Question to retrieve chunks for")],
    top_k: Annotated[int, typer.Option(min=1, help="Results per leg")] = DEFAULT_TOP_K,
    mode: Annotated[
        Mode, typer.Option(help="Which searches to run")
    ] = Mode.HYBRID,
    preview: Annotated[int, typer.Option(help="Characters of chunk text shown")] = 70,
) -> None:
    """Run the lexical-and-semantic retrieval leg and print the ranked chunks.

    No generated answer — this is the leg on its own, which is a demo feature as much
    as a debugging aid: `--mode vector` is how you show an audience that dense
    retrieval alone brings back the wrong client's report, and the default hybrid run
    is how you show the full-text index rescuing it.
    """
    settings = _settings()
    # Only the semantic half needs a key, so the lexical leg stays demonstrable
    # without one — useful when the API is the thing that is down.
    if mode.runs(Leg.VECTOR) and not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — cannot embed the "
            "question. `--mode fulltext` needs no key."
        )
        raise typer.Exit(code=1)

    cache = _open_cache(settings, discard=False)
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            coverage = embedding_coverage(driver, settings.neo4j.database)
            result = search_chunks(
                question=question,
                index=GraphChunkIndex(driver, settings.neo4j.database),
                embedder=embedder_for(settings, cache),
                top_k=top_k,
                mode=mode,
            )
    except EmbeddingError as error:
        console.print(f"[bold red]Embedding failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    except ClientError as error:
        # Narrowed to a failure *inside* the search procedure: a ClientError is also
        # how a missing database arrives, and `_graph_failure` says something far more
        # useful about that one than a note about Lucene syntax would.
        if "ProcedureCallFailed" not in (error.code or ""):
            raise _graph_failure(settings, error) from error
        # Every special character is escaped before the query is sent, so a parse
        # error here is this project's bug and not the operator's phrasing.
        console.print(
            f"[bold red]The full-text index rejected the query[/bold red]: {error}"
        )
        console.print(f"Query sent: {lucene_query(question)}")
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    # Only when it is news. `status` is where coverage is reported; repeating it on
    # every search would bury the one line this command exists to print — but a leg
    # that retrieved nothing because half the corpus is unembedded must say so here,
    # not leave the operator guessing at the question.
    if not coverage.matches(settings.mistral.embedding_dimensions) or not result.results:
        _print_coverage(coverage, settings.mistral.embedding_dimensions)
    _print_retrieval(result, mode, preview)


def _ancestry_label(chunk: Chunk) -> str:
    """`Kunde: Nordlicht Energie AG > Jahr: 2023 > Projekt: Projekt Atlas`.

    The whole point of carrying ancestry rather than a path: this is a sentence an
    answer can cite, and `corpus/Nordlicht Energie AG/2023/Projekt Atlas` is not.
    """
    return " › ".join(ancestry_lines(chunk.ancestry)) or chunk.filename


def _print_retrieval(result: LegResult, mode: Mode, preview: int) -> None:
    if result.query:
        console.print(f"[dim]Volltext-Query: {result.query}[/dim]")
    elif mode.runs(Leg.FULLTEXT):
        # Otherwise an empty lexical leg reads as an index that is not answering.
        console.print(
            "[yellow]Nothing in that question is searchable as text[/yellow] — the "
            "full-text leg was skipped."
        )
    table = Table(title=f"{mode.value} — {len(result.results)} chunks")
    table.add_column("#", justify="right")
    table.add_column("rrf", justify="right")
    for leg in mode.legs:
        table.add_column(_LEG_COLUMNS[leg], justify="right")
    table.add_column("Ablage")
    table.add_column("Seite", justify="right")
    table.add_column("Auszug")
    for position, found in enumerate(result.results, start=1):
        table.add_row(
            str(position),
            f"{found.score:.4f}",
            *(_rank_cell(found.rank_in(leg)) for leg in mode.legs),
            _ancestry_label(found.chunk),
            str(found.chunk.page) if found.chunk.page is not None else "-",
            found.chunk.text[:preview].replace("\n", " "),
        )
    console.print(table)
    _print_branches([found.chunk for found in result.results])
    _print_leg_contribution(result, mode)


#: The rank columns, narrow enough to leave the terminal's width to the ancestry.
_LEG_COLUMNS = {Leg.VECTOR: "vec", Leg.FULLTEXT: "txt"}


def _rank_cell(rank: int | None) -> str:
    """A leg that never returned this chunk, said as much rather than left blank."""
    return str(rank) if rank is not None else "[dim]-[/dim]"


def _print_leg_contribution(result: LegResult, mode: Mode) -> None:
    """What each leg contributed to the top result — the line the demo is about.

    Recorded every run rather than measured once by hand: whether dense retrieval alone
    would have found the right document is the project's central claim, and a claim
    that is re-derived on every question cannot go stale.
    """
    if not result.results:
        # Not when the lexical leg was the only one asked and had nothing to ask with
        # — the reason was already printed, and the corpus is not the suspect.
        if result.query or mode.runs(Leg.VECTOR):
            console.print(
                "[yellow]Nothing retrieved.[/yellow] Has the corpus been ingested?"
            )
        return
    if mode is not Mode.HYBRID:
        return
    top = result.results[0]
    vector_rank = top.rank_in(Leg.VECTOR)
    console.print(
        f"Top: [bold]{_ancestry_label(top.chunk)}[/bold] — "
        f"vektorisch {_place(vector_rank)}, "
        f"lexikalisch {_place(top.rank_in(Leg.FULLTEXT))}."
    )
    semantic = result.found_by(Leg.VECTOR)
    if vector_rank is None:
        console.print(
            "[bold yellow]Vector-only retrieval would have missed this chunk "
            "entirely.[/bold yellow] This is the case for the full-text index."
        )
    elif vector_rank > 1 and semantic:
        console.print(
            f"[yellow]Vector-only would have put "
            f"{_ancestry_label(semantic[0].chunk)} first instead.[/yellow]"
        )
    if not all(result.found_by(leg) for leg in mode.legs):
        console.print(
            "[yellow]One of the two searches returned nothing.[/yellow] The lexical "
            "leg is not optional on this corpus — see the risks in the README."
        )


def _place(rank: int | None) -> str:
    return f"Platz {rank}" if rank is not None else "nicht gefunden"


def _print_coverage(coverage, dimensions: int) -> None:
    widths = ", ".join(f"{width}-dim" for width in coverage.widths) or "none"
    colour = "green" if coverage.matches(dimensions) else "red"
    console.print(
        f"[{colour}]{coverage.embedded}/{coverage.total} chunks embedded[/{colour}] "
        f"({widths}; index expects {dimensions}-dim)"
    )


@app.command()
def traverse(
    question: Annotated[str, typer.Argument(help="Question to traverse the graph from")],
    hops: Annotated[
        int,
        typer.Option(
            min=0,
            help="Relation edges walked outward from the matched entities",
        ),
    ] = DEFAULT_HOPS,
    top_k: Annotated[int, typer.Option(min=1, help="Chunks returned")] = DEFAULT_TOP_K,
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help=(
                "Cosine similarity floor for matching a question entity by embedding. "
                "Defaults to TRAVERSAL_MATCH_THRESHOLD."
            ),
        ),
    ] = None,
    preview: Annotated[int, typer.Option(help="Characters of chunk text shown")] = 60,
) -> None:
    """Run the entity traversal leg and print the path it walked.

    The leg the project exists for, on its own: the chunks it returns are reached
    through the graph rather than through any resemblance to the question, which is why
    `just search` on the same question finds a different — and on a cross-client
    question, a worse — set. No answer is generated; `just query` does that.

    `--hops` is the knob to turn live. One hop is precise, two is where cross-document
    connections appear, three is noise.
    """
    settings = _settings()
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — this leg reads the "
            "entities out of the question and embeds them, and needs a key for both."
        )
        raise typer.Exit(code=1)

    cache = _open_cache(settings, discard=False)
    try:
        with graph_driver(settings.neo4j) as driver:
            driver.verify_connectivity()
            result = traverse_entities(
                question=question,
                graph=GraphEntityGraph(driver, settings.neo4j.database),
                embedder=embedder_for(settings, cache),
                model=CachingChatModel(
                    make_extraction_model(settings.mistral),
                    cache,
                    signature=settings.mistral.extraction_model,
                ),
                hops=hops,
                top_k=top_k,
                threshold=(
                    settings.traversal.match_threshold if threshold is None else threshold
                ),
                candidates=settings.traversal.candidates,
            )
    except EmbeddingError as error:
        console.print(f"[bold red]Embedding failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error

    _print_traversal(result, preview)
    _print_cache(cache)


def _print_traversal(result: TraversalResult, preview: int) -> None:
    """The leg's whole path, in the order it was walked.

    Printed in full rather than summarised: this command is the one place the walk can
    be inspected before an audience sees the same path drawn as a picture, and a leg
    that reports only its chunks cannot be checked for having started in the right
    place.
    """
    if not _print_path(result):
        return
    _print_traversed(result, preview)
    _print_branches(
        [found.chunk for found in result.results],
        hint=" Mehr Hops erreichen mehr.",
        via_graph=True,
    )


def _print_matches(matches: list[MatchedEntity]) -> None:
    table = Table(title="Einstiegspunkte im Graphen")
    table.add_column("Frage")
    table.add_column("Knoten")
    table.add_column("Typ")
    table.add_column("Kunde")
    table.add_column("über")
    table.add_column("Ähnlichkeit", justify="right")
    for match in sorted(matches, key=lambda found: (found.question.surface, found.entity.name)):
        table.add_row(
            match.question.surface,
            match.entity.name,
            match.entity.identity.type,
            match.entity.scope or "[dim]global[/dim]",
            "+".join(way.value for way in match.how),
            f"{match.similarity:.3f}" if match.similarity is not None else "[dim]-[/dim]",
        )
    console.print(table)
    # The planted collision, named when it happens: one question entity standing on two
    # scoped nodes is the hierarchy's verdict that two clients run the same-named
    # project, and it is the precondition for the cross-client answer below.
    scoped = [match for match in matches if match.entity.scope]
    if len({match.entity.identity.normalized_name for match in scoped}) < len(scoped):
        console.print(
            "[bold]One name in the question stands on more than one client's "
            "node.[/bold] The walk starts at every one of them."
        )


def _print_walk(result: TraversalResult) -> None:
    """Each hop as a line: how far out, how many edges, where it got to."""
    for hop in range(1, result.hops + 1):
        edges = [edge for edge in result.edges if edge.hop == hop]
        arrived = [reached for reached in result.reached if reached.hop == hop]
        if not edges:
            break
        names = ", ".join(
            sorted({reached.entity.name for reached in arrived})[:6]
        ) or "[dim]nothing new[/dim]"
        console.print(
            f"Hop {hop}: {len(edges)} Kanten → {len(arrived)} neue Entitäten — {names}"
        )


def _print_traversed(result: TraversalResult, preview: int) -> None:
    table = Table(title=f"{len(result.results)} Chunks über {result.hops} Hops")
    table.add_column("#", justify="right")
    table.add_column("Pfad", justify="right")
    table.add_column("Hop", justify="right")
    table.add_column("Ablage")
    table.add_column("Entitäten")
    table.add_column("Auszug")
    for position, found in enumerate(result.results, start=1):
        table.add_row(
            str(position),
            f"{found.score:.2f}",
            str(found.hop),
            _ancestry_label(found.chunk),
            ", ".join(reached.entity.name for reached in found.via[:3]),
            found.chunk.text[:preview].replace("\n", " "),
        )
    console.print(table)
    if not result.results:
        console.print(
            "[yellow]The matched entities are mentioned by nothing.[/yellow] Has "
            "`just extract` run since the last ingest?"
        )


def _print_branches(
    chunks: list[Chunk], hint: str = "", *, via_graph: bool = False
) -> None:
    """How many clients' folders a leg's results came from.

    Printed by *both* legs, which is what makes them comparable at a glance: the
    traversal leg's case is that it answers from folders the lexical leg never reaches,
    and two commands reporting the same number is how that is checked rather than
    asserted. Re-derived every run — a claim recomputed on each question cannot go
    stale.

    `via_graph` is why the count is reported and not what causes it. The line *reads*
    as the project's thesis, so only the walk may claim it: a lexical-only run that
    happened to land on two clients because both reports share a word would otherwise
    print the graph's case for it, in green, with nothing having been traversed. That
    is the one defect in this file that would mislead the audience rather than the
    operator.
    """
    branches = sorted({name for name in (branch(chunk.ancestry) for chunk in chunks) if name})
    if len(branches) > 1:
        console.print(
            f"[bold green]Die Chunks stammen aus {len(branches)} Kundenablagen[/bold green]: "
            + ", ".join(branches)
            + (
                " — verbunden über den Graphen, nicht über gemeinsame Wörter."
                if via_graph
                else " — gefunden über gemeinsame Wörter, ohne den Graphen."
            )
        )
    elif branches:
        console.print(f"[dim]Alle Chunks aus einer Kundenablage ({branches[0]}).{hint}[/dim]")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to answer")],
    legs: Annotated[
        Legs, typer.Option(help="Which retrieval legs run")
    ] = Legs.BOTH,
    mode: Annotated[
        Mode, typer.Option(help="Which searches the lexical leg runs")
    ] = Mode.HYBRID,
    top_k: Annotated[
        int, typer.Option(min=1, help="Results per retrieval leg, before the merge")
    ] = DEFAULT_TOP_K,
    hops: Annotated[
        int, typer.Option(min=0, help="Relation edges walked from the matched entities")
    ] = DEFAULT_HOPS,
    max_chunks: Annotated[
        int, typer.Option(min=1, help="Chunks in the context handed to the model")
    ] = DEFAULT_MAX_CHUNKS,
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help=(
                "Cosine similarity floor for matching a question entity by embedding. "
                "Defaults to TRAVERSAL_MATCH_THRESHOLD."
            ),
        ),
    ] = None,
    preview: Annotated[int, typer.Option(help="Characters of chunk text shown")] = 60,
    no_clarify: Annotated[
        bool,
        typer.Option(
            "--no-clarify",
            help="Answer even when the question names something the corpus cannot resolve",
        ),
    ] = False,
) -> None:
    """Answer a question from both retrieval legs, with its sources and its path.

    The whole system in one command: both legs run, their chunks are merged and capped,
    and the chat tier answers in German citing the client, year and project each fact
    came from.

    Every tuning parameter is here because every one of them is turned during a demo.
    `--legs search` and `--legs traversal` answer the same question from one leg alone,
    which is how an audience sees what the graph contributes rather than being told;
    `--legs search --mode vector` is the nearest thing to the flat-RAG baseline.
    """
    settings = _settings()
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — answering embeds the "
            "question and generates prose, and needs a key for both. `just search "
            "--mode fulltext` retrieves without one."
        )
        raise typer.Exit(code=1)

    cache = _open_cache(settings, discard=False)
    parameters = Parameters(
        legs=legs, mode=mode, top_k=top_k, hops=hops, max_chunks=max_chunks
    )
    with _answering(settings, cache, question, threshold) as pipeline:
        clarification = None if no_clarify else pipeline.clarify(question)
        if clarification is not None:
            _print_clarification(clarification)
            return
        answer = _ask(pipeline, question, parameters)

    _print_query(answer, preview)
    _print_cache(cache)


def _print_clarification(clarification: Clarification) -> None:
    """The question put back, with what the corpus actually holds.

    Printed instead of an answer, not alongside one: the point is that the question
    cannot be answered as asked, and printing a guess underneath would undo that.
    `--no-clarify` is there for the operator who wants to see the guess anyway.
    """
    console.print(f"[bold yellow]{clarification.question}[/bold yellow]")
    table = Table(box=None, pad_edge=False)
    table.add_column(clarification.choices[0].level or "Treffer", style="bold")
    table.add_column("Ablage")
    for choice in clarification.choices:
        table.add_row(choice.name, choice.context)
    console.print(table)
    console.print(
        "[dim]Die Frage mit einem dieser Namen erneut stellen — oder mit "
        "`--no-clarify` trotzdem antworten lassen.[/dim]"
    )


@contextmanager
def _answering(
    settings: Settings, cache: DiskCache, question: str, threshold: float | None
) -> Iterator[Pipeline]:
    """The answer path, open for as long as it is asked — with its failures as prose.

    The wiring itself is `pipeline.open_pipeline`, shared verbatim with the HTTP
    service so that a question asked from the browser runs the code a question asked
    from the terminal runs. What is left here is the part that is genuinely the CLI's:
    turning a dead Neo4j or a rejected Lucene query into a message and an exit code
    rather than into a traceback in front of an audience.

    `question` is only here for the diagnostics — a rejected Lucene query is reported
    with the query that was sent — and a session that asks several is reported against
    the one it was opened for, which is close enough to find the escaping bug that a
    parse error here would mean.
    """
    try:
        with open_pipeline(settings, cache, threshold) as pipeline:
            yield pipeline
    except EmbeddingError as error:
        console.print(f"[bold red]Embedding failed[/bold red]: {error}")
        raise typer.Exit(code=1) from error
    except ClientError as error:
        if "ProcedureCallFailed" not in (error.code or ""):
            raise _graph_failure(settings, error) from error
        console.print(
            f"[bold red]The full-text index rejected the query[/bold red]: {error}"
        )
        console.print(f"Query sent: {lucene_query(question)}")
        raise typer.Exit(code=1) from error
    except (Neo4jError, DriverError) as error:
        raise _graph_failure(settings, error) from error


def _ask(pipeline: Pipeline, question: str, parameters: Parameters) -> Answer:
    """One question, with the spinner an operator needs while two legs run."""
    with console.status(f"{parameters.legs.value}: Abruf und Antwort…"):
        return pipeline.ask(question, parameters)


def _print_query(answer: Answer, preview: int) -> None:
    """The answer, and everything that produced it.

    Printed in this order deliberately: the path first, the sources next, the prose
    last. The answer is what the audience reads, so it ends up at the bottom of the
    terminal where it stays on screen — and it arrives *after* the evidence for it,
    which is the argument this demo is making.
    """
    if answer.traversal is not None:
        _print_path(answer.traversal)
    _print_sources(answer, preview)
    _print_leg_balance(answer)
    _print_branches(
        [found.chunk for found in answer.context],
        # Only the walk may claim the graph joined two clients' folders, and only when
        # a chunk it reached is actually in the context the answer was written from.
        via_graph=any(found.found_by(Source.TRAVERSAL) for found in answer.context),
    )
    _print_answer_text(answer)


def _print_path(result: TraversalResult) -> bool:
    """Where the walk started and how far it got — or why it never started.

    Returns whether it started, so the caller prints what it reached only when there
    was a walk. Shared with `traverse` rather than written twice: a question that
    returns nothing has two quite different causes — nothing named, or nothing matched
    — and an answer path that printed an empty table for both would send the operator
    to the wrong knob mid-demo.
    """
    if not result.entities:
        console.print(
            "[yellow]No named entity in that question[/yellow] — this leg has nothing "
            "to start from. It answers questions that name a person, a project or an "
            "artefact; the lexical leg answers the rest."
        )
        return False
    console.print(
        "Erkannte Entitäten: "
        + ", ".join(
            f"[bold]{entity.surface}[/bold] ({entity.type})"
            for entity in result.entities
        )
    )
    if not result.matches:
        console.print(
            "[yellow]None of them is in the graph.[/yellow] Either the corpus does not "
            "name them, or --threshold is too high for the way the question spells them."
        )
        return False
    _print_matches(result.matches)
    _print_walk(result)
    return True


def _print_sources(answer: Answer, preview: int) -> None:
    """The numbered context, in the order the model was given it.

    The numbers are the citation scheme: `[Quelle 3]` in the answer is row 3 here, so
    a claim can be checked against the chunk it came from without leaving the terminal.
    """
    cited = set(answer.citations)
    table = Table(title=f"Kontext — {len(answer.context)} Chunks")
    table.add_column("Quelle", justify="right")
    table.add_column("zitiert", justify="center")
    table.add_column("rrf", justify="right")
    for source in answer.parameters.legs.sources:
        table.add_column(_SOURCE_COLUMNS[source], justify="right")
    table.add_column("Ablage")
    table.add_column("Seite", justify="right")
    table.add_column("Auszug")
    for position, found in enumerate(answer.context, start=1):
        table.add_row(
            str(position),
            "[green]✓[/green]" if position in cited else "[dim]-[/dim]",
            f"{found.score:.4f}",
            *(
                _rank_cell(found.rank_in(source))
                for source in answer.parameters.legs.sources
            ),
            _ancestry_label(found.chunk),
            str(found.chunk.page) if found.chunk.page is not None else "-",
            found.chunk.text[:preview].replace("\n", " "),
        )
    console.print(table)
    if not answer.context:
        # Which legs were asked, not "nothing was found": with `--legs traversal` the
        # usual cause is a question naming nothing the graph holds, and the remedy is
        # the other leg rather than a re-ingest.
        if answer.parameters.legs is Legs.BOTH:
            console.print(
                "[yellow]Neither leg retrieved anything.[/yellow] Has the corpus been "
                "ingested, and has `just extract` run since?"
            )
        else:
            console.print(
                f"[yellow]The {answer.parameters.legs.value} leg retrieved "
                "nothing[/yellow], and it was the only one asked — `--legs both` runs "
                "the other one too."
            )


#: Narrow, to leave the terminal's width to the ancestry — which is the column the
#: whole project is about.
_SOURCE_COLUMNS = {Source.SEARCH: "such", Source.TRAVERSAL: "graph"}


def _print_leg_balance(answer: Answer) -> None:
    """What each leg contributed to the context the answer was written from.

    The demo's central claim, re-derived on every question rather than measured once:
    a chunk in the context that the lexical leg never returned is a chunk the graph
    found, and it is the only honest way to say so without a flat-RAG baseline to put
    beside it.
    """
    if not answer.context or answer.parameters.legs is not Legs.BOTH:
        return
    graph_only = [
        found
        for found in answer.context
        if found.found_by(Source.TRAVERSAL) and not found.found_by(Source.SEARCH)
    ]
    both = [found for found in answer.context if len(found.ranks) == 2]
    console.print(
        f"Kontext: {len(answer.context)} Chunks — "
        f"{len([f for f in answer.context if f.found_by(Source.SEARCH)])} über die "
        f"Suche, {len([f for f in answer.context if f.found_by(Source.TRAVERSAL)])} "
        f"über den Graphen, {len(both)} über beide."
    )
    if graph_only:
        console.print(
            f"[bold green]{len(graph_only)} Chunks hätte die Suche allein nicht "
            f"gefunden[/bold green]: "
            + ", ".join(sorted({_ancestry_label(found.chunk) for found in graph_only}))
        )
    elif answer.traversal is not None and answer.traversal.results:
        console.print(
            "[dim]Beide Beine fanden dieselben Chunks — diese Frage braucht den "
            "Graphen nicht.[/dim]"
        )


def _print_answer_text(answer: Answer) -> None:
    console.print(
        Panel(
            answer.text or "[dim](keine Antwort)[/dim]",
            title=f"[bold]{answer.question}[/bold]",
            border_style="green" if answer.grounded else "yellow",
        )
    )
    if answer.uncited:
        # The model citing a source that does not exist has stopped quoting the context,
        # and an invented citation reads exactly like a real one.
        console.print(
            "[bold red]Die Antwort zitiert Quellen, die es nicht gibt[/bold red]: "
            + ", ".join(f"Quelle {number}" for number in answer.uncited)
            + " — die Antwort ist an dieser Stelle nicht belegt."
        )
    elif answer.grounded and not answer.citations:
        console.print(
            "[yellow]Die Antwort zitiert keine Quelle.[/yellow] Sie ist damit nicht "
            "gegen den Kontext prüfbar."
        )


@app.command()
def chat(
    opening: Annotated[
        str | None,
        typer.Argument(help="First question; omit to be prompted for it"),
    ] = None,
    legs: Annotated[Legs, typer.Option(help="Which retrieval legs run")] = Legs.BOTH,
    mode: Annotated[
        Mode, typer.Option(help="Which searches the lexical leg runs")
    ] = Mode.HYBRID,
    top_k: Annotated[
        int, typer.Option(min=1, help="Results per retrieval leg, before the merge")
    ] = DEFAULT_TOP_K,
    hops: Annotated[
        int, typer.Option(min=0, help="Relation edges walked from the matched entities")
    ] = DEFAULT_HOPS,
    max_chunks: Annotated[
        int, typer.Option(min=1, help="Chunks in the context handed to the model")
    ] = DEFAULT_MAX_CHUNKS,
    window: Annotated[
        int, typer.Option(min=0, help="Recent turns a follow-up is resolved against")
    ] = DEFAULT_WINDOW,
    threshold: Annotated[
        float | None,
        typer.Option(
            min=0.0,
            max=1.0,
            help=(
                "Cosine similarity floor for matching a question entity by embedding. "
                "Defaults to TRAVERSAL_MATCH_THRESHOLD."
            ),
        ),
    ] = None,
    preview: Annotated[int, typer.Option(help="Characters of chunk text shown")] = 60,
) -> None:
    """Ask follow-up questions, resolved against the conversation so far.

    `query` with a memory. Every turn after the first is rewritten into a standalone
    question before retrieval runs — "und 2023?" becomes "Was geschah 2023 im Projekt
    Atlas der Nordlicht Energie AG?" — and the rewrite is printed, because watching it
    happen is the point of the feature and because a follow-up that answers oddly is
    almost always a rewrite that went somewhere unexpected.

    An empty line or Ctrl-D ends the session. `--window 0` turns the memory off, which
    is how to show the same follow-up failing.
    """
    settings = _settings()
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — a conversation rewrites "
            "each follow-up and generates each answer, and needs a key for both."
        )
        raise typer.Exit(code=1)

    cache = _open_cache(settings, discard=False)
    parameters = Parameters(
        legs=legs, mode=mode, top_k=top_k, hops=hops, max_chunks=max_chunks
    )
    conversation = Conversation(window=window)
    console.print(
        "[dim]Rückfragen werden gegen den Verlauf aufgelöst. Leere Zeile oder "
        "Strg-D beendet.[/dim]"
    )
    with _answering(settings, cache, opening or "", threshold) as pipeline:
        # An empty opening is no opening, not an empty question: the task runner passes
        # `''` when it was given no question, and treating that as one would end the
        # session on the spot instead of prompting for the first one.
        asked = opening or None
        while True:
            if asked is None:
                asked = _next_question()
            if not asked:
                break
            # The two lines this ticket is about, in this order: the rewrite happens
            # *before* the pipeline and the pipeline is handed a question it cannot
            # tell apart from a first one.
            rewrite = pipeline.rewrite(asked, conversation)
            answer = _ask(pipeline, rewrite.standalone, parameters)

            _print_rewrite(rewrite)
            _print_query(answer, preview)
            conversation = conversation.extend(Turn(rewrite, answer).exchange)
            asked = None
    _print_cache(cache)


def _next_question() -> str:
    """The next question from the operator, or `""` when the session is over.

    Ctrl-D and Ctrl-C end a conversation rather than raising: this runs in front of an
    audience, and a traceback is a worse ending than a prompt that simply stops.
    """
    try:
        return console.input("\n[bold]Frage[/bold] › ").strip()
    except (EOFError, KeyboardInterrupt):
        console.print()
        return ""


def _print_rewrite(rewrite: Rewrite) -> None:
    """What the follow-up was resolved to, when it was resolved to anything.

    Silent when nothing changed — which is the common case and the correct one for an
    already-standalone question — so that the line appearing means something happened,
    rather than becoming a row the operator learns to ignore.
    """
    if rewrite.changed:
        console.print(
            f"[dim]Rückfrage aufgelöst:[/dim] [italic]{rewrite.asked}[/italic] → "
            f"[bold]{rewrite.standalone}[/bold]"
        )
        return
    # A rewrite that was attempted and came back unusable is not the same as one that
    # was never attempted, and only the first is worth interrupting the demo for: it
    # means the next follow-up will be answered as if it stood alone.
    if rewrite.note and rewrite.note != NOTHING_TO_RESOLVE:
        console.print(
            f"[yellow]Die Rückfrage wurde nicht aufgelöst[/yellow] ({rewrite.note}) — "
            "sie wird so beantwortet, wie sie gestellt wurde."
        )


@app.command()
def serve(
    host: Annotated[
        str | None, typer.Option(help="Bind host. Defaults to API_HOST.")
    ] = None,
    port: Annotated[
        int | None, typer.Option(help="Bind port. Defaults to API_PORT.")
    ] = None,
) -> None:
    """Serve the HTTP API: the same pipeline, plus the subgraph behind each answer.

    The service is the frontend's half of the demo — `POST /api/ask` answers a
    question and returns the nodes and edges that produced it, with the edges that
    cross between branches of the hierarchy already flagged. `GET /api/status` is the
    operator's view: what has been ingested, and what each level of the tree means.

    The graph driver and both model tiers are opened once at startup and held for the
    life of the process, so the first question costs no more than the tenth.
    """
    settings = _settings()
    if not settings.mistral.api_key:
        console.print(
            "[bold red]MISTRAL_API_KEY is not set[/bold red] — the service embeds "
            "every question and generates every answer, and needs a key for both."
        )
        raise typer.Exit(code=1)

    # Imported here rather than at module scope: FastAPI and uvicorn are a second of
    # import time that `just status` has no reason to pay.
    import uvicorn

    from graph_rag.api import create_app

    where = host or settings.service.host
    listening = port or settings.service.port
    console.print(
        f"[bold]Serving[/bold] on http://{where}:{listening} — "
        f"docs at http://{where}:{listening}/docs"
    )
    console.print(
        f"[dim]Graph: {settings.neo4j.uri}; CORS open to "
        f"{', '.join(settings.service.origins) or 'nothing'}.[/dim]"
    )
    # The app is built here and handed over as an object rather than as an import
    # string: the settings are already loaded and validated, and an import string
    # would make uvicorn load them a second time in its own process.
    uvicorn.run(create_app(settings), host=where, port=listening)


if __name__ == "__main__":
    app()
