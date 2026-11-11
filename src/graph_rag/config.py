"""Environment-backed configuration.

Every identifier and endpoint is read from the environment; nothing is hardcoded at a
call site. Settings are loaded once at the composition root (the CLI) and passed down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class MistralSettings:
    api_key: str
    extraction_model: str
    chat_model: str
    embedding_model: str
    embedding_dimensions: int
    #: Inputs per embedding request. Embedding is the first per-chunk API cost in the
    #: pipeline, so chunks go up in batches rather than one call each.
    embedding_batch_size: int
    #: Second cap on the same batch. The API rejects a request whose inputs exceed a
    #: total token budget ("Too many tokens overall, split into more batches"), which
    #: a count alone cannot prevent on long chunks. Characters stand in for tokens
    #: because German token density varies and the budget is generous either way.
    embedding_batch_characters: int


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str


@dataclass(frozen=True)
class ParsingSettings:
    """Self-hosted Unstructured transport. Defaults match the production service."""

    url: str
    strategy: str
    languages: tuple[str, ...]
    image_block_types: tuple[str, ...]
    timeout_seconds: float
    max_retries: int
    #: Documents parsed at once. Parsing dominates ingest wall-clock (seconds per PDF
    #: against a layout model), and a demo corpus of a hundred-odd files is otherwise
    #: slow enough that nobody re-runs ingestion while iterating.
    concurrency: int


@dataclass(frozen=True)
class ChunkingSettings:
    """Tuned German defaults, carried over verbatim from the production service.

    German is token-dense (~3-4 characters per token from compound words), so the caps
    are set in characters. Changing these decouples demo chunk boundaries from
    production ones, which is the whole reason the chunker was vendored.
    """

    strategy: str
    max_characters: int
    new_after_n_chars: int
    combine_under_n_chars: int
    overlap: int


@dataclass(frozen=True)
class SummarySettings:
    """One generated summary per document, written at ingestion time.

    The excerpt caps are what keep this one call per document rather than one per
    chunk: a long report is summarised from sections spread across it, not from all of
    it.
    """

    #: Sections of the document shown to the model, drawn evenly across it.
    excerpt_chunks: int
    #: Combined length of those sections, shared out between them.
    excerpt_characters: int
    #: Cap on the stored summary. A summary is read back into the context of a later
    #: answer alongside a dozen chunks, so a runaway one costs the budget it exists to
    #: save.
    max_characters: int
    #: Documents summarised at once.
    concurrency: int


@dataclass(frozen=True)
class EntitySettings:
    """Entity and relation extraction, and the resolution thresholds.

    The thresholds are the highest-risk numbers in the project. Too loose and two
    clients' similarly-named projects collapse into one node that asserts a connection
    nobody made; too strict and the graph fragments into spellings of one organisation,
    which demonstrates the opposite of this project's thesis. Both failures are silent,
    so these are tuned against the real corpus and the measurements are recorded in the
    README rather than left as folklore in a constant.
    """

    #: Chunks extracted at once. One model call each — the high-volume call site in the
    #: system and the main cost of an ingest.
    concurrency: int
    #: Cap on a stored evidence span. Evidence is read back into an answer's context
    #: and rendered on an edge in the UI, so a whole paragraph costs both.
    evidence_characters: int
    #: At or above this cosine similarity, two names are the same entity outright.
    merge_threshold: float
    #: Below this, they are different entities outright. Between the two lies the band
    #: the model adjudicates.
    review_threshold: float
    #: Whether the band is put to the model at all. Off leaves band pairs unmerged —
    #: visible as two nodes rather than quietly fused — which is the safe way to run
    #: without a key, and the way to measure what stage three is actually doing.
    adjudicate: bool


@dataclass(frozen=True)
class TraversalSettings:
    """The entity traversal retrieval leg.

    Hop depth is deliberately *not* here. It is adjusted live, question by question,
    and belongs on the call rather than in the environment; these two are the matching
    stage's risk knobs, which are the ones worth being able to retune without touching
    code — matching too loosely starts the walk from the wrong node and answers
    confidently out of genuinely connected material about something nobody asked after.
    """

    #: Cosine similarity below which a question entity does not match a graph entity.
    #: The band this sits in is narrow — 0.9001 to 0.9065 on the real corpus — which is
    #: the reason it is a setting and not a constant. See the traversal section of the
    #: README for the two pairs that bound it.
    match_threshold: float
    #: Graph entities considered per question entity when matching by embedding.
    candidates: int


@dataclass(frozen=True)
class ServiceSettings:
    """Where the HTTP service listens, who may call it, and what it serves at `/`.

    The origins are here and not hardcoded because the frontend of ticket 15 is served
    by Vite on its own port during development: a demo machine where that port differs
    would otherwise fail with an error visible only in the browser console, which is
    the worst possible place for it on the morning of a demo.

    `frontend` is the built bundle. One process serving both the API and the UI is one
    thing to start on stage instead of two, and it removes cross-origin from the demo
    path entirely — the browser calls `/api/ask` on the origin it was loaded from. A
    missing directory is not an error: the service is perfectly usable without a UI,
    and during development Vite is serving it anyway.
    """

    host: str
    port: int
    origins: tuple[str, ...]
    frontend: Path


@dataclass(frozen=True)
class CacheSettings:
    """Where the ingestion cache lives.

    One directory holding the result of every external call an ingest makes, so a
    re-run over an unchanged corpus costs neither minutes nor model spend. It is
    content-addressed and disposable: deleting it costs one slow ingest and nothing
    else, which is what `--no-cache` does.
    """

    directory: Path


@dataclass(frozen=True)
class Settings:
    neo4j: Neo4jSettings
    mistral: MistralSettings
    parsing: ParsingSettings
    chunking: ChunkingSettings
    summaries: SummarySettings
    entities: EntitySettings
    traversal: TraversalSettings
    service: ServiceSettings
    cache: CacheSettings


# Neo4j rejects a vector index outside this range, and it does so *after* `apply_schema`
# has already dropped the old one — so the value is validated here, before anything is
# destroyed.
MIN_EMBEDDING_DIMENSIONS = 1
MAX_EMBEDDING_DIMENSIONS = 4096


def _int_env(name: str, default: int) -> int:
    """`os.getenv` returns "" for `NAME=` in a .env file, which `int` rejects. Every
    subcommand loads settings first, so a crash here would also take out the diagnostics
    the user would reach for next."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from error


def _list_env(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _positive_env(name: str, default: int) -> int:
    """A batch size or concurrency of zero makes a loop do nothing at all, forever —
    caught here rather than as a hang with no output."""
    value = _int_env(name, default)
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}")
    return value


def _flag_env(name: str, default: bool) -> bool:
    raw = (os.getenv(name) or "").strip().casefold()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _similarity_env(name: str, default: float) -> float:
    """A cosine similarity threshold. Outside 0-1 it would silently merge everything or
    nothing, and either way the graph looks plausible until someone reads it."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {raw!r}") from error
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")
    return value


def _embedding_dimensions() -> int:
    value = _int_env("EMBEDDING_DIMENSIONS", 1024)
    if not MIN_EMBEDDING_DIMENSIONS <= value <= MAX_EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"EMBEDDING_DIMENSIONS must be between {MIN_EMBEDDING_DIMENSIONS} and "
            f"{MAX_EMBEDDING_DIMENSIONS}, got {value}"
        )
    return value


def load_settings() -> Settings:
    """Read settings from `.env` plus the process environment."""
    load_dotenv()
    return Settings(
        neo4j=Neo4jSettings(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "graphragdemo"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
        ),
        mistral=MistralSettings(
            api_key=os.getenv("MISTRAL_API_KEY", ""),
            extraction_model=os.getenv("MISTRAL_EXTRACTION_MODEL", "mistral-small-latest"),
            chat_model=os.getenv("MISTRAL_CHAT_MODEL", "mistral-large-latest"),
            embedding_model=os.getenv("MISTRAL_EMBEDDING_MODEL", "mistral-embed"),
            embedding_dimensions=_embedding_dimensions(),
            embedding_batch_size=_positive_env("EMBEDDING_BATCH_SIZE", 32),
            embedding_batch_characters=_positive_env("EMBEDDING_BATCH_CHARACTERS", 64_000),
        ),
        parsing=ParsingSettings(
            url=os.getenv("UNSTRUCTURED_URL", "http://localhost:8100"),
            strategy=os.getenv("PARSING_STRATEGY", "hi_res"),
            languages=_list_env("PARSING_LANGUAGES", ("deu", "eng")),
            image_block_types=_list_env("PARSING_IMAGE_BLOCK_TYPES", ("Image", "Table")),
            timeout_seconds=float(os.getenv("PARSING_TIMEOUT_SECONDS") or 1800.0),
            max_retries=_int_env("PARSING_MAX_RETRIES", 2),
            concurrency=_positive_env("PARSING_CONCURRENCY", 4),
        ),
        summaries=SummarySettings(
            excerpt_chunks=_positive_env("SUMMARY_EXCERPT_CHUNKS", 10),
            excerpt_characters=_positive_env("SUMMARY_EXCERPT_CHARACTERS", 8000),
            max_characters=_positive_env("SUMMARY_MAX_CHARACTERS", 1200),
            concurrency=_positive_env("SUMMARY_CONCURRENCY", 4),
        ),
        entities=_entity_settings(),
        traversal=TraversalSettings(
            match_threshold=_similarity_env("TRAVERSAL_MATCH_THRESHOLD", 0.905),
            candidates=_positive_env("TRAVERSAL_CANDIDATES", 5),
        ),
        service=ServiceSettings(
            host=os.getenv("API_HOST", "127.0.0.1"),
            port=_positive_env("API_PORT", 8000),
            origins=_list_env(
                "API_CORS_ORIGINS",
                ("http://localhost:5173", "http://127.0.0.1:5173"),
            ),
            frontend=Path(os.getenv("FRONTEND_DIST") or "frontend/dist").expanduser(),
        ),
        cache=CacheSettings(
            directory=Path(os.getenv("CACHE_DIR") or ".cache/ingest").expanduser()
        ),
        chunking=ChunkingSettings(
            strategy=os.getenv("CHUNK_STRATEGY", "by_title"),
            max_characters=_int_env("CHUNK_MAX_CHARACTERS", 2000),
            new_after_n_chars=_int_env("CHUNK_NEW_AFTER_N_CHARS", 1600),
            combine_under_n_chars=_int_env("CHUNK_COMBINE_UNDER_N_CHARS", 500),
            overlap=_int_env("CHUNK_OVERLAP", 250),
        ),
    )


def _entity_settings() -> EntitySettings:
    merge = _similarity_env("ENTITY_MERGE_THRESHOLD", 0.92)
    review = _similarity_env("ENTITY_REVIEW_THRESHOLD", 0.75)
    if review > merge:
        raise ValueError(
            f"ENTITY_REVIEW_THRESHOLD ({review}) is above ENTITY_MERGE_THRESHOLD "
            f"({merge}); the adjudication band runs from review up to merge, and an "
            "inverted pair silently means no band at all"
        )
    return EntitySettings(
        concurrency=_positive_env("ENTITY_CONCURRENCY", 8),
        evidence_characters=_positive_env("ENTITY_EVIDENCE_CHARACTERS", 300),
        merge_threshold=merge,
        review_threshold=review,
        adjudicate=_flag_env("ENTITY_ADJUDICATION", True),
    )
