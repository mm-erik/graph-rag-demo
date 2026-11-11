# Graph RAG Demo — Design

**Purpose:** a throwaway proof-of-concept demonstrating that graph RAG beats flat
RAG on deeply nested, highly self-similar document corpora (`Client/Year/Project/*.pdf`).
Not production code. Will be discarded.

## 1. Corpus

Real corpus (German, `Client/Year/Project/report.pdf`) is not currently accessible,
so ingestion is built against a **synthetic German corpus generated to be adversarial**:

- 4 clients x 3 years x 3-4 projects, text PDFs.
- A project name (`Projekt Atlas`) deliberately reused across two different clients.
- A consultant appearing under multiple clients — the cross-folder link the demo hinges on.
- Near-identical quarterly status-report boilerplate, so dense retrieval genuinely confuses documents.

Hierarchy depth is **not** fixed: ingestion handles arbitrary depth.

## 2. Pipeline

```
dir -> walk -> Unstructured (self-hosted :8100) -> elements
    -> element-aware chunking -> mistral-embed
    -> Mistral entity/relation extraction -> Neo4j
```

### Parsing — vendored from ../monorepo/apps/ai

Copied wholesale into `vendored/` (clearly marked, origin noted), stripped of the
monorepo's settings/DI framework:

- `providers/parsing/unstructured.py`, `unstructured_self_hosted_http.py`, `normalize.py`, `base.py`
- `ingestion/chunking.py` + its element/chunk models

Transport: **self-hosted Unstructured HTTP API at `localhost:8100`**, started via the
monorepo's process-compose. URL configurable in `.env`.

Chunking keeps the monorepo's tuned German defaults: `by_title`, `max_characters=2000`,
`new_after_n_chars=1600`, `combine_under_n_chars=500`, `overlap=250`, tables/images
isolated as citable units, chunk id = `uuid5(NAMESPACE_URL, "urn:chunk:{document_id}:{index}")`.

### Adaptive level inference

Folders are generic at any depth. After the walk, **one cheap LLM call per depth level**
inspects that level's sibling names and infers its semantics ("depth 1 looks like Client",
"depth 2 looks like Year"), stored as `inferred_level` + confidence on each Folder node.
Overridable via `ingest --levels Client,Year,Project` (falls back to inference beyond the
declared depth).

### Entity extraction

Fixed taxonomy, not open-ended: `Person`, `Organization`, `System/Product`, `Concept`,
`Metric/Deliverable`, `Risk`, plus a typed relation list.

Entity resolution, in order: exact normalized-name match -> embedding-similarity merge
above threshold -> Mistral tiebreak for the ambiguous middle band only. Getting this
wrong fragments the graph and makes it look *worse* than flat RAG, so it is the highest-risk
component.

### Caching

Extraction results cached by file sha256. No incrementality, no resumability, no deletion
handling. Progress bar on long ingests.

## 3. Storage — Neo4j only

Single container. Native **vector index** (cosine, 1024-dim) and a Lucene **full-text index**
over chunk text, so the lexical retrieval leg survives dropping Qdrant.

### Schema

Nodes:
- `Folder {name, path, depth, inferred_level, inferred_level_confidence}`
- `Document {filename, path, sha256, page_count, summary}`
- `Chunk {id, text, index, page, embedding}`
- `Entity {name, normalized_name, type, embedding}`

Edges:
- `(:Folder)-[:CONTAINS]->(:Folder|:Document)`
- `(:Document)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:NEXT]->(:Chunk)`
- `(:Chunk)-[:MENTIONS {evidence}]->(:Entity)`
- `(:Entity)-[:RELATES {type, evidence, source_chunk}]->(:Entity)`

**Entities are global** — never owned by a folder. This is what makes the cross-client
path real structure rather than a query-time join.

Each Document carries an LLM-written summary property for folder-level questions.

## 4. Retrieval — hybrid

Two legs in parallel:

1. **Lexical/semantic:** vector top-k (k=10) + full-text top-k (k=10), fused by RRF.
2. **Entity:** Mistral extracts entities from the question -> match `Entity` by normalized
   name + embedding -> traverse `RELATES` 1-2 hops -> collect `MENTIONS` chunks.

Merge both sets, attach each chunk's folder ancestry as context, cap at ~12 chunks.
All of k / hop depth / cap exposed as query params for live tuning. Each leg can also be
run alone via a flag, to show the difference.

**Multi-turn:** last few turns retained; a follow-up is rewritten into a standalone
question by Mistral before retrieval.

## 5. Models — Mistral API direct (`MISTRAL_API_KEY`)

| Role | Model |
|---|---|
| Entity/relation extraction (high volume) | `mistral-small-latest` |
| Chat answers, level inference, query rewriting | `mistral-large-latest` |
| Embeddings (1024-dim) | `mistral-embed` |

All IDs overridable via `.env`; exact IDs verified against the live API before wiring.
`.env.example` committed.

## 6. Interface

- **Core:** Python CLI — `ingest <dir> [--levels ...] [--no-cache]`, `query "<q>"`, `serve`.
- **API:** FastAPI on top of the same code paths.
- **Frontend:** React + Vite + `react-force-graph`.

The **traversal panel is the primary visual, not a side panel** — answer left, live
subgraph right, showing vector hits, the entities they led to, folder ancestry, and
**cross-folder edges in a distinct color**. With the flat-RAG baseline cut, this panel
carries the entire burden of proof: the target demo question is "what connects Client A's
Atlas project to Client B's?", whose answer should render a shape flat RAG visibly
could not produce.

## 7. Running it

- Neo4j: `docker compose up` (Docker is installed but not currently running).
- Unstructured: monorepo process-compose, :8100.
- App: `uv`.
- `justfile` wraps it: `just up` / `just ingest <dir>` / `just serve`. README documents the exact sequence.

## 8. Tests

~6 unit tests, only on the silent-failure-prone pure logic:

- path -> folder-node construction at arbitrary depth
- entity normalization and merge
- RRF fusion

Everything else is verified by running a real ingest.

## 9. Explicitly out of scope

Flat-RAG side-by-side baseline (cut deliberately), text-to-Cypher, community detection /
global summarization (Microsoft-GraphRAG style), OCR for scanned PDFs, incremental
re-ingest, auth, deployment.
