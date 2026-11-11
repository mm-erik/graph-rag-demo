# Graph RAG Demo — Ticket Set

Shared context for every ticket in `issues/`. **Read this file first.** Individual tickets
assume you have read it and do not repeat what is here.

## What we are building and why

A throwaway proof of concept demonstrating that graph RAG beats flat RAG on deeply nested,
highly self-similar document corpora — specifically `Client / Year / Project / report.pdf`
trees of German consulting documents where every quarterly report looks like every other one.

Flat RAG fails here *silently*: every chunk embeds to nearly the same point, so a question
about one client retrieves near-identical passages from another client, and the model answers
confidently from the wrong source. The structural knowledge that would disambiguate it lives
in the filesystem and is thrown away at ingestion time.

The demo preserves that structure as a graph, extracts global entities that join separate
branches of the tree, and **visualises the subgraph behind every answer** so the audience
watches the system reach across the hierarchy rather than taking it on faith.

**This code will be thrown away.** Optimise for demo clarity and speed of construction, not
longevity. Where a decision deliberately trades robustness for speed, the ticket says so.

## Authoritative documents

- **`SPEC.md`** (repo root) — the full specification. User stories, implementation decisions,
  testing decisions, out-of-scope, risks. **This is the source of truth.** If a ticket and the
  spec disagree, the spec wins; flag the discrepancy.
- **`DESIGN.md`** (repo root) — condensed design rationale, including rejected alternatives and
  *why* they were rejected. Read this before proposing a different approach — it has probably
  already been considered and declined for a stated reason.

## Architecture in one paragraph

Ingestion is a linear pipeline: directory walk → PDF parsing into layout-aware elements →
element-aware chunking → embedding → entity and relation extraction → graph write. Query is a
hybrid retrieval (lexical/semantic leg fused with an entity-traversal leg) followed by
generation. Everything lands in a single Neo4j instance. A CLI does the real work; a FastAPI
service wraps it; a React frontend renders the chat and the traversal subgraph.

## The seam

The architectural seam is the **pipeline boundary**: ingestion and question-answering are each
a single function, with the three external dependencies — the model client, the parser client,
and the graph driver — injected as protocols rather than constructed internally. Respect this.
Do not reach for a global client or construct a driver inside a helper.

Tests, however, are written at the pure-function level plus one end-to-end smoke test — not
with fakes at the top seam. This is a deliberate trade for a throwaway project; see
"Testing posture" below.

## Graph schema

Nodes:

- **Folder** — name, path, depth, inferred level name, inferred level confidence
- **Document** — filename, path, content hash, page count, generated summary
- **Chunk** — deterministic id, text, index within document, page, embedding
- **Entity** — surface name, normalised name, type, embedding

Relationships:

- Folder *contains* Folder or Document
- Document *has chunk* Chunk
- Chunk *next* Chunk (reading order within a document)
- Chunk *mentions* Entity, carrying evidence
- Entity *relates to* Entity, carrying relation type, evidence, and source chunk

**Entities are global and never attached to a folder.** This is the single most important
modelling decision in the project: it is what makes a cross-client path *real structure* in
the graph rather than a join computed at query time, and therefore what makes the
visualisation honest. Do not add a folder ownership edge to Entity.

## Models

All generation and embedding goes directly to the Mistral API with a single `MISTRAL_API_KEY`.

| Role | Model |
|---|---|
| Entity/relation extraction (high volume, per chunk) | `mistral-small-latest` |
| Chat answers, level inference, query rewriting | `mistral-large-latest` |
| Embeddings (1024-dim) | `mistral-embed` |

All identifiers are environment-configurable, never hardcoded. **The exact IDs are unverified
— confirm them against the live API before wiring them** and report any that have changed.

## External dependencies

- **Neo4j** in a container. Holds everything: hierarchy, documents, chunks with their embedding
  vectors, and entities. Native vector index *and* a Lucene full-text index over chunk text.
- **Unstructured**, self-hosted, reached over HTTP at a configurable URL (default `localhost:8100`).
  It is started by the production system's own process manager, in the sibling `monorepo`
  checkout. It is **not** vendored or run in-process — the in-process library pulls a
  multi-gigabyte ML stack, and the hosted vendor API needs a key we do not have.
- **Mistral API**, direct.

Docker is installed but was not running at time of writing, and neither was the parsing
service. Expect to start both.

## Known risks — read before you start

1. **Entity resolution is the highest-risk component in the system.** On a corpus engineered
   around near-duplicates, too-loose merging collapses distinct entities and too-strict merging
   fragments the graph into spelling variants. A fragmented graph demonstrates the *opposite*
   of this project's thesis. Both failure modes are silent. Expect to tune thresholds
   empirically; do not assume the first attempt is correct.
2. **The traversal visualisation carries the entire burden of proof.** A side-by-side flat-RAG
   baseline was considered and deliberately cut from scope. If the visual does not land, the
   demo reads as "a chatbot that answered a question." This is a single point of failure by
   choice.
3. **Dense retrieval on near-identical boilerplate is inherently weak**, which is why the
   full-text index is not optional despite the single-store decision. "Neo4j only" must not
   quietly become "dense-only".

## Testing posture

Coverage is deliberately narrow. The criterion for writing a test is **silent failure**: logic
whose bugs would not announce themselves and would instead make the graph quietly wrong in a
way first noticed during a live demo.

Tests assert external, observable behaviour — given this directory, the graph contains these
nodes and relationships; given this question, retrieval returns these chunks in this order.
They never assert how a result was reached and never reach into private state.

Three pure-logic units are tested directly (folder-node construction at arbitrary depth;
entity normalisation and merging; reciprocal rank fusion), plus one end-to-end smoke test
against real infrastructure. Everything else is verified by running a real ingestion.

Do not add broad coverage beyond this. It is not a quality signal here; it is waste on a
project being discarded.

## Ticket dependency graph

```
01 skeleton + Neo4j infra ──┬── 04 folder graph ──┬── 05 level inference
                            │                     │
02 synthetic corpus ────────┤                     │
                            │                     │
03 vendored parsing ────────┴── 06 chunks + embeddings ──┬── 07 document summaries
                                                         │
                                                         ├── 08 entity extraction ──┬── 09 cache + progress
                                                         │                          │
                                                         └── 10 lexical/semantic leg │
                                                                                     │
                                                           11 entity traversal leg ◄─┘
                                                                     │
                              10 ──────────────────┬──────────────── 11
                                                   │
                                      12 hybrid retrieval + answer (CLI)
                                                   │
                                   ┌───────────────┴───────────────┐
                                   │                               │
                         13 multi-turn + rewriting      14 FastAPI + subgraph payload
                                                                   │
                                                          15 React traversal UI
                                                                   │
                                                       16 smoke test + runbook
```

**Ticket 12 is the first user-facing tracer bullet.** Tickets 01–11 are stages of a pipeline:
each is independently verifiable (in the Neo4j Browser, or via a CLI subcommand) but none is a
complete user-facing path on its own. This is an acknowledged departure from strict vertical
slicing — an ingestion pipeline is a chain, and faking the stages ahead of you to force an
end-to-end slice would cost more than it is worth on a throwaway project.

## Parallelism

Tickets **01, 02 and 03** have no blockers and can run concurrently.
Tickets **05, 07, 08 and 10** can run concurrently once their blockers are done.
Tickets **13 and 14** can run concurrently once 12 is done.

## Conventions

- Python managed with `uv`. Frontend with Vite.
- All configuration via environment variables, with a committed `.env.example` (note the
  spelling — `.env.example`, not `.example.env`).
- A task-runner file wraps the multi-step startup into single commands. For a demo that will be
  run in front of people after weeks of neglect, a one-command start matters more than it sounds.
- Corpus and prompts are **German**. Chunking parameters are tuned for German token density.
- Do not add authentication, deployment, CI, incremental re-ingestion, crash resumption,
  deleted-file handling, OCR, or performance work. All explicitly out of scope — see `SPEC.md`.
