# 01: Project skeleton and Neo4j infrastructure

**Read `../README.md` first.**

**What to build:** a developer can clone this directory, run one command, and have a running
Neo4j instance with every index and constraint the project needs already created — then run a
second command that reports the graph is reachable and correctly indexed. Nothing is ingested
yet; this is the ground the rest of the pipeline stands on.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Context

The project is greenfield — the directory contains only `SPEC.md`, `DESIGN.md` and this
scratchpad. Every decision about stack and storage is already made and recorded; your job is to
realise them, not to revisit them.

Storage is a **single Neo4j instance** holding the hierarchy, documents, chunks with their
embedding vectors, and entities. A separate dedicated vector store was considered and rejected
to keep the demo focused on the graph and avoid a second store that could drift out of sync.

The consequence of that rejection is the part people get wrong: the sparse/lexical retrieval leg
must be recovered **inside** Neo4j via a Lucene full-text index over chunk text, alongside the
native vector index. Without it, "Neo4j only" quietly becomes "dense-only retrieval", which is
the weakest possible configuration on a corpus specifically designed to defeat dense retrieval.
Both indexes are your responsibility in this ticket, even though nothing writes to them yet.

The vector index must match the embedding model's dimensionality (1024) and use cosine
similarity. Confirm the embedding model identifier against the live Mistral API rather than
trusting the table in the README, and report if it has changed.

Index and constraint creation must be **idempotent** — this will be re-run constantly during
development and must never fail on a second invocation.

Respect the seam described in the README: the graph driver is injected, not constructed inside
helpers or held in a module global.

## Acceptance criteria

- [ ] `uv`-managed Python project with a dependency set that installs cleanly
- [ ] Container definition for Neo4j, with volumes so the graph survives a restart
- [ ] Task-runner file with a single command that brings the infrastructure up
- [ ] A CLI entry point exists with subcommands stubbed for the pipeline (ingest, query, serve)
- [ ] A CLI subcommand creates all constraints, the vector index (1024-dim, cosine) and the chunk full-text index, idempotently
- [ ] Running that subcommand twice in a row succeeds both times
- [ ] A CLI subcommand reports connectivity and lists which indexes exist, so a demo operator can verify readiness
- [ ] `.env.example` committed, covering the Mistral key and model identifiers, the Neo4j connection, and the Unstructured service URL
- [ ] No secrets committed; real `.env` is git-ignored
- [ ] README at repo root documenting the exact startup sequence
- [ ] The graph driver is passed in as a dependency rather than constructed internally
- [x] Verified embedding model identifier against the live Mistral API; any discrepancy reported — all three identifiers available, embedding width measured at 1024 as configured; no discrepancy
