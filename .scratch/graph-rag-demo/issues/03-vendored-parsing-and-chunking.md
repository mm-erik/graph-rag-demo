# 03: Vendored parsing and chunking

**Read `../README.md` first.**

**What to build:** a developer points a CLI subcommand at a single PDF and gets back the ordered,
retrieval-sized chunks that document would contribute to the graph — with tables and images
isolated as their own units. This is the input side of the pipeline, proven in isolation before
anything is written to the graph.

**Blocked by:** 01 (project skeleton and Neo4j infrastructure).

**Status:** ready-for-agent

## Context

**Do not write this from scratch.** Parsing, element normalisation and chunking are lifted
**wholesale** from the existing production ingestion service and vendored into this project.

Source: the sibling checkout at `../monorepo`, in its AI application's parsing providers and
ingestion chunking modules. Copy them into a clearly marked vendored directory, strip that
system's settings and dependency-injection framework, and note their origin in a header comment
so nobody mistakes them for demo-quality code written here.

The rationale matters and should restrain your instinct to improve things: this path is solved,
boring, and not what the demo is about. Reusing it means chunk boundaries match production, so
**demo behaviour predicts real behaviour**. Rewriting it would silently decouple the two.

Preserve the tuned German defaults exactly: title-aware chunking strategy, a hard character cap
with a lower soft cap so chunks trend to uniform size rather than always maxing out, a floor
below which consecutive small sections are combined, and a character overlap carried across
splits. Tables and images are isolated as their own chunks so each stays individually citable.

Chunk identifiers are derived **deterministically** from document identity and chunk index, so
re-ingesting a document replaces its chunks instead of accumulating duplicates. This property is
load-bearing for later tickets — do not replace it with random identifiers.

Parsing is delegated to the **self-hosted Unstructured HTTP service** at a configurable URL
(default `localhost:8100`), started by the production system's own process manager in the
sibling monorepo. Do not install the in-process library (multi-gigabyte ML stack) and do not use
the hosted vendor API (needs a key we do not have). Expect to start that service yourself; it
has a slow cold start and its config warns that running more than one worker has OOM-killed the
machine.

Verify the production service starts cleanly in isolation — this project depends on it without
the rest of that system — and report what you find.

The parser client is injected as a protocol, per the seam in the README.

## Acceptance criteria

- [ ] Parsing, normalisation and chunking modules vendored from the monorepo into a clearly marked directory
- [ ] Vendored files carry a header noting their origin and that they are not demo-authored code
- [ ] The monorepo's settings/DI framework is stripped; configuration comes from this project's environment
- [ ] German chunking defaults preserved exactly as in the source (strategy, hard cap, soft cap, combine floor, overlap)
- [ ] Tables and images are isolated into their own chunks
- [ ] Chunk identifiers are deterministic from document identity and index; re-parsing yields identical ids
- [ ] Parsing reaches the self-hosted Unstructured service over HTTP at a configurable URL
- [ ] A CLI subcommand parses one PDF and prints its chunks with their ids and ordering
- [ ] Verified against a PDF from the corpus generator (ticket 02) if available, otherwise any born-digital PDF
- [ ] Confirmed whether the production parsing service starts cleanly standalone; findings reported
- [ ] The parser client is injected rather than constructed internally
