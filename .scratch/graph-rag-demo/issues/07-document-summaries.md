# 07: Document summaries

**Read `../README.md` first.**

**What to build:** a demo operator can ask "what is this report about?" or "what happened on this
project in 2023?" and get an answer without the system having to read every chunk of every
document. Each document carries a generated summary of its own content.

**Blocked by:** 06 (chunks and embeddings into the graph).

**Status:** ready-for-agent

## Context

A small, self-contained ticket that can run in parallel with 08 and 10.

Each document node gains a generated summary property, written at ingestion time from that
document's chunks. The purpose is folder-level and document-level questions: without summaries,
answering "what did we do for this client in 2023" means pulling every chunk of every report,
which blows the context budget and retrieves nothing useful.

Use the chat-tier model — this is one call per document, not per chunk, so it is low-volume and
summary quality is directly visible in answers.

Summaries are German, matching the corpus.

This is also the obvious consumer of the inferred level names from ticket 05 if it has landed:
a summary that knows it describes a project report for a particular client and year is more
useful than one that does not. Treat that as a nice-to-have, not a blocker — do not couple to
ticket 05.

Summary generation should respect the ingestion cache once ticket 09 lands; keying is by
document content hash, which already exists on the document node from ticket 04.

## Acceptance criteria

- [ ] Each document node carries a generated summary written at ingestion time
- [ ] Summaries are produced by the chat-tier model, one call per document
- [ ] Summaries are in German
- [ ] Summaries describe the document's actual content specifically enough to distinguish it from a sibling report with near-identical boilerplate
- [ ] Re-ingestion does not regenerate a summary for an unchanged document once caching lands
- [ ] Verified by inspecting summaries for two near-identical reports from different clients and confirming they are distinguishable
