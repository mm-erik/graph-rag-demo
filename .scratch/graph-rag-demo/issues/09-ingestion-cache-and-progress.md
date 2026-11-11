# 09: Ingestion cache and progress reporting

**Read `../README.md` first.**

**What to build:** a demo operator re-runs ingestion and it completes in seconds instead of
minutes, at near-zero model cost — and when a genuinely fresh ingestion does take minutes, a
progress indicator makes it obvious the system is working rather than hung.

**Blocked by:** 08 (entity extraction and resolution).

**Status:** ready-for-agent

## Context

This ticket is worth more than it looks. It pays for itself the first time you re-run ingestion
while iterating on extraction prompts, it keeps Mistral spend near zero across development, and
it makes a stage demo re-runnable in seconds rather than requiring a nervous wait.

Caching is keyed by **document content hash**, which already exists on document nodes from
ticket 04. Cache the expensive derived artefacts: parse results, and above all the per-chunk
entity extraction results from ticket 08, which are the dominant cost. Document summaries from
ticket 07 should also be cached if that ticket has landed.

A flag bypasses the cache entirely, for forcing a clean re-run after changing extraction prompts.
Make sure this genuinely invalidates rather than merely appending — a stale-prompt cache that
cannot be cleared will waste a lot of your time.

Progress reporting exists for a specific failure mode: a multi-minute ingestion in front of an
audience that looks frozen. It needs to convey what stage is running and roughly how far through
it is.

**Deliberately not in scope**, and do not add them: incremental re-ingestion, crash resumption,
and handling of deleted files. All explicitly cut in `SPEC.md`. If a document changes, its hash
changes and it is reprocessed — that is the entire invalidation story.

## Acceptance criteria

- [ ] Parse results cached by document content hash
- [ ] Per-chunk entity extraction results cached, keyed so an unchanged document is never re-extracted
- [ ] Document summaries cached if ticket 07 has landed
- [ ] A flag bypasses and invalidates the cache for a forced clean re-run
- [ ] Re-running ingestion over an unchanged corpus completes in seconds and issues no extraction API calls
- [ ] Changing one document's content causes exactly that document to be reprocessed
- [ ] Progress indicator shows current stage and approximate completion during a fresh ingestion
- [ ] No incremental re-ingestion, resumption, or deletion handling added
