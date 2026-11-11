# 10: Lexical and semantic retrieval leg

**Read `../README.md` first.**

**What to build:** a developer asks a question from the CLI and gets back a ranked list of the
chunks most likely to answer it — each annotated with which client, year and project it came
from — found by combining semantic similarity with literal keyword matching. No generated answer
yet; this ticket proves retrieval itself works on a corpus designed to defeat it.

**Blocked by:** 06 (chunks and embeddings into the graph).

**Status:** ready-for-agent

## Context

This is the first of two retrieval legs. The second (ticket 11) traverses the entity graph; they
are fused in ticket 12. Build this one to stand alone and be independently inspectable, because
being able to run each leg in isolation is a demo feature, not just a debugging aid.

The mechanism: a vector similarity search and a full-text search over chunk text each return a
ranked list, fused by **reciprocal rank fusion**.

Why both, specifically on this corpus: every chunk embeds to nearly the same point because the
documents are near-identical boilerplate, so dense retrieval alone is weak here by construction.
A distinctive literal token — a project code, a client name — is often the only signal that
separates the right document from four lookalikes. The full-text leg is what rescues those
queries. If you find yourself tempted to drop it for simplicity, re-read the risks in the README.

Each retrieved chunk carries its **folder ancestry** into the result. This is what later lets an
answer state which client and year a fact came from instead of reciting a filesystem path — and
it is the structural advantage made legible.

The number of results per leg is a query parameter with a sensible default (around ten), exposed
so it can be adjusted during a live demo.

This ticket owns one of the project's three unit-tested pure functions: **reciprocal rank
fusion**. It is tested because incorrect fusion degrades retrieval quality without ever raising
an error — a purely silent failure. Test that two ranked lists fuse into the expected combined
order. Observable behaviour only.

## Acceptance criteria

- [ ] A CLI subcommand takes a question and returns a ranked list of chunks
- [ ] Vector similarity search over chunk embeddings implemented
- [ ] Full-text search over chunk text implemented
- [ ] The two ranked lists are fused by reciprocal rank fusion
- [ ] Each result carries its full folder ancestry, not just a path string
- [ ] Results-per-leg is a query parameter with a default, adjustable at call time
- [ ] This leg is runnable in isolation and its output inspectable
- [ ] Unit tests cover reciprocal rank fusion against known input orderings
- [ ] Demonstrated on the synthetic corpus that a question containing a distinctive literal token retrieves the correct client's document, and recorded whether vector-only would have
