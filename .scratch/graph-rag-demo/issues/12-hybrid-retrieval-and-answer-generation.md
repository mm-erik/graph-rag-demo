# 12: Hybrid retrieval and answer generation

**Read `../README.md` first.**

**What to build:** **the first complete user-facing path.** A demo operator asks a question in
natural language from the CLI and gets a cited natural-language answer, drawn from both
retrieval legs working together, stating which client and year each fact came from. This is the
tracer bullet — after this ticket, there is a working graph RAG system.

**Blocked by:** 10 (lexical and semantic retrieval leg), 11 (entity traversal retrieval leg).

**Status:** ready-for-agent

## Context

Tickets 01–11 are pipeline stages, each verifiable but none user-facing. This one joins them
into something you can actually demo.

The mechanism: run both legs in parallel, merge their chunk sets, annotate each chunk with its
folder ancestry, cap the combined context at roughly a dozen chunks, and generate an answer with
the chat-tier model.

All of the tuning parameters — results per leg, hop depth, context cap — are exposed as query
parameters with defaults, because they will be adjusted during a live demo. Flags to run either
leg alone must survive into this layer too: showing "here is leg A, here is leg B, here is both"
is how the audience sees what the graph contributes.

### Return structured provenance, not just prose

This is the requirement most likely to be under-built, and retrofitting it is painful. The
answer path must return a **structured result**, not a string:

- the generated answer text
- which chunks were retrieved, from which leg, and in what rank order
- which entities were matched and which edges were traversed
- the folder ancestry of every retrieved chunk

Tickets 14 and 15 consume exactly this to render the traversal subgraph, which — with the
flat-RAG baseline cut from scope — carries the entire burden of proof for the demo. Build the
structured result now, and let the CLI render a human-readable view of it rather than being the
only thing that exists.

Answers must cite their sources so claims can be verified, and should use the inferred level
names from ticket 05 where available ("Client X's 2023 Atlas project") rather than reciting
paths.

Answers are in German, matching the corpus.

## Acceptance criteria

- [ ] A CLI subcommand takes a natural-language question and returns a generated answer
- [ ] Both retrieval legs run and their results are merged into one context
- [ ] Each context chunk is annotated with its folder ancestry
- [ ] Combined context is capped, with the cap exposed as a parameter
- [ ] Results-per-leg, hop depth and context cap are all adjustable at call time
- [ ] Either leg can be run alone via a flag, through this layer
- [ ] The answer path returns a structured result containing answer text, retrieved chunks with leg and rank, matched entities, traversed edges, and folder ancestry
- [ ] The CLI renders that structured result readably rather than being the only representation
- [ ] Answers cite their source documents
- [ ] Answers are in German and use inferred level names where available
- [ ] Demonstrated answering the target demo question about what connects two clients' same-named projects
