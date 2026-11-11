# 11: Entity traversal retrieval leg

**Read `../README.md` first.**

**What to build:** a developer asks a question naming a person or project, and the system finds
relevant material in documents that **share no vocabulary with the question at all** — by
matching the named entities into the graph and walking their relationships to whatever those
entities connect to. This is the leg that makes cross-client answers possible.

**Blocked by:** 08 (entity extraction and resolution).

**Status:** ready-for-agent

## Context

The second retrieval leg, fused with the first in ticket 12. Like ticket 10, build it to stand
alone and be independently runnable — showing each leg separately is a demo feature.

The mechanism: entities are extracted from the question, matched against graph entities by
normalised name *and* embedding, then traversed one to two hops along relation edges, collecting
the chunks that mention the entities reached.

This leg is where the project's thesis actually cashes out. The target demo question — *"What
connects Client A's Atlas project to Client B's?"* — is unanswerable by any amount of vector
search, because no single chunk contains the connection. It exists only as a path through the
graph, and only because entities are global rather than folder-owned (ticket 08).

**Hop depth is a query parameter and will be adjusted live on stage.** The defaults were chosen
with that in mind: one hop is precise, two hops is where cross-document connections appear,
three hops is noise. Default to two. Make it genuinely tunable at call time, not a constant.

Question-side entity extraction uses the extraction-tier model and is a single call per
question — cheap, unlike ingestion-side extraction.

Matching questions to entities needs to tolerate the user not naming entities exactly as the
documents do. Use both normalised-name matching and embedding similarity, and be aware this is
the same resolution hazard as ticket 08 in miniature: match too loosely and the traversal starts
from the wrong node, which produces confidently wrong answers.

Results carry folder ancestry, consistent with ticket 10, and must also carry **the path taken**
— which entities were matched, which edges were walked, which chunks they led to. Ticket 14 and
the visualisation in ticket 15 depend on this provenance existing. Capture it from the start;
retrofitting it is painful.

## Acceptance criteria

- [ ] A CLI subcommand takes a question, extracts its entities, and returns chunks reached by graph traversal
- [ ] Question entities are matched to graph entities by both normalised name and embedding similarity
- [ ] Traversal walks relation edges to a configurable hop depth, defaulting to two
- [ ] Hop depth is adjustable at call time, not a constant
- [ ] Chunks mentioning the reached entities are collected as results
- [ ] Each result carries its folder ancestry
- [ ] The traversal path is captured: matched entities, edges walked, chunks reached
- [ ] This leg is runnable in isolation and its output inspectable
- [ ] Demonstrated on the synthetic corpus that this leg surfaces material connecting two different clients' same-named projects, which the lexical/semantic leg alone does not
