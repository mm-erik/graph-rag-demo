# 14: HTTP service with traversal subgraph payload

**Read `../README.md` first.**

**What to build:** the graph RAG system becomes reachable over HTTP — a client can post a
question and receive both the generated answer *and* a complete description of the subgraph that
produced it, ready to be drawn.

**Blocked by:** 12 (hybrid retrieval and answer generation).

**Status:** ready-for-agent

## Context

A thin layer over the same code paths the CLI already uses. Do not fork the logic — the service
and the CLI are two front ends onto one pipeline, per the seam described in the README. If you
find yourself reimplementing retrieval here, stop.

Ticket 12 was required to return a **structured result** containing the answer text, retrieved
chunks with their leg and rank, matched entities, traversed edges, and folder ancestry. This
ticket's main job is shaping that into a payload a graph renderer can consume directly: nodes
and edges with types and identities, flagged so the frontend can distinguish folders, documents,
chunks and entities — and, critically, **flag edges that cross between different branches of the
hierarchy**, since ticket 15 renders those distinctly and they are the visual payoff of the
entire project.

Work out the cross-branch determination here, on the server, where the folder ancestry of every
node is already known. Do not push that computation into the frontend.

The service also needs to expose enough for the operator: triggering or reporting on ingestion,
and surfacing the inferred level names from ticket 05.

Tuning parameters from ticket 12 — results per leg, hop depth, context cap, single-leg flags —
must be settable per request so they can be adjusted from the UI during a demo.

Conversation history from ticket 13 should be accepted if that ticket has landed; do not block
on it.

No authentication, no deployment concerns, no production error handling. All out of scope.

## Acceptance criteria

- [ ] An HTTP service exposes question answering over the existing pipeline
- [ ] Responses include the generated answer text
- [ ] Responses include a renderable subgraph: typed nodes and edges with stable identities
- [ ] Node types distinguish folders, documents, chunks and entities
- [ ] Edges that cross between different branches of the hierarchy are flagged server-side
- [ ] Retrieved chunks are identified with their retrieval leg and rank
- [ ] Folder ancestry is included for every relevant node
- [ ] Tuning parameters are settable per request
- [ ] Conversation history is accepted if ticket 13 has landed
- [ ] Ingestion status and inferred level names are exposed to the operator
- [ ] Retrieval and generation logic is shared with the CLI, not duplicated
- [ ] A command starts the service
