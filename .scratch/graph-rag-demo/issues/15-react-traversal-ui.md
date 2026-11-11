# 15: React chat and traversal visualisation

**Read `../README.md` first.**

**What to build:** **the demo itself.** A chat interface where every answer appears beside a live
rendering of the subgraph that produced it — the retrieved chunks, the entities they led to, the
folder ancestry that scoped them, and, highlighted distinctly, every edge that crossed from one
branch of the hierarchy into another. The audience watches the system reach across the tree.

**Blocked by:** 14 (HTTP service with traversal subgraph payload).

**Status:** ready-for-agent

## Context

**Read the risks section of `../README.md` before starting.** A side-by-side comparison against
flat RAG was considered and deliberately cut from scope. The consequence, recorded explicitly in
the spec: **this visualisation now carries the entire burden of proof.** If it does not land, the
demo reads as "a chatbot that answered a question." Everything upstream of this ticket is
plumbing in service of this screen.

Accordingly, the traversal panel is the **primary** visual, not a side panel. Answer on one side,
live subgraph on the other, with comparable weight. Resist the instinct to make this a
chat app with a small diagram tucked into a corner.

Stack: React with Vite, using a force-directed graph renderer. That renderer was chosen
deliberately: the structure is a mixed hierarchy-plus-entity-web, which strictly hierarchical
layouts handle poorly. The server already flags cross-branch edges (ticket 14) — do not
recompute that client-side.

Cross-hierarchy edges are drawn in a distinct colour. This is the single most important visual
affordance in the project: it is the thing that makes "it found a connection flat RAG could not"
legible at a glance, from the back of a room, to someone who has not read the spec.

The target demo question — *"What connects Client A's Atlas project to Client B's?"* — should
produce a picture whose shape is self-evidently impossible for flat retrieval to have produced.
Test against that question specifically and judge the result by that standard.

Practical demo constraints worth designing for: it will be shown on a projector, possibly at
poor contrast, to people standing some distance away. Legibility beats density. An audience
member should be able to trace the reasoning path node by node — which chunks were retrieved,
which entities they led to — without the operator narrating every step.

Expose the tuning parameters (results per leg, hop depth, context cap) in the UI, since hop
depth in particular will be adjusted live on stage.

If ticket 13 has landed, support multi-turn conversation and show the rewritten standalone
question alongside the answer.

## Acceptance criteria

- [ ] React and Vite frontend with a chat interface served alongside the HTTP service
- [ ] Every answer renders its traversal subgraph with a force-directed layout
- [ ] The graph panel has primary visual weight, comparable to the answer, not a corner widget
- [ ] Cross-hierarchy edges are rendered in a visually distinct colour, using the server's flag
- [ ] Node types (folder, document, chunk, entity) are visually distinguishable
- [ ] Retrieved chunks and the entities they led to can be traced node by node
- [ ] Source citations from the answer are visible and connected to the graph
- [ ] Tuning parameters including hop depth are adjustable from the UI
- [ ] Multi-turn conversation supported, showing the rewritten question, if ticket 13 has landed
- [ ] Legible on a projector at a distance: verified at presentation scale, not just on a laptop
- [ ] Verified against the target demo question, producing a visibly cross-branch result
