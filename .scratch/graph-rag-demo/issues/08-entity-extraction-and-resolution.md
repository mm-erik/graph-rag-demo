# 08: Entity extraction and resolution

**Read `../README.md` first.**

**What to build:** the graph stops being a folder tree and becomes a knowledge graph. Entities
named in the documents — people, organisations, systems, concepts, metrics, risks — become
nodes linked to the chunks that mention them and to each other, and a consultant who worked for
two different clients becomes **one node joining two separate branches of the hierarchy**. That
cross-folder edge is the thing the whole demo exists to show.

**Blocked by:** 06 (chunks and embeddings into the graph).

**Status:** ready-for-agent

## Context

**This is the highest-risk ticket in the project. Read the risks section of `../README.md`
before starting.**

### Extraction

Extraction uses a **fixed taxonomy**, not open-ended extraction: person, organisation, system or
product, concept, metric or deliverable, and risk — together with a fixed list of relation
types. Open-ended extraction was explicitly considered and rejected: it produces inconsistent
types that visualise badly, and the visualisation is the entire point of this demo.

Use the extraction-tier model. This is the one high-volume call site in the system — one call
per chunk — so it is also the main cost driver. Ticket 09 adds caching; until then expect real
spend while iterating.

Each extracted relation records the **evidence span and the source chunk** it came from, so any
edge displayed on screen can be justified when an audience member challenges it. This is not
optional metadata; it is what makes the visualisation defensible.

### Resolution — the dangerous part

Resolution runs in three escalating stages:

1. Exact match on a normalised name
2. Merge by embedding similarity above a threshold
3. Model adjudication for the ambiguous middle band only

Both failure modes are silent and both destroy the demo:

- **Too loose** — distinct entities collapse together. Two different clients' similarly-named
  projects become one node, and the graph asserts connections that do not exist.
- **Too strict** — the graph fragments into spelling variants of the same organisation. A
  fragmented graph demonstrates the *opposite* of this project's thesis and looks worse than
  flat RAG.

The synthetic corpus (ticket 02) is built with planted collisions precisely to exercise this.
Use it. Do not assume your first thresholds are correct; tune them empirically against the
corpus and record what you settled on and why.

### The non-negotiable modelling rule

**Entities are global and are never attached to a folder.** No folder ownership edge on Entity,
ever. This is what makes a cross-client path real structure in the graph rather than a join
computed at query time — and therefore what makes the visualisation honest rather than staged.

### Tests

This ticket owns one of the project's three unit-tested pure functions: **entity normalisation
and merging**. Test that spelling variants of one entity collapse to a single node, *and* that
genuinely distinct similarly-named entities do not. Both directions. Test observable behaviour
only.

## Acceptance criteria

- [ ] Entities extracted from every chunk using the fixed taxonomy; no open-ended types appear in the graph
- [ ] Relations extracted using the fixed relation type list
- [ ] Entity nodes record surface name, normalised name, type and embedding
- [ ] Chunks link to the entities they mention, carrying evidence
- [ ] Entities link to each other with relation type, evidence and source chunk
- [ ] Entity nodes have no folder ownership edge of any kind
- [ ] Resolution implements all three stages, with model adjudication restricted to the ambiguous band
- [ ] A person planted in multiple clients' documents resolves to exactly one node joining both subtrees, verified in the Neo4j Browser
- [ ] Two distinct same-named projects from different clients remain separate nodes
- [ ] Chosen similarity thresholds are recorded, with the reasoning and the corpus evidence behind them
- [ ] Unit tests cover normalisation and merging in both directions (should-merge and must-not-merge)
