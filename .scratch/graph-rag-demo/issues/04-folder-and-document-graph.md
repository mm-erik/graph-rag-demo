# 04: Folder and document graph from a directory walk

**Read `../README.md` first.**

**What to build:** a demo operator points ingestion at any directory, of any depth, and sees the
corpus's structure appear in the Neo4j Browser as a navigable tree of folder nodes with the PDF
documents hanging off the leaves. No document content is read yet — this ticket makes the
*structure* itself queryable knowledge.

**Blocked by:** 01 (project skeleton and Neo4j infrastructure).

**Status:** done

## Context

The adaptivity requirement lives here and is a headline feature: the hierarchy depth is **not**
fixed at `Client / Year / Project`. A corpus nested deeper must work without configuration or
code change.

The decision that makes this possible: folders are modelled **generically** — a single node type
at any depth, linked by containment — rather than as typed per-level entities. Recovering what
each level *means* is a separate concern handled in ticket 05; do not pre-empt it here, and do
not introduce per-level node labels.

Each document node records its filename, path, and **content hash**. The hash is load-bearing
for the caching in ticket 09 — compute it now even though nothing consumes it yet.

Ingestion must be re-runnable: running it twice over the same directory must not duplicate nodes.

This ticket owns one of the project's three unit-tested pure functions: **folder-node
construction at arbitrary depth**. It is tested because its failure is silent — if it is wrong
at depth four, nothing above it raises an error; the graph is simply quietly misshapen, and you
find out during a live demo. Test that a tree of arbitrary shape and depth produces the correct
node and containment structure. Test external behaviour only; see the testing posture in the
README.

## Acceptance criteria

- [x] Ingestion walks a target directory to arbitrary depth and writes folder nodes linked by containment
- [x] PDF files at any leaf become document nodes contained by their parent folder
- [x] Folder nodes record name, path and depth
- [x] Document nodes record filename, path and content hash
- [x] A single generic folder node type is used; no per-level labels
- [x] Re-running ingestion over the same directory does not duplicate nodes or edges
- [x] Verified against a tree deeper than three levels, not just the demo shape
- [x] Unit tests cover folder-node construction at arbitrary depth, asserting observable structure
- [x] The resulting hierarchy is visible and navigable in the Neo4j Browser
