# 06: Chunks and embeddings into the graph

**Read `../README.md` first.**

**What to build:** ingestion now reads the PDFs it discovered, and every document in the Neo4j
Browser can be expanded into its ordered chunks, each carrying an embedding and each findable by
both vector similarity and full-text search. This closes the ingestion path from "we know the
files exist" to "we can retrieve their content".

**Blocked by:** 03 (vendored parsing and chunking), 04 (folder and document graph).

**Status:** ready-for-agent

## Context

This joins the two halves built so far: the structural walk from 04 and the parse/chunk path
from 03. Document nodes already exist; this ticket hangs their content off them.

Chunk nodes record their deterministic id, text, index within the document, page, and embedding.
The deterministic id from ticket 03 is what makes re-ingestion an upsert rather than an
accumulation — preserve that property end to end.

Adjacent chunks within a document are linked in reading order. This exists so retrieval can
later expand a hit to its neighbouring passages; it is cheap to add now and awkward to backfill.

Embeddings come from the Mistral embedding model at 1024 dimensions, which must match the vector
index created in ticket 01. A mismatch here fails loudly if you are lucky and silently degrades
retrieval if you are not — verify the dimensionality rather than assuming it.

Both indexes created in ticket 01 must actually be populated and queryable by the end of this
ticket. Confirm a vector query and a full-text query each return sensible results over real
ingested chunks. The full-text index is not decorative — see the risks in the README.

Embedding is the first place in the pipeline with meaningful per-chunk API cost. Batch requests
rather than issuing one call per chunk.

## Acceptance criteria

- [ ] Ingestion parses each discovered PDF and writes its chunks to the graph, linked to their document
- [ ] Chunk nodes record deterministic id, text, index, page and embedding
- [ ] Adjacent chunks within a document are linked in reading order
- [ ] Embeddings are produced by the configured Mistral embedding model at 1024 dimensions
- [ ] Embedding requests are batched rather than one call per chunk
- [ ] Vector index dimensionality verified to match the embeddings actually produced
- [ ] A vector similarity query over ingested chunks returns plausible results
- [ ] A full-text query over ingested chunks returns plausible results
- [ ] Re-running ingestion replaces a document's chunks rather than duplicating them
- [ ] Verified end to end on the synthetic corpus
