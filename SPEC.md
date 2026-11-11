# Spec: Graph RAG Demo for Nested, Self-Similar Document Corpora

## Problem Statement

We hold large volumes of client documentation organised as a deep folder hierarchy —
`Client / Year / Project / report.pdf` — where the documents are overwhelmingly similar to
one another. Every project produces quarterly status reports with near-identical structure,
boilerplate and vocabulary. Project names repeat across clients. The same people appear
under several clients.

Conventional flat RAG performs badly here, and it fails *silently*. Because every chunk
embeds to nearly the same point in vector space, a question about one client's project
retrieves near-identical passages from a different client's project, and the model answers
confidently from the wrong source. Worse, the structural knowledge that would disambiguate
it — which client, which year, which project a passage belongs to — is present in the
filesystem but discarded at ingestion time.

Separately, questions that span the hierarchy are simply unanswerable. "What connects these
two clients' Atlas projects?" requires joining facts that live in different folders; flat
retrieval has no mechanism to make that hop.

We need to demonstrate — to stakeholders, convincingly, in a live demo — that graph RAG
solves this class of problem, before committing to it.

## Solution

A demo application that ingests an arbitrary directory tree of PDFs into a knowledge graph
which preserves *both* the folder structure and the entities described inside the documents,
then answers questions against that graph through a chat interface that **visualises the
part of the graph it used**.

The user points the ingestion pipeline at a directory. The pipeline walks it to any depth,
builds a node per folder, infers what each level of the hierarchy *means* (client, year,
project) without being told, parses every leaf PDF, chunks it, and extracts a typed set of
entities and relationships from each chunk. Entities are global rather than folder-owned,
so a consultant who worked for two clients becomes one node joining two subtrees.

The user then asks questions in a chat UI. Each answer appears alongside a live rendering of
the subgraph that produced it: the retrieved chunks, the entities they led to, the folder
ancestry that scoped them, and — highlighted distinctly — any edge that crossed from one
part of the hierarchy into another. The audience does not have to take the advantage on
faith; they watch the system reach across the tree.

This is a proof of concept. It will be thrown away.

## User Stories

### Ingestion

1. As a demo operator, I want to point the pipeline at any directory and have it ingest everything beneath it, so that I do not have to pre-process or restructure a corpus before showing it.
2. As a demo operator, I want ingestion to handle a hierarchy of any depth, so that a corpus deeper than `Client/Year/Project` does not break the demo.
3. As a demo operator, I want the system to infer what each level of the hierarchy means from the folder names themselves, so that I can demo on an unfamiliar corpus without configuring a schema first.
4. As a demo operator, I want to override the inferred level semantics with an explicit flag, so that a bad inference does not derail a live demo.
5. As a demo operator, I want to see the inferred level names and their confidence, so that I know whether to trust them before presenting.
6. As a demo operator, I want a progress indicator during ingestion, so that a multi-minute run does not look hung in front of an audience.
7. As a demo operator, I want extraction results cached by file content hash, so that re-running ingestion while iterating is fast and does not re-incur model cost.
8. As a demo operator, I want to bypass the cache with a flag, so that I can force a clean re-run when I have changed extraction prompts.
9. As a developer, I want PDF parsing to use the same parser and chunking parameters as our production system, so that what the demo shows predicts how the real corpus will behave.
10. As a developer, I want tables and images to be isolated into their own chunks, so that they stay individually citable rather than being smeared across prose.
11. As a developer, I want chunk identifiers to be deterministic, so that re-ingesting a document replaces its chunks instead of accumulating duplicates.
12. As a developer, I want ingestion to record each document's content hash, so that cache invalidation is correct when a file changes.
13. As a developer, I want a per-document summary generated at ingestion time, so that questions about a whole document or folder can be answered without reading every chunk.

### Graph construction

14. As a demo operator, I want every folder to become a node linked to its parent, so that the structure of the corpus is itself queryable knowledge.
15. As a demo operator, I want entities extracted from document text with a consistent, fixed set of types, so that the rendered graph is legible rather than an inconsistent mush of labels.
16. As a demo operator, I want entities to be global rather than owned by a folder, so that the same person or system appearing under two clients becomes a single node joining both subtrees.
17. As a demo operator, I want near-duplicate entity mentions to be merged into one node, so that the graph does not fragment into dozens of spellings of the same organisation.
18. As a demo operator, I want entity merging to be conservative enough not to fuse genuinely distinct entities, so that two different clients' similarly-named projects stay separate.
19. As a demo operator, I want relationships between entities to carry the evidence and source chunk they were derived from, so that any edge shown on screen can be justified when challenged.
20. As a demo operator, I want adjacent chunks within a document linked in reading order, so that retrieved context can be expanded to neighbouring passages.

### Retrieval and chat

21. As a demo audience member, I want the chatbot to answer questions about the corpus in natural language, so that I can judge the system without learning a query language.
22. As a demo operator, I want retrieval to combine semantic and lexical search, so that a distinctive token such as a project code still finds the right document even when every chunk embeds almost identically.
23. As a demo operator, I want retrieval to also traverse the entity graph from entities named in my question, so that answers can draw on documents that share no vocabulary with the question.
24. As a demo operator, I want the two retrieval strategies fused into one ranked context, so that the answer benefits from both without me choosing between them.
25. As a demo operator, I want to run each retrieval leg in isolation via a flag, so that I can show the difference each one makes.
26. As a demo operator, I want to tune the number of retrieved chunks and the graph hop depth at query time, so that I can adjust live if results are too noisy or too narrow.
27. As a demo operator, I want each retrieved chunk to carry its folder ancestry into the model's context, so that the answer can state which client and year a fact came from.
28. As a demo audience member, I want to ask a follow-up question using pronouns and ellipsis, so that the conversation feels natural.
29. As a developer, I want follow-up questions rewritten into standalone questions before retrieval, so that conversational shorthand does not break search.
30. As a demo audience member, I want answers to cite which documents they came from, so that I can verify them.

### Visualisation

31. As a demo audience member, I want to see the subgraph that produced each answer, so that I understand *how* the system reached its conclusion rather than trusting an opaque response.
32. As a demo audience member, I want edges that cross between different branches of the hierarchy highlighted distinctly, so that the cross-client connection is immediately obvious.
33. As a demo operator, I want the graph panel to be the primary visual rather than an afterthought, so that the structural advantage is the thing the audience looks at.
34. As a demo audience member, I want to see which chunks were retrieved and which entities they led to, so that I can trace the reasoning path node by node.
35. As a demo operator, I want the graph to render legibly for a mixed hierarchy-plus-entity-web structure, so that it reads clearly on a projector.

### Corpus

36. As a demo operator, I want a synthetic corpus generated that is deliberately adversarial — repeated project names across clients, shared personnel, near-identical boilerplate — so that the demo exercises exactly the failure mode we are claiming to solve.
37. As a demo operator, I want the synthetic corpus in German, matching our real corpus, so that chunking and extraction behaviour is representative.
38. As a demo operator, I want the pipeline to work unchanged on the real corpus once it is accessible, so that the synthetic corpus is a stand-in and not a special case.

### Operation

39. As a demo operator, I want a single command to bring up the infrastructure, so that I can start the demo reliably after not touching it for weeks.
40. As a demo operator, I want separate commands for ingestion, one-off querying, and serving the UI, so that I can script a demo or debug retrieval without the frontend.
41. As a demo operator, I want all model identifiers and connection URLs in environment configuration with a committed example file, so that setup is self-documenting.
42. As a developer, I want a README documenting the exact startup sequence, so that someone else can run the demo without me.

## Implementation Decisions

### Scope and posture

This is an explicitly throwaway proof of concept. Decisions trade longevity for demo clarity
and speed of construction. Where that trade has been made deliberately it is noted.

### Pipeline shape

Ingestion is a linear pipeline: directory walk, PDF parsing into layout-aware elements,
element-aware chunking, embedding, entity and relation extraction, graph write. Query is a
hybrid retrieval followed by generation.

The architectural seam is the pipeline boundary: ingestion and question-answering are each a
single function, with the three external dependencies — the model client, the parser client
and the graph driver — injected as protocols rather than constructed internally.

### Parsing and chunking: reuse, not rewrite

Parsing, element normalisation and chunking are lifted wholesale from the existing production
ingestion service and vendored into this project in a clearly marked directory, stripped of
that system's settings and dependency-injection framework, with their origin noted. Rationale:
this path is solved, boring, and not what the demo is about; reusing it means chunk boundaries
match production, so demo behaviour predicts real behaviour.

The vendored chunking retains its tuned German defaults: title-aware strategy, a hard character
cap with a lower soft cap so chunks trend to uniform size, a floor below which consecutive small
sections are combined, and a character overlap carried across splits. Tables and images are
isolated as their own chunks. Chunk identifiers are derived deterministically from document
identity and chunk index, so re-ingestion upserts rather than accumulates.

Everything graph-related is written fresh; there is no precedent to copy.

### PDF parsing transport

Parsing is delegated to the self-hosted Unstructured HTTP service already run by the production
system, reached over HTTP at a configurable URL. Rejected alternatives: the in-process library
(pulls a multi-gigabyte ML stack into the project's environment) and the hosted vendor API
(requires a paid key we do not have). The cost of this decision is that the demo has a second
service dependency that must be running.

OCR for scanned PDFs is out of scope; the corpus is born-digital.

### Adaptive hierarchy

Folders are modelled generically — a single node type at any depth, linked by containment —
rather than as typed per-level entities. This is what makes arbitrary depth work without
configuration.

Level *semantics* are recovered separately: after the walk, one inexpensive model call per
depth level inspects the sibling folder names at that level and infers what the level
represents, storing the inferred name and a confidence score on each folder node at that depth.
An explicit ordered list of level names may be supplied at ingestion time to override inference,
falling back to inference for any depth beyond the supplied list.

Rationale: "point it at any folder tree and it works out what the levels mean" is a materially
better demo than "configure your schema first", and costs one small call per level.

### Entity extraction

Extraction uses a **fixed taxonomy**, not open-ended extraction: person, organization,
system or product, concept, metric or deliverable, and risk — together with a fixed list of
relation types. Open-ended extraction was rejected because it produces inconsistent types that
visualise badly, and the visualisation is the point.

Each extracted relation records the evidence span and the chunk it came from, so any edge
displayed can be justified.

### Entity resolution

Resolution runs in three escalating stages: exact match on a normalised name; then merge by
embedding similarity above a threshold; then a model adjudication for the ambiguous middle band
only.

This is identified as the **highest-risk component in the system**. On a corpus engineered
around near-duplicates, too-loose merging collapses distinct entities together and too-strict
merging fragments the graph into spelling variants — and a fragmented graph demonstrates the
*opposite* of the intended thesis. Thresholds are expected to require tuning against the real
generated corpus rather than being correct on the first attempt.

### Storage

A single Neo4j instance holds everything: the folder hierarchy, documents, chunks (including
their embedding vectors), and entities. A separate dedicated vector store was considered and
rejected in favour of keeping the demo focused on the graph and avoiding a second store that
could drift out of sync.

The consequence of that choice is that the sparse/lexical retrieval leg must be recovered
inside the graph database: a Lucene full-text index over chunk text, alongside the native
vector index. Without it, "Neo4j only" would quietly mean "dense-only retrieval", which is the
weakest possible configuration on precisely this corpus.

Schema — nodes:

- **Folder** — name, path, depth, inferred level name, inferred level confidence
- **Document** — filename, path, content hash, page count, generated summary
- **Chunk** — deterministic id, text, index within document, page, embedding
- **Entity** — surface name, normalised name, type, embedding

Schema — relationships:

- Folder *contains* Folder or Document
- Document *has chunk* Chunk
- Chunk *next* Chunk (reading order within a document)
- Chunk *mentions* Entity, carrying evidence
- Entity *relates to* Entity, carrying relation type, evidence and source chunk

Entities are deliberately **not** attached to any folder. This is the single most important
modelling decision in the spec: it is what makes a cross-client path real structure in the
graph rather than a join computed at query time, and therefore what makes the visualisation
honest.

### Retrieval

Hybrid, two legs run in parallel.

The first leg is lexical-plus-semantic: a vector similarity search and a full-text search over
chunk text, each returning a ranked list, fused by reciprocal rank fusion.

The second leg is entity-driven: entities are extracted from the question, matched against
graph entities by normalised name and embedding, then traversed one to two hops along relation
edges, collecting the chunks that mention the entities reached.

The two chunk sets are merged, each chunk annotated with its folder ancestry, and the combined
context capped before generation. Default parameters — ten results per leg, two hops, a cap of
roughly a dozen chunks — are exposed as query parameters so they can be adjusted during a live
demo. Hop depth in particular is expected to be adjusted on stage: one hop is precise, two is
where cross-document connections appear, three is noise.

Each leg can also be run in isolation by flag, to demonstrate the contribution of each.

Text-to-Cypher retrieval was considered and rejected: flashy but brittle, with ugly failure
modes in front of an audience.

### Conversation

Multi-turn, with a bounded window of recent turns retained. Before retrieval, a follow-up
question is rewritten by the model into a standalone question. Rationale: an audience will
reliably ask a pronoun-laden second question, and a standalone-only system faceplants visibly
at that moment.

### Models

All generation and embedding goes directly to the Mistral API with a single API key. A smaller,
cheaper model handles the high-volume per-chunk extraction; a larger model handles chat answers,
level-semantics inference and query rewriting; the vendor's embedding model produces 1024-dimension
vectors, which the graph's vector index is configured to match. All model identifiers are
environment-configurable rather than hardcoded, and will be verified against the live API before
wiring.

### Interface

Three layers over one set of code paths. A command-line interface provides ingestion, one-off
querying and serving. An HTTP API exposes the same operations. A single-page frontend provides
the chat and the graph visualisation, built with a force-directed graph renderer chosen because
the structure is a mixed hierarchy-plus-entity-web that hierarchical layouts handle poorly.

The traversal visualisation is the **primary** visual, not a side panel: answer on one side,
live subgraph on the other, with cross-hierarchy edges drawn in a distinct colour.

### Note on the dropped baseline

A side-by-side comparison against flat RAG was considered and deliberately cut from scope. The
consequence, recorded explicitly: the traversal visualisation now carries the entire burden of
proof. If that visual does not land, the demo reads as "a chatbot that answered a question."
This is a single point of failure by choice.

### Caching and ingestion robustness

Extraction results are cached keyed by document content hash — this both makes iteration fast
and keeps model spend near zero across repeated runs, and lets a stage demo be re-run in seconds.
Incremental re-ingestion, crash resumption and deletion handling are deliberately not implemented.

### Operation

The graph database runs in a container. The Python application is managed with its standard
project tooling. The parsing service is started by the production system's own process manager.
A task-runner file wraps the multi-step startup into single commands, and a README documents the
sequence — for a demo that will be run in front of people after weeks of neglect, a one-command
start matters more than it sounds.

## Testing Decisions

### What makes a good test here

Tests assert external, observable behaviour: given this input directory, the graph contains
these nodes and these relationships; given this question, retrieval returns these chunks in
this order. They do not assert how a function reached that result, do not reach into private
state, and do not mirror the implementation's structure. A test that would need rewriting
because the implementation was refactored without changing behaviour is a bad test.

Because this is an explicitly throwaway proof of concept, test coverage is deliberately narrow.
The criterion for including a test is **silent failure**: logic whose bugs would not announce
themselves, and would instead make the graph quietly wrong in a way first noticed during a
live demo.

### What is tested

Three pieces of pure logic, tested directly:

- **Folder-node construction at arbitrary depth** — that a tree of any shape and depth produces
  the correct node and containment structure. This is the adaptivity claim; if it is wrong at
  depth four, nothing above surfaces an error.
- **Entity normalisation and merging** — that spelling variants of one entity collapse to a
  single node, and that genuinely distinct similarly-named entities do not. This is the
  highest-risk component; both of its failure modes are silent and both destroy the demo.
- **Reciprocal rank fusion** — that two ranked lists fuse into the expected combined order.
  An incorrect fusion degrades retrieval quality without ever raising an error.

Plus one **end-to-end smoke test**: a real ingestion of a tiny corpus of a few PDFs against the
real graph database and real model API, asserting that nodes and relationships of each expected
kind exist and that a known question retrieves a known document. This is the only test that
catches wiring and configuration errors, and it is cheap because the infrastructure is running
anyway.

Everything else is verified by running a real ingestion and using the application.

### Prior art

There is no prior art within this project — it is greenfield. The production ingestion service
from which parsing and chunking are vendored carries its own extensive unit tests for parsing
providers, chunking and normalisation; the vendored code is treated as already-tested and is not
re-covered here.

## Out of Scope

- **Flat-RAG side-by-side baseline.** Considered and deliberately cut. See the note under
  Implementation Decisions regarding the consequence.
- **Community detection and global summarisation** (Microsoft GraphRAG style) — expensive to
  ingest and slow, with little payoff for the demo's specific thesis.
- **Text-to-Cypher retrieval.**
- **OCR for scanned PDFs** — the parser is configured for born-digital documents. A corpus of
  scans would require revisiting this.
- **Incremental re-ingestion, crash resumption, and handling of deleted files.**
- **Authentication, authorisation, multi-tenancy.**
- **Deployment, hosting, CI.**
- **Production-grade error handling and observability.**
- **Performance and scale work.** The corpus is demo-sized.
- **Use of the real client corpus**, which is not currently accessible. The pipeline is built to
  work on it unchanged, but this is unverified until it can be run against it.

## Further Notes

**Known risks, in order of severity.**

Entity resolution thresholds are the most likely thing to go wrong, for the reasons given above,
and should be the first thing tuned once the real corpus is available.

The traversal visualisation is a single point of failure for the demo's persuasiveness, having
absorbed the role the dropped baseline would have played.

Dense retrieval on a corpus of near-identical boilerplate is inherently weak, which is why the
full-text index is not optional despite the single-store decision.

**Unverified assumptions, to be confirmed before or during implementation.** The exact current
model identifiers should be checked against the live vendor API rather than assumed. The
production parsing service should be confirmed to start cleanly in isolation, since this project
depends on it without the rest of that system.

**Environment.** Requires the container runtime to be running and the parsing service to be up.
Neither is currently running.
