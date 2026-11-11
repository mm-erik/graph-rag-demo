# Graph RAG Demo

A throwaway proof of concept: graph RAG over a deeply nested, highly self-similar German
document corpus (`Client / Year / Project / report.pdf`), where flat RAG fails silently.
See `SPEC.md` (source of truth) and `DESIGN.md` (rationale and rejected alternatives).

**Status:** tickets 01–16 — infrastructure, the synthetic corpus, parsing/chunking, the
folder/document graph, adaptive level inference, chunks with their embeddings, a generated
summary per document, the entity/relation graph extracted from those chunks, the ingestion
cache that makes re-running all of it cost seconds, and both retrieval legs: `search`
(lexical and semantic, fused) and `traverse` (the entity graph walk). `ingest` writes the
whole corpus, `query` fuses both legs into a cited German answer, and `chat` resolves
follow-up questions against the conversation. `serve` puts the same pipeline behind HTTP
and serves the React frontend beside it: the answer on one side, the subgraph that
produced it on the other, with every edge that leaves the hierarchy drawn in gold. One
end-to-end smoke test runs that whole chain against the real infrastructure.

> **Running the demo?** `RUNBOOK.md` is the operational document: the cold-start sequence
> from a machine where nothing is up, what to check when something is wrong, the settled
> entity-resolution thresholds, and the demo script — which questions to ask, in which
> order, and what each one is supposed to show.

> **Verified against the live API:** `mistral-small-latest`, `mistral-large-latest` and
> `mistral-embed` all exist, and the embedding width really is 1024 — no discrepancy with
> the spec's table. Re-check any time with `just verify-models`.

## Prerequisites

- [Docker](https://www.docker.com/) running (Neo4j runs in a container)
- [`uv`](https://docs.astral.sh/uv/) and [`just`](https://just.systems/)
- A Mistral API key
- The self-hosted Unstructured service from the sibling `monorepo` checkout (only needed
  from ticket 03 onwards), reachable at `UNSTRUCTURED_URL`
- Node 20+ and npm, for the frontend

## Startup sequence

```sh
cp .env.example .env     # then fill in MISTRAL_API_KEY
just up                  # start Neo4j, install deps, create constraints + indexes
just verify-models       # confirm the Mistral model identifiers against the live API
just status              # confirm connectivity and that every index is ONLINE
just corpus              # generate the synthetic corpus into ./corpus
just ingest corpus --levels "Kunde,Jahr,Projekt"   # walk the tree into the graph
just ui-install          # once per checkout: the frontend's dependencies
just demo                # build the UI and serve it with the API on :8000
```

`just up` is safe to re-run: the schema statements are idempotent, and the graph lives in
named Docker volumes so it survives `just down` and machine restarts.

This is the short form. From a genuinely cold machine — Docker stopped, the parsing
service in the sibling `monorepo` checkout not running — follow `RUNBOOK.md` instead: it
starts the two slow things first and says how long each one takes.

Neo4j Browser: <http://localhost:7474> (user `neo4j`, password from `.env`).

**Changing `NEO4J_PASSWORD` after the first start does not work.** Neo4j only reads
`NEO4J_AUTH` when it initialises an empty data volume, so the container keeps the original
password while the healthcheck starts using the new one — `just up` then hangs until it
times out. To actually change it, remove the volume first (`docker compose down -v`, which
discards the graph) and re-ingest.

## Commands

| Command | What it does |
|---|---|
| `just up` | Start Neo4j, sync dependencies, apply the schema |
| `just init` | Apply constraints and indexes (idempotent) |
| `just status` | Report connectivity, index state and node counts; exits non-zero if not ready |
| `just verify-models` | Check configured Mistral model IDs and the real embedding width |
| `just corpus [dir]` | Generate the synthetic adversarial German corpus (default `./corpus`) |
| `just parse <pdf>` | Parse one PDF and print the chunks it would contribute (needs Unstructured, unless the parse is already cached — so chunk parameters can be tuned against a real document for free) |
| `just ingest <dir>` | Walk a document hierarchy into the graph: structure, level names, chunks with embeddings, a summary per document, and the entities and relations they name. Cached — a re-run over an unchanged corpus takes seconds and makes no API calls; `--no-cache` forces a clean run |
| `just summaries` | Print the stored document summaries, to compare near-identical reports |
| `just extract` | Re-run entity extraction over the chunks already in the graph (no re-parsing); `--no-cache` forces re-extraction |
| `just entities [--bridges]` | Print the extracted entities; `--bridges` shows only those joining two branches |
| `just search "<q>"` | Retrieve chunks for a question — vector and full-text search fused, each result labelled with its client, year and project; `--mode vector` / `--mode fulltext` runs one leg alone, `--top-k` sets results per leg |
| `just traverse "<q>"` | Retrieve chunks for a question by walking the entity graph from the names in it — the second leg, alone; `--hops` sets how far the walk goes, `--threshold` how loosely a name has to match |
| `just query "<q>"` | Answer a question from both legs, merged and cited — the whole system in one command; `--legs search` / `--legs traversal` answer from one leg alone, `--max-chunks` caps the context, `--hops` and `--top-k` as above |
| `just chat ["<q>"]` | Ask follow-up questions, resolved against the conversation so far — "und 2023?" is rewritten into a standalone question before retrieval; `--window` bounds the turns remembered, `--window 0` turns the memory off, plus every `just query` option |
| `just serve` | Run the HTTP API: question answering plus the subgraph behind each answer, and an operator status endpoint — and the built frontend at `/`, if `just ui-build` has run; `--port` and `--host` override `API_PORT` and `API_HOST` |
| `just ui-install` | Install the frontend's dependencies. Once per checkout |
| `just ui-build` | Build the frontend into `frontend/dist`, which `just serve` then serves at `/` |
| `just ui` | The frontend with hot reload on :5173, proxying `/api` to the service. For working on it, not for the demo |
| `just demo` | **What to run on stage.** Builds the UI and serves it with the API, one process |
| `just test` | The fast suite: pure logic only, no infrastructure, under a second |
| `just smoke` | The end-to-end smoke test, against real Neo4j, Mistral and parsing service. Expects a cold graph — see `RUNBOOK.md` |
| `just down` / `just logs` | Stop the container (volumes kept) / tail Neo4j logs |

## Graph schema

Constraints: `Folder.path`, `Document.path` and `Chunk.id` unique; `Entity` unique on
`(normalized_name, type)`.

Indexes:

- `chunk_embedding` — native vector index on `Chunk.embedding`, 1024-dim, cosine
- `entity_embedding` — native vector index on `Entity.embedding`, 1024-dim, cosine
  (entity resolution and the traversal leg match entities by embedding)
- `chunk_text_fulltext` — Lucene full-text index on `Chunk.text`, German analyzer

Relationships: `(:Folder)-[:CONTAINS]->(:Folder|:Document)`, `(:Document)-[:HAS_CHUNK]->(:Chunk)`,
`(:Chunk)-[:NEXT]->(:Chunk)`, `(:Chunk)-[:MENTIONS]->(:Entity)` and
`(:Entity)-[:RELATES]->(:Entity)`. There is deliberately **no** edge between `Folder` and
`Entity` — see "Entities and relations" below.

The full-text index is not optional. A single store was chosen deliberately, so the
sparse/lexical retrieval leg has to be recovered *inside* Neo4j; without it, "Neo4j only"
quietly becomes "dense-only", which is the weakest possible configuration on a corpus
engineered out of near-identical boilerplate.

`CREATE ... IF NOT EXISTS` ignores the `OPTIONS` of an index that already exists, so a
vector index at the wrong width or a full-text index with the wrong analyzer would survive
re-initialisation unnoticed — and both fail silently at query time rather than erroring.
`just init` therefore detects that drift, drops the index and recreates it correctly, and
`just status` fails on it. Note that recreating a vector index leaves any embeddings
written at the old width invisible to vector search: re-ingest after a dimension change.

## The folder and document graph

`just ingest <dir>` walks a directory to **any depth** and writes what it finds as the
structure of the graph: one `Folder` node per folder, linked by `CONTAINS`, with the PDFs
hanging off whichever folder holds them.

```
(:Folder {name, path, depth}) -[:CONTAINS]-> (:Folder)
(:Folder) -[:CONTAINS]-> (:Document {filename, path, content_hash})
```

Browse it in the Neo4j Browser at <http://localhost:7474>:

```cypher
MATCH p=(:Folder)-[:CONTAINS*]->(n) RETURN p LIMIT 200
```

Three properties of the walk are deliberate and load-bearing:

- **Folders are generic.** One `Folder` label at every depth, never `Client`/`Year`/
  `Project` labels. That is the whole reason a corpus nested deeper than the demo shape
  works without configuration. What each level *means* is recovered separately (ticket 05)
  and is not encoded in the node type.
- **Folders come from the documents, not from the disk.** Only folders on the path to a
  PDF become nodes, so a `.git` directory or an empty branch never reaches the graph.
- **Re-running is safe.** Every write is a `MERGE` on the node's path, so a second ingest
  over the same tree creates nothing and reports as much. A file whose bytes changed keeps
  its node and has its `content_hash` updated in place.
- **Depth comes from the graph, not from the walk.** The walk only knows depth relative to
  the root it was pointed at, so ingesting a subdirectory of an already-ingested tree
  (`just ingest "corpus/Nordlicht Energie AG"`) would stamp that branch a level too
  shallow — silently, with the old containment edge still attached, and ticket 05 groups
  sibling folders by exactly this property. Depth is therefore recomputed from the
  containment chain on every ingest, which both prevents and repairs it; the command says
  how many depths it corrected.

The content hash is SHA-256 over the file's bytes. It is what the ingestion cache keys
on — both the summary stored on the node and, below, the parse of the file itself — so a
document that changed is reprocessed and one that did not is never touched again.

The walked root itself is the depth-0 folder, so the corpus is a single navigable tree
rather than a handful of disconnected client subtrees.

## What the levels mean

The folder graph is deliberately untyped — one `Folder` label at every depth — so the
graph alone cannot say that depth 1 is a client and depth 2 a year. `ingest` recovers that
afterwards and stores it on every folder as `inferred_level_name` and
`inferred_level_confidence`.

```
(:Folder {name, path, depth, inferred_level_name, inferred_level_confidence})
```

**One model call per depth, not per folder.** Each call sees the sibling folder names at
one depth (up to 40 of them, plus how many folders and how many distinct names there are)
and answers with a noun and a confidence. A four-level corpus of a thousand folders costs
four calls. The chat-tier model is used rather than the extraction-tier one: this is
low-volume and its output is read aloud to the audience.

**The override is the escape hatch.** A bad inference must not derail a live demo, so
`--levels` bypasses the model entirely for the depths it covers:

```sh
just ingest corpus --levels "Kunde,Jahr,Projekt"   # 0 model calls
just ingest corpus --levels "Kunde"                # depth 1 fixed, 2 and 3 inferred
```

The nth name applies to the nth level **below the walked root**. The root is depth 0 — the
corpus itself, not a level of it — so `Kunde,Jahr,Projekt` means depths 1, 2 and 3. Names
past the last real depth are ignored rather than inventing a level.

Ingestion prints every level with its confidence and whether it was inferred or overridden,
and flags anything inferred below 0.9 — the operator needs to know a level is shaky
*before* presenting, not during. That threshold is tuned against what the model actually
emits: legible corpora come back at 0.95-0.98 and opaque ones (`XK-01`, `a7`) at
0.70-0.80.

Levels are scoped to **one tree**: the hierarchy containing the walked root. Two unrelated
corpora in the same graph have their own meanings for depth 1 — one's clients are another's
regions — so sampling or labelling across both would describe neither. Ingesting a
subdirectory of an already-ingested tree still names that whole tree, since the sibling
names that characterise a level live in the tree, not in the subdirectory.

If a level needs the model and `MISTRAL_API_KEY` is unset, ingestion writes the folder and
document structure, then stops with a non-zero exit rather than leaving a hierarchy whose
levels are silently unnamed. Set the key and re-run, or pass `--levels`; the structure
write is idempotent, so the second run only adds the names.

A level the model could not be reached for is recorded as `unbekannt` at confidence 0.0
and the error is printed, rather than discarding the levels that did resolve — losing
three paid-for calls because the fourth hit a rate limit is not a good trade mid-demo.

## Document summaries

Every `Document` node carries a generated German summary of its own content, written at
ingestion time, plus the content hash that summary was written from.

```
(:Document {filename, path, content_hash, page_count, summary, summary_hash})
```

They exist for the questions chunks cannot answer. "Worum geht es in diesem Bericht?" or
"was haben wir 2023 für diesen Kunden gemacht?" are about whole documents and whole
folders; answering them from chunks means pulling every chunk of every document in a
branch, which exhausts the context budget and retrieves nothing useful. **One chat-tier
call per document**, not per chunk — low volume, and the output is read back to the
audience, which is why the chat tier rather than the extraction tier.

Two decisions are load-bearing, and both fail silently:

- **The excerpt is spread across the document, not taken from its front.** Every report in
  this corpus opens with the same cover page and the same preamble, so a summary written
  from the first few thousand characters describes the *template*. Sections are drawn at
  evenly spaced positions, first and last always included, and the character budget is
  shared out between them rather than spent front to back. Nothing raises when this is
  wrong — the summaries simply stop telling documents apart, which is the one job they have.
- **The prompt is given the document's place in the hierarchy** (`Kunde: Nordlicht Energie
  AG`, `Jahr: 2023`, `Projekt: Atlas`), labelled with the inferred level names from ticket
  05 where they exist and with bare folder names where they do not. A report that says
  "der Auftraggeber" throughout never names its client in its own text; the client lives in
  the folder structure. The labels are a nice-to-have, not a dependency.

**Summaries are cached by the document's content hash.** The summary is stored next to the
hash of the bytes it describes, so re-ingesting an unchanged corpus makes no chat calls at
all, and a document whose bytes changed is re-summarised. `--no-cache` forces regeneration
— the escape hatch for a changed *prompt*, which no content hash can see.

This is the one cached artefact that lives in the graph rather than in the ingestion
cache described below, because the summary is a property of the document either way and
a sidecar copy of it could only drift.

A document whose summary fails — a rate limit, an unusable reply, a PDF that parsed to
nothing — is reported and skipped. It keeps no summary rather than a bad one, since at
query time a stored apology reads exactly like a real summary.

Compare two near-identical reports from different clients:

```sh
just summaries --contains projektbericht-2023 --limit 4
```

## The synthetic corpus

The real client corpus is not currently accessible, so the demo is developed against a
generated stand-in — `just corpus` writes ~160 born-digital German PDFs in two halves.
Generation is seeded, so the same seed yields byte-identical files.

The **client half** is ~120 PDFs as `Client / Year / Project / *.pdf` (4 clients x 3 years
x 3-4 projects x 3 documents). It is deliberately hostile to flat retrieval:

- `Projekt Atlas` exists at two unrelated clients with entirely different work behind it.
- The only thing joining those two projects is a **shared entity** — the consultant
  Dr. Katharina Vogt and the `Atlas-Referenzarchitektur` she carried across. **No document
  names the other client**, so the connection exists only in the graph; no single chunk
  contains it. That is the demo's whole claim, and tests guard it — including that she is
  staffed *only* on the projects she bridges, since a consultant on every client's work
  would be a hub that connects nothing in particular.
- A second, weaker bridge (`Projekt Datenschild`, Milan Horvat) shows the first is not a fluke.
- Every status report reuses the same headings and sentence frames, varying only names and
  numbers, so dense retrieval genuinely confuses one client's report for another's.

The **machine half** is 42 PDFs as `Manufacturer / Machine type / Model / *.pdf`
(3 manufacturers x 2 machine types x 2-3 models x 3 documents: a data sheet, a variant
overview and a maintenance plan). It is three folder levels deep, like the client half,
but the levels *mean* something else — so one corpus now carries two branches whose
depth-1/2/3 labels do not agree, which is exactly the case a single flat `--levels` list
cannot describe.

Where the client half is hostile because *names collide*, this half is hostile because
*names mean nothing*:

- Machine types, models and assemblies are opaque codes — `XTC27`, `B2BC`, `HX-114`. An
  embedding of the question sits nowhere near the right document, and a lexical hit on a
  code is worthless, because the same code names more than one thing.
- `B2BC` is a model of the `XTC27` vacuum mixer-dryer **and** of the `ZM3K` assembly cell,
  at different manufacturers, with different everything. `XTC27` itself is a mixer-dryer at
  one manufacturer and a roller-track transfer unit at another. The option code `HK` means
  *Heizkreis* at one and *Hubkonsole* at the others. Only the branch above a code
  disambiguates it.
- Each model is sold in several variants, and a variant's empty weight and height are
  **additive**: base model plus the deltas of the options it carries. So „was wiegt ein
  XTC27 B2BC mit Fahrwerk und Ölauffangwanne?" has exactly one right answer (1.625 kg,
  one table row in one document) and a plausible answer from the base row, a sibling
  variant or the colliding `ZM3K B2BC` is wrong by a checkable amount.
- Two further cross-**manufacturer** bridges, built like the client half's: the
  `Planetengetriebe HX-114` carried by Dipl.-Ing. Reinhard Oswald from `XTC27-B2BC` to
  `QH4L-P6V`, and the weaker `Spannsatz KN-30` carried by Beate Lindqvist. **No document
  names the other manufacturer** — the join is the assembly and the engineer, and tests
  guard that, including that each engineer appears only on the two models he or she
  bridges.
- All three document kinds reuse one set of headings and sentence frames across all
  fourteen models, so what separates two data sheets is a handful of codes and a column of
  numbers.

Re-running over an existing corpus requires `--force` (`just corpus corpus --force`): a
different seed selects a different project set, and ingestion walks the directory rather
than the plan, so orphaned folders would be ingested as if they belonged.

`corpus/KOLLISIONEN.md` (written by the generator) documents every planted collision and
the questions to ask on stage. The generator lives in `src/graph_rag/corpus/` and nothing in
the ingestion path imports it: the synthetic corpus is a stand-in for the real one, never a
special case the pipeline knows about.

## Parsing and chunking

Parsing, element normalisation and chunking are **vendored wholesale** from the production
ingestion service in the sibling `monorepo` checkout, into `src/graph_rag/vendored/`. Each
file names its origin and marks every deviation `LOCAL CHANGE`. Do not improve them: the
point of copying is that chunk boundaries here match production, so demo behaviour predicts
real behaviour — a rewrite silently decouples the two.

The tuned German defaults are carried over exactly (`by_title`, hard cap 2000, soft cap
1600, combine floor 500, overlap 250). German is token-dense, so the caps are expressed in
characters. Tables and images are isolated into their own chunks so each stays individually
citable, and chunk ids are `uuid5` of document identity plus index, so re-ingesting a
document replaces its chunks rather than accumulating duplicates.

Document identity is the **resolved path**, not a content hash: this corpus is built out of
near-identical reports, and hashing content would give two different clients' reports the
same chunk ids and silently collapse them.

### The parsing service

Parsing goes over HTTP to the self-hosted Unstructured service (`UNSTRUCTURED_URL`,
default `localhost:8100`), which is **not** part of this project — it is run by the
monorepo. To start it on its own:

```sh
cd ../monorepo
nix develop --command process-compose -f nix/services/unstructured.yml up --tui=false
```

Then `just parse "corpus/Nordlicht Energie AG/2023/Projekt Atlas/projektbericht-2023.pdf"`.

Notes from running it standalone: it comes up cleanly without the rest of that system
(it needs nothing from the backend, the database or NATS), and answers `/healthcheck`
within a few seconds on a warm Nix store and model cache — the service config budgets 20s
before its readiness probe starts, because a cold start imports a multi-gigabyte ML stack.
Keep it at **one worker**: each worker is a full copy of that stack, and two have
OOM-killed a worker mid-parse on this machine. A `hi_res` parse of a two-page German PDF
takes roughly 10-15 seconds.

## Entities and relations

This is where the graph stops being a folder tree and becomes a knowledge graph. Entities
named in the documents become nodes linked to the chunks that mention them and to each
other, and a consultant who worked for two different clients becomes **one node joining
two separate branches of the hierarchy**. That cross-folder edge is the thing the whole
demo exists to show.

```
(:Entity {name, normalized_name, type, scope, mentions, aliases, embedding})
   identity = (normalized_name, type, scope) — unique
(:Chunk)-[:MENTIONS {evidence, surface}]->(:Entity)
(:Entity)-[:RELATES {type, evidence, source_chunk, support}]->(:Entity)
```

**Entities are global and carry no folder edge of any kind.** This is the single most
important modelling decision in the project: it is what makes a cross-client path *real
structure* in the graph rather than a join computed at query time, and therefore what
makes the visualisation honest rather than staged. `just entities` asserts it on every
run and says so in red if it is ever violated.

### The fixed taxonomy

Six entity types — `person`, `organisation`, `system`, `konzept`, `ergebnis`, `risiko` —
and nine relation types: `arbeitet_fuer`, `leitet`, `mitarbeit_in`, `verantwortet`,
`setzt_ein`, `liefert`, `hat_risiko`, `basiert_auf`, `teil_von`. Open-ended extraction was
considered and rejected: it produces inconsistent types that visualise badly, and the
visualisation is the entire point. Anything outside the lists is dropped rather than
stored, and each relation is checked against the kinds of entity it may join — a
`arbeitet_fuer` edge pointing at a document is a claim its own evidence does not support.

One extraction-tier call per chunk. This is the **high-volume call site in the system**
and the main cost of an ingest (766 chunks ≈ 2m50s at `ENTITY_CONCURRENCY=8`) — and
therefore the call the ingestion cache below exists for. Re-running `just extract` over
an unchanged corpus replays those 766 results from disk in under three seconds and calls
the API not at all.

Every mention and every relation carries **evidence taken from the chunk itself**, not
from the model's reply. A paraphrased span is replaced by the chunk's own sentence, and a
mention whose name appears nowhere in the chunk is dropped outright — the model invented
it. This is not decorative metadata: it is what lets any edge on screen survive an
audience member asking where it says that.

### How two clients' `Projekt Atlas` stay apart

The corpus plants a name collision: `Projekt Atlas` exists under **Nordlicht Energie AG**
and under **Brandt & Söhne Maschinenbau**, on entirely unrelated work, and *no document
names the other client*. It also plants the opposite case: the `Atlas-Referenzarchitektur`
is named identically under both clients and genuinely **is** the same artefact, carried
across by Dr. Katharina Vogt. Both are the same taxonomy type, so no rule based on type
can separate them.

The folder tree decides. A name is scoped to its branch when, and only when, it matches a
folder on its own document's path *and* that folder name occurs under more than one
top-level branch:

| Name | Matches a folder on its path? | Ambiguous across branches? | Result |
|---|---|---|---|
| `Projekt Atlas` | yes | yes | two nodes, scoped per client |
| `Atlas-Referenzarchitektur` | no | – | one global node, joining both |
| `Projekt Windfeld` | yes | no (one client only) | one global node |
| `Dr. Katharina Vogt` | no | – | one global node, joining both |

Scope is a third component of the entity's *identity*, alongside the normalised name
and the type — never an edge. `normalized_name` stays a normalised name, so the
traversal leg can match "Projekt Atlas" from a question and reach *both* projects rather
than neither. Comparison never crosses a scope: once the hierarchy has ruled two names
distinct, no threshold and no model call may overturn it.

This rule is an addition to the spec, which expects the three resolution stages to keep
same-named entities apart on their own. They cannot: the two `Projekt Atlas` strings are
*identical*, so their cosine similarity is exactly 1.0 and an adjudicator shown two
identical names will rightly call them the same thing. Nothing inside the text can
separate them — the disambiguating knowledge lives in the folder tree and nowhere else,
which is this project's whole thesis. The honest cost is that a scoped node is pinned to
its branch and can never join another; it is applied only to names the hierarchy proves
ambiguous, which on this corpus is `Projekt Atlas` and `Projekt Datenschild` and nothing
else.

### Resolution, and the thresholds it runs on

Four stages, cheapest first:

0. **One name, one type.** The taxonomy is applied per chunk, so the same artefact comes
   back as `ergebnis` in one chunk and `konzept` in the next. Since comparison never
   crosses a type, that would be two nodes nothing could ever rejoin — the quiet way to
   cut the demo's one cross-client edge in half. The commoner type wins.
1. **Exact match** on the normalised name. German-aware: `ä→ae`, `ß→ss`, `&→und`, legal
   forms stripped from organisations and titles from people, so `Brandt & Söhne
   Maschinenbau GmbH` and `Brandt und Soehne Maschinenbau` are one entity for free.
2. **Embedding similarity** at or above `ENTITY_MERGE_THRESHOLD`.
3. **Model adjudication**, for the band between `ENTITY_REVIEW_THRESHOLD` and the merge
   threshold only — the one expensive stage, asked only about pairs the thresholds
   genuinely cannot decide (66 pairs on this corpus).

Both failure modes are silent and both destroy the demo: too loose and distinct entities
collapse, too strict and the graph fragments into spelling variants — and a fragmented
graph demonstrates the *opposite* of this project's thesis.

**The thresholds are `merge = 0.92`, `review = 0.75`, measured against the generated
corpus rather than guessed:**

| Pair | Cosine | Verdict |
|---|---|---|
| `Schnittstellenverzeichnis` / `Schnittstellenverträge` | **0.9013** | must *not* merge |
| `Vektor Consulting GmbH` / `Vektor Con` | **0.7990** | must merge |
| `SAP IS-U` / `SAP S/4HANA` | **0.8076** | must *not* merge |

The first pair sets the floor for `merge`: at 0.90 the two deliverables fuse, so 0.92. The
second sets the ceiling for `review`: a parser-truncated spelling of the consultancy sits
at 0.799, so anything at or above 0.80 leaves it as its own node and the client fragments.

The last two are the whole argument for stage three. They are **0.009 apart and on
opposite sides of the answer** — no threshold can separate them, and only a model asked
about that one pair can. Its prompt is deliberately biased against merging, and told that
names truncated mid-word come from PDF table cells: of the two mistakes, a fragment is
visible on screen and recoverable, while a wrongly fused node silently asserts a
connection nobody made and keeps no record that it was ever two things.

Also deterministic, and also tuned against the corpus: a **boilerplate list**. Every
report carries the same `Risikoregister`, `Maßnahmenliste` and `Lenkungsausschuss`, so as
nodes they join all four clients to each other for no defensible reason and bury the two
bridges that mean something. The extraction prompt forbids them and the model extracts
them anyway in a few chunks per hundred; a list is deterministic where a prompt is not. It
is explicitly shaped by *this* corpus — an accepted trade on a throwaway project, named
here rather than hidden in a constant.

### Verifying it

```sh
just entities --bridges              # every node joining two or more top-level folders
just entities --contains Atlas       # what merged into what, and under which scope
```

On the generated corpus this resolves to one `Dr. Katharina Vogt`, one
`Atlas-Referenzarchitektur`, one `Milan Horvat`, one `Prüfschema Berechtigungsrollen` —
and *two* `Projekt Atlas` nodes and *two* `Projekt Datenschild` nodes, one per client.

In the Neo4j Browser, the picture the demo is built on — one person, two clients:

```cypher
MATCH p = (b:Folder {depth: 1})-[:CONTAINS*]->(:Document)-[:HAS_CHUNK]->(:Chunk)
          -[:MENTIONS]->(e:Entity {name: "Dr. Katharina Vogt"})
RETURN p LIMIT 60
```

and the two same-named projects she leads, which stay apart:

```cypher
MATCH (v:Entity {name: "Dr. Katharina Vogt"})-[r:RELATES]->(p:Entity)
WHERE p.name = "Projekt Atlas"
RETURN v, r, p
```

### Known residue

A handful of nodes are imperfect and are left so deliberately:

- The consultancy appears in every document and its name is truncated by the PDF parser in
  a dozen different ways (`Vekt`, `Vektor Consulti`, `Vektor Consulting Gmb`). Twelve of
  those fold into one node; one or two stragglers with a single mention each survive a
  run. They are a parsing artefact, the consultancy is a useless discriminator by design,
  and chasing the last of them means loosening thresholds that are correctly placed.
- Resolution runs over the whole corpus in one pass and does not seed itself from entities
  already in the graph, so ingesting one branch at a time can leave stage-2 variants
  unmerged across runs. Incremental re-ingestion is out of scope; re-run `just extract`.

## The ingestion cache

A full ingest of the 120-document corpus is minutes of layout analysis and a model call
per chunk. Almost none of that work changes between two runs, so none of it is paid for
twice: every external call the pipeline makes is stored on disk under `CACHE_DIR`
(default `.cache/ingest`), keyed by everything that call depended on.

| Cached | Keyed by | Why it is the expensive one |
|---|---|---|
| The parsing service's reply | the file's SHA-256 + strategy, languages, image block types | seconds of `hi_res` layout analysis per PDF |
| Each embedding | the text + model + width | one vector per chunk, and one per distinct entity name — not the dominant cost, but what stands between "no extraction calls" and a re-run that is actually measured in seconds |
| Each model reply | the system prompt + the user prompt + the tier + JSON mode | **one call per chunk** — the dominant cost of an ingest |

Two rules decide what goes in, and both matter more than they look.

**The raw result of the call is cached, never the artefact derived from it.** The
elements the parser returned, not the chunks; the model's reply, not the entities parsed
out of it. Everything downstream of a call is pure and cheap, so it re-runs on every
cache hit. Chunk sizes, the evidence limit, the ambiguity rule, the taxonomy filter and
both resolution thresholds can therefore all be changed and re-measured against a warm
cache, at no cost and with no bust — which is the difference between tuning them and
guessing.

**The key is every input the call depended on, and nothing else.** A key that forgets an
input does not fail; it answers with the result of a *different* call, and the graph
that comes out looks entirely plausible. Two consequences are worth knowing: an edited
extraction prompt invalidates its own entries, because the prompt is part of the key;
and a changed `PARSING_STRATEGY` re-parses, because it is too.

The cache lives on disk rather than in the graph because the graph is the thing that
gets dropped. `just init` rebuilds the schema from empty; re-filling it afterwards costs
seconds and not one API call, and does not need the parsing service to be running at all.

It is a few tens of megabytes for this corpus — mostly the stored vectors, which are
1024 floats of JSON apiece — and it is gitignored and disposable: deleting the directory
costs one slow ingest and nothing else. Nothing is ever evicted, so a week of iterating
on the extraction prompt leaves every superseded reply on disk; `--no-cache`, or `rm -rf`
on the directory, reclaims it. Eviction is not worth building on a corpus this size.

**What a re-run actually costs:**

```sh
just ingest corpus     # first run:  minutes, ~1 model call per chunk
just ingest corpus     # unchanged:  seconds, zero API calls
```

Changing one document's bytes changes its content hash, which misses its parse, which
produces different chunk texts, which miss their embeddings and their extraction calls —
so exactly that document is reprocessed and the other 119 are not. That is the entire
invalidation story: there is deliberately **no incremental re-ingestion, no crash
resumption and no deleted-file handling**, all cut in `SPEC.md`.

`--no-cache` **empties** the cache and then refills it as it goes. It deletes rather than
bypasses, because a flag that only skipped reads would leave the old entries underneath
the new ones and the next run without the flag would answer from whichever it found
first — a cache that cannot be cleared. It removes only the directories it wrote, so a
mis-set `CACHE_DIR` cannot take out an unrelated tree. Reach for it when something a key
cannot see has changed: a model whose answers drifted under the same identifier, or
entries written by a version of this code that read replies differently.

Each command discards only what it could actually re-do, which is why the flag is safe
to reach for:

| | `--no-cache` discards |
|---|---|
| `just ingest <dir>` | everything |
| `just extract` | the model replies — the parses it does not use are left alone |

`just parse` has no such flag on purpose: a diagnostic over one file should not be able
to discard an ingest's worth of parses, and a changed parsing strategy re-parses by
itself anyway.

Every run reports what the cache did, because a cache that has quietly stopped hitting
looks exactly like one that is working, right up until the ingest takes ten minutes:

```
Cache .cache/ingest — parsed documents: 120 reused / 0 new; embeddings: 1206 reused / 0 new; model replies: 1086 reused / 0 new.
```

## Progress during a long ingest

A fresh ingest takes minutes, and a run that prints nothing for minutes in front of an
audience is indistinguishable from one that has hung. Each of the six stages announces
itself with its number and, where it is iterating over a known number of things, a bar
and a count:

```
[1/6] Walking the corpus
[2/6] Writing the structure
[3/6] Inferring what each level means
[4/6] Parsing and embedding documents ━━━━━━━━━━━━━━━━ 120/120 0:03:41
[5/6] Summarising documents           ━━━━━━━━━━━━━━━━ 120/120 0:01:12
[6/6] Extracting entities             ━━━━━━━━━━━━━━━━ 1086/1086 0:06:55
```

A document or chunk that fails prints under the bar and the run carries on; the exit code
is decided once, at the end. On a cached re-run the same bars appear and fill in seconds,
which is itself the proof that the cache is working.

## Retrieval: the lexical and semantic leg

`just search "<Frage>"` runs the first of the two retrieval legs and prints what it found.
No answer is generated — this leg on its own is the thing being shown:

```
$ just search "Welche Risiken nennt der Bericht zu Projekt Taktzeit?" --top-k 5
Volltext-Query: Welche Risiken nennt der Bericht zu Projekt Taktzeit\?

  #     rrf   txt   vec   Ablage                                                 Auszug
  1  0.0164     1     -   … Brandt und Soehne Maschinenbau › 2024 › Taktzeit     Statusbericht Proj…
  2  0.0164     -     1   … Brandt und Soehne Maschinenbau › 2023 › Atlas        Arbeitsergebnisse…
  3  0.0161     2     -   … Brandt und Soehne Maschinenbau › 2022 › Taktzeit     Statusbericht Proj…
  4  0.0161     -     2   … Brandt und Soehne Maschinenbau › 2024 › Atlas        Arbeitsergebnisse…

Top: Kunde: Brandt und Soehne Maschinenbau › Jahr: 2024 › Projekt: Projekt Taktzeit —
vektorisch nicht gefunden, lexikalisch Platz 1.
Vector-only retrieval would have missed this chunk entirely. This is the case for the
full-text index.
```

(The `Ablage` column is shown here abbreviated; it prints each folder with its inferred
level, `Kunde: Brandt und Soehne Maschinenbau › Jahr: 2024 › Projekt: Projekt Taktzeit`.)

Two searches run, each returning its own ranked list:

- **Vector**, over the 1024-dim chunk embeddings in the Neo4j vector index.
- **Full text**, over chunk text through the German-analysed Lucene index. The question is
  escaped before it is sent — German prose is full of characters Lucene reads as syntax,
  and a leading `-` silently inverts a term into an exclusion rather than failing loudly.
  Terms are OR-ed and left to the index's own analyser, which already stems German and
  strips German stopwords.

The two lists are fused by **reciprocal rank fusion** (`score = Σ 1/(60 + rank)`), on rank
rather than on score: a cosine similarity and a Lucene BM25 score share no scale, so no
weighting of the two numbers would mean anything. `--top-k` is per leg, so the fused list
can be up to twice as long; capping the combined context is the answer path's job.

Every result carries its **folder ancestry** — `Kunde: … › Jahr: … › Projekt: …`, names
and level labels, not a path string. That is what later lets an answer say which client
and year a fact came from.

### What each leg actually contributes

Measured on the 766-chunk synthetic corpus, `--top-k 10`:

| Question | Vector rank 1 | Full-text rank 1 |
|---|---|---|
| *Welche Risiken nennt der Bericht zu Projekt **Taktzeit**?* | Brandt / **Atlas** | Brandt / **Taktzeit** |
| *Was hat **Milan Horvat** 2024 gemacht?* | **Sanitas** / Datenschild | **Finanzkontor** / Datenschild |

In both, the distinctive literal token is the only thing separating the right document
from four lookalikes, and dense retrieval does not find the right document *at all* — it
answers from a sibling report of the same client in a different project or a different
year. This is the risk the ticket README names, reproduced on demand: the
full-text index is not optional here.

Run `--mode vector` to show it, live, going wrong.

The year is left out of that table deliberately. Two quarterly reports of the same
project differ by a few names and numbers, so they tie to three decimal places in both
indexes — BM25 scores them 2.963 either way — and which of them surfaces first is
arbitrary and shifts with `--top-k`. The *project* is the claim; among genuinely tied
chunks the order means nothing, and ticket 11's entity leg is what separates them.

**Ties are broken toward the full-text leg** — which is why `txt` is the left-hand rank
column. Two chunks each ranked first by one leg and absent from the other score
identically, and RRF has nothing further to say about them, so the order of the legs
decides, and that order is one line: `Mode.legs`. A full-text hit is a literal token the document contains; a
vector hit on this corpus is a 0.93 similarity in a field of 0.92s. Both rows above are
exactly that tie, and vector-first would have put the wrong project on top of each.

### The vector index needs a running start

Neo4j's vector index is approximate, and the `limit` it is called with is also roughly
what it spends searching. On a corpus where every chunk embeds into the same small region
of the sphere, a small limit makes it settle in the wrong part of the graph and return a
worse neighbourhood entirely — with no error and plausible-looking results. Measured, for
*Prüfschema Berechtigungsrollen*, before the fix:

| Asked for | Best similarity | Right project |
|---|---|---|
| 3 | 0.8822 | 0/3 |
| 5 | 0.8822 | 0/5 |
| 8 | 0.9321 | 8/8 |
| 10 | 0.9321 | 10/10 |

So `vector_search` never asks the index for fewer than `VECTOR_CANDIDATE_FLOOR` (12)
candidates and truncates afterwards — twelve being the smallest value that recovered every
question tried, including at `--top-k 3`. Without it, turning `--top-k` down mid-demo to
fit the screen would quietly destroy dense recall.

## Retrieval: the entity traversal leg

`just traverse "<Frage>"` runs the second leg. Where the first finds chunks that *resemble*
the question, this one finds chunks that are *connected* to it — and that is the whole
thesis of the project, because the connection the target demo question asks about is in no
chunk at all. It is a path through the graph.

Four steps, printed in the order they happen:

1. **Extract** the entities the question names — one extraction-tier call per question,
   cheap, unlike the per-chunk extraction an ingest pays for.
2. **Match** them into the graph by normalised name *and* by embedding.
3. **Walk** `RELATES` edges outward to `--hops`.
4. **Collect** the chunks mentioning everything reached, each with its folder ancestry.

```
$ just traverse "Worauf basiert das methodische Vorgehen im Projekt Atlas bei Brandt & Söhne Maschinenbau GmbH?"
Erkannte Entitäten: Atlas (konzept), Brandt & Söhne Maschinenbau GmbH (organisation)

  Frage                 Knoten                 Typ      Kunde                      über             Ähnlichkeit
  Atlas                 Projekt Atlas          konzept  Brandt und Soehne Maschi…  embedding              0.923
  Atlas                 Projekt Atlas          konzept  Nordlicht Energie AG       embedding              0.923
  Brandt & Söhne …      Brandt & Söhne …       organi…  global                     name+embedding         1.000

One name in the question stands on more than one client's node. The walk starts at every one of them.
Hop 1: 35 Kanten → 24 neue Entitäten — Apache Kafka, Atlas-Referenzarchitektur, Dr. Katharina Vogt, …

Die Chunks stammen aus 2 Kundenablagen: Brandt und Soehne Maschinenbau, Nordlicht Energie AG
— über den Graphen verbunden, nicht über gemeinsame Wörter.
```

The question names **one** client. The answer comes back from two.

Note what matched how: the question says `Atlas` where the folders and the documents say
`Projekt Atlas`, so the normalised name does not match and the embedding half is what finds
the node — twice, once per client. That is the case for running both matches rather than
either.

### What this leg finds that the other cannot

Run the two legs against each other and read the last line of each — both print how many
clients' folders their results came from:

```
just search   "Worauf basiert das methodische Vorgehen im Projekt Atlas bei Brandt & Söhne Maschinenbau GmbH?"
just traverse "Worauf basiert das methodische Vorgehen im Projekt Atlas bei Brandt & Söhne Maschinenbau GmbH?"
```

Measured on the 766-chunk synthetic corpus, `--top-k 10`. "Brückenchunks" counts how many
of the ten returned chunks name the planted bridge — `Atlas-Referenzarchitektur` or
`Dr. Katharina Vogt` — which is what `KOLLISIONEN.md` says the connection consists of:

| Question | Leg | Kunden | Brückenchunks |
|---|---|---|---|
| *Worauf basiert das methodische Vorgehen im Projekt Atlas bei **Brandt & Söhne**?* | lexical/semantic | 1 | 1/10 |
| | traversal, 0 hops | 1 | 5/10 |
| | traversal, **1 hop** | **2** | **10/10** |
| *Was verbindet das Projekt Atlas von **Nordlicht** mit dem von **Brandt & Söhne**?* | lexical/semantic | 2 | 4/10 |
| | traversal, 2 hops | 2 | 10/10 |

The first row is the point. The question names one client, so the lexical leg can only
answer from that client's folder — there is no term in it that any of Nordlicht's documents
contain, and no chunk anywhere states the connection. One hop along `RELATES` reaches the
`Atlas-Referenzarchitektur` and the consultant who carried it, and the material that
actually answers the question arrives from a client the question never mentions.

The second row shows the same leg winning a question the lexical leg can do something with:
naming both clients lets full-text find both folders, but only 4 of its 10 chunks say
anything about the connection, against 10 of 10.

### Hop depth, and why two

`--hops` is a query parameter, not a constant, and it is expected to be turned up and down
live. Walking outward from the four matched nodes of the question above:

| Hop | Edges crossed | New entities | Reached, cumulative |
|---|---|---|---|
| — | — | 3 matched | 3 of 110 |
| 1 | 34 | 24 | 27 of 110 |
| 2 | 140 | 33 | 60 of 110 |
| 3 | 47 | 18 | 78 of 110 |

(Each edge is counted once, at the hop that first crossed it — which is why hop 3 adds
few: by then the walk is mostly re-meeting edges it already has. 221 of the graph's 231
relations have been crossed.)

One hop is precise and already crosses to the other client. Two is the default: it is where
connections that run through an intermediate artefact appear, and it still leaves nearly half
the graph out. Three reaches 71% of every entity in the corpus — at that point the walk has
stopped selecting anything and the leg is returning the archive.

### Matching a question to a node

Two matches, and they fail differently.

**By normalised name**, through the same normaliser the ingestion side uses — so a question's
`Projekt Atlas` arrives as the exact string the graph stored. This match deliberately ignores
**scope** and **type**:

- Scope, because `Projekt Atlas` is two nodes the hierarchy ruled apart, and a question asked
  from outside the hierarchy names neither in particular. Starting at both is the demo.
- Type, because the question side and the ingestion side are two independent extractions of
  one name. An artefact one calls `ergebnis` and the other `konzept` is still the node the
  walk has to start from, and a type check would drop exactly the bridge entity while
  reporting nothing wrong.

**By embedding**, for the names a question spells differently from the documents. This is the
loose half, and it is ticket 08's resolution hazard in miniature with the same silent failure:
match too loosely and the walk starts from the wrong node, then answers confidently out of
genuinely connected material about something nobody asked after. Measured, the usable band is
narrow:

| Pair | Cosine | |
|---|---|---|
| `Vogt` / `Dr. Katharina Vogt` | 0.9065 | the lowest real match found |
| `Prüfschema Berechtigungsrollen` / `Berechtigungskonzept Patientendaten` | 0.9001 | a different artefact |
| `Atlas-Referenz` / `Altbestände` | 0.8847 | unrelated |
| `Atlas` / `Snowflake` | 0.8556 | unrelated |

`TRAVERSAL_MATCH_THRESHOLD` sits at **0.905**, between the first two. Short German names embed
close together — nothing unrelated scores below about 0.85 — so the threshold has 0.006 of
headroom and is a setting rather than a constant for that reason. `--threshold` overrides it
per question.

### What a result is scored on

A chunk scores `Σ 1/(1 + hop)` over the reached entities it names, which rewards proximity and
breadth at once: the bridge chunk this project exists to surface is precisely the one naming
two reached entities, the named project and the artefact one hop out from it.

Scoring sees **every** mention. An earlier version read chunk text and ancestry in the same
query and capped the rows, which looked harmless and was not: on a question naming two
clients the cap filled with the alphabetically-first client's mentions and the leg returned an
answer about one of them, with nothing on screen to say the other had been cut. The mention
rows are now ids only and uncapped; text and ancestry are fetched for the ten chunks that
survive.

Every result carries its **folder ancestry** and the leg carries **the path** — which entities
matched, which edges were walked at which hop, which chunks they led to. Tickets 14 and 15
render that path, and it is the demo's entire burden of proof.

## Answering a question

`just query "<frage>"` is the whole system in one command, and the first thing in this project
that a demo audience sees rather than an operator. Both legs run, their chunks are merged into
one context, and the chat tier answers in German citing the client, year and project each fact
came from.

```
just query "Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem von Brandt & Söhne?"
```

### Merging the two legs

The legs are fused on **rank**, by the same reciprocal rank fusion that fuses vector against
full text one level down. They have to be: the lexical leg scores in RRF units and the entity
leg in hop weights, and no arithmetic over those two numbers means anything.

Two decisions in the merge are load-bearing.

**The cap is applied after the fusion, never per leg.** Capping each leg first decides the
context budget before either leg knows what the other found; on a cross-client question that
spends the whole budget on whichever leg is merged first, and the other client silently
disappears from an answer that still reads well.

**A tie falls to the traversal leg.** Each leg's first place scores identically and the formula
has nothing further to say, so the order is declared rather than left to emerge: the traversal
leg's claim on a chunk is the stronger one, because the question *named* an entity that chunk
mentions, while a vector hit on this corpus is a 0.93 similarity against a field of 0.92s.

### What the demo question looks like

Measured on the real corpus, `--max-chunks 12`:

| Legs | Kundenablagen in context | Answer |
|---|---|---|
| `--legs search` | 2 (both named in the question) | "Eine inhaltliche Verknüpfung lässt sich nicht belegen." |
| `--legs both` | 2 | Both projects run by Vektor Consulting under Dr. Katharina Vogt, Atlas-Referenzarchitektur as the shared standard |

Six of the twelve context chunks came from the walk and would not have been retrieved by search
at all — the line the command prints after every question, re-derived rather than measured once.
The lexical-only answer is the honest failure mode of flat retrieval on this corpus: it finds
both clients' reports, because the question names both clients, and still concludes there is no
connection, because no single chunk contains one.

### Isolating a leg

`--legs search` and `--legs traversal` answer the same question from one leg alone, and
`--legs search --mode vector` is the nearest thing this project has to the flat-RAG baseline
that was cut from scope. A leg that is excluded is **not run**, rather than run and discarded:
the traversal leg costs a model call, and the comparison is only honest if the other leg really
was absent.

### Grounding, and how it is checked

The answer cites its sources as `[Quelle N]`, numbered by position in the context table the
command prints — so every claim can be read back against the chunk it came from without leaving
the terminal. Two checks run on every answer:

- A citation pointing at a source that does not exist is reported in red. A model citing
  `[Quelle 9]` against seven sources has stopped quoting the context, and an invented citation
  reads exactly like a real one.
- An empty context is **never put to the model**. Handed no sources and a rule saying "answer
  only from the sources", the chat tier reliably produces a fluent paragraph out of its own
  knowledge of consulting reports — about a client that does not exist, in front of an
  audience, with nothing to check it against.

The prompt's fourth rule is the one that earns its place on this corpus: a context of a dozen
chunks routinely holds two clients' versions of the same boilerplate paragraph, and a model
that is not told to keep them apart will answer as though there were one project. Asked
"welche Risiken nennt der Bericht zu Projekt Taktzeit?", the answer says the Taktzeit reports
name none and that the risks in context belong to Projekt Atlas — rather than quietly
transplanting them.

### The result is a structure

`answer_question` returns an `Answer`, not a string: the merged context with each chunk's leg,
rank and folder ancestry, the entities the question named, where they matched in the graph,
every edge the walk crossed and at which hop, and the generated prose. The CLI renders a view
of it. Tickets 14 and 15 consume the same object to draw the traversal subgraph, which — with
the flat-RAG baseline cut — carries the demo's entire burden of proof.

## Follow-up questions

`just chat` is `query` with a memory. An audience *will* ask the second question with a pronoun
in it, and a system that only accepts standalone questions faceplants visibly at exactly that
moment — there is no recovering the room from "und 2023?" retrieving nothing.

```
just chat "Wer leitet das Projekt Atlas bei der Nordlicht Energie AG?"
Frage › Und 2023?
Rückfrage aufgelöst: Und 2023? → Wer leitet das Projekt Atlas bei der Nordlicht Energie AG im Jahr 2023?
```

One chat-tier call resolves the follow-up against a bounded window of recent turns, and
**that happens before the pipeline, never inside it**: `answer.py` does not import
`conversation.py`, so retrieval and generation are unchanged and cannot tell a resolved
follow-up from a first question. The rewrite is printed because watching it happen is the
moment on stage, and because a follow-up that answers oddly is almost always a rewrite that
went somewhere unexpected.

Measured over the three-turn exchange above:

| Turn | Asked | Retrieved on |
|---|---|---|
| 1 | Wer leitet das Projekt Atlas bei der Nordlicht Energie AG? | unchanged — a first question is never sent to the rewriter |
| 2 | Und 2023? | Wer leitet das Projekt Atlas bei der Nordlicht Energie AG im Jahr 2023? |
| 3 | Wer hat sonst noch daran gearbeitet? | Welche weiteren Personen haben neben Dr. Katharina Vogt und Henrike Baumann am Projekt Atlas der Nordlicht Energie AG in den Jahren 2022, 2023 und 2024 mitgearbeitet? |

Turn 2 narrows to the year, and every source the answer cites comes from that client's 2023
folder. Turn 3 is the case that decides what the window holds: "daran" and "sonst noch" refer to
people the *answer* named, not the question — so the window keeps the answer text, not only the
questions. It also keeps each question in its **resolved** form, because follow-ups chain: a
history of shorthand degrades one turn at a time, which reads as the model getting worse rather
than as a bug.

`--window 0` turns the memory off, which is how to show the same follow-up failing: "und 2023?"
then retrieves 2023 reports from whichever clients happen to rank, and answers about all of them.

### Every failure falls back to the question as asked

The floor is the behaviour of a system without this feature, because the one outcome worse than
not rewriting is retrieval running on an empty string. A reply that is not JSON, holds no
question, drops most of the question, or runs past 300 characters leaves the question exactly as
typed and says so. The model is asked for `{"frage": "..."}` rather than a bare line for the same
reason every other model call in this project is: it removes the whole class of "Hier ist die
umformulierte Frage:" preambles.

Two guards came out of watching it run rather than from reasoning about it:

- **Markdown is stripped.** Asked to return an already-standalone question unchanged, the chat
  tier returns `Projekt **Taktzeit** der **Brandt & Söhne**?` — it decorates proper nouns because
  that is how it writes prose everywhere else. Both `*` and `` ` `` are Lucene syntax, so the
  escaper faithfully passes `\*\*Taktzeit\*\*` to the full-text index, which contains no such
  token, while entity matching normalises the asterisks into the surface name and misses the
  node. Two legs fail at once, silently.
- **A rewrite may not drop most of the question.** Resolving shorthand only ever *adds* what it
  pointed at, so a reply much shorter than the question summarised or truncated it — a
  two-clause question coming back with one clause, answered as though that were all that was
  asked.

An already-standalone question is asked to come back word for word, and `changed` is compared on
the text rather than taken from the fact that a call was made — so the common case prints no
rewrite line at all, and a line appearing means something actually happened.

## The HTTP service

`just serve` puts the same pipeline behind HTTP. Three endpoints, no authentication, no
sessions:

| Endpoint | What it is |
|---|---|
| `POST /api/ask` | A question in; the answer, its numbered sources, and the subgraph that produced it out |
| `GET /api/status` | The operator's view: what has been ingested, whether every index is online, and what each level of the hierarchy was inferred to mean |
| `GET /api/health` | Is the process up. Says nothing about the graph |

Interactive docs at `/docs`.

**The service is not a second implementation.** `pipeline.py` is the composition root for
answering — driver, both graph adapters, the embedder and both model tiers behind the cache —
and `just query`, `just chat` and `POST /api/ask` all run it. The CLI keeps only what is
genuinely its own: turning a dead Neo4j into a message and an exit code rather than a 503. The
pipeline is opened once at startup and held for the life of the process, so the first question
over HTTP costs no more than the tenth.

### Asking

```sh
curl -s localhost:8000/api/ask -H 'content-type: application/json' -d '{
  "question": "Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem von Brandt und Soehne Maschinenbau?",
  "tuning": {"hops": 2, "top_k": 10, "max_chunks": 12, "legs": "both", "mode": "hybrid"},
  "history": [],
  "window": 4
}'
```

Every tuning parameter of `just query` is settable per request, because hop depth in particular
is turned live on stage. What was actually used comes back in `parameters` — a demo where the
operator cannot see which knob produced which picture is one that cannot be debugged in front of
an audience.

Conversation history is accepted and not held: the client sends the turns it has, the service
resolves the follow-up against them and returns the standalone question in `rewrite`. **Send
back `rewrite.standalone`, not what the user typed** — a history of shorthand questions degrades
one turn at a time, which looks like the model getting worse rather than like a bug.

### The subgraph payload

`subgraph.nodes` and `subgraph.edges` are what ticket 15 draws. Nodes are typed `folder`,
`document`, `chunk` or `entity` and carry stable identities (`chunk:<uuid>`,
`entity:<type>:<scope>:<normalised name>`, a folder's name chain). Every node in the hierarchy
carries its folder ancestry, and a chunk carries its source number — `[Quelle 3]` in the prose
is the chunk node whose `source` is 3, which is how a citation finds its node.

The chunks drawn are **the context the answer was written from**, not everything the legs
ranked: the picture is the evidence behind this answer, and showing chunks that were retrieved
and then capped away would overstate the case by exactly the number the cap removed. The
entities and relation edges are the whole walk, which at `hops: 2` on this corpus is around
sixty entities and two hundred-odd edges — turn `hops` down to 1 for a picture that reads from
the back of a room.

A chunk's `mentions` edges are read from the graph, not taken from the walk. The traversal leg
only knows which entities led to the chunks *it* returned, so a chunk the lexical leg found
alone would otherwise hang in the picture unconnected — and on this corpus the lexical leg is
the one that rescues a question spelled with a literal project name. That costs one extra read
per question, the same `mentions_of` the walk already makes, and it is what stops a real
crossing from rendering one-sided exactly when the lexical leg did the work.

### Cross-branch edges, decided on the server

`crosses_branches` is the demo's one visual claim, and it is computed here rather than in the
browser — the branch a node belongs to is known on this side and is not in the payload unless
the server puts it there.

Every node carries a `branches` list. A folder, a document and a chunk each sit under exactly
one top-level folder — the client. An entity sits under none: entities are global by design,
which is the modelling decision that makes a cross-client path real structure rather than a
query-time join. So **an entity's branches are the branches of the drawn chunks that mention
it**, which keeps every flag traceable on screen: an edge coloured as crossing is one whose
crossing the audience can follow node by node. Deriving it from the whole graph instead would
flag edges whose second branch is nowhere in the picture.

An edge crosses when **its two ends, taken together, sit under more than one branch** — not when
they are disjoint. The bridge entity is mentioned from both clients, so the edge from one
client's project into it overlaps on one branch and is still the edge on which the walk leaves
that client's subtree; a disjointness rule would leave the demo's one interesting edge
uncoloured. Containment edges therefore never cross, which is the point: the hierarchy is the
thing being escaped. An entity no drawn chunk mentions has no branches and its edges are left
unflagged rather than guessed at.

`crossing_edges` counts edges, not connections: one crossing is usually several edges — the
mention into the bridge and the relation out of it.

On the target demo question the payload comes back with both clients in `branches`, bridge
entities led by `Dr. Katharina Vogt` and `Vektor Consulting GmbH`, and the `leitet` edges from
each client's `Projekt Atlas` to Vogt among the crossings — with six of the twelve context
chunks found by the entity leg alone.

### Status

`GET /api/status` answers the questions an operator asks before going on stage: node counts,
index state, whether every chunk carries a vector the index can see, how many documents have a
summary, how many entities bridge two branches, and `levels` — what each depth of the hierarchy
was inferred to mean, read back from the graph rather than re-inferred. `entities.folder_owned`
must be zero; anything else means an entity acquired a folder edge and the cross-client path has
stopped being real structure.

Ingestion is **reported, not triggered**. An ingest is a multi-minute pipeline with a progress
display built for a terminal, and a POST that returns in twenty minutes is worse than the
command — so the endpoint returns the command to run instead.

## The screen

`just demo` builds the frontend and serves it with the API on one port. The answer is on
the left, the subgraph that produced it on the right, and **the graph is the larger of the
two** — a 36/64 split, fixed in the layout rather than left to whoever edits the markup
next. That is the ticket's instruction and it follows from a decision taken in the spec:
the side-by-side comparison against flat RAG was cut from scope, so this screen carries the
entire burden of proof. A chat app with a diagram tucked into a corner would read as "a
chatbot that answered a question."

### What the picture encodes

| Channel | Meaning |
|---|---|
| **Shape** | The node's kind: folder, document, chunk (circle, white core if the answer cited it), entity (diamond) |
| **Hue** | The top-level branch — the client — a node sits under. Two clients read as two coloured clusters before a single label has been read |
| **Gold** | Reserved, completely: bridge entities, and every edge the server flagged `crosses_branches`. Nothing else is ever gold |
| **Motion** | Particles, on crossing edges only. The strongest signal available, spent entirely on the one thing the demo exists to show |

Gold is the whole argument, so it is spent on nothing else. The flag itself is **read, never
recomputed** — `subgraph.py` decides it, for the reasons in "Cross-branch edges, decided on
the server" above, and the browser only colours by it.

The layout does argumentative work too. Containment edges are short and at full strength, so
each client's folder tree collapses into a tight cluster that reads as one place in the
hierarchy; crossing edges are given the longest rest length in the graph and the bridges
they end at repel hardest, so the two clusters are held apart with the gold stretched between
them. A collision force keeps the bridge labels off each other. What the audience ends up
looking at — two tight trees, a span of gold across the gap — is the claim, drawn.

### Tracing it node by node

Clicking a `[Quelle 3]` pill in the prose, or its card below, lights that chunk's whole
story: the folder chain above it — client, year, project — and the entities it led to, with
everything else dimmed to a shadow. The camera frames that subgraph and shifts the centre
left by half the detail card, because the first version of this reliably parked the selected
node underneath the card describing it.

`Nur die Übergänge` dims everything except the crossings and their endpoints. That is the
button for the moment when the room needs to see the claim with nothing else on screen.

Each source carries the leg that found it. **`nur Graph`** means the lexical leg never
returned that chunk — with no flat-RAG baseline to put beside it, that badge is the most
direct statement of the thesis anywhere in the UI, and on the target question six of twelve
sources carry it.

### The knobs, and why they are on screen

Hop depth, results per leg, context cap, which legs run, which searches run, and the
conversation window are all in the header, because hop depth in particular is turned live on
stage and an operator who cannot see which knob produced which picture cannot debug the demo
in front of an audience. What was actually used is printed above the graph.

Two knobs change nothing on the server. **Dichte** keeps the relations that are doing work —
the ones that cross a branch, and the ones between two entities a drawn chunk actually
mentions — and drops the fan running outward from the walk's frontier to entities nothing on
screen names. At `hops: 2` the target question returns 180 relations and that fan is most of
them: unlabelled dots costing exactly as much room as the crossing they surround. The rule is
"both ends named" rather than "crosses" alone on purpose, because the hop-by-hop route between
entities *is* relation edges, and keeping only the crossings would colour the payoff correctly
while deleting the path to it. The count of what was dropped — relations and entities both —
stays on screen, so the picture never silently claims to be everything, and `alles` puts them
back.

**Bühne** scales the drawing, the answer and the legend for a room, and deliberately not the
control strip: scaling the chrome as well grows the header until it has eaten a third of the
canvas and the graph has *shrunk* to fit what is left. It scales the collision force too but
not the link distances — scaling the distances would enlarge the whole layout, the camera
would zoom out by exactly as much, and the control would do nothing at all.

The hop slider defaults to 2, matching the CLI, and that was measured rather than assumed.
At one hop the target question draws a sparser, prettier picture and gets a visibly worse
answer — the model reports that no connection between the two projects can be established
from the sources. The second hop is what reaches the shared reference architecture.

### Verified at presentation scale

The one criterion that cannot be checked by reading code: it will be shown on a projector,
at poor contrast, to people standing some distance away. `frontend/verify.mjs` drives the
built UI with headless Chrome and writes a screenshot, so the judgement is made by looking
at the right size rather than at a developer's monitor:

```sh
just serve                                    # in another terminal
cd frontend && node verify.mjs 1280 720 stage # a projector, at presentation scale
```

Checked at 1280×720 with `Bühne: groß`, on the target question: both clients present, the
bridge entities labelled and legible, and the gold unmistakable from across a room. Actions
are `empty`, `ask`, `crossings`, `cite`, `stage` and `followup`. It asserts nothing — the
project's testing posture is deliberately narrow and this is a verification tool, not a
test.

### Multi-turn

Each turn keeps its own subgraph, so clicking back through the conversation redraws the
picture that question actually produced rather than re-asking it — a re-ask would cost a
model call and might answer differently, on stage, in front of the thing the audience just
looked at. History goes back in its **standalone** form, and a rewritten follow-up is shown
above the answer rather than applied silently: the audience asked "und 2023?", something
else was retrieved, and watching the substitution is the moment.

## Tests

Two commands, and deliberately little between them. The criterion for writing a test here
is **silent failure**: logic whose bugs would not announce themselves and would instead
make the graph quietly wrong in a way first noticed during a live demo.

`just test` is the fast suite — pure logic, no driver, no client, no HTTP, under a second.
`just smoke` is the one test that catches **wiring and configuration** errors: it runs the
real `ingest` command over four PDFs against **real Neo4j, the real Mistral API and the
real parsing service**, asserts that every node and relationship kind landed, and asks a
question whose answer is known through the real pipeline. No fakes anywhere — that is the
deliberate alternative to building fake implementations at the pipeline seam, and it is
cheap because all three are running anyway.

It is deselected from `just test`, skips with the reason and the fix when a dependency is
down, and expects a **cold graph**: entity resolution runs over every chunk in the graph
by design, so it refuses to re-resolve a corpus that is already ingested. `RUNBOOK.md`
has the details and the override.

Do not add coverage beyond these. On a project being discarded, breadth is not a quality
signal; it is waste.

## Layout

```
src/graph_rag/
  cli.py              composition root: loads settings, builds the driver, injects it
  config.py           environment-backed settings
  cache.py            pays for each external call once; wraps the parser, embedder and chat model
  mistral_check.py    verifies model identifiers against the live API
  corpus/             synthetic corpus generator (plan -> text -> pdf); ingestion never imports it
  parsing.py          PDF -> ordered chunks; parser and chunker injected
  vendored/           VENDORED from the production monorepo — parsing, normalisation, chunking
  graph/driver.py     the only place a Neo4j driver is constructed
  graph/schema.py     constraints and indexes, idempotent
  graph/health.py     connectivity and index readiness reporting
  hierarchy.py        directory walk + pure folder/document node construction
  graph/hierarchy.py  writes that structure into Neo4j, idempotently
  levels.py           infers what each depth of the hierarchy means; model injected
  graph/levels.py     reads sibling names per depth, writes level names back
  chunks.py           vendored chunks -> the records the graph stores
  ingestion.py        parse -> chunk -> embed -> write, per document; everything injected
  embedding.py        the embedding tier behind a narrow protocol, batched
  graph/chunks.py     writes a document's chunks and reading order; embedding coverage
  summaries.py        one generated summary per document; model and store injected
  graph/summaries.py  reads what a summary is written from, writes it back
  entities.py         fixed taxonomy, normalisation, scoping, extraction; model injected
  resolution.py       the four resolution stages; similarity and adjudicator injected
  graph/entities.py   writes entities and their edges; bridge and coverage reads
  retrieval.py        the lexical/semantic leg: RRF, query escaping; index injected
  graph/retrieval.py  the two chunk searches, each carrying folder ancestry back
  traversal.py        the entity leg: question entities, matching, the walk, the path
  graph/traversal.py  the entity reads the walk makes, one per hop
  ancestry.py         a document's place in the hierarchy, as an answer would cite it
  graph/ancestry.py   the one Cypher projection that reads it, shared by both readers
  answer.py           both legs merged into one capped context, and the cited answer
  conversation.py     a bounded window of turns; rewrites a follow-up before retrieval
  chat.py             the chat-tier Mistral model behind a narrow protocol
  pipeline.py         the answer path wired up once, shared by the CLI and the service
  subgraph.py         an answer as drawable nodes and edges; flags the cross-branch ones
  api.py              the HTTP service over that pipeline; no second code path
tests/                pure-logic tests, plus the one end-to-end smoke test
  test_smoke.py       the only test that opens a driver or calls a model; `just smoke`
frontend/
  src/api.ts          the payload, typed; the cross-branch flag is read, never recomputed
  src/theme.ts        the palette; gold is reserved for crossings and bridges alone
  src/graph.ts        the payload as a drawable model: colour, size, pruning, focus
  src/App.tsx         the split screen, the turns, and what the camera frames
  src/components/     the graph canvas, the chat panel, the knobs, the overlays
  verify.mjs          drives the built UI in headless Chrome at projector size
```

The graph driver is injected into every function that touches the graph; no module holds a
global client.
