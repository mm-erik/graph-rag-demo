# Runbook

How to get from a cold machine to a working demo, what to do when it is not working, and
what to say while it runs. Written for the case this project was specified against: the
demo is run in front of people after weeks of not being touched, on a machine where
**Docker is not running and the parsing service is not up** — which is exactly the state
both were in on the day the spec was written.

`README.md` explains how the system works and why it is built this way. This file
explains how to *operate* it. Where it says "see the README", the detail is there and is
not repeated here.

---

## 1. Cold start

In this order. Steps 1 and 2 are the two that are never already done, and step 2 is the
slow one — start it first and let it warm up while you do the rest.

### 0. The repository and the key

```sh
cd graph_rag
cp .env.example .env          # then put a real MISTRAL_API_KEY in it
```

Everything else in `.env.example` has a working default. The only value that must be
filled in is the key.

### 1. Docker

Neo4j runs in a container, so the daemon has to be up before anything else works.

```sh
open -a Docker                # macOS; or start Docker Desktop by hand
docker info > /dev/null       # succeeds once the daemon is actually ready
```

Docker Desktop takes a minute or two from a cold launch, and `docker info` failing with
*Cannot connect to the Docker daemon* means it is not ready yet rather than broken.

### 2. The parsing service — the external dependency, in the sibling checkout

Parsing goes over HTTP to the self-hosted Unstructured service. It is **not part of this
project**: it belongs to the production system and is started by that system's own
process manager, in the sibling `monorepo` checkout.

```sh
cd ../monorepo
nix develop --command process-compose -f nix/services/unstructured.yml up --tui=false
```

Leave it running in its own terminal. Then, from this repository:

```sh
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8100/healthcheck   # 200
```

Three things about it that cost time if you do not know them:

- **A cold start imports a multi-gigabyte ML stack.** On a warm Nix store and model cache
  it answers `/healthcheck` within a few seconds; the service config budgets 20 seconds
  before its readiness probe starts for exactly this reason. On a cold store, budget
  minutes and start it before you need it.
- **Keep it at one worker.** Each worker is a full copy of that stack, and two have
  OOM-killed a worker mid-parse on this machine.
- A `hi_res` parse of a two-page German PDF takes roughly **10–15 seconds**. That is the
  dominant cost of a first ingest, and the reason the ingestion cache exists.

It is only needed for parsing — that is, for `just parse` and for a `just ingest` whose
documents are not already in the cache. The demo itself (`just query`, `just chat`,
`just serve`) never calls it.

### 3. Neo4j, the schema, and the corpus

```sh
just up                       # start Neo4j, sync dependencies, apply constraints + indexes
just verify-models            # the Mistral identifiers, against the live API
just status                   # must end with "Graph is ready."
just corpus                   # generate the synthetic German corpus into ./corpus
```

`just up` waits on the container's healthcheck and is safe to re-run; the graph lives in
named Docker volumes and survives `just down` and machine restarts.

### 4. Optional, and the best minute you can spend: the smoke test

**On a cold graph, before ingesting anything**, this is the one check that exercises the
whole chain — Neo4j, the parsing service, both Mistral tiers, the writes, and both
retrieval legs:

```sh
just smoke
```

It ingests four PDFs, asserts every node and relationship kind landed, asks a question
whose answer is known, and removes its own tree again. Measured: **55 seconds** on a
completely cold cache, 2 seconds once the four parses are in it. It skips, with a message naming
the dependency and the command that fixes it, if any of the three is down.

It **skips on a graph that already holds another corpus** — entity resolution runs over
every chunk in the graph by design, so running it after `just ingest` would re-resolve
the demo's entities with four extra documents on the scale. That is not something to
discover on stage. See "Testing posture" below.

### 5. Ingest

```sh
just ingest corpus --levels "Kunde,Jahr,Projekt"
```

Expect, on a cold cache, ten to fifteen minutes for 120 documents: layout analysis per
PDF (10–15 seconds each at four at a time — the single largest share), one embedding per
chunk, one summary per document (about 2 minutes for the pass), and one extraction call
per chunk (about 3½ minutes for 766 of them). **Every one of those
calls is cached on disk** under `CACHE_DIR`, so a re-run over an unchanged corpus takes
seconds and makes no API calls at all. Measured on this machine with the parses and
embeddings cached and the summaries cold: **2 minutes 08 seconds**, of which 2 minutes
was the summary pass — one chat call per document is the pass to expect to wait for
when only part of the cache survives. `--levels` is passed because the level names of
this corpus are known; without it they are inferred by the model.

Then confirm it landed:

```sh
just status                   # Folder=57, Document=120, Chunk=766, Entity≈110-115
just entities --bridges       # the cross-client nodes — this is the demo
```

`just entities --bridges` must list **`Atlas-Referenzarchitektur`** (Brandt & Söhne,
Nordlicht) and **`Prüfschema Berechtigungsrollen`** (Finanzkontor Weser, Sanitas). Those
are the two planted bridges; if they are not there, the demo has nothing to show and the
thresholds in section 4 are where to look.

### 6. The screen

```sh
just ui-install               # once per checkout
just demo                     # builds the UI and serves it with the API on :8000
```

Open <http://localhost:8000>. That one process serves both the API and the built
frontend, which is deliberate: one thing to start on stage, and no cross-origin in the
demo path.

### This sequence has been rehearsed

Last walked end to end on **2026-11-11**, from Docker Desktop quit and the Neo4j volume
discarded: Docker up, parsing service answering, `just up` (9 seconds), `verify-models`,
`status`, `corpus`, `just smoke` against the cold graph (55 seconds on a cold cache, 2
seconds warm), `just ingest`, `just ui-build`, `just serve` — then the demo question of
section 3 answered over HTTP with 35 cross-branch edges in the subgraph payload. Every
command below is one that was actually run, in this order.

### The whole sequence, for copying

```sh
open -a Docker && docker info > /dev/null
# in another terminal, in ../monorepo:
#   nix develop --command process-compose -f nix/services/unstructured.yml up --tui=false
cp .env.example .env          # fill in MISTRAL_API_KEY
just up && just verify-models && just status
just corpus
just ingest corpus --levels "Kunde,Jahr,Projekt"
just ui-install
just demo
```

---

## 2. When something is wrong

`just status` is the first command to run, always. It reports connectivity, every index
and its state, the constraints, node counts and embedding coverage, and exits non-zero
when the graph is not ready. Nearly everything below shows up there first.

| Symptom | What it is | What to do |
|---|---|---|
| `Cannot connect to the Docker daemon` | Docker is not running | Start Docker Desktop; wait for `docker info` to succeed |
| `just up` hangs on the healthcheck | `NEO4J_PASSWORD` was changed after the volume was first initialised. Neo4j only reads `NEO4J_AUTH` when it initialises an *empty* data volume, so the container keeps the old password while the healthcheck starts using the new one | `docker compose down -v` (**this discards the graph**) and re-ingest — or put the original password back |
| `just status`: an index is `MISSING`, or the detail column shows drift | `CREATE ... IF NOT EXISTS` ignores the `OPTIONS` of an index that already exists, so a vector index at the wrong width or a full-text index with the wrong analyzer survives re-initialisation | `just init` — it detects the drift, drops and recreates. After a *dimension* change, re-ingest: embeddings written at the old width are invisible to the new index |
| `just status`: an index is `POPULATING` | Normal right after a large ingest | Wait and re-run; re-running `just init` does nothing |
| `just status`: *Chunks are missing an embedding* | Chunks were written at a width the index cannot see | Re-ingest the corpus |
| `just ingest` / `just parse`: connection refused on `:8100` | The parsing service is not running, or not warm yet | Start it in `../monorepo` (step 2 above) and wait for `/healthcheck` to answer 200 |
| A parse dies mid-run, or a worker disappears | Two Unstructured workers on this machine have OOM-killed each other | Run it with **one** worker |
| Mistral `401` | `MISTRAL_API_KEY` is missing or wrong in `.env` | Fix the key. `just verify-models` is the quickest check |
| Mistral `429`, or an ingest crawling | Rate limits | Lower `ENTITY_CONCURRENCY`, `SUMMARY_CONCURRENCY`, `EMBEDDING_BATCH_SIZE`. The cache means a re-run resumes rather than restarts |
| An answer is *"Zu dieser Frage wurde im Archiv nichts gefunden"* | Retrieval returned nothing — an empty graph, or an index that is not online. It is deliberately **not** passed to the model: asked with no sources, the chat tier writes a fluent answer about a client that does not exist | `just status`, then `just search "<the question>"` to see which leg is empty |
| An answer is confident and about the wrong client | The full-text index is missing or has the wrong analyzer, so "Neo4j only" has quietly become "dense-only" — the weakest possible configuration on this corpus | `just status`; the `chunk_text_fulltext` row must read `analyzer=german`. Compare `just search --mode vector` against `--mode fulltext` |
| The graph has *two* `Dr. Katharina Vogt`, or one node for two different artefacts | Entity resolution thresholds — see section 4 | `just entities --contains <name>` to see what merged; the thresholds are environment variables and re-deciding them is free |
| `http://localhost:8000/` returns 404 | The UI has not been built | `just ui-build`, or just use `just demo`, which builds first |
| The browser console shows a CORS error | The frontend is being served by Vite on a port not in `API_CORS_ORIGINS` | Add it, or use `just demo` — one origin, no CORS at all |
| `just smoke` skips | It says which of the four reasons: no API key, Neo4j unreachable, schema not ready, parsing service down — or the graph already holds another corpus | Fix what it names. The last one is by design |

**Before going on stage**, two commands and nothing else:

```sh
just status                   # "Graph is ready."
just query "Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem Projekt Atlas von Brandt und Soehne Maschinenbau?"
```

If both of those are right, the demo is right. Do **not** re-ingest, re-extract or run
the smoke test on the morning of the demo: the one genuinely model-decided step in
ingestion is the entity adjudication band, and re-running it can move the entity set by a
few nodes (see section 4).

---

## 3. The demo script

Six questions, in this order, each one setting up the next. Every one of them has been
run against the generated corpus; what each is *supposed* to show is below it, so a
different answer on the day is recognisable as such.

Run them in the UI (`just demo`) — the subgraph beside the answer is the entire burden of
proof — and keep a terminal open for the two that are shown twice with a knob changed.

### 1. The archive is real, and it cites

> **„Woran hat Dr. Katharina Vogt gearbeitet?"**

Answers in German, with `[Quelle n]` citations, and says for every statement which
*Ablage* it came from — client, year, project. **Point at:** the context summary line,
which reports chunks from **three** client folders, "verbunden über den Graphen, nicht
über gemeinsame Wörter". One person, several clients, already visible.

### 2. The trap: two clients, one project name

> **„Wie war der Status von Projekt Atlas im zweiten Quartal 2023?"**

The corpus contains two unrelated `Projekt Atlas` projects, and this question does not
say which. The answer **names both clients and keeps their statements apart** instead of
merging them into one project that does not exist. **Point at:** the two *Ablage* labels
in one answer. This is the failure the whole project exists to disprove, declined in the
last step rather than by luck.

### 3. The claim, stated as a question flat retrieval cannot answer

> **„Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem Projekt Atlas von Brandt und Soehne Maschinenbau?"**

Ask it **first with the graph leg switched off** — the `Legs` control in the UI, or:

```sh
just query "Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem Projekt Atlas von Brandt und Soehne Maschinenbau?" --legs search
```

> *„Die verfügbaren Quellen geben keinen Aufschluss darüber … Es handelt sich um separate
> Vorhaben bei unterschiedlichen Kunden."*

That is flat RAG, answering honestly and wrongly. No document names the other client; the
connection is not in any single chunk, so lexical and dense retrieval cannot reach it.

### 4. The same question, with the graph

Switch `Legs` back to both and ask it again.

Now the answer names **Dr. Katharina Vogt**, who led both, and the
**Atlas-Referenzarchitektur** carried from one to the other. **Point at:** the subgraph —
two client branches, and the gold edges that leave the hierarchy to meet at one entity
node. Then at the line *„n Chunks hätte die Suche allein nicht gefunden"*, which names
the folders the graph reached and the search did not.

This is the moment the demo exists for. Everything before it is setup.

### 5. The encore: the artefact, not just the person

> **„Wo wurde die Atlas-Referenzarchitektur wiederverwendet und was wurde daraus übernommen?"**

The whole story, unhedged: developed in `Projekt Atlas` at Nordlicht, adopted as an
internal standard, reused as the methodical basis at Brandt & Söhne, where the interface
contracts, the Kafka error-handling pattern and the event-type naming convention were
taken over unchanged and only the domain objects were swapped. **Point at:** the second
bridge node. A consultant moving between clients is a coincidence; a document doing it is
an argument.

### 6. The follow-up, to show the conversation is real

Ask, as a follow-up to question 1:

> **„Und Milan Horvat?"**

The rewritten standalone question is shown above the answer *before* retrieval runs —
watching the substitution is the point. **Point at:** the rewrite, then the answer, which
finds the second planted bridge: Milan Horvat carried the *Prüfschema
Berechtigungsrollen* between the two unrelated `Projekt Datenschild` projects at Sanitas
Klinikverbund and Finanzkontor Weser.

Two honest notes on this one. The rewrite sometimes inherits the *previous* turn's client
("… bei der Nordlicht Energie AG?"), and the answer then correctly says there is no
evidence for that and reports where he did work — which is the system refusing to invent,
on camera, and is worth narrating rather than hiding. And `--window 0` shows the same
follow-up failing without the memory, if someone asks whether the rewrite is doing
anything.

### Knobs worth having a hand on

| Knob | When to turn it |
|---|---|
| `Legs` / `--legs search` | The comparison in questions 3 and 4. The single most valuable control on the screen |
| `hops` / `--hops` | A question that reaches too far (hop 2 pulls in 30+ entities on this corpus) or not far enough. Two is the default for a reason — see the README |
| `max-chunks` / `--max-chunks` | A cross-client question needs room for both sides. The default 12 is sometimes filled by near-identical title pages; 20 helps |
| `--mode vector` / `--mode fulltext` | Only if someone asks what each search contributes |

### Questions to avoid on stage

Questions about a **specific number or table cell** in a status report — "wie hoch war der
Fortschritt", "welche Systeme wurden betrachtet". The corpus has 120 near-identical title
pages, and on a bare fact question they crowd the context and the answer comes back
truthfully as "the sources do not say". It is the corpus being adversarial exactly as
designed, but it reads on stage as the system not working. Ask about **connections**,
which is what it is for. If you must, add the client and the year to the question and
raise `--max-chunks`.

---

## 4. Handover notes

Everything below is a decision that was made empirically and would otherwise have to be
rediscovered.

### The entity resolution thresholds, and why these numbers

This is the highest-risk component in the system, and both of its failure modes are
silent: merge too loosely and two clients' similarly-named projects collapse into one
node asserting a connection nobody made; merge too strictly and the graph fragments into
spellings of one organisation — which demonstrates the *opposite* of this project's
thesis.

**Settled values: `ENTITY_MERGE_THRESHOLD=0.92`, `ENTITY_REVIEW_THRESHOLD=0.75`,
`TRAVERSAL_MATCH_THRESHOLD=0.905`.** They were measured against the generated corpus, not
guessed:

| Pair | Cosine | Verdict | What it fixes |
|---|---|---|---|
| `Schnittstellenverzeichnis` / `Schnittstellenverträge` | **0.9013** | must *not* merge | Sets the floor for `merge`: at 0.90 two different deliverables fuse, so 0.92 |
| `Vektor Consulting GmbH` / `Vektor Con` | **0.7990** | must merge | Sets the ceiling for `review`: a parser-truncated spelling sits at 0.799, so anything at or above 0.80 leaves it as its own node and the consultancy fragments |
| `SAP IS-U` / `SAP S/4HANA` | **0.8076** | must *not* merge | With the pair above: **0.009 apart and on opposite sides of the answer**. No threshold separates them — this pair is the entire argument for the model adjudication stage |

The adjudication prompt is deliberately **biased against merging**, and is told that names
truncated mid-word come from PDF table cells. Of the two mistakes, a fragment is visible
on screen and recoverable, while a wrongly fused node silently asserts a connection and
keeps no record that it was ever two things.

`TRAVERSAL_MATCH_THRESHOLD` sits in a band of its own — 0.9001 to 0.9065 on this corpus —
which is why it is a setting rather than a constant. The README's traversal section names
the two pairs that bound it.

None of the three thresholds is part of any cache key. Moving one and re-running
`just extract` re-reads the same vectors and re-decides the band for free, which is the
whole point of being able to tune them. Check the result with:

```sh
just entities --bridges              # every node joining two or more clients
just entities --contains Atlas       # what merged into what, and under which scope
```

On the generated corpus that resolves to one `Dr. Katharina Vogt`, one
`Atlas-Referenzarchitektur`, one `Milan Horvat`, one `Prüfschema Berechtigungsrollen` —
and **two** `Projekt Atlas` nodes and **two** `Projekt Datenschild` nodes, one per client.

### Also tuned against this corpus, and not transferable

- **The boilerplate list.** Every report carries the same `Risikoregister`,
  `Maßnahmenliste` and `Lenkungsausschuss`; as nodes they join all four clients for no
  defensible reason and bury the two bridges that mean something. The extraction prompt
  forbids them and the model extracts them anyway in a few chunks per hundred, so a
  deterministic list does what a prompt cannot. It is shaped by *this* corpus, by choice,
  and is named in the code rather than hidden.
- **Hop depth 2** for the traversal leg, **12 chunks** of context, **5** entity
  candidates per question entity. Reasoning for each is in the README; all are call-time
  parameters because they are expected to be turned live.
- **Chunking caps** (`by_title`, 2000/1600/500, overlap 250) are **not** tuned here —
  they are carried over verbatim from the production service so that demo chunk
  boundaries match production. Do not improve them.

### The entity count moves slightly between re-extractions

Expect `Entity` between roughly **110 and 115** on the full corpus. The band pairs are
decided by a model, and until a given pair is in the cache the decision can differ between
runs; a re-extraction consequently resolves a few nodes differently and removes a few that
nothing mentions any more. This is why section 2 says not to re-extract on demo morning —
not because it breaks, but because there is no reason to re-roll a graph that is already
right.

### The two assumptions the spec flagged — both now resolved

1. **The exact Mistral model identifiers were unverified.** They are correct.
   `mistral-small-latest` (extraction), `mistral-large-latest` (chat, level inference,
   query rewriting, adjudication) and `mistral-embed` all exist on the live API, and the
   embedding width really is **1024** — no discrepancy with the spec's table. Verified
   2026-11-11; re-check any time with `just verify-models`, which also *measures* the
   embedding width rather than trusting the configured one.
2. **Whether the production parsing service starts cleanly standalone.** It does. It
   comes up from the sibling monorepo on its own, needing nothing from the backend, the
   database or NATS, and answers `/healthcheck` within a few seconds on a warm Nix store
   and model cache. The caveats are operational and are in step 2 above: cold start
   imports a multi-gigabyte ML stack, keep it at one worker, 10–15 seconds per two-page
   `hi_res` parse.

### Known residue, left deliberately

- The consultancy's name is truncated by the PDF parser in a dozen different ways
  (`Vekt`, `Vektor Consulti`, `Vektor Consulting Gmb`). Twelve of those fold into one
  node; one or two stragglers with a single mention each survive a run. They are a
  parsing artefact, the consultancy is a useless discriminator by design, and chasing the
  last of them means loosening thresholds that are correctly placed.
- Resolution runs over the whole corpus in one pass and does not seed itself from
  entities already in the graph, so ingesting one branch at a time can leave variants
  unmerged across runs. Incremental re-ingestion is out of scope; re-run `just extract`.
- Retrieval is weak on bare-fact questions about near-identical status reports. See
  "Questions to avoid on stage".

### Testing posture — and why there is so little of it

Deliberately narrow, per `SPEC.md`. The criterion for writing a test is **silent
failure**: logic whose bugs would not announce themselves and would instead make the
graph quietly wrong in a way first noticed during a live demo.

| | What it covers | Cost |
|---|---|---|
| `just test` | Three pure-logic units — folder-node construction at arbitrary depth, entity normalisation and merging, reciprocal rank fusion — plus the pure helpers around them. No driver, no client, no HTTP | under a second |
| `just smoke` | The one test that catches **wiring and configuration** errors: the real `ingest` command against real Neo4j, real Mistral and the real parsing service, then a known question through the real pipeline | seconds on a warm cache, about a minute cold |

Everything between those pure functions — the composition root, the Cypher, the clients,
the cache, the service — is covered by the smoke test and by running a real ingestion, and
by nothing else. That is the deliberate alternative to building fake implementations at
the pipeline seam, and it is a trade made because this project is being discarded. **Do
not add broad coverage.** It is not a quality signal here; it is waste.

The smoke test's own rules, which are in its docstring and worth knowing before you run
it:

- Neo4j Community has one database, so the test isolates itself by **corpus root**: its
  four PDFs live under `.cache/smoke-corpus`, every assertion is scoped to that path, and
  the tree is deleted again afterwards.
- It **skips on a graph that already holds another corpus**, because entity resolution is
  the one stage that is not path-scoped. `SMOKE_ALLOW_POPULATED=1` runs it anyway and
  re-resolves the surviving corpus's entities on the way out — which costs seconds
  against a warm cache, and is still not a thing to do on demo morning.

---

## 5. Probing the retrieval

Material for testing the system rather than demonstrating it. Everything below is a fact
about the corpus that is actually in the graph, so each probe has a knowable right
answer — which is the only way to tell retrieval working from retrieval being plausible.

### What is in the archive

4 clients, 14 projects, 3 years each, 3 documents per project-year (one
`projektbericht`, two `statusbericht` for Q2 and Q4). 120 documents, 766 chunks.

| Kunde | Projekte | Jahre |
|---|---|---|
| **Brandt und Soehne Maschinenbau** | Atlas, Serviceportal, Taktzeit | 2022–2024 |
| **Finanzkontor Weser AG** | Datenschild, Kontrapunkt, Pruefstand | 2022–2024 |
| **Nordlicht Energie AG** | Atlas, Kompass, Windfeld, Zaehlerwerk | 2022–2024 |
| **Sanitas Klinikverbund** | Datenschild, Triage, Vitalis, Logistik Plus | 2022–2024 (Logistik Plus: 2023 only) |

Note the two **name collisions**: `Projekt Atlas` exists at Brandt *and* Nordlicht,
`Projekt Datenschild` at Finanzkontor *and* Sanitas — in each case entirely different
work under the same name. That is what makes a wrong answer here recognisable as wrong.

### What crosses a client boundary, and what only looks like it does

```sh
just entities --bridges
```

| Entity | Type | Clients | What it is |
|---|---|---|---|
| `Dr. Katharina Vogt` | person | Brandt, Nordlicht | **Planted bridge 1.** Leads both `Projekt Atlas` projects |
| `Atlas-Referenzarchitektur` | konzept | Brandt, Nordlicht | **Planted bridge 1.** The artefact she carried across |
| `Fehlerbehandlungsmuster`, `Schnittstellenverträge`, `Namenskonvention der Ereignistypen` | konzept / ergebnis | Brandt, Nordlicht | What was reused from it, unchanged |
| `Milan Horvat` | person | Finanzkontor, Sanitas | **Planted bridge 2.** Leads both `Projekt Datenschild` projects |
| `Prüfschema Berechtigungsrollen` | ergebnis | Finanzkontor, Sanitas | **Planted bridge 2.** The artefact he carried across |
| `Luis Arndt`, `Sophie Lehmann`, `Jens Petersen`, `Tobias Kern`, `Miriam Falk`, `Dr. Annika Roth` | person | all four | **Noise, by design.** Team members staffed everywhere. A node touching all four clients is not a finding |
| `Schnittstellenverzeichnis` | ergebnis | all four | Noise. Every project keeps one |

A **bridge that means something touches exactly two clients.** That distinction is the
single most useful thing to hold on to when reading the graph.

And what must stay *apart*: two `Projekt Atlas` nodes and two `Projekt Datenschild`
nodes, one per client. `just entities --contains Atlas` shows the scope each was given.

### Probes, with the answer they should give

Run each with `just query "…"`, or in the UI. The right answer is stated so a wrong one
is visible.

| # | Probe | What it tests | Expected |
|---|---|---|---|
| 1 | „Wer leitet das Projekt Datenschild und worum geht es dabei?" | Name collision, kept apart | Milan Horvat at **both**, then *Berechtigungskonzept Fallakte* for Sanitas and *Löschkonzept Kundendaten* for Finanzkontor — never merged into one project |
| 2 | „Welche Projekte hat die Meridian Chemie AG beauftragt?" | **Negative control** — hallucination under pressure | An explicit "no information about this client". If this invents a project, grounding is broken |
| 3 | „Was verbindet das Projekt Atlas von Nordlicht Energie AG mit dem Projekt Atlas von Brandt und Soehne Maschinenbau?" | The thesis | Dr. Katharina Vogt and the Atlas-Referenzarchitektur. Run it with `--legs search` first: that must answer "no connection" |
| 4 | „Wo wurde das Prüfschema Berechtigungsrollen wiederverwendet?" | Bridge 2, artefact-first | Carried by Milan Horvat from one `Projekt Datenschild` to the other |
| 5 | „Woran hat Dr. Annika Roth gearbeitet?" | Noise node — the contrast with a real bridge | A broad spread of unrelated projects over several clients (she is mentioned under all four), with no shared artefact joining any two. Compare with probe 3: that is the difference between a staffing coincidence and a bridge |
| 6 | „Welche Systeme wurden im Projekt Zaehlerwerk der Nordlicht Energie AG eingesetzt?" | Single-branch fact retrieval | Nordlicht/Zaehlerwerk only. Weakest case — see "Questions to avoid on stage" |
| 7 | „Und 2024?" as a follow-up to any of the above | Rewriting | The standalone rewrite is shown before retrieval runs. `--window 0` turns the memory off, for the comparison |

### Looking underneath an answer

| Command | What it shows |
|---|---|
| `just search "<q>" --mode vector` / `--mode fulltext` | One search leg alone, and therefore which of the two actually found a chunk |
| `just traverse "<q>" --hops 1` / `--hops 2` | The entity walk alone: which nodes the question matched, every edge crossed, and what each hop added |
| `just query "<q>" --legs search` / `--legs traversal` | The same question answered from one leg only. The comparison that makes the graph's contribution visible |
| `just entities --contains <name>` | What merged into one node, and under which scope |
| `just summaries --contains projektbericht-2023` | The generated summaries side by side — the quickest way to see how near-identical these reports are |
| `POST /api/ask` | The same thing as JSON, including `subgraph.edges[].crosses_branches`, which is what is drawn in gold |

### Reading the context table

Every `just query` prints the merged context before the answer. Four things in it are
worth more than the prose:

- **`graph` and `such` columns** — the chunk's rank in each leg. A chunk with a rank in
  `graph` and a dash in `such` is one only the entity walk found.
- **„n Chunks hätte die Suche allein nicht gefunden"**, with the folders they came from.
  This is the measurement the whole project exists to produce.
- **„Die Chunks stammen aus n Kundenablagen"** — more than one is the cross-client claim,
  re-derived for each question rather than asserted once.
- **`zitiert`** — which sources the answer actually cited. A cited number pointing at no
  source would be flagged; an uncited chunk just means it was in context and unused.
