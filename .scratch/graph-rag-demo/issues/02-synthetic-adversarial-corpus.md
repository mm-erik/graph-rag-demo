# 02: Synthetic adversarial German corpus generator

**Read `../README.md` first.**

**What to build:** a demo operator runs one command and gets a directory tree of German PDF
consulting reports that is *deliberately hostile* to flat retrieval — ready to be pointed at the
ingestion pipeline. Without this, there is nothing to demo against.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

## Context

The real client corpus is **not currently accessible**, so the entire demo is developed against
a synthetic stand-in. The pipeline must work on the real corpus unchanged once it is available —
so the synthetic corpus is a stand-in, never a special case. Do not let any knowledge of the
generator leak into the ingestion code.

This corpus is the experimental apparatus, not set dressing. The project's thesis is that graph
RAG wins on nested, self-similar data; if the corpus is not genuinely confusing to flat
retrieval, the demo proves nothing. Generate it adversarially **on purpose**:

- Roughly 4 clients × 3 years × 3–4 projects, shaped `Client / Year / Project / report.pdf`
- A project name (e.g. `Projekt Atlas`) deliberately reused across two *different* clients —
  this is the collision the money demo question depends on
- A consultant appearing under multiple clients — the shared entity that makes a cross-folder
  edge exist at all
- Near-identical quarterly status-report boilerplate across projects, so dense retrieval
  genuinely confuses documents rather than merely theoretically being able to

Content is **German**, matching the real corpus. Chunking parameters downstream are tuned for
German token density (compound words run 3–4 characters per token), and an English corpus would
misrepresent how the system behaves on the real thing.

PDFs must be **born-digital with a real text layer**. OCR is explicitly out of scope, so a
generator that rasterises text will silently break every downstream ticket.

The target demo question the corpus must be able to answer is: *"What connects Client A's Atlas
project to Client B's?"* Make sure a genuine, traceable connection exists in the generated
content — and make sure a flat keyword or embedding search would plausibly surface the wrong
client's report for a naively-phrased question.

Keep generation deterministic (seeded) so a demo is reproducible and so downstream tickets can
assert against known content.

## Acceptance criteria

- [ ] One command generates the full corpus tree to a configurable output directory
- [ ] Hierarchy is `Client / Year / Project / *.pdf`, ~4 × 3 × 3–4
- [ ] All content is German and reads as plausible consulting documentation
- [ ] At least one project name is shared across two different clients, with genuinely different content
- [ ] At least one person appears across multiple clients, with enough context to be extracted as an entity
- [ ] Status reports share heavy near-identical boilerplate across projects
- [ ] A documented, traceable connection exists between the two same-named projects of different clients
- [ ] PDFs have an extractable text layer (verified by extracting text from a generated file)
- [ ] Generation is seeded and reproducible
- [ ] A short note records what collisions were planted and where, so the demo operator knows what to ask
- [ ] No generator-specific knowledge is required by, or leaks into, the ingestion pipeline
