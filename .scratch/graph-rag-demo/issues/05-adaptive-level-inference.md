# 05: Adaptive level inference

**Read `../README.md` first.**

**What to build:** a demo operator points ingestion at an unfamiliar corpus and the system works
out by itself what each level of the hierarchy *means* — "depth 1 looks like Client, depth 2
looks like Year" — displaying the inferred names and its confidence so the operator knows
whether to trust them before going on stage. An explicit override exists for when inference
guesses wrong.

**Blocked by:** 04 (folder and document graph).

**Status:** done

## Context

This is a headline demo feature, not a convenience. "Point it at any folder tree and it works
out what the levels mean" is a materially better story than "configure your schema first", and
it costs almost nothing.

The mechanism: after the directory walk, **one inexpensive model call per depth level** inspects
the sibling folder names at that level and infers what the level represents. The inferred name
and a confidence score are stored on every folder node at that depth. Note the economics —
one call *per level*, not per folder. A corpus of hundreds of folders and four levels costs four
calls.

An explicit ordered list of level names may be supplied at ingestion time to override inference
entirely, falling back to inference for any depth beyond the supplied list. This override exists
specifically so a bad inference does not derail a live demo — it is the escape hatch, so make
sure it genuinely bypasses the model rather than merely biasing it.

Surfacing confidence matters for the same reason: the operator needs to know *before* presenting
whether the inference is shaky.

Use the chat-tier model for this, not the extraction-tier model — it is low-volume and the
quality of the inference is visible to the audience.

Downstream, these inferred level names are what let answers say "this came from Client X's 2023
Atlas project" in natural language rather than reciting a path.

## Acceptance criteria

- [x] After the walk, one model call per depth level infers that level's semantic name
- [x] Inferred name and confidence are stored on every folder node at the corresponding depth
- [x] Cost scales with depth, not with folder count
- [x] An ingestion option accepts an explicit ordered list of level names that bypasses inference
- [x] The explicit list falls back to inference for depths beyond those supplied
- [x] Inferred level names and confidences are reported to the operator at the end of ingestion
- [x] Verified on a corpus whose level names are not `Client/Year/Project` — live inference on a 4-level Region/Abteilung/Quartal/Revision tree returned all four correctly at 0.95-0.98, 120 folders labelled in 4 calls
- [x] Verified that the override fully bypasses the model for the levels it covers
