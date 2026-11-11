# 16: End-to-end smoke test and demo runbook

**Read `../README.md` first.**

**What to build:** confidence that the whole thing actually works together, and that someone can
run the demo weeks from now without the person who built it. One test exercises the real path
end to end against real infrastructure, and a runbook documents the exact sequence to go from a
cold machine to a working demo.

**Blocked by:** 15 (React chat and traversal visualisation).

**Status:** ready-for-agent

## Context

### The smoke test

This is the **only test in the project that catches wiring and configuration errors**. The other
three test targets are pure functions — folder construction, entity merging, rank fusion — and
by design none of them touches a driver, a client, or a model. Everything between those pure
functions is unverified until this test exists.

It runs against **real infrastructure**: real Neo4j, real Mistral API, real parsing service. No
fakes. This is cheap precisely because the infrastructure is running anyway, and it is the
deliberate alternative to building fake implementations at the top seam — a trade made because
this project is being discarded.

Scope it to a tiny corpus of a few PDFs so it runs in reasonable time. Assert that nodes and
relationships of each expected kind exist after ingestion, and that a known question retrieves a
known document.

Do not expand test coverage beyond this and the three pure-function test sets. Broad coverage is
not a quality signal here; it is waste on a throwaway project. See the testing posture in the
README.

### The runbook

The failure mode this guards against is concrete: the demo is run in front of people after weeks
of not being touched, on a machine where Docker is not running and the parsing service is not
up. Both were in exactly that state when this project was specified.

Document the real sequence, in order, including the external dependency that lives in the
sibling monorepo checkout and its slow cold start. Include what to check when something is
wrong — the connectivity and index-listing command from ticket 01 exists for this.

Also record, for whoever picks this up next: the entity resolution thresholds that were settled
on and why, anything that was tuned empirically against the synthetic corpus, and the state of
the two assumptions flagged in the spec (the exact Mistral model identifiers, and whether the
production parsing service starts cleanly standalone).

Finally, write down the demo script itself: which questions to ask, in what order, and what each
one is supposed to show. The target question about what connects two clients' same-named
projects is the finale; the earlier questions should build to it.

## Acceptance criteria

- [ ] One end-to-end smoke test ingests a tiny real corpus against real Neo4j, Mistral and parsing service
- [ ] It asserts nodes and relationships of each expected kind exist after ingestion
- [ ] It asserts a known question retrieves a known document
- [ ] It runs in reasonable time and is documented as requiring live infrastructure
- [ ] No test coverage added beyond this and the three pure-function test sets
- [ ] Runbook documents the exact cold-start sequence including the sibling parsing service
- [ ] Runbook documents troubleshooting for the common failure modes
- [ ] Settled entity resolution thresholds and their rationale recorded
- [ ] Status of the two flagged assumptions recorded: Mistral model identifiers, standalone parsing service
- [ ] A demo script records which questions to ask in which order and what each demonstrates
- [ ] Verified by following the runbook from a cold machine, with Docker stopped, and reaching a working demo
