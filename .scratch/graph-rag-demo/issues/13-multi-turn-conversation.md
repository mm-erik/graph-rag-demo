# 13: Multi-turn conversation with query rewriting

**Read `../README.md` first.**

**What to build:** a demo audience member asks a follow-up question using pronouns and
shorthand — "and what about 2023?", "who else worked on it?" — and the system resolves it
against the conversation so far instead of searching for the literal fragment.

**Blocked by:** 12 (hybrid retrieval and answer generation).

**Status:** ready-for-agent

## Context

A small ticket with outsized demo value, and it can run in parallel with ticket 14.

The reasoning, stated plainly in the spec: an audience **will** reliably ask a pronoun-laden
second question, and a standalone-only system faceplants visibly at exactly that moment. There
is no recovering the room from "and what about 2023?" returning nothing.

The mechanism: retain a bounded window of recent turns. Before retrieval, have the chat-tier
model rewrite the follow-up into a standalone question. Retrieval and generation then proceed
unchanged — the rewrite happens *before* the pipeline from ticket 12, not inside it.

Keep the window bounded. This is a demo; unbounded history is cost and confusion with no payoff.

The rewritten question is worth surfacing: seeing "and what about 2023?" become "what happened
on Client A's Atlas project in 2023?" is itself a nice moment on stage, and it makes debugging a
bad follow-up answer far easier.

A question that is already standalone should pass through substantially unchanged — the rewrite
must not mangle well-formed questions.

## Acceptance criteria

- [ ] A bounded window of recent conversation turns is retained
- [ ] Follow-up questions are rewritten into standalone questions before retrieval
- [ ] Rewriting uses the chat-tier model and happens before the existing retrieval pipeline, not inside it
- [ ] The rewritten question is available for display alongside the answer
- [ ] An already-standalone question passes through substantially unchanged
- [ ] Verified with a multi-turn exchange using pronouns and ellipsis against the synthetic corpus
- [ ] Verified that a follow-up narrowing by year correctly scopes to that year's folder
