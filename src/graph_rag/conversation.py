"""Follow-up questions, resolved against the conversation before retrieval runs.

An audience *will* ask the second question with a pronoun in it — "und 2023?", "wer hat
sonst noch daran gearbeitet?" — and a system that only accepts standalone questions
faceplants visibly at exactly that moment. There is no recovering the room from "und
2023?" retrieving nothing.

The mechanism is one chat-tier call *before* the pipeline, never inside it: the window
of recent turns plus the shorthand question go to the model, a standalone question comes
back, and ticket 12's `answer_question` runs on that, unchanged and unaware. That
direction of dependency is the whole design — `answer.py` does not import this module,
so nothing about retrieval or generation is now conditional on there being a
conversation at all.

Two properties keep it from being worse than no rewrite:

**An already-standalone question must survive it.** A rewrite that "improves" a
well-formed question is a silent regression — it retrieves something slightly different
from what was asked, and the only symptom is a slightly wrong answer.

**Every failure falls back to the question as asked.** The model returning prose,
nothing, or a paragraph is recoverable; the pipeline running on an empty string is not.

The history is a list of plain question/answer pairs rather than a live object, so
ticket 14 can accept it in a request body and the CLI can hold it in a local variable,
without either one owning the conversation.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from graph_rag.answer import Answer

#: Turns the rewrite is shown. Four is two or three follow-ups deep, which is as far as
#: a demo exchange ever gets, and bounded because an unbounded history is cost and
#: confusion with no payoff: a pronoun in question seven refers to something in question
#: six, never to something in question one.
DEFAULT_WINDOW = 4

#: Cap on one answer inside the rewrite prompt. The answer is there because the
#: antecedent often lives in it rather than in the question — "wer hat sonst noch daran
#: gearbeitet?" refers to a person the answer named — and the opening sentences are
#: where those names are. The cap exists for the runaway answer, not the normal one:
#: generation is capped at eight sentences, so this trims almost nothing in practice.
ANSWER_EXCERPT_CHARACTERS = 600

#: Longest rewrite accepted. A model that stuffs the entire history into one question
#: produces a query whose every term is OR-ed by the lexical leg, which retrieves the
#: whole corpus weakly rather than the right document strongly — a worse failure than
#: not rewriting at all, and an invisible one.
MAX_REWRITE_CHARACTERS = 300

#: Shortest rewrite accepted, as a fraction of the question asked. Resolving a
#: shorthand question only ever *adds* what it was pointing at, so a reply much shorter
#: than the question is one that summarised or truncated it — a two-clause question
#: coming back with one clause answered, silently, as though that were all that was
#: asked. Half is loose enough that German function words moving around never trips it.
MIN_REWRITE_RATIO = 0.5

#: The note on a question that was never put to the model, because it opened the
#: conversation. Named rather than written inline because a reader has to be able to
#: tell it apart from the notes that mean a rewrite was *attempted and failed*: the
#: first is the normal course of events and the second is worth interrupting a demo
#: for, and both leave the question unchanged.
NOTHING_TO_RESOLVE = "there was nothing to resolve it against"


@dataclass(frozen=True)
class Exchange:
    """One question-and-answer pair, as the rewrite prompt reads it.

    `question` is the **standalone** form, not the shorthand the user typed. Follow-ups
    chain — "und 2023?" then "wer hat das geleitet?" — and the second one resolves
    against the first only if the first was recorded resolved. Storing the shorthand
    makes the history degrade one turn at a time, which looks like the model getting
    worse rather than like a bug.

    Deliberately plain strings: ticket 14 accepts this in a request body, and a history
    that carried an `Answer` with its chunks and ancestry would be a megabyte of
    round-tripped payload to resolve one pronoun.
    """

    question: str
    answer: str


@dataclass(frozen=True)
class Conversation:
    """The turns so far, bounded.

    Immutable: `extend` returns a new conversation rather than mutating this one, so
    the CLI's loop and an HTTP handler hold the history exactly the same way and
    neither can be surprised by the other's idea of what was said.
    """

    exchanges: tuple[Exchange, ...] = ()
    window: int = DEFAULT_WINDOW

    def extend(self, exchange: Exchange) -> Conversation:
        """This conversation with one more turn, dropping anything past the window."""
        kept = (*self.exchanges, exchange)
        return Conversation(
            exchanges=kept[-self.window :] if self.window > 0 else (),
            window=self.window,
        )

    @property
    def recent(self) -> tuple[Exchange, ...]:
        """The turns the rewrite is shown, oldest first."""
        return self.exchanges[-self.window :] if self.window > 0 else ()


# German, per the project convention, and the corpus's language — the rewrite must not
# quietly translate the question into the language the model prefers to reason in.
#
# Rule 2 is the one that earns its place. Asked to "make this question standalone", a
# model handed a question that already is one will still rewrite it, because that is
# what it was asked to do: it expands abbreviations, adds a year it inferred, and
# retrieval then answers a question nobody asked. Nothing about that failure is visible
# — the answer is fluent and cites real documents.
#
# Rule 3 is the corpus's: this archive's whole difficulty is that `Nordlicht Energie
# AG` and `Brandt & Söhne Maschinenbau GmbH` have near-identical reports, and the
# lexical leg matches literal tokens while the entity leg matches normalised names.
# A rewrite that shortens a client's name to "Nordlicht" degrades both at once.
REWRITE_SYSTEM_PROMPT = (
    "Du formulierst Rückfragen an ein Projektarchiv so um, dass sie ohne den "
    "bisherigen Gesprächsverlauf verständlich sind.\n"
    "\n"
    "Regeln:\n"
    "1. Löse Pronomen und Auslassungen auf ('dort', 'das Projekt', 'und 2023?') und "
    "setze ein, worauf sie sich im Verlauf beziehen.\n"
    "2. Ist die Frage bereits ohne den Verlauf verständlich, gib sie Wort für Wort "
    "unverändert zurück. Formuliere sie dann nicht um, kürze sie nicht und ergänze "
    "nichts.\n"
    "3. Übernimm Eigennamen — Kunden, Projekte, Personen, Systeme — exakt so, wie sie "
    "im Verlauf geschrieben stehen. Kürze sie nicht ab.\n"
    "4. Übernimm Einschränkungen der Rückfrage, besonders Jahreszahlen und Zeiträume.\n"
    "5. Beantworte die Frage nicht und füge nichts hinzu, was nicht in der Frage oder "
    "im Verlauf steht.\n"
    "6. Antworte auf Deutsch, mit genau einer Frage, als reiner Text — keine "
    "Hervorhebungen, keine Sternchen, keine Anführungszeichen.\n"
    "\n"
    'Antworte ausschliesslich als JSON: {"frage": "..."}'
)


#: Markdown emphasis, stripped out of a rewrite. Observed, not hypothetical: asked to
#: return a question about "Projekt Taktzeit von Brandt & Söhne", the chat tier returns
#: "Projekt **Taktzeit** der **Brandt & Söhne**?" — it decorates proper nouns because
#: that is how it writes prose everywhere else. Both characters are Lucene syntax, so
#: `lucene_query` escapes them into the search term and the full-text leg then looks
#: for a token no document contains; entity matching normalises them into the surface
#: name and misses the node. Neither fails loudly, and neither has any business in a
#: German question.
_DECORATION = ("*", "`")


def _plain(text: str) -> str:
    """A model's question as plain text, with its formatting removed."""
    for character in _DECORATION:
        text = text.replace(character, "")
    return " ".join(text.split())


def _excerpt(text: str, limit: int) -> str:
    """An answer trimmed to the last whole sentence that fits inside `limit`."""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= limit:
        return collapsed
    head = collapsed[:limit]
    stop = max(head.rfind(". "), head.rfind("! "), head.rfind("? "))
    return head[: stop + 1] if stop > 0 else head.rstrip() + " […]"


def build_rewrite_prompt(question: str, recent: Sequence[Exchange]) -> str:
    """The window and the shorthand question, oldest turn first.

    Chronological, and numbered. A pronoun refers to the *most recent* thing that could
    bear it, so a history handed over newest-first resolves "das Projekt" against the
    wrong turn — and produces a perfectly well-formed standalone question about
    something nobody asked after.
    """
    lines: list[str] = ["Bisheriger Verlauf:"]
    for position, exchange in enumerate(recent, start=1):
        lines.append(f"--- Runde {position} von {len(recent)} ---")
        lines.append(f"Frage: {exchange.question}")
        lines.append(f"Antwort: {_excerpt(exchange.answer, ANSWER_EXCERPT_CHARACTERS)}")
    lines.append("")
    lines.append(f"Neue Rückfrage: {question}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Rewrite:
    """What the question became, and whether anything happened to it."""

    #: The question as the user typed it.
    asked: str
    #: What retrieval is actually run on. Equal to `asked` when nothing was rewritten,
    #: including when the rewrite failed — never empty.
    standalone: str
    #: Why the question was passed through unchanged, when it was. `None` means it was
    #: genuinely rewritten. Carried rather than raised: a failed rewrite still has a
    #: usable question behind it, and losing a turn mid-demo over it is the worse
    #: outcome.
    note: str | None = None

    @property
    def changed(self) -> bool:
        """Whether retrieval runs on something other than what was typed.

        Compared on the text rather than trusted from the model: a model that echoes
        the question back — which rule 2 asks it to do, and which is the right answer
        for an already-standalone question — must not be reported on stage as having
        rewritten anything.
        """
        return self.standalone.strip() != self.asked.strip()


class RewriteModel(Protocol):
    """The chat tier, narrowed to the one call a follow-up makes.

    The chat tier rather than the cheaper extraction tier, deliberately: resolving
    "und 2023?" against a German answer is the same reading comprehension the answer
    itself needs, it is one call per *turn* rather than per chunk, and getting it wrong
    is visible to the audience immediately.
    """

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...


def parse_rewrite(reply: str, asked: str) -> Rewrite:
    """A model reply as the question to retrieve on.

    Total, like `parse_inference`: every reply yields a usable question, because the
    alternative to a usable question is a turn that visibly does nothing. Prose,
    truncated JSON, an empty object and a paragraph-long "question" all fall back to
    what the user typed — which is the behaviour of a system without this feature, and
    therefore the right floor.
    """
    try:
        payload = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return Rewrite(asked=asked, standalone=asked, note="the reply was not JSON")
    if not isinstance(payload, dict):
        return Rewrite(asked=asked, standalone=asked, note="the reply was not an object")
    standalone = _plain(str(payload.get("frage") or ""))
    if not standalone:
        return Rewrite(asked=asked, standalone=asked, note="the reply held no question")
    if len(standalone) < len(asked.strip()) * MIN_REWRITE_RATIO:
        return Rewrite(
            asked=asked,
            standalone=asked,
            note="the rewrite dropped most of the question rather than resolving it",
        )
    if len(standalone) > MAX_REWRITE_CHARACTERS:
        return Rewrite(
            asked=asked,
            standalone=asked,
            note=(
                f"the rewrite ran to {len(standalone)} characters, past the "
                f"{MAX_REWRITE_CHARACTERS}-character limit"
            ),
        )
    return Rewrite(asked=asked, standalone=standalone)


def rewrite_question(
    question: str,
    conversation: Conversation,
    model: RewriteModel,
) -> Rewrite:
    """A follow-up as a standalone question, resolved against the conversation.

    **The first question of a conversation is never sent to the model.** There is
    nothing to resolve it against, the call would be latency an audience watches for
    no result, and the only thing a rewrite could do to it is change it.
    """
    recent = conversation.recent
    if not recent:
        return Rewrite(
            asked=question, standalone=question, note=NOTHING_TO_RESOLVE
        )
    return parse_rewrite(
        model.complete(
            REWRITE_SYSTEM_PROMPT,
            build_rewrite_prompt(question, recent),
            json_object=True,
        ),
        asked=question,
    )


@dataclass(frozen=True)
class Turn:
    """One exchange: what was asked, what it was resolved to, and what came back.

    The pair is the unit the CLI prints and ticket 14 serialises — seeing "und 2023?"
    become "Was geschah 2023 im Projekt Atlas der Nordlicht Energie AG?" is both the
    moment on stage and the first thing to look at when a follow-up answers oddly.
    """

    rewrite: Rewrite
    answer: Answer

    @property
    def exchange(self) -> Exchange:
        """This turn as the next turn's history."""
        return Exchange(question=self.rewrite.standalone, answer=self.answer.text)
