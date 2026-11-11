"""The silent failures of rewriting a follow-up before retrieval runs.

Every defect here produces a well-formed German question, which retrieves real chunks
from real documents and is answered fluently. None of them raises, and none is visible
in the answer: the system simply answered a question nobody asked. That is strictly
worse than having no rewrite at all, because without one the failure is loud — "und
2023?" returns nothing and everyone can see why.

Three properties are guarded: that a rewrite only ever happens when there is something
to resolve against, that every unusable reply falls back to the question as typed, and
that the history reaches the model bounded, in order, and in its resolved form.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from graph_rag.conversation import (
    DEFAULT_WINDOW,
    MAX_REWRITE_CHARACTERS,
    Conversation,
    Exchange,
    build_rewrite_prompt,
    parse_rewrite,
    rewrite_question,
)


@dataclass
class FakeModel:
    """A chat tier that answers with whatever it was given, and records the prompt."""

    reply: str = '{"frage": "umformuliert"}'
    prompts: list[str] = field(default_factory=list)

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str:
        self.prompts.append(user)
        return self.reply


def _conversation(*pairs: tuple[str, str], window: int = DEFAULT_WINDOW) -> Conversation:
    conversation = Conversation(window=window)
    for question, answer in pairs:
        conversation = conversation.extend(Exchange(question=question, answer=answer))
    return conversation


def test_the_first_question_of_a_conversation_is_never_sent_to_the_model():
    """There is nothing to resolve it against, so the only thing a rewrite could do to
    it is change it — and a first question silently altered before retrieval is the
    one failure this whole feature must not introduce."""
    model = FakeModel()

    rewrite = rewrite_question("Was ist Projekt Atlas?", Conversation(), model)

    assert model.prompts == []
    assert rewrite.standalone == "Was ist Projekt Atlas?"
    assert not rewrite.changed


def test_a_question_the_model_returns_unchanged_is_not_reported_as_rewritten():
    """Rule 2 asks the model to echo an already-standalone question back, which is the
    right answer — and `changed` is read off the text rather than off the fact that a
    call was made, so that echo is not announced on stage as a rewrite that happened."""
    asked = "Welche Risiken nennt der Bericht zu Projekt Atlas?"
    model = FakeModel(reply=json.dumps({"frage": asked}))

    rewrite = rewrite_question(asked, _conversation(("Frage", "Antwort")), model)

    assert rewrite.standalone == asked
    assert not rewrite.changed
    assert rewrite.note is None


def test_every_unusable_reply_falls_back_to_the_question_as_typed():
    """The floor is the behaviour of a system without this feature.

    Retrieval running on an empty string returns nothing and answers nothing, which is
    the one outcome worse than not rewriting — and three of these four replies would
    produce exactly that if the parse trusted them.
    """
    asked = "Und 2023?"
    for reply in ("", "Hier ist die Frage: ...", "[]", '{"frage": "   "}'):
        rewrite = parse_rewrite(reply, asked=asked)

        assert rewrite.standalone == asked, reply
        assert rewrite.note is not None, reply


def test_a_rewrite_that_swallows_the_whole_history_is_refused():
    """A model that concatenates every turn into one question produces a query whose
    every term the lexical leg ORs together — the whole corpus retrieved weakly instead
    of one document retrieved strongly. Worse than no rewrite, and invisible."""
    asked = "Und 2023?"
    runaway = "Was " + "x" * MAX_REWRITE_CHARACTERS + "?"

    rewrite = parse_rewrite(json.dumps({"frage": runaway}), asked=asked)

    assert rewrite.standalone == asked
    assert "characters" in (rewrite.note or "")


def test_markdown_emphasis_is_stripped_out_of_a_rewrite():
    """Observed against the live model, not hypothetical: asked to return an
    already-standalone question unchanged, the chat tier bolds the proper nouns in it
    because that is how it writes prose everywhere else. Both characters are Lucene
    syntax, so the full-text leg goes looking for `\\*\\*Taktzeit\\*\\*` — a token no
    document contains — and entity matching normalises the asterisks into the surface
    name and misses the node. Neither failure says anything."""
    reply = json.dumps({"frage": "Welche Risiken nennt der Bericht zu **Taktzeit**?"})

    rewrite = parse_rewrite(reply, asked="Und die Risiken?")

    assert rewrite.standalone == "Welche Risiken nennt der Bericht zu Taktzeit?"


def test_a_rewrite_that_drops_most_of_the_question_is_refused():
    """Resolving a shorthand question only ever adds what it pointed at. A reply much
    shorter than the question summarised or truncated it — a two-clause question coming
    back with one clause, answered as though that were all that was asked."""
    asked = "Welche Risiken nennt der Bericht zu Projekt Atlas, und wer hat sie erfasst?"

    rewrite = parse_rewrite(json.dumps({"frage": "Welche Risiken?"}), asked=asked)

    assert rewrite.standalone == asked
    assert rewrite.note is not None


def test_the_prompt_carries_the_window_oldest_first():
    """A pronoun refers to the most recent thing that could bear it. A history handed
    over newest-first resolves "das Projekt" against the wrong turn and produces a
    perfectly well-formed question about something nobody asked after."""
    model = FakeModel()
    conversation = _conversation(("Erste Frage", "A"), ("Zweite Frage", "B"))

    rewrite_question("Und 2023?", conversation, model)

    prompt = model.prompts[0]
    assert prompt.index("Erste Frage") < prompt.index("Zweite Frage")
    assert prompt.index("Zweite Frage") < prompt.index("Und 2023?")


def test_only_the_window_reaches_the_model():
    """Bounded, because an unbounded history is cost and confusion with no payoff — and
    because the oldest turns are the ones most likely to supply a *wrong* antecedent
    for a pronoun that meant the last thing said."""
    model = FakeModel()
    conversation = _conversation(
        *((f"Frage {number}", f"Antwort {number}") for number in range(1, 7)),
        window=2,
    )

    rewrite_question("Und 2023?", conversation, model)

    prompt = model.prompts[0]
    assert "Frage 5" in prompt and "Frage 6" in prompt
    assert "Frage 4" not in prompt


def test_the_history_holds_the_resolved_question_not_the_shorthand():
    """Follow-ups chain. "und 2023?" then "wer hat das geleitet?" resolves only if the
    first was recorded resolved — a history of shorthand degrades one turn at a time,
    which reads as the model getting worse rather than as a bug."""
    conversation = _conversation(
        ("Was ist Projekt Atlas der Nordlicht Energie AG?", "Eine Migration.")
    )

    prompt = build_rewrite_prompt("Wer hat das geleitet?", conversation.recent)

    assert "Nordlicht Energie AG" in prompt


def test_an_overlong_answer_is_trimmed_at_a_sentence_boundary():
    """The answer is in the prompt because the antecedent often lives in it. Cutting it
    mid-clause would hand the model a sentence that says something different from what
    was actually answered."""
    long_answer = "Dr. Katharina Vogt leitete das Projekt. " + "Weiterer Satz. " * 100

    prompt = build_rewrite_prompt("Wer war das?", [Exchange("Frage", long_answer)])

    answer_line = next(
        line for line in prompt.splitlines() if line.startswith("Antwort: ")
    )
    assert "Dr. Katharina Vogt" in answer_line
    assert answer_line.rstrip().endswith(".")
    assert len(answer_line) < len(long_answer)


def test_a_conversation_never_grows_past_its_window():
    conversation = _conversation(
        *((f"Frage {number}", "A") for number in range(1, 10)), window=3
    )

    assert len(conversation.exchanges) == 3
    assert conversation.exchanges[-1].question == "Frage 9"
