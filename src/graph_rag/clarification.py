"""Asking back, when a question names a folder the corpus cannot resolve on its own.

This corpus is built so that a name often does *not* identify one thing: `B2BC` is a
model of two unrelated machines at two manufacturers, `XTC27` is a machine type at two
of them, and `Projekt Atlas` belongs to two clients. Answering such a question picks a
branch and sounds certain about it — the worst failure available here, because the
answer is fluent, cited, and about the wrong machine.

So the check is structural rather than semantic: the hierarchy already knows which
names it cannot tell apart, and it knows which folders have something below them. Two
situations are worth interrupting for:

* **ambiguous** — the name matches folders in more than one place. Answering would
  choose one silently.
* **underspecified** — the name matches one folder, that folder has children, and the
  question names none of them. The answer would be drawn from whichever child's
  documents happened to rank, which reads as a statement about the whole family.

Nothing here knows what a machine or a client *is*: it reads folder names, depths and
containment, so a corpus nested differently gets the same behaviour without a change.

Two things keep it from crying wolf. Names in a question **narrow each other** — asked
about `XTC27 B2BC`, only one `XTC27` has a `B2BC` under it and only one `B2BC` has an
`XTC27` above it, so neither is ambiguous any more. And a question that is plainly
*asking to enumerate* ("welche Modelle gibt es…") is answered rather than bounced: the
list it wants is the answer, and demanding one of its items first would be absurd.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

_NON_WORD = re.compile(r"[^0-9a-z]+")
_GERMAN_FOLDING = {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}

#: A question containing one of these is asking for the list, not for one of its items.
#: Deliberately a short, readable list of German cues rather than a model call: the
#: check runs on every question before anything is retrieved, and a clarifier that
#: costs a round trip would be turned off.
ENUMERATING_CUES: tuple[str, ...] = (
    "welche modelle",
    "welche varianten",
    "welche ausfuehrungen",
    "welche maschinen",
    "welche projekte",
    "welche optionen",
    "welche baureihen",
    "liste",
    "uebersicht",
    "alle modelle",
    "alle varianten",
    "gibt es",
)


def _fold(text: str) -> str:
    lowered = text.casefold()
    for letter, replacement in _GERMAN_FOLDING.items():
        lowered = lowered.replace(letter, replacement)
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def tokens(text: str) -> tuple[str, ...]:
    """Folded word tokens. Splitting on non-alphanumerics is what makes `P6V` fail to
    match `P6VX` — a substring test would match it, and would answer a question about
    one model from the documents of another."""
    return tuple(token for token in _NON_WORD.split(_fold(text)) if token)


@dataclass(frozen=True)
class Folder:
    """One folder of the corpus, as the clarifier needs to see it."""

    name: str
    path: str
    depth: int
    #: Folder names from the top down to and including this one.
    ancestry: tuple[str, ...]
    children: tuple[str, ...]
    #: The inferred name of this depth — "Modell", "Projekt" — or None if unlabelled.
    level: str | None = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def branch(self) -> str:
        return self.ancestry[0] if self.ancestry else self.name


@dataclass(frozen=True)
class Choice:
    """One thing the user could have meant."""

    name: str
    path: str
    #: Where it sits, for a reader who needs to tell two identical names apart.
    context: str
    level: str | None = None


@dataclass(frozen=True)
class Clarification:
    """A question to put back to the user instead of guessing."""

    #: "ambiguous" or "underspecified" — which of the two situations this is.
    kind: str
    #: The word in the question that could not be resolved.
    named: str
    question: str
    choices: tuple[Choice, ...]

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(choice.path for choice in self.choices)


def _named_in(question: str, folders: Sequence[Folder]) -> dict[str, list[Folder]]:
    """Folder names the question contains, each with every folder carrying that name.

    Matched on whole token runs, so a folder named `P6V` is found in "QH4L P6V" and a
    folder named `P6VX` is not.
    """
    asked = tokens(question)
    found: dict[str, list[Folder]] = {}
    for folder in folders:
        needle = tokens(folder.name)
        if not needle:
            continue
        if any(
            asked[index : index + len(needle)] == needle
            for index in range(len(asked) - len(needle) + 1)
        ):
            found.setdefault(folder.name, []).append(folder)
    return found


def _compatible(candidate: Folder, other: str, folders: Sequence[Folder]) -> bool:
    """Whether `candidate` can still be meant when the question also names `other`.

    True when the other name sits above it on its own path, or below it somewhere in
    its subtree. This is what lets two individually ambiguous names resolve each other.
    """
    if other == candidate.name:
        return True
    if other in candidate.ancestry:
        return True
    prefix = candidate.path
    return any(
        folder.name == other and folder.path.startswith(prefix) for folder in folders
    )


def _narrow(
    candidates: Sequence[Folder], others: Iterable[str], folders: Sequence[Folder]
) -> list[Folder]:
    narrowed = [
        candidate
        for candidate in candidates
        if all(_compatible(candidate, other, folders) for other in others)
    ]
    # Narrowing to nothing means the question combined names that never co-occur. The
    # unnarrowed set is the honest thing to offer — it is what the corpus does have.
    return narrowed or list(candidates)


def is_enumerating(question: str) -> bool:
    folded = " ".join(tokens(question))
    return any(" ".join(tokens(cue)) in folded for cue in ENUMERATING_CUES)


def _choice(folder: Folder) -> Choice:
    return Choice(
        name=folder.name,
        path=folder.path,
        context=" › ".join(folder.ancestry[:-1]) or folder.name,
        level=folder.level,
    )


def _child_choices(targets: Sequence[Folder], folders: Sequence[Folder]) -> list[Choice]:
    by_path = {folder.path: folder for folder in folders}
    choices = []
    for target in targets:
        for child in target.children:
            child_folder = by_path.get(f"{target.path}/{child}")
            if child_folder is not None:
                choices.append(_choice(child_folder))
    return choices


def _level_word(choices: Sequence[Choice], fallback: str) -> str:
    named = {choice.level for choice in choices if choice.level}
    return named.pop() if len(named) == 1 else fallback


def clarify(question: str, folders: Sequence[Folder]) -> Clarification | None:
    """The question to ask back, or None when the question resolves on its own.

    The *deepest* named folder decides: a question naming both a machine type and a
    model is a question about the model, and asking which type it belongs to would be
    answering a question nobody asked.
    """
    named = _named_in(question, folders)
    if not named:
        return None

    deepest = max(named, key=lambda name: max(f.depth for f in named[name]))
    others = [name for name in named if name != deepest]
    targets = _narrow(named[deepest], others, folders)

    # Several places carry this name, and nothing in the question says which.
    if len(targets) > 1:
        children = _child_choices(targets, folders)
        named_children = {
            name for name in named if any(c.name == name for c in children)
        }
        if named_children:
            return None  # the question already names one of them
        choices = tuple(children or [_choice(target) for target in targets])
        word = _level_word(choices, "Treffer")
        return Clarification(
            kind="ambiguous",
            named=deepest,
            question=(
                f"»{deepest}« kommt {len(targets)}-mal im Korpus vor. "
                f"Bitte das gemeinte {word} angeben:"
            ),
            choices=choices,
        )

    target = targets[0]
    if target.is_leaf:
        return None
    if is_enumerating(question):
        return None  # the list below it *is* the answer
    if any(name in target.children for name in named):
        return None

    choices = tuple(_child_choices([target], folders))
    if not choices:
        return None
    word = _level_word(choices, "Eintrag")
    return Clarification(
        kind="underspecified",
        named=target.name,
        question=(
            f"Zu »{target.name}« gibt es {len(choices)} Einträge "
            f"auf der Ebene {word}. Bitte einen davon angeben:"
        ),
        choices=choices,
    )
