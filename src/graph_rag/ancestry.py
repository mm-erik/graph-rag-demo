"""A document's place in the hierarchy, as read back out of the graph.

The structural advantage this project exists to demonstrate is legible in exactly one
place: a fact that arrives labelled *Kunde: Nordlicht Energie AG, Jahr: 2023* rather
than as a filesystem path. The folder names come from ticket 04 and what each depth
*means* from ticket 05, and both a document summary and a retrieved chunk carry them —
hence a shared home rather than a copy in each.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Ancestor:
    """One folder on the path from the corpus root to the document."""

    name: str
    #: What that depth of the hierarchy means, from ticket 05, when it is known. A
    #: nice-to-have, deliberately not a dependency: the names still read usefully
    #: unlabelled, and level inference may have been skipped or have failed.
    level: str | None


def ancestry_lines(ancestry: Sequence[Ancestor]) -> list[str]:
    """The document's place in the hierarchy, labelled where the level is known.

    This is what an answer needs in order to say *whose* project a fact came from: the
    client and the year live in the folder names and nowhere in the text of a document
    that refers to "der Auftraggeber" throughout.
    """
    return [
        f"{ancestor.level}: {ancestor.name}" if ancestor.level else ancestor.name
        for ancestor in ancestry
    ]


def branch(ancestry: Sequence[Ancestor]) -> str | None:
    """The top-level folder this document sits under — the client, in this corpus.

    Both retrieval legs report which branches their results came from, and that count
    is the project's central claim rather than a detail: it is how an audience sees the
    traversal leave the subtree the question named. One accessor so neither leg has to
    know that depth one is the client.
    """
    return ancestry[0].name if ancestry else None
