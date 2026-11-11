"""Reading a document's folder ancestry out of the graph, once.

Two reads need it — the summary a document is written from, and every chunk a
retrieval leg returns — and both need it the same way, down to which folders count and
what each one is paired with. Written twice it would be a Folder property added in one
place and missing in the other: an answer that can say *Kunde Nordlicht, Jahr 2023* on
one path and only a path string on the other, with nothing reporting the difference.
"""

from __future__ import annotations

from graph_rag.ancestry import Ancestor


def ancestry_projection(carry: str) -> str:
    """Cypher collecting `d`'s ancestry into a row named `ancestry`.

    `carry` is the comma-separated variables the caller needs to keep alive across the
    aggregation, which always includes `d`.

    Ordered by the folder's *recorded* depth, never by the length of the match: depth
    is normalised from the containment chain on every ingest and is the property the
    level names are stamped against, so ordering by anything else could pair a folder
    name with another level's label. Depth 0 is skipped — it is the corpus root, which
    is the archive itself rather than a level of it.
    """
    return f"""
OPTIONAL MATCH (f:Folder)-[:CONTAINS*]->(d)
WITH {carry}, f ORDER BY f.depth
WITH {carry}, [folder IN collect(f) WHERE folder.depth > 0 |
     {{name: folder.name, level: folder.inferred_level_name}}] AS ancestry
"""


def to_ancestors(rows) -> list[Ancestor]:
    """The `ancestry` column of such a row, as the shared dataclass."""
    return [Ancestor(name=row["name"], level=row["level"]) for row in rows]
