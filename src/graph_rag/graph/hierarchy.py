"""Writing the folder and document structure into Neo4j.

Every statement is a MERGE on the node's identifying path, so re-running ingestion over
the same directory updates nodes in place instead of duplicating them. That matters more
than it looks: ingestion is re-run constantly while iterating, and a duplicated folder
tree is not an error anyone sees — it is a demo where the same client appears twice.

The driver is injected; nothing here constructs one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from neo4j import Driver, ManagedTransaction

from graph_rag.clarification import Folder
from graph_rag.hierarchy import Hierarchy

_MERGE_FOLDERS = """
UNWIND $folders AS folder
MERGE (f:Folder {path: folder.path})
SET f.name = folder.name
"""

# Containment is written as its own pass so every folder exists before anything points
# at it, whatever order the rows arrive in.
_MERGE_FOLDER_CONTAINMENT = """
UNWIND $folders AS folder
WITH folder WHERE folder.parent_path IS NOT NULL
MATCH (parent:Folder {path: folder.parent_path})
MATCH (child:Folder {path: folder.path})
MERGE (parent)-[:CONTAINS]->(child)
"""

# The containing folder is matched *before* the document is merged: were it the other
# way round, a folder that did not exist would leave behind an orphan Document node and
# no error, which is exactly the kind of quietly misshapen graph this pipeline must not
# produce.
_MERGE_DOCUMENTS = """
UNWIND $documents AS document
MATCH (folder:Folder {path: document.folder_path})
MERGE (d:Document {path: document.path})
SET d.filename = document.filename, d.content_hash = document.content_hash
MERGE (folder)-[:CONTAINS]->(d)
"""


# Depth is derived from the containment chain rather than written from the walk,
# because the walk only knows depth *relative to the root it was pointed at*. Ingesting
# a subdirectory of an already-ingested tree — "just re-do this one client" is an
# obvious thing to do mid-demo — would otherwise stamp that branch with depths one or
# more levels shallower than its structural peers, silently and with the old containment
# edge still in place. Ticket 05 groups sibling folders by exactly this property, so a
# wrong depth is not cosmetic.
_NORMALISE_DEPTH = """
MATCH (root:Folder) WHERE NOT (:Folder)-[:CONTAINS]->(root)
MATCH path = (root)-[:CONTAINS*0..]->(f:Folder)
WITH f, length(path) AS depth
WHERE f.depth IS NULL OR f.depth <> depth
WITH f, depth, f.depth IS NOT NULL AS disagreed
SET f.depth = depth
RETURN sum(CASE WHEN disagreed THEN 1 ELSE 0 END) AS corrected
"""


@dataclass(frozen=True)
class HierarchyWriteReport:
    """What the write itself learned. The shape that was written is the caller's
    `Hierarchy`, and is deliberately not restated here."""

    nodes_created: int
    relationships_created: int
    #: Folders whose *already recorded* depth disagreed with their place in the
    #: containment chain — non-zero only when a previous ingest was rooted somewhere
    #: else. A folder first given its depth this run is not a correction.
    depths_corrected: int

    @property
    def unchanged(self) -> bool:
        """Nothing about the structure moved — the visible proof that ingestion is
        re-runnable.

        A corrected depth counts as a change, so the one line an operator reads to
        confirm re-runnability cannot claim nothing happened while the shape of the
        tree was being rewritten underneath it. A document whose *bytes* changed keeps
        its node and has its hash updated in place; that is content, not structure, and
        is deliberately invisible here.
        """
        return (
            self.nodes_created == 0
            and self.relationships_created == 0
            and self.depths_corrected == 0
        )


def write_hierarchy(
    driver: Driver, database: str, hierarchy: Hierarchy
) -> HierarchyWriteReport:
    """Write folders, their containment, and the documents at the leaves.

    All three passes share one transaction, so a failure cannot leave folders stranded
    without the containment edges that make them navigable.
    """
    parameters = {
        "folders": [asdict(folder) for folder in hierarchy.folders],
        "documents": [asdict(document) for document in hierarchy.documents],
    }

    def write(transaction: ManagedTransaction) -> HierarchyWriteReport:
        nodes = relationships = 0
        for cypher in (_MERGE_FOLDERS, _MERGE_FOLDER_CONTAINMENT, _MERGE_DOCUMENTS):
            counters = transaction.run(cypher, parameters).consume().counters
            nodes += counters.nodes_created
            relationships += counters.relationships_created
        # Last, so every edge this run created is already in place to be measured along.
        depths = transaction.run(_NORMALISE_DEPTH).single()["corrected"]
        return HierarchyWriteReport(
            nodes_created=nodes,
            relationships_created=relationships,
            depths_corrected=depths,
        )

    with driver.session(database=database) as session:
        return session.execute_write(write)


# Every folder with its parent, its children and its depth — the whole shape of the
# corpus in one round trip. Small enough to read once and hold: a corpus of a hundred
# thousand documents still has a folder tree an operator could print.
_READ_FOLDER_INDEX = """
MATCH (f:Folder) WHERE f.depth > 0
OPTIONAL MATCH (f)-[:CONTAINS]->(child:Folder)
RETURN f.name AS name, f.path AS path, f.depth AS depth,
       f.inferred_level_name AS level,
       [c IN collect(child.name) WHERE c IS NOT NULL] AS children
ORDER BY f.path
"""


def read_folder_index(driver: Driver, database: str) -> list[Folder]:
    """The corpus's folder tree, shaped for the clarifier.

    Ancestry is derived from the stored path rather than walked with a second query:
    `CONTAINS` already made the path authoritative when the hierarchy was written, and
    a variable-length match per folder would cost one traversal each to rediscover
    what the string already says.
    """
    records, _, _ = driver.execute_query(_READ_FOLDER_INDEX, database_=database)
    by_path = {record["path"]: record for record in records}
    folders = []
    for record in records:
        parts = record["path"].split("/")
        ancestry = []
        for index in range(len(parts)):
            candidate = "/".join(parts[: index + 1])
            found = by_path.get(candidate)
            if found is not None:
                ancestry.append(found["name"])
        folders.append(
            Folder(
                name=record["name"],
                path=record["path"],
                depth=record["depth"],
                ancestry=tuple(ancestry),
                children=tuple(sorted(record["children"])),
                level=record["level"],
            )
        )
    return folders
