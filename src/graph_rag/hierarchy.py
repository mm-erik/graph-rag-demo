"""The corpus's folder structure, recovered from a directory walk.

Folders are modelled *generically*: one node type at any depth, linked by containment,
rather than typed per-level entities. That is what makes a corpus nested deeper than
`Client / Year / Project` work without configuration or code change. What each level
*means* is a separate concern (ticket 05) and deliberately absent here.

`build_hierarchy` is pure — it is given already-resolved paths and never touches the
filesystem — because its failure is silent: a tree misshapen at depth four raises
nothing, it simply answers from the wrong branch during the demo.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

DOCUMENT_SUFFIX = ".pdf"


@dataclass(frozen=True)
class DocumentFile:
    """A document found on disk. `path` must be absolute and under the walked root."""

    path: Path
    content_hash: str


@dataclass(frozen=True)
class FolderNode:
    name: str
    path: str
    depth: int
    #: None for the walked root — the only folder nothing contains.
    parent_path: str | None


@dataclass(frozen=True)
class DocumentNode:
    filename: str
    path: str
    #: What the ingestion cache keys on: a document whose hash changed is reparsed,
    #: re-summarised and re-extracted, and one whose hash did not is never touched.
    content_hash: str
    folder_path: str


@dataclass(frozen=True)
class Hierarchy:
    folders: list[FolderNode]
    documents: list[DocumentNode]

    @property
    def max_depth(self) -> int:
        return max((folder.depth for folder in self.folders), default=0)

    def folders_at(self, depth: int) -> list[FolderNode]:
        return [folder for folder in self.folders if folder.depth == depth]


def _ancestors(root: Path, file_path: Path) -> Iterator[Path]:
    """The root and every folder between it and `file_path`, outermost first."""
    relative = file_path.relative_to(root)
    current = root
    yield current
    for part in relative.parts[:-1]:
        current = current / part
        yield current


def build_hierarchy(root: Path, files: Sequence[DocumentFile]) -> Hierarchy:
    """The folder and document nodes a directory of documents becomes.

    `root` must be absolute and resolved, matching the paths in `files` — pair it with
    `walk_documents`, which resolves the same root.

    Folders are derived *from* the documents rather than from the walk, so a branch
    holding no documents never reaches the graph — the structure stays the structure of
    the corpus, not of whatever else happens to sit on disk beside it.
    """
    folders: dict[str, FolderNode] = {}
    documents: list[DocumentNode] = []
    for file in files:
        if not file.path.is_relative_to(root):
            # `relative_to` would raise here anyway, but with a message that reads like
            # a path bug rather than what it is: a root that was never resolved, while
            # `walk_documents` resolves its own. The two must agree.
            raise ValueError(
                f"{file.path} is not under the walked root {root}. Both must be "
                "absolute and resolved; see walk_documents."
            )
        parent = file.path.parent
        for depth, ancestor in enumerate(_ancestors(root, file.path)):
            path = str(ancestor)
            if path not in folders:
                folders[path] = FolderNode(
                    name=ancestor.name or path,
                    path=path,
                    depth=depth,
                    parent_path=str(ancestor.parent) if depth else None,
                )
        documents.append(
            DocumentNode(
                filename=file.path.name,
                path=str(file.path),
                content_hash=file.content_hash,
                folder_path=str(parent),
            )
        )
    return Hierarchy(folders=list(folders.values()), documents=documents)


def hash_file(path: Path) -> str:
    """SHA-256 of the file's bytes, read in blocks so a large PDF is not held in memory."""
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def walk_documents(root: Path) -> list[DocumentFile]:
    """Every PDF under `root`, to any depth, with its content hash.

    Hidden directories are pruned: a corpus checked out of version control carries a
    `.git` full of files that are not documents. Results are sorted so two runs over an
    unchanged tree produce the same order, and so a `--limit`-style truncation during a
    demo is at least predictable.
    """
    root = root.resolve()
    files = [
        path
        for path in root.rglob("*")
        if path.suffix.lower() == DOCUMENT_SUFFIX
        and path.is_file()
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    return [
        DocumentFile(path=path, content_hash=hash_file(path)) for path in sorted(files)
    ]
