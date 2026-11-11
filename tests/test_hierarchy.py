"""The adaptivity claim lives in this function: a corpus nested deeper than
`Client / Year / Project` must work without configuration. Its failure mode is silent —
a tree misshapen at depth four raises nothing, it just quietly answers from the wrong
branch during a live demo. These tests assert the structure the graph will be built
from, for trees of arbitrary shape and depth."""

from pathlib import Path

import pytest

from graph_rag.hierarchy import DocumentFile, build_hierarchy, walk_documents

ROOT = Path("/corpus")


def _file(relative: str, content_hash: str = "hash") -> DocumentFile:
    return DocumentFile(path=ROOT / relative, content_hash=content_hash)


def _folders_by_path(hierarchy) -> dict[str, object]:
    return {folder.path: folder for folder in hierarchy.folders}


def test_a_tree_deeper_than_the_demo_shape_produces_a_folder_per_level():
    hierarchy = build_hierarchy(
        ROOT, [_file("Kunde/2024/Projekt/Phase 2/Anhang/bericht.pdf")]
    )

    folders = _folders_by_path(hierarchy)
    assert [(f.name, f.depth) for f in sorted(hierarchy.folders, key=lambda f: f.depth)] == [
        ("corpus", 0),
        ("Kunde", 1),
        ("2024", 2),
        ("Projekt", 3),
        ("Phase 2", 4),
        ("Anhang", 5),
    ]
    assert folders["/corpus/Kunde/2024/Projekt/Phase 2/Anhang"].parent_path == (
        "/corpus/Kunde/2024/Projekt/Phase 2"
    )
    assert folders["/corpus"].parent_path is None


def test_siblings_share_one_parent_folder_node():
    hierarchy = build_hierarchy(
        ROOT,
        [
            _file("Kunde A/2024/Projekt Atlas/bericht.pdf"),
            _file("Kunde A/2024/Projekt Atlas/status.pdf"),
            _file("Kunde A/2024/Projekt Taktzeit/bericht.pdf"),
            _file("Kunde B/2024/Projekt Atlas/bericht.pdf"),
        ],
    )

    paths = [folder.path for folder in hierarchy.folders]
    assert len(paths) == len(set(paths))
    assert [f.path for f in hierarchy.folders if f.parent_path is None] == ["/corpus"]
    assert all(
        folder.parent_path in set(paths)
        for folder in hierarchy.folders
        if folder.parent_path is not None
    )
    assert sorted(paths) == [
        "/corpus",
        "/corpus/Kunde A",
        "/corpus/Kunde A/2024",
        "/corpus/Kunde A/2024/Projekt Atlas",
        "/corpus/Kunde A/2024/Projekt Taktzeit",
        "/corpus/Kunde B",
        "/corpus/Kunde B/2024",
        "/corpus/Kunde B/2024/Projekt Atlas",
    ]


def test_documents_hang_off_the_folder_that_contains_them():
    hierarchy = build_hierarchy(
        ROOT,
        [
            _file("Kunde/2024/Projekt/bericht.pdf", content_hash="abc"),
            _file("uebersicht.pdf", content_hash="def"),
        ],
    )

    documents = {document.filename: document for document in hierarchy.documents}
    assert documents["bericht.pdf"].folder_path == "/corpus/Kunde/2024/Projekt"
    assert documents["bericht.pdf"].path == "/corpus/Kunde/2024/Projekt/bericht.pdf"
    assert documents["bericht.pdf"].content_hash == "abc"
    assert documents["uebersicht.pdf"].folder_path == "/corpus"


def test_documents_at_different_depths_coexist_in_one_tree():
    """Nothing forces a corpus to put every document at the same level."""
    hierarchy = build_hierarchy(
        ROOT,
        [
            _file("Kunde/uebersicht.pdf"),
            _file("Kunde/2024/Projekt/Phase 2/bericht.pdf"),
        ],
    )

    depths = {folder.path: folder.depth for folder in hierarchy.folders}
    assert depths["/corpus/Kunde"] == 1
    assert depths["/corpus/Kunde/2024/Projekt/Phase 2"] == 4
    folder_paths = {document.folder_path for document in hierarchy.documents}
    assert folder_paths == {"/corpus/Kunde", "/corpus/Kunde/2024/Projekt/Phase 2"}


def test_the_walk_finds_pdfs_at_any_depth_and_ignores_everything_else(tmp_path):
    """One test guards the walk because its mistakes are silent too: a hash that does
    not track content makes ticket 09's cache serve a stale extraction, and swept-up
    junk files turn into document nodes nobody put in the corpus."""
    (tmp_path / "Kunde/2024/Projekt/Phase 2").mkdir(parents=True)
    (tmp_path / ".git/objects").mkdir(parents=True)
    (tmp_path / "Kunde/2024/Projekt/Phase 2/bericht.pdf").write_bytes(b"%PDF-1.7 eins")
    (tmp_path / "Kunde/2024/Projekt/status.PDF").write_bytes(b"%PDF-1.7 zwei")
    (tmp_path / "Kunde/notizen.txt").write_text("kein PDF")
    (tmp_path / ".git/objects/bericht.pdf").write_bytes(b"%PDF-1.7 drei")

    found = walk_documents(tmp_path)

    hashes = {file.path.name: file.content_hash for file in found}
    assert sorted(hashes) == ["bericht.pdf", "status.PDF"]
    assert len(set(hashes.values())) == 2
    assert all(file.path.is_absolute() for file in found)

    (tmp_path / "Kunde/2024/Projekt/status.PDF").write_bytes(b"%PDF-1.7 geaendert")
    changed = {file.path.name: file.content_hash for file in walk_documents(tmp_path)}
    assert changed["status.PDF"] != hashes["status.PDF"]
    assert changed["bericht.pdf"] == hashes["bericht.pdf"]


def test_a_root_that_does_not_contain_the_files_is_rejected():
    """`walk_documents` resolves its root; a caller pairing it with an unresolved one
    would otherwise get a bare path error from a function documented as pure."""
    with pytest.raises(ValueError, match="walked root"):
        build_hierarchy(Path("corpus"), [_file("Kunde/bericht.pdf")])
