"""The corpus is the experimental apparatus. Each failure mode below is silent: the
generator still produces a tree of plausible-looking PDFs, and only a live demo reveals
that the thing being demonstrated is no longer there."""

from pypdf import PdfReader

from graph_rag.corpus.pdf import render
from graph_rag.corpus.plan import (
    BRIDGE_CONSULTANT,
    BRIDGE_PROJECT,
    SECOND_BRIDGE_CONSULTANT,
    SHARED_PROJECT,
    build_plan,
    client_by_name,
)
from graph_rag.corpus.text import (
    BRIDGE_ORIGIN_CLIENT,
    BRIDGE_REUSE_CLIENT,
    SECOND_BRIDGE_ARTEFACT,
    SECOND_BRIDGE_ORIGIN_CLIENT,
    SECOND_BRIDGE_REUSE_CLIENT,
    project_report,
    status_report,
)

SEED = 20241101


def _report_for(client: str, project_name: str):
    project = next(
        p for p in build_plan(SEED) if p.client == client and p.name == project_name
    )
    return project_report(project, client_by_name(client), SEED)


def _all_text(document) -> str:
    """Every string the PDF renders — the metadata block and table cells are real,
    extractable text too, so a client name leaking into one of them would defeat the
    bridge just as thoroughly as one in a paragraph."""
    parts = [document.title, document.subtitle]
    parts += [value for pair in document.metadata for value in pair]
    for section in document.sections:
        parts.append(section.heading)
        parts.extend(section.paragraphs)
        for row in section.table or ():
            parts.extend(row)
    return " ".join(parts)


def test_generated_pdfs_have_an_extractable_text_layer(tmp_path):
    """OCR is out of scope, so a rasterising generator would break every downstream
    ticket without raising anything."""
    document = _report_for(BRIDGE_ORIGIN_CLIENT, SHARED_PROJECT)
    path = render(document, tmp_path / "bericht.pdf")

    extracted = "".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

    assert "Atlas-Referenzarchitektur" in extracted
    assert "Geschäftsjahr" in extracted  # umlauts survive the text layer
    assert len(extracted) > 1000


def test_the_same_seed_produces_identical_bytes(tmp_path):
    document = _report_for(BRIDGE_ORIGIN_CLIENT, SHARED_PROJECT)
    first = render(document, tmp_path / "a.pdf").read_bytes()
    second = render(document, tmp_path / "b.pdf").read_bytes()
    assert first == second


def test_a_different_seed_produces_a_different_corpus():
    assert build_plan(SEED) != build_plan(7)


def test_one_project_name_is_shared_by_two_different_clients():
    clients = {p.client for p in build_plan(SEED) if p.name == SHARED_PROJECT}
    assert clients == {BRIDGE_ORIGIN_CLIENT, BRIDGE_REUSE_CLIENT}


def test_the_bridge_is_a_shared_entity_and_never_a_shared_client_name():
    """If either document named the other client, a single chunk would contain the
    answer, flat retrieval would find it, and the demo would prove nothing."""
    origin = _report_for(BRIDGE_ORIGIN_CLIENT, SHARED_PROJECT)
    reuse = _report_for(BRIDGE_REUSE_CLIENT, SHARED_PROJECT)

    assert BRIDGE_REUSE_CLIENT not in _all_text(origin)
    assert BRIDGE_ORIGIN_CLIENT not in _all_text(reuse)
    assert "Brandt" not in _all_text(origin)
    assert "Nordlicht" not in _all_text(reuse)

    # ...and the entity that does join them is present on both sides.
    assert "Atlas-Referenzarchitektur" in _all_text(origin)
    assert "Atlas-Referenzarchitektur" in _all_text(reuse)
    assert origin.metadata[-1][1].startswith("Dr. Katharina Vogt")
    assert reuse.metadata[-1][1].startswith("Dr. Katharina Vogt")


def test_status_reports_are_near_identical_across_clients():
    """Dense retrieval must genuinely confuse these, not merely be able to in theory."""
    plan = build_plan(SEED)
    origin = next(
        p for p in plan if p.client == BRIDGE_ORIGIN_CLIENT and p.name == SHARED_PROJECT
    )
    reuse = next(
        p for p in plan if p.client == BRIDGE_REUSE_CLIENT and p.name == SHARED_PROJECT
    )
    a = status_report(origin, client_by_name(origin.client), 2, SEED)
    b = status_report(reuse, client_by_name(reuse.client), 2, SEED)

    assert [s.heading for s in a.sections] == [s.heading for s in b.sections]
    assert a.sections[0].paragraphs[0] == b.sections[0].paragraphs[0]


def test_a_bridge_consultant_appears_only_on_the_projects_she_bridges():
    """She is the join between two clients. Staffing her onto unrelated projects would
    make her a hub touching every client, so a traversal through her would reach
    everywhere and connect nothing in particular."""
    appearances = [
        p for p in build_plan(SEED) if any(x.name == BRIDGE_CONSULTANT for x in p.people)
    ]

    assert {p.name for p in appearances} == {SHARED_PROJECT}
    assert {p.client for p in appearances} == {BRIDGE_ORIGIN_CLIENT, BRIDGE_REUSE_CLIENT}


def test_the_second_bridge_carries_its_artefact_across_two_clients():
    """The weaker bridge shows the first is not a fluke, and nothing else guards it:
    its reports would quietly fall back to generic methodology prose while
    KOLLISIONEN.md kept promising the artefact."""
    origin = _report_for(SECOND_BRIDGE_ORIGIN_CLIENT, BRIDGE_PROJECT)
    reuse = _report_for(SECOND_BRIDGE_REUSE_CLIENT, BRIDGE_PROJECT)

    for document in (origin, reuse):
        assert SECOND_BRIDGE_ARTEFACT in _all_text(document)
        assert SECOND_BRIDGE_CONSULTANT in _all_text(document)

    assert SECOND_BRIDGE_REUSE_CLIENT not in _all_text(origin)
    assert SECOND_BRIDGE_ORIGIN_CLIENT not in _all_text(reuse)
