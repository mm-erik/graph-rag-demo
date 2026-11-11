"""The machine half of the corpus is the second experimental apparatus, and its failure
modes are just as silent as the client half's: the generator still writes a tidy tree of
plausible data sheets while the collision that made the tree worth building has quietly
dissolved into two documents that any flat index can tell apart."""

from pypdf import PdfReader

from graph_rag.corpus.machine_plan import (
    BRIDGE_ASSEMBLY,
    BRIDGE_ENGINEER,
    BRIDGE_ORIGIN,
    BRIDGE_REUSE,
    MACHINE_TYPES,
    SECOND_BRIDGE_ASSEMBLY,
    SECOND_BRIDGE_ENGINEER,
    SECOND_BRIDGE_ORIGIN,
    SECOND_BRIDGE_REUSE,
    SHARED_MODEL_CODE,
    SHARED_OPTION_CODE,
    SHARED_TYPE_CODE,
    build_machine_plan,
    machine_by,
    manufacturer_by_name,
    option_by_code,
    variant_spec,
)
from graph_rag.corpus.machine_text import (
    data_sheet,
    maintenance_plan,
    part_number,
    variant_overview,
)
from graph_rag.corpus.pdf import render

SEED = 20241101


def _documents_for(anchor: tuple[str, str, str]):
    machine = machine_by(*anchor, SEED)
    manufacturer = manufacturer_by_name(machine.manufacturer)
    return (
        data_sheet(machine, manufacturer, SEED),
        variant_overview(machine, manufacturer, SEED),
        maintenance_plan(machine, manufacturer, SEED),
    )


def _all_text(document) -> str:
    """Every string the PDF renders. Table cells and the metadata block are extractable
    text too, so a manufacturer name leaking into one of them would defeat a bridge as
    thoroughly as one in a paragraph."""
    parts = [document.title, document.subtitle]
    parts += [value for pair in document.metadata for value in pair]
    for section in document.sections:
        parts.append(section.heading)
        parts.extend(section.paragraphs)
        for row in section.table or ():
            parts.extend(row)
    return " ".join(parts)


def _text_of(anchor: tuple[str, str, str]) -> str:
    return " ".join(_all_text(document) for document in _documents_for(anchor))


def test_a_data_sheet_has_an_extractable_text_layer(tmp_path):
    """OCR is out of scope: a rasterising generator would break ingestion silently."""
    sheet, _, _ = _documents_for(BRIDGE_ORIGIN)
    path = render(sheet, tmp_path / "datenblatt.pdf")

    extracted = "".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)

    assert "XTC27-B2BC-FW/OW" in extracted  # the order key survives the text layer
    assert "Ölauffangwanne" in extracted  # umlauts do too
    assert "1625 kg" in extracted  # and so does the one number that answers the question
    assert len(extracted) > 1000


def test_the_same_seed_produces_identical_bytes(tmp_path):
    sheet, _, _ = _documents_for(BRIDGE_ORIGIN)
    first = render(sheet, tmp_path / "a.pdf").read_bytes()
    second = render(sheet, tmp_path / "b.pdf").read_bytes()
    assert first == second


def test_every_model_folder_holds_three_differently_named_documents():
    """Three documents share one folder, so a collision in the generated filenames would
    silently drop two thirds of the machine corpus on disk."""
    for machine in build_machine_plan(SEED):
        manufacturer = manufacturer_by_name(machine.manufacturer)
        names = {
            document.filename
            for document in (
                data_sheet(machine, manufacturer, SEED),
                variant_overview(machine, manufacturer, SEED),
                maintenance_plan(machine, manufacturer, SEED),
            )
        }
        assert len(names) == 3


def test_one_model_code_is_shared_by_two_unrelated_machine_types():
    """`B2BC` alone identifies nothing. A lexical hit on it is worthless and an answer
    taken from the wrong branch is wrong by a checkable amount."""
    owners = {
        (machine.manufacturer, machine.type_code)
        for machine in build_machine_plan(SEED)
        if machine.model.code == SHARED_MODEL_CODE
    }
    assert owners == {
        ("Hagedorn Verfahrenstechnik GmbH", "XTC27"),
        ("Pleickner Werke KG", "ZM3K"),
    }

    mixer = machine_by("Hagedorn Verfahrenstechnik GmbH", "XTC27", SHARED_MODEL_CODE, SEED)
    cell = machine_by("Pleickner Werke KG", "ZM3K", SHARED_MODEL_CODE, SEED)
    assert mixer.model.base_weight != cell.model.base_weight
    assert mixer.type_name != cell.type_name


def test_one_machine_type_code_is_shared_by_two_manufacturers():
    owners = {
        machine_type.manufacturer
        for machine_type in MACHINE_TYPES
        if machine_type.code == SHARED_TYPE_CODE
    }
    assert owners == {"Hagedorn Verfahrenstechnik GmbH", "Ostmark Fördertechnik AG"}

    names = {
        machine_type.name
        for machine_type in MACHINE_TYPES
        if machine_type.code == SHARED_TYPE_CODE
    }
    assert len(names) == 2  # same code, nothing in common


def test_one_option_code_means_different_things_per_manufacturer():
    """The code is resolved only in the variant overview of the branch it belongs to.
    If both manufacturers gave `HK` the same name, the code would become answerable
    without the branch and the collision would be decoration."""
    heizkreis = option_by_code("Hagedorn Verfahrenstechnik GmbH", SHARED_OPTION_CODE)
    hubkonsole = option_by_code("Pleickner Werke KG", SHARED_OPTION_CODE)

    assert heizkreis.name != hubkonsole.name
    assert heizkreis.weight_delta != hubkonsole.weight_delta


def test_variant_weights_are_additive_and_unique_to_their_row():
    """The demo question — weight with running gear *and* oil tray — must be unanswerable
    from the base row, from a sibling variant, or from the colliding model."""
    machine = machine_by(*BRIDGE_ORIGIN, SEED)
    base = variant_spec(machine, ())
    both = variant_spec(machine, ("FW", "OW"))
    running_gear = option_by_code(machine.manufacturer, "FW")
    oil_tray = option_by_code(machine.manufacturer, "OW")

    assert both.weight == (
        base.weight + running_gear.weight_delta + oil_tray.weight_delta
    )
    assert both.height == (
        base.height + running_gear.height_delta + oil_tray.height_delta
    )
    assert both.oil_volume == oil_tray.oil_delta

    weights = {
        variant_spec(machine, variant).weight for variant in machine.model.variants
    }
    assert len(weights) == len(machine.model.variants)


def test_an_option_that_overrides_the_protection_rating_wins_over_the_model():
    machine = machine_by(*BRIDGE_ORIGIN, SEED)
    assert variant_spec(machine, ()).protection == "IP55"
    assert variant_spec(machine, ("EX", "OW")).protection == "IP66"


def test_the_bridge_is_a_shared_assembly_and_never_a_shared_manufacturer_name():
    """If either side named the other manufacturer, one chunk would hold the whole
    answer, flat retrieval would find it, and the bridge would prove nothing."""
    origin = _text_of(BRIDGE_ORIGIN)
    reuse = _text_of(BRIDGE_REUSE)

    assert "Ostmark" not in origin
    assert "Hagedorn" not in reuse
    assert BRIDGE_REUSE[0] not in origin
    assert BRIDGE_ORIGIN[0] not in reuse

    # ...and the entities that do join them are present on both sides.
    for text in (origin, reuse):
        assert BRIDGE_ASSEMBLY in text
        assert BRIDGE_ENGINEER in text


def test_the_second_bridge_carries_its_assembly_across_two_manufacturers():
    """The weaker bridge shows the first is not a fluke, and nothing else guards it: the
    maintenance plans would quietly fall back to generic prose while KOLLISIONEN.md kept
    promising the assembly."""
    origin = _text_of(SECOND_BRIDGE_ORIGIN)
    reuse = _text_of(SECOND_BRIDGE_REUSE)

    for text in (origin, reuse):
        assert SECOND_BRIDGE_ASSEMBLY in text
        assert SECOND_BRIDGE_ENGINEER in text

    assert "Hagedorn" not in origin
    assert "Pleickner" not in reuse


def test_a_bridge_engineer_appears_only_on_the_two_models_he_bridges():
    """Staffing him onto unrelated models would make him a hub touching every
    manufacturer, so a traversal through him would reach everywhere and connect nothing
    in particular."""
    for engineer, anchors in (
        (BRIDGE_ENGINEER, {BRIDGE_ORIGIN, BRIDGE_REUSE}),
        (SECOND_BRIDGE_ENGINEER, {SECOND_BRIDGE_ORIGIN, SECOND_BRIDGE_REUSE}),
    ):
        appearances = {
            (machine.manufacturer, machine.type_code, machine.model.code)
            for machine in build_machine_plan(SEED)
            if engineer in (machine.designer.name, machine.inspector.name)
        }
        assert appearances == anchors


def test_a_shared_assembly_keeps_one_spare_part_number_across_the_bridge():
    """The number is derived, so a change to the derivation would give the same assembly
    two identities and quietly cut the bridge the graph is meant to traverse."""
    origin = machine_by(*BRIDGE_ORIGIN, SEED)
    reuse = machine_by(*BRIDGE_REUSE, SEED)

    assert BRIDGE_ASSEMBLY in origin.model.assemblies
    assert BRIDGE_ASSEMBLY in reuse.model.assemblies
    assert part_number(BRIDGE_ASSEMBLY) == "PLA-0114"


def test_documents_of_different_models_share_their_sentence_frames():
    """Dense retrieval must genuinely confuse these, not merely be able to in theory:
    what separates two data sheets is a handful of opaque codes and a column of numbers."""
    a, _, _ = _documents_for(BRIDGE_ORIGIN)
    b, _, _ = _documents_for(("Hagedorn Verfahrenstechnik GmbH", "XTC27", "B2BD"))

    assert [section.heading for section in a.sections] == [
        section.heading for section in b.sections
    ]
    assert a.sections[1].paragraphs[0] != b.sections[1].paragraphs[0]  # only codes differ
    assert a.sections[1].paragraphs[0].replace("B2BC", "") == b.sections[1].paragraphs[
        0
    ].replace("B2BD", "")


def test_the_generator_writes_the_machine_tree_three_levels_deep(tmp_path):
    """Same depth as the client half, different meaning per level: depth 2 is a year on
    one branch and a machine type on the other. The hierarchy is recovered from the walk
    and carries no per-branch meaning, so a generator that flattened the machine tree
    would quietly remove the only case where two branches disagree about their levels."""
    from graph_rag.corpus.generate import generate_corpus

    report = generate_corpus(tmp_path, seed=SEED)

    assert report.model_count == len(build_machine_plan(SEED))
    assert len(report.machine_documents) == 3 * report.model_count

    sheet = (
        tmp_path
        / "Hagedorn Verfahrenstechnik"
        / "XTC27"
        / "B2BC"
        / "datenblatt-XTC27-B2BC.pdf"
    )
    assert sheet.is_file()
    # Three folders then the file — the same shape as `Kunde / Jahr / Projekt`.
    assert len(sheet.relative_to(tmp_path).parts) == 4

    note = (tmp_path / "KOLLISIONEN.md").read_text(encoding="utf-8")
    assert BRIDGE_ASSEMBLY in note
    assert "XTC27-B2BC-FW/OW" in note
    assert "1625 kg" in note  # the note quotes the number the documents actually carry


def test_a_second_run_refuses_to_write_over_the_machine_tree(tmp_path):
    """`--force` clears only what this generator owns. A manufacturer folder missing from
    that list would be left behind on a reseed, and ingestion walks the directory rather
    than the plan — it would happily ingest documents belonging to no corpus."""
    from graph_rag.corpus.generate import CorpusExists, generate_corpus

    generate_corpus(tmp_path, seed=SEED)
    machine_folder = tmp_path / "Pleickner Werke"
    assert machine_folder.is_dir()

    try:
        generate_corpus(tmp_path, seed=SEED)
    except CorpusExists:
        pass
    else:  # pragma: no cover - the guard is the point of the test
        raise AssertionError("a second run must refuse rather than overwrite")

    generate_corpus(tmp_path, seed=7, force=True)
    assert machine_folder.is_dir()
