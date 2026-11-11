"""The override is the demo's escape hatch: it exists so a bad inference does not
derail a live demo. Both of these units fail silently — an off-by-one in the
depth mapping labels every level with its neighbour's name, and a malformed model reply
would otherwise store a confidence the operator reads as trustworthy. Neither raises."""

import pytest

from graph_rag.levels import (
    LevelSample,
    apply_overrides,
    parse_inference,
    parse_level_names,
)


def _samples(*depths: int) -> list[LevelSample]:
    return [
        LevelSample(depth=depth, folder_count=4, distinct_count=4, names=["a", "b"])
        for depth in depths
    ]


def test_the_first_override_names_the_first_level_below_the_root():
    """The walked root is depth 0 and is not a level, so `Kunde,Jahr,Projekt` means
    depth 1, 2, 3 — not 0, 1, 2."""
    overridden, remaining = apply_overrides(["Kunde", "Jahr", "Projekt"], _samples(1, 2, 3))

    assert [(level.depth, level.name) for level in overridden] == [
        (1, "Kunde"),
        (2, "Jahr"),
        (3, "Projekt"),
    ]
    assert all(level.source == "override" for level in overridden)
    assert remaining == []


def test_depths_beyond_the_supplied_list_fall_back_to_inference():
    overridden, remaining = apply_overrides(["Kunde", "Jahr"], _samples(1, 2, 3, 4))

    assert [level.depth for level in overridden] == [1, 2]
    assert [sample.depth for sample in remaining] == [3, 4]


def test_no_overrides_leaves_every_level_to_inference():
    overridden, remaining = apply_overrides([], _samples(1, 2, 3))

    assert overridden == []
    assert [sample.depth for sample in remaining] == [1, 2, 3]


def test_more_overrides_than_levels_ignores_the_surplus():
    """An operator who pastes a four-level list at a three-level corpus should still
    get an ingest, not a crash and not a phantom level."""
    overridden, remaining = apply_overrides(["Kunde", "Jahr", "Projekt", "Phase"], _samples(1, 2))

    assert [level.depth for level in overridden] == [1, 2]
    assert remaining == []


def test_a_well_formed_reply_yields_its_name_and_confidence():
    assert parse_inference('{"name": "Jahr", "confidence": 0.92}') == ("Jahr", 0.92)


def test_a_confidence_is_coerced_into_a_number_in_range():
    """A model that answers 7, or answers "0.8" as a string, would otherwise render as
    a 700%-confident level on stage or crash the formatter mid-ingest."""
    assert parse_inference('{"name": "Jahr", "confidence": 7}')[1] == 1.0
    assert parse_inference('{"name": "Jahr", "confidence": -2}')[1] == 0.0
    assert parse_inference('{"name": "Jahr", "confidence": "0.8"}') == ("Jahr", 0.8)


def test_an_unusable_reply_reports_no_confidence_rather_than_guessing():
    """Whatever comes back, the operator must see a level they can distrust — never a
    crash partway through an ingest, and never a confident-looking default."""
    for reply in ('{"confidence": 0.9}', "not json at all", "{}", '{"name": ""}'):
        name, confidence = parse_inference(reply)
        assert name == "unbekannt"
        assert confidence == 0.0


def test_an_empty_level_name_is_rejected_rather_than_shifting_the_rest():
    """`--levels "Kunde,,Projekt"` must not quietly label the years "Projekt" — a
    silent mislabel is exactly what the override exists to rule out."""
    with pytest.raises(ValueError, match="empty level name"):
        parse_level_names("Kunde,,Projekt")
    with pytest.raises(ValueError, match="empty level name"):
        parse_level_names("Kunde,Jahr,")
    assert parse_level_names("Kunde, Jahr ,Projekt") == ["Kunde", "Jahr", "Projekt"]
    assert parse_level_names(None) == []
