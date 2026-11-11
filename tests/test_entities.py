"""Entity identity: what collapses into one node and what must not.

Both failure modes here are silent and both destroy the demo. Too loose, and two
clients' similarly-named projects become one node asserting a connection that does not
exist. Too strict, and the graph fragments into spellings of the same organisation —
which demonstrates the *opposite* of this project's thesis and looks worse than flat
RAG. Nothing raises in either case; the graph is simply wrong on screen.
"""

import json
from math import cos, sin

from graph_rag.entities import (
    ChunkExtraction,
    ambiguous_names,
    identify,
    joins,
    normalise_name,
    parse_extraction,
)
from graph_rag.resolution import Candidate, Thresholds, resolve_entities


def test_spelling_variants_of_one_organisation_normalise_together():
    """The corpus writes the legal name in the documents and an ASCII-folded one on
    disk, so the same client arrives spelled three ways in a single ingest."""
    variants = [
        "Brandt & Söhne Maschinenbau GmbH",
        "Brandt und Soehne Maschinenbau",
        "brandt & soehne  maschinenbau gmbh",
    ]

    assert len({normalise_name(name, "organisation") for name in variants}) == 1


def test_a_titled_person_and_the_same_person_untitled_normalise_together():
    """The reports introduce the project lead with her title and then drop it."""
    variants = ["Dr. Katharina Vogt", "Katharina Vogt", "Frau Dr. Katharina Vogt"]

    assert len({normalise_name(name, "person") for name in variants}) == 1


def test_a_legal_form_is_not_stripped_from_something_that_is_not_an_organisation():
    """`AG` is a legal form in a client's name and part of the name in a product.
    Stripping it everywhere would merge two systems that share a first word."""
    assert normalise_name("AG Grid", "system") != normalise_name("Grid", "system")


def test_two_systems_sharing_a_vendor_prefix_stay_distinct():
    """The clearest must-not-merge in the corpus: both are SAP, neither is the other."""
    assert normalise_name("SAP IS-U", "system") != normalise_name(
        "SAP S/4HANA", "system"
    )


def test_a_name_that_is_nothing_but_a_legal_form_keeps_an_identity():
    """Everything strippable means the mention *was* the legal form. An empty identity
    would collapse every such mention into a single node."""
    assert normalise_name("GmbH", "organisation") == "gmbh"


def test_a_project_name_reused_by_two_clients_resolves_to_two_identities():
    """`Projekt Atlas` is run by two unrelated clients on entirely different work. One
    node for both would assert a connection that does not exist — and the folder tree
    is what knows the difference, because the text of neither report mentions the other
    client."""
    ambiguous = ambiguous_names([("Nordlicht", "Projekt Atlas"), ("Brandt", "Projekt Atlas")])

    nordlicht = identify("Projekt Atlas", "konzept", ["Nordlicht", "2022", "Projekt Atlas"], ambiguous)
    brandt = identify("Projekt Atlas", "konzept", ["Brandt", "2022", "Projekt Atlas"], ambiguous)

    assert nordlicht != brandt
    assert (nordlicht.scope, brandt.scope) == ("Nordlicht", "Brandt")


def test_an_artefact_carried_between_those_same_two_clients_stays_one_identity():
    """The counterweight, and the harder direction: the reference architecture is named
    identically under both clients *and is the same document*. Scoping it by client
    would cut the bridge the whole demo exists to show."""
    ambiguous = ambiguous_names([("Nordlicht", "Projekt Atlas"), ("Brandt", "Projekt Atlas")])

    nordlicht = identify(
        "Atlas-Referenzarchitektur", "ergebnis", ["Nordlicht", "2022", "Projekt Atlas"], ambiguous
    )
    brandt = identify(
        "Atlas-Referenzarchitektur", "ergebnis", ["Brandt", "2023", "Projekt Atlas"], ambiguous
    )

    assert nordlicht == brandt
    assert nordlicht.scope == ""


def test_a_folder_name_unique_to_one_branch_is_not_scoped():
    """Scoping is for names the hierarchy shows to be ambiguous. Applying it to every
    folder name would split consultants, systems and clients by branch as well."""
    ambiguous = ambiguous_names([("Nordlicht", "Projekt Windfeld"), ("Brandt", "Projekt Taktzeit")])

    identity = identify(
        "Projekt Windfeld", "konzept", ["Nordlicht", "2022", "Projekt Windfeld"], ambiguous
    )

    assert identity.scope == ""


def test_an_ambiguous_name_mentioned_outside_its_own_branch_is_left_unscoped():
    """Nothing on this document's path says *which* Atlas is meant, so inventing a
    scope would be a guess. One honest shared node beats two confident wrong ones."""
    ambiguous = ambiguous_names([("Nordlicht", "Projekt Atlas"), ("Brandt", "Projekt Atlas")])

    identity = identify("Projekt Atlas", "konzept", ["Sanitas", "2022", "Projekt Triage"], ambiguous)

    assert identity.scope == ""


def test_a_mention_of_a_type_outside_the_taxonomy_is_not_an_entity():
    """Open-ended types were rejected: they visualise as an inconsistent mush. Anything
    the model invents is dropped rather than stored."""
    assert identify("Q2/2022", "zeitraum", ["Nordlicht"], frozenset()) is None


def _candidate(surface: str, entity_type: str, angle: float, mentions: int = 1) -> Candidate:
    """A candidate whose embedding sits at a chosen cosine similarity to `angle=0`.

    Real vectors through the real cosine, rather than a stubbed score: the thresholds
    are the thing under test and a stub would let them drift away from the arithmetic
    they are compared against.
    """
    return Candidate(
        identity=identify(surface, entity_type, [], frozenset()),
        surface=surface,
        embedding=[cos(angle), sin(angle)],
        mentions=mentions,
    )


def _thresholds() -> Thresholds:
    return Thresholds(merge=0.92, review=0.80)


def test_mentions_sharing_a_normalised_name_resolve_without_comparing_embeddings():
    """Stage one. The two spellings carry deliberately unrelated vectors; an
    implementation that reached for similarity first would split them."""
    resolution = resolve_entities(
        [
            _candidate("Nordlicht Energie AG", "organisation", 0.0, mentions=9),
            _candidate("nordlicht energie", "organisation", 1.4),
        ],
        _thresholds(),
    )

    assert len(resolution.entities) == 1
    assert resolution.entities[0].surface == "Nordlicht Energie AG"
    assert resolution.entities[0].mentions == 10


def test_a_near_duplicate_above_the_merge_threshold_folds_into_the_same_node():
    """Stage two. Without it the graph fragments into spellings of one organisation,
    which demonstrates the opposite of this project's thesis."""
    resolution = resolve_entities(
        [
            _candidate("Nordlicht Energie AG", "organisation", 0.0, mentions=9),
            _candidate("Nordlicht Energie Aktiengesellschaft", "organisation", 0.2),
        ],
        _thresholds(),
    )

    assert len(resolution.entities) == 1


def test_two_systems_from_one_vendor_stay_separate_nodes():
    """The must-not-merge direction of stage two. Both are SAP and the corpus runs them
    in different projects; one node would assert a connection that does not exist."""
    resolution = resolve_entities(
        [
            _candidate("SAP IS-U", "system", 0.0),
            _candidate("SAP S/4HANA", "system", 0.9),
        ],
        _thresholds(),
    )

    assert len(resolution.entities) == 2


def test_only_the_ambiguous_band_is_put_to_the_model():
    """Stage three is the expensive one and exists for the middle band alone. Asked
    about pairs the thresholds already decide, it would cost a call per pair and
    overrule the tuning."""
    asked = []

    def adjudicate(entity_type: str, left: str, right: str) -> bool:
        asked.append((left, right))
        return True

    resolution = resolve_entities(
        [
            _candidate("Agfa Orbis", "system", 0.0),
            _candidate("Orbis", "system", 0.55),
            _candidate("Snowflake", "system", 1.5),
        ],
        _thresholds(),
        adjudicate=adjudicate,
    )

    assert asked == [("Agfa Orbis", "Orbis")]
    assert len(resolution.entities) == 2


def test_the_model_declining_a_band_pair_keeps_both_nodes():
    resolution = resolve_entities(
        [
            _candidate("Projekt Vitalis", "konzept", 0.0),
            _candidate("Projekt Triage", "konzept", 0.55),
        ],
        _thresholds(),
        adjudicate=lambda entity_type, left, right: False,
    )

    assert len(resolution.entities) == 2


def test_entities_of_different_types_are_never_candidates_for_each_other():
    """A person and the deliverable named after her sit at almost the same point in
    embedding space. Merging across types produces a node that is both and renders as
    neither, so similarity is never consulted across the boundary."""
    resolution = resolve_entities(
        [
            _candidate("Vogt", "person", 0.0),
            _candidate("Vogt-Methodik", "ergebnis", 0.05),
        ],
        _thresholds(),
        adjudicate=lambda entity_type, left, right: True,
    )

    assert len(resolution.entities) == 2


def test_one_name_read_as_two_types_becomes_one_node_of_the_commoner_type():
    """The taxonomy is applied per chunk, so the same artefact comes back as a result
    in one chunk and a concept in the next. Left alone that is two nodes no threshold
    can rejoin — comparison never crosses a type — and the cross-client edge the demo
    rests on is quietly cut in half."""
    resolution = resolve_entities(
        [
            _candidate("Atlas-Referenzarchitektur", "ergebnis", 0.0, mentions=7),
            _candidate("Atlas-Referenzarchitektur", "konzept", 1.4, mentions=2),
        ],
        _thresholds(),
    )

    assert len(resolution.entities) == 1
    assert resolution.entities[0].type == "ergebnis"
    assert resolution.entities[0].mentions == 9


def test_two_clients_projects_of_the_same_name_are_never_compared_at_all():
    """Identical names, identical vectors, and still two nodes: the hierarchy has
    already ruled them distinct, so no threshold and no model call can undo it."""
    ambiguous = ambiguous_names([("Nordlicht", "Projekt Atlas"), ("Brandt", "Projekt Atlas")])
    candidates = [
        Candidate(
            identity=identify("Projekt Atlas", "konzept", [branch, "2022", "Projekt Atlas"], ambiguous),
            surface="Projekt Atlas",
            embedding=[1.0, 0.0],
            mentions=4,
        )
        for branch in ("Nordlicht", "Brandt")
    ]

    resolution = resolve_entities(
        candidates, _thresholds(), adjudicate=lambda entity_type, left, right: True
    )

    assert len(resolution.entities) == 2
    assert {entity.scope for entity in resolution.entities} == {"Nordlicht", "Brandt"}


def test_every_mention_is_told_which_node_it_ended_up_on():
    """The mapping is what the graph write uses to attach MENTIONS edges. A merged-away
    identity missing from it silently drops every edge that mention would have made."""
    resolution = resolve_entities(
        [
            _candidate("Nordlicht Energie AG", "organisation", 0.0, mentions=9),
            _candidate("Nordlicht Energie Aktiengesellschaft", "organisation", 0.2),
        ],
        _thresholds(),
    )

    canonical = resolution.entities[0].identity
    assert set(resolution.canonical.values()) == {canonical}
    assert len(resolution.canonical) == 2


_CHUNK = (
    "Die Projektleitung liegt bei Dr. Katharina Vogt (Principal Architektin), Vektor "
    "Consulting GmbH. Im Projektverlauf entstand die Atlas-Referenzarchitektur."
)


def _reply(**payload) -> str:
    return json.dumps(payload)


def test_an_entity_the_chunk_does_not_name_is_dropped():
    """The guard behind every edge on screen. An invented name carries invented
    evidence, and an audience member who checks the chunk finds nothing there."""
    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Dr. Katharina Vogt", "type": "person", "evidence": "Die Projektleitung liegt bei Dr. Katharina Vogt"},
                {"name": "Henrike Baumann", "type": "person", "evidence": "Abnahme durch Henrike Baumann"},
            ]
        ),
        _CHUNK,
        evidence_limit=300,
    )

    assert [entity.name for entity in extraction.entities] == ["Dr. Katharina Vogt"]


def test_a_paraphrased_span_is_replaced_by_the_chunks_own_sentence():
    """The model summarises its evidence more often than it quotes it. Storing the
    paraphrase would put words on screen that the source document never used."""
    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Atlas-Referenzarchitektur", "type": "ergebnis", "evidence": "Sie wurde im Projekt erstellt."}
            ]
        ),
        _CHUNK,
        evidence_limit=300,
    )

    assert extraction.entities[0].evidence in _CHUNK


def test_a_type_outside_the_taxonomy_never_reaches_the_graph():
    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Vektor Consulting GmbH", "type": "beratungsunternehmen", "evidence": "Vektor Consulting GmbH"}
            ]
        ),
        _CHUNK,
        evidence_limit=300,
    )

    assert extraction.entities == []


def test_a_relation_to_an_entity_that_was_not_extracted_is_dropped():
    """An edge is only as defensible as the two nodes it joins."""
    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Dr. Katharina Vogt", "type": "person", "evidence": "Dr. Katharina Vogt"}
            ],
            relations=[
                {"from": "Dr. Katharina Vogt", "type": "arbeitet_fuer", "to": "Nordlicht Energie AG", "evidence": "x"}
            ],
        ),
        _CHUNK,
        evidence_limit=300,
    )

    assert extraction.relations == []


def test_an_invented_relation_type_is_dropped_rather_than_stored():
    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Dr. Katharina Vogt", "type": "person", "evidence": "Dr. Katharina Vogt"},
                {"name": "Vektor Consulting GmbH", "type": "organisation", "evidence": "Vektor Consulting GmbH"},
            ],
            relations=[
                {"from": "Dr. Katharina Vogt", "type": "ist_angestellt_bei", "to": "Vektor Consulting GmbH", "evidence": "x"}
            ],
        ),
        _CHUNK,
        evidence_limit=300,
    )

    assert extraction.relations == []


def test_a_reply_that_is_not_json_costs_one_chunk_rather_than_the_pass():
    assert parse_extraction("Es tut mir leid, …", _CHUNK, evidence_limit=300) == (
        ChunkExtraction(entities=[], relations=[])
    )


def test_report_furniture_named_by_every_client_never_becomes_an_entity():
    """`Risikoregister` appears in all hundred documents. As a node it joins all four
    clients to each other, which is a cross-branch edge that means nothing and buries
    the two that mean everything."""
    ambiguous = frozenset()

    assert identify("Risikoregister", "ergebnis", ["Nordlicht"], ambiguous) is None
    assert identify("Lenkungsausschuss", "organisation", ["Brandt"], ambiguous) is None


def test_a_figure_or_a_year_is_not_an_entity():
    """The most repeated strings in a corpus of quarterly reports are its numbers."""
    assert identify("2024", "konzept", ["Nordlicht"], frozenset()) is None
    assert identify("850 TEUR", "ergebnis", ["Nordlicht"], frozenset()) is None


def test_a_named_deliverable_sharing_a_boilerplate_word_still_counts():
    """The guard matches whole names, not substrings: `Fachkonzept` alone is a table
    row in every report, `Fachkonzept Kontoeröffnung` is one client's deliverable."""
    assert identify("Fachkonzept", "ergebnis", ["Finanzkontor"], frozenset()) is None
    assert (
        identify("Fachkonzept Kontoeröffnung", "ergebnis", ["Finanzkontor"], frozenset())
        is not None
    )


def test_evidence_is_not_cut_short_at_an_abbreviated_title():
    """Half the people in this corpus are introduced as `Dr. …`. Splitting the sentence
    at that full stop leaves `Dr.` standing as the proof behind the edge."""
    chunk = "Methodisch wurde das Vorgehen von Dr. Katharina Vogt verantwortet."

    extraction = parse_extraction(
        _reply(
            entities=[
                {"name": "Dr. Katharina Vogt", "type": "person", "evidence": "sie war es"}
            ]
        ),
        chunk,
        evidence_limit=300,
    )

    assert extraction.entities[0].evidence == chunk


def test_a_relation_joining_the_wrong_kinds_of_entity_is_dropped():
    """The model reads "Dr. Katharina Vogt, Vektor Consulting GmbH" and sometimes hangs
    `arbeitet_fuer` off the architecture named in the next clause. The edge still
    renders, and the evidence under it does not support what it claims."""
    assert joins("arbeitet_fuer", "person", "organisation")
    assert not joins("arbeitet_fuer", "person", "ergebnis")


# The machine half of the taxonomy. The failure it guards against was measured, not
# imagined: before `baugruppe` and `verbaut_in` existed, a shared assembly was named in
# two manufacturers' documents and had exactly one edge, so the traversal leg — which
# walks edges, not mentions — reached one side of the bridge and asserted the other did
# not exist. A taxonomy silently missing a relation looks identical to a corpus that
# genuinely has no connection.


def test_an_assembly_can_be_joined_to_the_machine_it_is_built_into():
    assert joins("verbaut_in", "baugruppe", "maschine")


def test_the_bridge_relation_refuses_endpoints_that_would_make_it_meaningless():
    """`verbaut_in` is the edge a cross-manufacturer question traverses. Pointed at an
    organisation or a person it would still render on the graph, as a claim the
    evidence underneath it does not support."""
    assert not joins("verbaut_in", "baugruppe", "organisation")
    assert not joins("verbaut_in", "maschine", "baugruppe")  # direction is load-bearing
    assert not joins("verbaut_in", "person", "maschine")


def test_an_engineer_can_be_made_responsible_for_an_assembly():
    """The second strand of each planted bridge is the person who carried the assembly
    across. Rejecting this edge would leave the bridge hanging on one relation."""
    assert joins("verantwortet", "person", "baugruppe")
    assert joins("verantwortet", "person", "maschine")


def test_the_machine_relations_join_only_what_they_name():
    assert joins("variante_von", "maschine", "maschine")
    assert joins("bietet_option", "maschine", "option")
    assert joins("hergestellt_von", "maschine", "organisation")
    assert not joins("bietet_option", "option", "maschine")
    assert not joins("hergestellt_von", "organisation", "maschine")


def test_two_models_sharing_a_code_under_different_machine_types_stay_distinct():
    """`B2BC` names a Hagedorn mixer-dryer and a Pleickner assembly cell. One node for
    both would merge two unrelated machines and make every spec answer ambiguous — and
    only the folder tree knows the difference, since neither data sheet mentions the
    other manufacturer."""
    ambiguous = ambiguous_names([("Hagedorn", "B2BC"), ("Pleickner", "B2BC")])

    hagedorn = identify("B2BC", "maschine", ["Hagedorn", "XTC27", "B2BC"], ambiguous)
    pleickner = identify("B2BC", "maschine", ["Pleickner", "ZM3K", "B2BC"], ambiguous)

    assert hagedorn != pleickner
    assert (hagedorn.scope, pleickner.scope) == ("Hagedorn", "Pleickner")


def test_an_assembly_shared_by_two_manufacturers_stays_one_identity():
    """The mirror of the test above, and the reason the bridge exists: the same
    assembly named under two branches must be *one* node, or there is nothing to
    traverse between them."""
    ambiguous = ambiguous_names([("Hagedorn", "B2BC"), ("Pleickner", "B2BC")])

    origin = identify(
        "Planetengetriebe HX-114", "baugruppe", ["Hagedorn", "XTC27", "B2BC"], ambiguous
    )
    reuse = identify(
        "Planetengetriebe HX-114", "baugruppe", ["Ostmark", "QH4L", "P6V"], ambiguous
    )

    assert origin == reuse
    assert origin.scope == ""


def test_the_machine_documents_furniture_is_not_extractable():
    """Every data sheet carries these words. One left in becomes a hub joining all
    three manufacturers — and the real bridges here are four mentions wide, so a hub
    does not merely add noise, it outranks them."""
    for word in ("Baugruppe", "Wartungsplan", "Grundausführung", "Bestellschlüssel"):
        assert identify(word, "baugruppe", ["Hagedorn", "XTC27", "B2BC"], frozenset()) is None

    # ...while a code-bearing name of the same shape survives.
    assert identify(
        "Planetengetriebe HX-114", "baugruppe", ["Hagedorn", "XTC27", "B2BC"], frozenset()
    ) is not None
    assert identify("Option FW", "option", ["Hagedorn", "XTC27", "B2BC"], frozenset()) is not None
