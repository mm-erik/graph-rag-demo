"""Entities as the graph sees them: a fixed taxonomy, and a normalised identity.

Extraction uses a **fixed taxonomy** rather than open-ended types. Open-ended
extraction was considered and rejected: it produces inconsistent labels that visualise
badly, and the visualisation is the entire point of this demo. Anything the model
returns outside the taxonomy is dropped rather than stored.

Normalisation is the first of the three resolution stages and the cheapest: two
mentions that normalise to the same string are the same entity, with no embedding and
no model call. It is deliberately German-aware — the corpus writes `Brandt & Söhne
Maschinenbau GmbH` in the documents and `Brandt und Soehne Maschinenbau` on disk, and a
normaliser that does not fold those together fragments the client into three nodes.

The extraction model is injected as a protocol; nothing here constructs one, and
nothing here touches the graph.
"""

from __future__ import annotations

import asyncio
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

#: The fixed entity taxonomy, in German because every label on this screen is read by a
#: German-speaking audience. The first six are the consulting archive's concepts:
#: person, organisation, system or product, concept, metric or deliverable, risk.
#:
#: The last three exist because the machine half of the corpus is *built* from opaque
#: codes — `XTC27`, `B2BC`, `HX-114` — and the traversal leg walks edges, not mentions.
#: Without a type for an assembly there is no relation that can carry one, so a shared
#: assembly ends up named in two branches and traversable from neither: measured, the
#: `HX-114` bridge had four mentions across two manufacturers and exactly one edge, and
#: `--legs traversal` answered "ausschliesslich Ostmark" while the flat legs got it
#: right. A taxonomy that cannot express the join is the whole reason the graph lost.
ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "organisation",
    "system",
    "konzept",
    "ergebnis",
    "risiko",
    "maschine",
    "baugruppe",
    "option",
)

#: German letter folding. Applied before the accent strip below, so `ü` becomes `ue`
#: rather than `u` — `Söhne` and `Soehne` are the same word and `Sohne` is not how
#: anyone writes it.
_GERMAN_FOLDING = {
    "ä": "ae",
    "ö": "oe",
    "ü": "ue",
    "ß": "ss",
}

#: Legal forms, stripped from organisation names only. `AG` is a legal form in
#: `Nordlicht Energie AG` and part of the name in a system called `AG Grid`, so the
#: strip is scoped to the one type where it is unambiguous.
_LEGAL_FORMS = frozenset(
    {
        "gmbh",
        "ggmbh",
        "mbh",
        "ag",
        "kgaa",
        "kg",
        "ohg",
        "gbr",
        "se",
        "ev",
        "co",
        "inc",
        "ltd",
    }
)

#: Titles and salutations, stripped from person names only, for the same reason. The
#: corpus writes `Dr. Katharina Vogt` in one sentence and `Katharina Vogt` in the next.
_TITLES = frozenset(
    {
        "dr",
        "prof",
        "med",
        "dipl",
        "ing",
        "herr",
        "frau",
        "mag",
        "phd",
        "mba",
    }
)

#: Articles, stripped from every type. A model told to return a bare name still
#: occasionally returns `die Atlas-Referenzarchitektur`.
_ARTICLES = frozenset({"der", "die", "das", "den", "dem", "des"})

_NON_WORD = re.compile(r"[^0-9a-z]+")

#: Report furniture, as normalised names. Every document in this corpus carries the
#: same headings, the same committee and the same appendices, so these are named by all
#: four clients — and an entity named by all four clients becomes a hub node joining
#: every branch of the hierarchy to every other, for no reason anyone can defend. That
#: is the opposite of the one picture this project exists to draw.
#:
#: The extraction prompt already forbids them and the model extracts them anyway, a few
#: chunks in a hundred. A list is deterministic where a prompt is not. It is explicitly
#: shaped by *this* corpus, which is a trade this throwaway project is happy to make:
#: see the entity section of the README.
BOILERPLATE = frozenset(
    {
        "risikoregister",
        "massnahmenliste",
        "lenkungsausschuss",
        "testsystem",
        "testfaelle",
        "fachkonzept",
        "umsetzung",
        "test und abnahme",
        "abnahme",
        "abstimmungstermin",
        "woechentlicher abstimmungstermin",
        "projekthandbuch",
        "projektbericht",
        "statusbericht",
        "berichtszeitraum",
        "meilenstein",
        "teilprojekt",
        "fachbereich",
        "fachbereiche",
        "bestandssysteme",
        "workshop",
        "workshops",
        "projekt",
        # The machine half's furniture. Every data sheet, variant overview and
        # maintenance plan carries these words, so each one left in would become a hub
        # joining all three manufacturers — the same failure the consulting terms above
        # guard against, in a corpus where the real bridges are four mentions wide.
        "wartungsplan",
        "datenblatt",
        "betriebsanleitung",
        "maschinenbuch",
        "ergaenzungsblaetter",
        "aufstellplan",
        "lieferprogramm",
        "grundausfuehrung",
        "ausfuehrung",
        "ausfuehrungen",
        "bestellschluessel",
        "baugruppe",
        "baugruppen",
        "ersatzteil",
        "ersatzteile",
        "ersatzteilnummer",
        "variante",
        "varianten",
        "baureihe",
        "modell",
        "maschine",
        "option",
        "optionen",
        "pruefintervall",
        "schutzart",
        "leergewicht",
        "gesamthoehe",
    }
)


def _fold(text: str) -> str:
    """Lowercase, with German letters spelled out and remaining accents dropped."""
    lowered = text.casefold()
    for letter, replacement in _GERMAN_FOLDING.items():
        lowered = lowered.replace(letter, replacement)
    # `&` is read as a word rather than punctuation: dropping it would leave
    # `brandt soehne` against `brandt und soehne`, which no longer match.
    lowered = lowered.replace("&", " und ")
    decomposed = unicodedata.normalize("NFKD", lowered)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _droppable(entity_type: str) -> frozenset[str]:
    if entity_type == "organisation":
        return _ARTICLES | _LEGAL_FORMS
    if entity_type == "person":
        return _ARTICLES | _TITLES
    return _ARTICLES


def normalise_name(name: str, entity_type: str) -> str:
    """The identity string two mentions must share to be the same entity outright.

    Type-aware, because the tokens worth discarding differ by type and discarding the
    wrong ones merges entities that are genuinely distinct. Returns `""` for a name
    that is nothing but noise, which the caller drops rather than storing an entity
    with no identity at all.
    """
    words = [word for word in _NON_WORD.split(_fold(name)) if word]
    droppable = _droppable(entity_type)
    kept = [word for word in words if word not in droppable]
    # Everything dropped means the name *was* the title or the legal form — `GmbH` on
    # its own. Keeping the original words is better than an empty identity, which would
    # collapse every such mention into one node.
    return " ".join(kept or words)


@dataclass(frozen=True)
class EntityIdentity:
    """What makes two mentions the same node — before any embedding is involved.

    All three fields together are the identity, and all three are the graph's
    uniqueness constraint. `normalized_name` deliberately stays a *normalised name*
    rather than absorbing the scope: the traversal leg matches a name taken from a
    question against this property, and a question naming "Projekt Atlas" should reach
    both clients' projects rather than neither.
    """

    type: str
    normalized_name: str
    #: The branch this name belongs to, or `""` when the entity is global. A *name*
    #: component, deliberately not an edge: entities are never attached to a folder,
    #: which is what makes a cross-client path real structure rather than a query-time
    #: join.
    scope: str = ""


def ambiguous_names(folders: Sequence[tuple[str, str]]) -> frozenset[str]:
    """Normalised folder names that occur under more than one top-level branch.

    `folders` is `(branch, folder name)` for every folder in the corpus. This is the
    hierarchy telling us which names it cannot distinguish on their own — the corpus is
    engineered so that no document naming `Projekt Atlas` mentions the other client, so
    the folder tree is the *only* source for that fact.
    """
    branches: dict[str, set[str]] = {}
    for branch, name in folders:
        # Folder names are compared as concepts: a folder is a thing in the archive,
        # and the entity that collides with it arrives through the same normaliser.
        branches.setdefault(normalise_name(name, "konzept"), set()).add(branch)
    return frozenset(name for name, owners in branches.items() if len(owners) > 1)


def identify(
    surface: str,
    entity_type: str,
    ancestry: Sequence[str],
    ambiguous: frozenset[str],
) -> EntityIdentity | None:
    """The identity one extracted mention claims, or None when it claims none.

    Scope is applied only where the hierarchy proves it is needed: the name matches a
    folder on this document's own path *and* that folder name exists under more than
    one branch. Scoping every folder name instead would split the consultants and the
    systems by client too, and scoping none of them would merge two clients' unrelated
    projects into one node.
    """
    if entity_type not in ENTITY_TYPES:
        return None
    normalised = normalise_name(surface, entity_type)
    if not normalised or normalised in BOILERPLATE or _is_quantity(surface):
        return None
    return EntityIdentity(
        type=entity_type,
        normalized_name=normalised,
        scope=_scope_for(normalised, ancestry, ambiguous),
    )


#: A figure, optionally with its unit: `2024`, `850 TEUR`, `15 Monate`. Matched rather
#: than "starts with a digit", which would also discard `3M` and `8D-Report`.
_QUANTITY = re.compile(r"^\d[\d.,]*\s*[A-Za-zÄÖÜäöüß]{0,8}$")


def _is_quantity(surface: str) -> bool:
    """Whether the name is a number rather than a thing.

    Figures and years are the most repeated strings in a corpus of quarterly reports,
    and a year extracted as a concept becomes a node every client is joined to.
    """
    return bool(_QUANTITY.match(surface.strip()))


def _scope_for(
    normalised: str, ancestry: Sequence[str], ambiguous: frozenset[str]
) -> str:
    """The branch an ambiguous name belongs to, or `""`.

    An ambiguous name mentioned outside any branch that owns it is left unscoped: the
    document's path says nothing about which one is meant, and a guess would put a real
    edge on the wrong client's subtree.
    """
    if normalised not in ambiguous or not ancestry:
        return ""
    on_path = any(
        normalise_name(folder, "konzept") == normalised for folder in ancestry
    )
    return ancestry[0] if on_path else ""


#: The fixed relation list. Open-ended relations have the same problem as open-ended
#: types — an edge label nobody has seen before cannot be given a colour, a legend entry
#: or an explanation to an audience. German, because these labels are rendered on the
#: traversal graph.
RELATION_TYPES: tuple[str, ...] = (
    "arbeitet_fuer",
    "leitet",
    "mitarbeit_in",
    "verantwortet",
    "setzt_ein",
    "liefert",
    "hat_risiko",
    "basiert_auf",
    "teil_von",
    # The machine half. `verbaut_in` is the load-bearing one: it is the edge a question
    # about a shared assembly traverses to reach the other manufacturer.
    "verbaut_in",
    "variante_von",
    "bietet_option",
    "hergestellt_von",
)


#: What each relation may join. The types are already spelled out in the extraction
#: prompt; this enforces them, because a model that reads "Dr. Katharina Vogt, Vektor
#: Consulting GmbH" sometimes emits `arbeitet_fuer` pointing at the architecture named
#: in the next clause. An edge like that is not merely noise — it is a claim on screen
#: that the evidence underneath it does not support, and the evidence is the whole
#: reason the edge is shown.
#:
#: `teil_von` is deliberately unconstrained: containment is the one relation whose
#: endpoints vary by what is being contained.
RELATION_ENDPOINTS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "arbeitet_fuer": (frozenset({"person"}), frozenset({"organisation"})),
    "leitet": (frozenset({"person"}), frozenset({"konzept", "organisation"})),
    "mitarbeit_in": (frozenset({"person"}), frozenset({"konzept"})),
    "verantwortet": (
        frozenset({"person"}),
        # Machines and assemblies included: the engineer who designed an assembly is
        # the *second* strand of a bridge, and a constraint that rejected that edge
        # would leave the bridge hanging on one relation.
        frozenset({"ergebnis", "konzept", "risiko", "maschine", "baugruppe"}),
    ),
    "setzt_ein": (
        frozenset({"konzept", "organisation", "person"}),
        frozenset({"system"}),
    ),
    "liefert": (frozenset({"konzept", "organisation"}), frozenset({"ergebnis"})),
    "hat_risiko": (
        frozenset({"konzept", "organisation", "system"}),
        frozenset({"risiko"}),
    ),
    "basiert_auf": (
        frozenset({"ergebnis", "konzept"}),
        frozenset({"ergebnis", "konzept"}),
    ),
    "teil_von": (frozenset(ENTITY_TYPES), frozenset(ENTITY_TYPES)),
    "verbaut_in": (frozenset({"baugruppe"}), frozenset({"maschine"})),
    "variante_von": (frozenset({"maschine"}), frozenset({"maschine"})),
    "bietet_option": (frozenset({"maschine"}), frozenset({"option"})),
    "hergestellt_von": (frozenset({"maschine"}), frozenset({"organisation"})),
}


def joins(relation_type: str, source_type: str, target_type: str) -> bool:
    """Whether this relation is allowed to join these two kinds of entity."""
    endpoints = RELATION_ENDPOINTS.get(relation_type)
    if endpoints is None:
        return False
    sources, targets = endpoints
    return source_type in sources and target_type in targets


@dataclass(frozen=True)
class MentionDraft:
    """One entity the model claims the chunk names, before identity is assigned."""

    name: str
    type: str
    evidence: str


@dataclass(frozen=True)
class RelationDraft:
    """One relation the model claims, named by surface rather than by identity."""

    source: str
    target: str
    type: str
    evidence: str


@dataclass(frozen=True)
class ChunkExtraction:
    entities: list[MentionDraft]
    relations: list[RelationDraft]


_WHITESPACE = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s")

#: Abbreviations whose full stop is not the end of a sentence. Without these, the
#: evidence behind "Dr. Katharina Vogt" is the single word "Dr." — which proves
#: nothing to anyone who asks where the edge came from.
_ABBREVIATIONS = frozenset(
    {"dr", "prof", "med", "ca", "bzw", "z", "b", "usw", "nr", "abs", "ggf", "inkl", "ff"}
)
_LAST_WORD = re.compile(r"([A-Za-zÄÖÜäöüß]+)\.$")


def _flatten(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def locate_evidence(claim: str, name: str, chunk_text: str, limit: int) -> str | None:
    """The chunk's own words backing this mention, or None when there are none.

    Every edge shown on screen has to survive an audience member asking "where does it
    say that?", so evidence is taken *from the chunk* rather than from the model's
    reply. A model that paraphrased its span still gets its mention, recovered from the
    sentence the name appears in; a model that invented the name outright gets nothing,
    and the mention is dropped.
    """
    haystack = _flatten(chunk_text)
    if not haystack:
        return None
    quoted = _flatten(claim)
    if quoted and quoted in haystack:
        return _shorten(quoted, limit)
    position = haystack.casefold().find(_flatten(name).casefold())
    if position < 0 or not name.strip():
        return None
    return _shorten(_sentence_around(haystack, position), limit)


def _sentence_around(haystack: str, position: int) -> str:
    """The sentence containing `position`, so recovered evidence still reads as prose."""
    breaks = [
        match.end()
        for match in _SENTENCE_END.finditer(haystack)
        if not _is_abbreviation(haystack[: match.start()])
    ]
    start = max([0, *(stop for stop in breaks if stop <= position)])
    end = next((stop for stop in breaks if stop > position), len(haystack))
    return haystack[start:end].strip()


def _is_abbreviation(text: str) -> bool:
    """Whether the full stop ending `text` abbreviates a word rather than a sentence."""
    match = _LAST_WORD.search(text)
    return bool(match) and match.group(1).casefold() in _ABBREVIATIONS


#: Marks evidence the cap cut short, so a truncated span does not read as a whole
#: sentence the document never finished.
ELLIPSIS = " […]"


def _shorten(text: str, limit: int) -> str:
    """Trim to `limit` characters *including* the ellipsis — a cap that its own marker
    pushes back over the limit is not a cap."""
    if limit < 1 or len(text) <= limit:
        return text
    return text[: max(0, limit - len(ELLIPSIS))].rstrip() + ELLIPSIS


def parse_extraction(reply: str, chunk_text: str, evidence_limit: int) -> ChunkExtraction:
    """A model reply as the entities and relations to store.

    Total, like `parse_inference` and `clean_summary`: one odd answer out of several
    hundred chunks must cost that chunk's entities, not the whole pass. Everything
    unusable is dropped rather than raised — an invented type, a relation between
    entities the model did not extract, a mention whose name is nowhere in the chunk.
    """
    try:
        payload = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return ChunkExtraction(entities=[], relations=[])
    if not isinstance(payload, dict):
        return ChunkExtraction(entities=[], relations=[])

    entities: list[MentionDraft] = []
    seen: set[tuple[str, str]] = set()
    for row in _rows(payload.get("entities")):
        name = str(row.get("name") or "").strip()
        entity_type = str(row.get("type") or "").strip().casefold()
        if entity_type not in ENTITY_TYPES or not name:
            continue
        evidence = locate_evidence(
            str(row.get("evidence") or ""), name, chunk_text, evidence_limit
        )
        if evidence is None:
            continue
        key = (normalise_name(name, entity_type), entity_type)
        if key in seen:
            continue
        seen.add(key)
        entities.append(MentionDraft(name=name, type=entity_type, evidence=evidence))

    return ChunkExtraction(
        entities=entities,
        relations=_relations(payload.get("relations"), entities, chunk_text, evidence_limit),
    )


def _rows(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _relations(
    value: object,
    entities: Sequence[MentionDraft],
    chunk_text: str,
    evidence_limit: int,
) -> list[RelationDraft]:
    """Relations whose endpoints the model also extracted, deduplicated.

    An endpoint that is not among the extracted entities is dropped rather than
    invented as a new one: an entity nothing mentions has no evidence behind it, and an
    edge is only as defensible as the two nodes it joins.
    """
    by_name = {
        normalise_name(entity.name, entity.type): entity for entity in entities
    }
    # Matched without the type, because the model names an endpoint rather than
    # repeating its type, and the same string under two types does not occur in
    # practice. Ambiguity falls to whichever was extracted first, deterministically.
    by_any_type: dict[str, MentionDraft] = {}
    for entity in entities:
        for entity_type in ENTITY_TYPES:
            by_any_type.setdefault(normalise_name(entity.name, entity_type), entity)

    relations: list[RelationDraft] = []
    seen: set[tuple[str, str, str]] = set()
    for row in _rows(value):
        relation_type = str(row.get("type") or "").strip().casefold()
        if relation_type not in RELATION_TYPES:
            continue
        source = _endpoint(row.get("from"), by_name, by_any_type)
        target = _endpoint(row.get("to"), by_name, by_any_type)
        if source is None or target is None or source is target:
            continue
        evidence = locate_evidence(
            str(row.get("evidence") or ""),
            f"{source.name} {target.name}",
            chunk_text,
            evidence_limit,
        )
        if evidence is None:
            # Both endpoints are in the chunk but no single sentence covers the pair.
            # The relation is still the model's reading of this chunk, so it is kept
            # with the source entity's own sentence standing as its evidence.
            evidence = source.evidence
        key = (source.name, relation_type, target.name)
        if key in seen:
            continue
        seen.add(key)
        relations.append(
            RelationDraft(
                source=source.name,
                target=target.name,
                type=relation_type,
                evidence=evidence,
            )
        )
    return relations


def _endpoint(
    value: object,
    by_name: dict[str, MentionDraft],
    by_any_type: dict[str, MentionDraft],
) -> MentionDraft | None:
    name = str(value or "").strip()
    if not name:
        return None
    for entity_type in ENTITY_TYPES:
        found = by_name.get(normalise_name(name, entity_type))
        if found is not None:
            return found
    return by_any_type.get(normalise_name(name, "konzept"))


# German, per the project convention. The taxonomy and the relation list are spelled out
# in the prompt rather than left implicit: a model given "extract the entities" invents
# a type per document, and an inconsistent set of labels is exactly what the fixed
# taxonomy exists to prevent.
#
# The instruction to skip boilerplate is load-bearing on this corpus. Every status
# report carries the same headings and the same sentence frames, so a model left to
# itself extracts "Lenkungsausschuss", "Berichtszeitraum" and "Maßnahmenliste" from all
# hundred of them — a hub node joining every client to every other for no reason, which
# ruins the one picture this project exists to draw.
EXTRACTION_SYSTEM_PROMPT = (
    "Du extrahierst Entitäten und Beziehungen aus Textabschnitten eines "
    "Projektarchivs einer Unternehmensberatung.\n"
    "\n"
    "Erlaubte Entitätstypen — nichts anderes ist zulässig:\n"
    "- person: benannte Menschen\n"
    "- organisation: Unternehmen, Behörden, Kliniken, Geschäftsbereiche\n"
    "- system: Software, Plattformen, Produkte, technische Standards\n"
    "- konzept: Projekte, Vorhaben, Methoden, Verfahren\n"
    "- ergebnis: Arbeitsergebnisse, Dokumente, Konzepte, Kennzahlen, Meilensteine\n"
    "- risiko: benannte Risiken, Hindernisse, offene Punkte\n"
    "- maschine: Maschinentypen, Modelle und bestellbare Varianten, fast immer "
    "Buchstaben-Zahlen-Codes wie 'XTC27', 'B2BC', 'QH4L-P6V' oder "
    "'XTC27-B2BC-FW/OW'\n"
    "- baugruppe: Baugruppen, Bauteile und Ersatzteile mit eigener Bezeichnung, etwa "
    "'Planetengetriebe HX-114' oder 'Spannsatz KN-30'\n"
    "- option: bestellbare Ausstattungsoptionen mit Kurzzeichen, etwa 'Option FW' "
    "oder 'Option OW'\n"
    "\n"
    "Erlaubte Beziehungstypen — nichts anderes ist zulässig:\n"
    "- arbeitet_fuer: Person -> Organisation\n"
    "- leitet: Person -> Projekt oder Organisation\n"
    "- mitarbeit_in: Person -> Projekt\n"
    "- verantwortet: Person -> Ergebnis\n"
    "- setzt_ein: Projekt oder Organisation -> System\n"
    "- liefert: Projekt -> Ergebnis\n"
    "- hat_risiko: Projekt -> Risiko\n"
    "- basiert_auf: Ergebnis oder Konzept -> Ergebnis oder Konzept\n"
    "- teil_von: Zugehörigkeit zwischen zwei Entitäten\n"
    "- verbaut_in: Baugruppe -> Maschine\n"
    "- variante_von: Maschine -> Maschine (Variante oder Modell zu seiner Baureihe)\n"
    "- bietet_option: Maschine -> Option\n"
    "- hergestellt_von: Maschine -> Organisation\n"
    "\n"
    "Regeln:\n"
    "1. Nenne jede Entität mit ihrem Eigennamen, so wie er im Text steht, und immer "
    "vollständig, auch wenn der Abschnitt ihn abkürzt. Lasse Zusätze wie 'in der "
    "Fassung 2', 'Version 3', Jahreszahlen oder Quartalsangaben weg — das Artefakt "
    "bleibt dasselbe.\n"
    "2. Gib zu jeder Entität und jeder Beziehung einen wörtlichen Beleg aus dem "
    "Abschnitt an — ein Satz, unverändert abgeschrieben.\n"
    "3. Erfinde nichts. Was nicht im Abschnitt steht, wird nicht extrahiert, auch wenn "
    "du es aus der Ablage erschliessen könntest.\n"
    "4. Keine Entitäten sind: Funktionsbezeichnungen ('Principal Architektin', "
    "'Geschäftsführer'), Orte, Branchen, Zeiträume und Beträge.\n"
    "5. Extrahiere nur, was einen Eigennamen trägt. Überspringe Textbausteine, die in "
    "jedem Bericht stehen: Überschriften, Berichtszeiträume, Gremien wie "
    "'Lenkungsausschuss', Begriffe wie 'Projekt', 'Maßnahmenliste', 'Statusbericht', "
    "'Risikoregister' oder 'Fachbereich' ohne eigenen Namen.\n"
    "6. Beide Enden einer Beziehung müssen in deiner Entitätenliste stehen.\n"
    "7. Codes wie 'XTC27', 'B2BC' oder 'HX-114' sind Eigennamen und werden immer "
    "extrahiert, auch wenn sie nichts bedeuten. Schreibe jeden Code Zeichen für "
    "Zeichen so ab, wie er im Abschnitt steht — ergänze und entferne keinen "
    "Buchstaben und keine Ziffer. Ein Bestellschlüssel wie 'XTC27-B2BC' oder "
    "'QH4L-P6V' ist EINE Entität vom Typ maschine; zerlege ihn nicht in Baureihe "
    "und Modell.\n"
    "8. Nennt ein Abschnitt eine Baugruppe und die Maschine, in der sie sitzt, dann "
    "gib dazu immer die Beziehung verbaut_in an — auch wenn die Baugruppe aus einem "
    "früheren Einsatz übernommen wurde. Diese Kante ist der Zweck der Extraktion.\n"
    "\n"
    'Antworte ausschliesslich als JSON: {"entities": [{"name": "...", "type": "...", '
    '"evidence": "..."}], "relations": [{"from": "...", "type": "...", "to": "...", '
    '"evidence": "..."}]}'
)


def build_extraction_prompt(chunk_text: str, ancestry: Sequence[str]) -> str:
    """The chunk, with its place in the archive.

    The ancestry is context, not content: it tells the model that "der Auftraggeber" in
    this passage is the client named on the path, which is what lets a chunk that never
    names its client still contribute edges to the right one. Rule 3 in the system
    prompt stops it from being extracted *as* entities — the folder tree is already in
    the graph and does not need recovering from the text.
    """
    lines = ["Ablage:", *(f"- {folder}" for folder in ancestry), "", "Abschnitt:", chunk_text]
    return "\n".join(lines)


class ExtractionModel(Protocol):
    """The extraction-tier model. One call per chunk — the high-volume call site in the
    system, and therefore the one place where the cheaper tier is worth it."""

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...


@dataclass(frozen=True)
class ChunkSource:
    """One chunk, with everything the extraction call needs."""

    id: str
    text: str
    #: Folder names outermost first, excluding the corpus root.
    ancestry: list[str]


@dataclass(frozen=True)
class Mention:
    """A chunk naming an entity, with the chunk's own words as evidence."""

    chunk_id: str
    identity: EntityIdentity
    surface: str
    evidence: str


@dataclass(frozen=True)
class Relation:
    """An edge between two entities, and the chunk that justifies it."""

    chunk_id: str
    source: EntityIdentity
    target: EntityIdentity
    type: str
    evidence: str


@dataclass(frozen=True)
class ChunkResult:
    chunk_id: str
    mentions: list[Mention]
    relations: list[Relation]
    #: Why this chunk yielded nothing, when it failed. Reported, not raised: a rate
    #: limit on chunk six hundred must not discard the five hundred already paid for.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def extract_chunk(
    source: ChunkSource,
    model: ExtractionModel,
    ambiguous: frozenset[str],
    evidence_limit: int,
) -> ChunkResult:
    """One chunk's entities and relations, with identities assigned."""
    reply = model.complete(
        EXTRACTION_SYSTEM_PROMPT,
        build_extraction_prompt(source.text, source.ancestry),
        json_object=True,
    )
    extraction = parse_extraction(reply, source.text, evidence_limit)
    identities: dict[str, EntityIdentity] = {}
    mentions: list[Mention] = []
    for draft in extraction.entities:
        identity = identify(draft.name, draft.type, source.ancestry, ambiguous)
        if identity is None:
            continue
        identities[draft.name] = identity
        mentions.append(
            Mention(
                chunk_id=source.id,
                identity=identity,
                surface=draft.name,
                evidence=draft.evidence,
            )
        )
    relations = [
        Relation(
            chunk_id=source.id,
            source=identities[draft.source],
            target=identities[draft.target],
            type=draft.type,
            evidence=draft.evidence,
        )
        for draft in extraction.relations
        if draft.source in identities
        and draft.target in identities
        # Two surfaces that resolve to one identity — "Projekt Atlas" and "Atlas" in
        # one chunk — would otherwise write an edge from a node to itself.
        and identities[draft.source] != identities[draft.target]
        and joins(
            draft.type,
            identities[draft.source].type,
            identities[draft.target].type,
        )
    ]
    return ChunkResult(chunk_id=source.id, mentions=mentions, relations=relations)


@dataclass(frozen=True)
class ExtractionReport:
    chunks: list[ChunkResult]

    @property
    def mentions(self) -> list[Mention]:
        return [mention for chunk in self.chunks for mention in chunk.mentions]

    @property
    def relations(self) -> list[Relation]:
        return [relation for chunk in self.chunks for relation in chunk.relations]

    @property
    def failures(self) -> list[ChunkResult]:
        return [chunk for chunk in self.chunks if not chunk.ok]


async def extract_chunks(
    sources: Sequence[ChunkSource],
    *,
    model: ExtractionModel,
    ambiguous: frozenset[str],
    concurrency: int,
    evidence_limit: int,
    on_chunk: Callable[[ChunkResult], None] | None = None,
) -> ExtractionReport:
    """Extract from every chunk, a bounded number at a time.

    One call per chunk, which makes this the main cost driver of an ingest. The model
    client is synchronous, so each call is handed to a thread — on the event loop one
    chunk's call would stall every other chunk's.
    """
    limit = asyncio.Semaphore(concurrency)

    async def one(source: ChunkSource) -> ChunkResult:
        async with limit:
            try:
                result = await asyncio.to_thread(
                    extract_chunk, source, model, ambiguous, evidence_limit
                )
            # Deliberately broad, as in `ingest_contents`: one chunk that cannot be
            # extracted is a few missing edges, not a failed ingest.
            except Exception as error:  # noqa: BLE001
                result = ChunkResult(
                    chunk_id=source.id, mentions=[], relations=[], error=str(error)
                )
            if on_chunk is not None:
                on_chunk(result)
            return result

    results = await asyncio.gather(*(one(source) for source in sources))
    return ExtractionReport(chunks=list(results))


@dataclass(frozen=True)
class NamedEntity:
    """One distinct identity and the name the corpus actually uses for it."""

    identity: EntityIdentity
    surface: str
    mentions: int


def candidate_entities(mentions: Sequence[Mention]) -> list[NamedEntity]:
    """Distinct identities, each with the surface the corpus uses most and a count.

    This is stage one of resolution applied across the whole corpus at once: everything
    that normalises alike is already one candidate before a single embedding is
    computed. The surface chosen is the most-mentioned spelling, ties broken on the
    longer name — `Nordlicht Energie AG` reads better on screen than `Nordlicht`.
    """
    counts: dict[EntityIdentity, Counter] = defaultdict(Counter)
    for mention in mentions:
        counts[mention.identity][mention.surface] += 1
    candidates = []
    for identity, surfaces in counts.items():
        surface = max(surfaces.items(), key=lambda item: (item[1], len(item[0]), item[0]))[0]
        candidates.append(
            NamedEntity(
                identity=identity, surface=surface, mentions=sum(surfaces.values())
            )
        )
    return sorted(candidates, key=lambda named: named.identity.normalized_name)


@dataclass(frozen=True)
class GraphRelation:
    """A `RELATES` edge, after resolution, with one representative justification.

    The same relation is usually asserted by several chunks — every status report of a
    project says who leads it. One edge per `(source, type, target)` keeps the picture
    legible; `support` records how many chunks agreed and `source_chunk` names the one
    whose evidence is carried, so the edge can still be traced back to a document.
    """

    source: EntityIdentity
    target: EntityIdentity
    type: str
    evidence: str
    source_chunk: str
    support: int


def canonical_edges(
    mentions: Sequence[Mention],
    relations: Sequence[Relation],
    canonical: dict[EntityIdentity, EntityIdentity],
) -> tuple[list[Mention], list[GraphRelation]]:
    """Rewrite every edge onto the node its endpoints resolved to.

    Self-loops are dropped here rather than earlier: two surfaces in one chunk can be
    distinct identities at extraction time and the same node after merging, and an
    entity related to itself renders as a stray circle with no meaning.
    """
    seen_mentions: set[tuple[str, EntityIdentity]] = set()
    graph_mentions: list[Mention] = []
    for mention in mentions:
        identity = canonical.get(mention.identity, mention.identity)
        key = (mention.chunk_id, identity)
        if key in seen_mentions:
            continue
        seen_mentions.add(key)
        graph_mentions.append(
            Mention(
                chunk_id=mention.chunk_id,
                identity=identity,
                surface=mention.surface,
                evidence=mention.evidence,
            )
        )

    collapsed: dict[tuple[EntityIdentity, str, EntityIdentity], GraphRelation] = {}
    for relation in relations:
        source = canonical.get(relation.source, relation.source)
        target = canonical.get(relation.target, relation.target)
        if source == target:
            continue
        key = (source, relation.type, target)
        existing = collapsed.get(key)
        if existing is None:
            collapsed[key] = GraphRelation(
                source=source,
                target=target,
                type=relation.type,
                evidence=relation.evidence,
                source_chunk=relation.chunk_id,
                support=1,
            )
        else:
            collapsed[key] = replace(existing, support=existing.support + 1)
    return graph_mentions, list(collapsed.values())
