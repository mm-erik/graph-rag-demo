"""German document text for the machine half of the corpus.

Three documents per model, all three built from the same sentence frames across all
fourteen models: a data sheet, a variant overview and a maintenance plan. The frames are
near-identical on purpose — what distinguishes two documents is a handful of opaque codes
and a column of numbers, which is precisely what an embedding smears away.

The maintenance plan carries the cross-manufacturer bridges, in the section on where an
assembly came from. As on the client half, **neither side names the other manufacturer**:
the join is the shared assembly and the engineer who carried it, so it exists in the graph
and never in a single chunk.
"""

from __future__ import annotations

from random import Random

from graph_rag.corpus.machine_plan import (
    BRIDGE_ASSEMBLY,
    BRIDGE_ENGINEER,
    BRIDGE_ORIGIN,
    BRIDGE_REUSE,
    SECOND_BRIDGE_ASSEMBLY,
    SECOND_BRIDGE_ENGINEER,
    SECOND_BRIDGE_ORIGIN,
    SECOND_BRIDGE_REUSE,
    Machine,
    Manufacturer,
    options_for,
    variant_spec,
    variant_specs,
)
from graph_rag.corpus.text import Document, Section

#: Spare part numbers are derived from the assembly name so the same assembly keeps the
#: same number wherever it appears — including across the bridge.
def part_number(assembly: str) -> str:
    digits = "".join(character for character in assembly if character.isdigit()) or "0"
    letters = "".join(
        character for character in assembly if character.isalpha()
    ).upper()
    return f"{letters[:3]}-{digits[:4].zfill(4)}"


def _rng(machine: Machine, salt: str, seed: int) -> Random:
    return Random(f"{seed}:{machine.manufacturer}:{machine.order_key}:{salt}")


def _header(machine: Machine, manufacturer: Manufacturer) -> tuple[tuple[str, str], ...]:
    """The metadata block every machine document carries. The three codes that identify
    the machine are here and nowhere else in extractable form, which is what makes a
    document retrievable by structure rather than by topic."""
    return (
        ("Hersteller", manufacturer.name),
        ("Werk", manufacturer.city),
        ("Maschinentyp", f"{machine.type_code} — {machine.type_name}"),
        ("Modell", machine.model.code),
        ("Bestellschlüssel", machine.order_key),
        ("Konstruktion", f"{machine.designer.name}, {machine.designer.role}"),
    )


def _variant_table(machine: Machine) -> tuple[tuple[str, ...], ...]:
    rows = [
        (
            "Variante",
            "Ausstattung",
            "Leergewicht",
            "Gesamthöhe",
            "Auffangvolumen",
            "Schutzart",
        )
    ]
    for spec in variant_specs(machine):
        rows.append(
            (
                spec.code,
                spec.label,
                f"{spec.weight} kg",
                f"{spec.height} mm",
                f"{spec.oil_volume} l" if spec.oil_volume else "—",
                spec.protection,
            )
        )
    return tuple(rows)


def data_sheet(machine: Machine, manufacturer: Manufacturer, seed: int) -> Document:
    """The technical data sheet: base data plus one row per orderable variant."""
    rng = _rng(machine, "datenblatt", seed)
    model = machine.model
    anschluss = rng.choice((400, 400, 690))
    gewaehrleistung = rng.choice((12, 24, 24, 36))
    aufstellflaeche = model.footprint

    return Document(
        filename=f"datenblatt-{machine.order_key}.pdf",
        title=f"Technisches Datenblatt {machine.type_code} {model.code}",
        subtitle=f"{machine.type_name}, Bestellschlüssel {machine.order_key}",
        metadata=_header(machine, manufacturer),
        sections=(
            Section(
                "Einsatzbereich",
                (
                    f"Die Baureihe {machine.type_code} der {manufacturer.name} dient "
                    f"für {machine.purpose}. Das Modell {model.code} ist die "
                    f"{model.description}.",
                    f"Innerhalb der Baureihe {machine.type_code} stehen daneben die "
                    f"Modelle {', '.join(machine.siblings) or '— keine weiteren —'} zur "
                    f"Verfügung. Die Modelle sind in Aufstellmaß und Anschlussleistung "
                    f"nicht untereinander austauschbar.",
                ),
            ),
            Section(
                "Technische Grunddaten",
                (
                    f"Die nachfolgenden Werte gelten für die Grundausführung "
                    f"{machine.order_key}-00 ohne Zusatzausstattung. Abweichungen je "
                    f"Variante sind im Abschnitt 'Ausführungsvarianten' angegeben.",
                ),
                table=(
                    ("Merkmal", "Wert"),
                    ("Leergewicht Grundausführung", f"{model.base_weight} kg"),
                    ("Gesamthöhe Grundausführung", f"{model.base_height} mm"),
                    ("Aufstellfläche", aufstellflaeche),
                    ("Leistungsdaten", model.capacity),
                    ("Antrieb", model.drive),
                    ("Steuerung", model.control),
                    ("Schutzart", model.protection),
                    ("Emissionsschalldruckpegel", f"{model.noise} dB(A)"),
                    ("Netzanschluss", f"{anschluss} V, 50 Hz"),
                ),
            ),
            Section(
                "Ausführungsvarianten",
                (
                    f"Das Modell {model.code} wird in {len(model.variants)} "
                    f"Ausführungen geliefert. Leergewicht und Gesamthöhe ergeben sich "
                    f"additiv aus der Grundausführung und den gewählten Optionen.",
                ),
                table=_variant_table(machine),
            ),
            Section(
                "Aufstellbedingungen",
                (
                    f"Die Maschine ist auf ebenem, tragfähigem Untergrund aufzustellen. "
                    f"Für die Grundausführung {machine.order_key}-00 ist ein "
                    f"Fundament nach Aufstellplan vorzusehen.",
                    f"Die Gewährleistung beträgt {gewaehrleistung} Monate ab "
                    f"Inbetriebnahme. Ansprechpartner ist "
                    f"{manufacturer.contact.name} ({manufacturer.contact.role}).",
                ),
            ),
        ),
    )


def variant_overview(
    machine: Machine, manufacturer: Manufacturer, seed: int
) -> Document:
    """One paragraph per option and one per orderable variant.

    The option codes repeat across manufacturers with different meanings, so the text
    that resolves a code is only ever in this document, under this machine type.
    """
    model = machine.model
    options = options_for(machine)

    option_paragraphs = tuple(
        f"Option {option.code} — {option.name}: {option.description}. "
        f"{option.constraint}"
        for option in options
    )

    variant_paragraphs = []
    for variant in model.variants:
        spec = variant_spec(machine, variant)
        if not variant:
            variant_paragraphs.append(
                f"{spec.code} ist die Grundausführung ohne Zusatzausstattung. "
                f"Leergewicht {spec.weight} kg, Gesamthöhe {spec.height} mm, "
                f"Schutzart {spec.protection}."
            )
            continue
        enthaelt = " und ".join(
            f"{option.name} ({option.code})" for option in spec.options
        )
        variant_paragraphs.append(
            f"{spec.code} umfasst {enthaelt}. Gegenüber der Grundausführung steigt das "
            f"Leergewicht um {spec.weight - model.base_weight} kg auf {spec.weight} kg, "
            f"die Gesamthöhe um {spec.height - model.base_height} mm auf "
            f"{spec.height} mm. "
            + (
                f"Das Auffangvolumen beträgt {spec.oil_volume} l. "
                if spec.oil_volume
                else ""
            )
            + f"Die Schutzart der Variante ist {spec.protection}."
        )

    return Document(
        filename=f"ausstattungsvarianten-{machine.order_key}.pdf",
        title=f"Ausstattungsvarianten {machine.type_code} {model.code}",
        subtitle=f"Optionen und Bestellschlüssel, {machine.type_name}",
        metadata=_header(machine, manufacturer),
        sections=(
            Section(
                "Aufbau des Bestellschlüssels",
                (
                    f"Der Bestellschlüssel setzt sich aus Maschinentyp, Modell und den "
                    f"Kurzzeichen der gewählten Optionen zusammen, getrennt durch "
                    f"Schrägstriche: {machine.type_code}-{model.code}-Option/Option. "
                    f"Die Grundausführung trägt die Kennung 00.",
                    f"Die Kurzzeichen gelten ausschließlich innerhalb des "
                    f"Lieferprogramms der {manufacturer.name}. Gleiche Kurzzeichen "
                    f"anderer Hersteller bezeichnen andere Ausstattungen.",
                ),
            ),
            Section("Verfügbare Optionen", option_paragraphs),
            Section("Lieferbare Varianten", tuple(variant_paragraphs)),
            Section(
                "Nicht lieferbare Kombinationen",
                (
                    f"Kombinationen, die in der Variantenliste nicht aufgeführt sind, "
                    f"werden für das Modell {model.code} nicht freigegeben. "
                    f"Sonderausführungen bedürfen der Freigabe durch "
                    f"{machine.designer.name}.",
                ),
            ),
        ),
    )


def _origin_paragraphs(machine: Machine) -> tuple[str, ...]:
    """Where the assemblies came from — the section carrying the two bridges.

    Never names the other manufacturer: the assembly and the engineer are the whole
    join, and a sentence naming both sides would hand the answer to flat retrieval.
    """
    anchor = (machine.manufacturer, machine.type_code, machine.model.code)
    designer = machine.designer.name

    if anchor == BRIDGE_ORIGIN:
        return (
            f"Das {BRIDGE_ASSEMBLY} ist im {machine.order_key} verbaut und wurde "
            f"dafür von {BRIDGE_ENGINEER} konstruiert. Ausgelegt ist es auf ein "
            f"Abtriebsmoment von 2.400 Nm bei "
            f"einer Übersetzung von 1:31,5; die Lagerung ist für Dauerbetrieb bei "
            f"wechselnder Lastrichtung freigegeben.",
            "Die Baugruppe wurde nach der Erstabnahme als hausinterner Standard für "
            "Antriebe gleicher Momentenklasse übernommen. Übernommen werden die "
            "Flanschmaße, das Schmierstoffintervall von 4.000 Betriebsstunden und die "
            "Werkstoffpaarung der Planetenstufe.",
        )
    if anchor == BRIDGE_REUSE:
        return (
            f"Für den Kettenantrieb des {machine.order_key} wird das "
            f"{BRIDGE_ASSEMBLY} eingesetzt. "
            f"{BRIDGE_ENGINEER} brachte die Baugruppe aus einem früheren Einsatz in der "
            f"Verfahrenstechnik ein; Flanschmaße und Werkstoffpaarung der Planetenstufe "
            f"wurden unverändert übernommen.",
            "Angepasst wurde allein die Abtriebswelle: an die Stelle der Rührwerks"
            "welle tritt eine Steckwelle für das Kettenrad. Das Schmierstoffintervall "
            "von 4.000 Betriebsstunden gilt unverändert, sodass die "
            "Wartungsunterlagen weitgehend weiterverwendet werden konnten.",
        )
    if anchor == SECOND_BRIDGE_ORIGIN:
        return (
            f"Der {SECOND_BRIDGE_ASSEMBLY} ist im {machine.order_key} verbaut und "
            f"wurde im Rahmen der Baumusterprüfung durch {SECOND_BRIDGE_ENGINEER} "
            f"freigegeben. Der Spannsatz überträgt 1.150 Nm "
            f"reibschlüssig und ersetzt die zuvor verwendete Passfederverbindung.",
            f"Die Freigabe umfasst das Anzugsmoment von 59 Nm je Spannschraube sowie "
            f"die vorgeschriebene Nachkontrolle nach den ersten 50 Betriebsstunden. "
            f"Die Konstruktion verantwortet {designer}.",
        )
    if anchor == SECOND_BRIDGE_REUSE:
        return (
            f"An den Mischwellen des {machine.order_key} kommt der "
            f"{SECOND_BRIDGE_ASSEMBLY} zum Einsatz. "
            f"{SECOND_BRIDGE_ENGINEER} übertrug die Freigabe aus einer Prüfung an "
            f"einem Pressenantrieb auf diese Anwendung; Anzugsmoment und Nachkontrolle "
            f"nach 50 Betriebsstunden wurden unverändert übernommen.",
            f"Abweichend ist allein die Wellenschulter ausgeführt, die hier beidseitig "
            f"anliegt. Die Konstruktion verantwortet {designer}.",
        )
    return (
        f"Die im {machine.order_key} verbauten Baugruppen entsprechen dem "
        f"Konstruktionsstand der Baureihe {machine.type_code}. Verantwortlich für "
        f"Änderungen am Konstruktionsstand ist {designer}.",
    )


def maintenance_plan(
    machine: Machine, manufacturer: Manufacturer, seed: int
) -> Document:
    """Assemblies, spare part numbers and intervals — plus where an assembly came from."""
    rng = _rng(machine, "wartung", seed)
    model = machine.model
    intervals = ("täglich", "wöchentlich", "500 h", "2.000 h", "4.000 h")

    rows = [("Baugruppe", "Ersatzteilnummer", "Prüfintervall", "Tätigkeit")]
    for assembly in model.assemblies:
        rows.append(
            (
                assembly,
                part_number(assembly),
                rng.choice(intervals),
                rng.choice(
                    (
                        "Sichtprüfung auf Leckage",
                        "Schmierstoff wechseln",
                        "Verschleißmaß prüfen",
                        "Schraubverbindungen nachziehen",
                        "Funktionsprüfung der Verriegelung",
                    )
                ),
            )
        )

    oil_variants = [
        spec for spec in variant_specs(machine) if spec.oil_volume
    ]
    wanne = (
        f"Bei den Varianten {', '.join(spec.code for spec in oil_variants)} ist die "
        f"Auffangwanne vor jeder Reinigung zu entleeren."
        if oil_variants
        else "Eine Auffangwanne ist für dieses Modell nicht vorgesehen."
    )

    return Document(
        filename=f"wartungsplan-{machine.order_key}.pdf",
        title=f"Wartungsplan {machine.type_code} {model.code}",
        subtitle="Baugruppen, Ersatzteile und Prüfintervalle",
        metadata=_header(machine, manufacturer),
        sections=(
            Section(
                "Geltungsbereich",
                (
                    f"Dieser Wartungsplan gilt für alle Varianten des Modells "
                    f"{model.code} der Baureihe {machine.type_code}. Er ersetzt die "
                    f"Angaben der Betriebsanleitung nicht, sondern ergänzt sie um die "
                    f"baugruppenbezogenen Intervalle.",
                    f"Die Prüfungen sind von unterwiesenem Personal durchzuführen und "
                    f"im Maschinenbuch zu dokumentieren. Die Abnahme der Erstprüfung "
                    f"erfolgte durch {machine.inspector.name} "
                    f"({machine.inspector.role}).",
                ),
            ),
            Section(
                "Baugruppen und Ersatzteile",
                (
                    f"Die nachfolgend aufgeführten Baugruppen sind wartungsrelevant. "
                    f"Ersatzteile sind unter Angabe des Bestellschlüssels "
                    f"{machine.order_key} zu bestellen.",
                ),
                table=tuple(rows),
            ),
            Section("Konstruktive Herkunft der Baugruppen", _origin_paragraphs(machine)),
            Section(
                "Variantenbezogene Hinweise",
                (
                    wanne,
                    f"Abweichende Wartungsumfänge der Optionen "
                    f"{', '.join(model.options)} sind in den jeweiligen "
                    f"Ergänzungsblättern beschrieben. Für Rückfragen zur Konstruktion "
                    f"steht {machine.designer.name} zur Verfügung.",
                ),
            ),
        ),
    )
