"""German document text.

Status reports share heavy, near-identical boilerplate across every project and client:
same headings, same sentence frames, only names and numbers differ. That is what makes
dense retrieval genuinely confuse one client's report for another's rather than merely
being theoretically able to.

The longer project reports carry the entity-dense content — people, systems, deliverables,
risks — including the bridge passages that join the two same-named projects. Those
passages never name the other client.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from graph_rag.corpus.plan import (
    BRIDGE_ARTEFACT,
    BRIDGE_PROJECT,
    CONSULTANCY,
    SECOND_BRIDGE_CONSULTANT,
    SHARED_PROJECT,
    Client,
    Project,
)


@dataclass(frozen=True)
class Section:
    heading: str
    paragraphs: tuple[str, ...] = ()
    table: tuple[tuple[str, ...], ...] | None = None


@dataclass(frozen=True)
class Document:
    filename: str
    title: str
    subtitle: str
    metadata: tuple[tuple[str, str], ...]
    sections: tuple[Section, ...] = field(default=())


AMPEL = ("grün", "gelb", "gelb", "grün", "rot")


def _rng(project: Project, salt: str, seed: int) -> Random:
    return Random(f"{seed}:{project.client}:{project.year}:{project.name}:{salt}")


def _team_sentence(project: Project) -> str:
    members = ", ".join(person.name for person in project.team[:-1])
    return (
        f"Die Projektleitung liegt bei {project.lead.name} ({project.lead.role}), "
        f"{CONSULTANCY}. Im Team arbeiten {members}. "
        f"Ansprechpartner auf Kundenseite ist {project.team[-1].name} "
        f"({project.team[-1].role})."
    )


def status_report(project: Project, client: Client, quarter: int, seed: int) -> Document:
    """A quarterly status report. Deliberately formulaic."""
    rng = _rng(project, f"status-{quarter}", seed)
    ampel = rng.choice(AMPEL)
    fortschritt = rng.randrange(25, 95, 5)
    plan = fortschritt + rng.choice((-10, -5, 0, 5))
    budget = rng.randrange(120, 780, 10)
    verbraucht = round(budget * rng.uniform(0.3, 0.9))
    personentage = rng.randrange(40, 260, 5)
    offene_punkte = rng.randint(3, 17)

    system = project.systems[0]
    zweitsystem = project.systems[-1]

    return Document(
        filename=f"statusbericht-{project.year}-Q{quarter}.pdf",
        title=f"Statusbericht {project.name}",
        subtitle=f"Berichtszeitraum Q{quarter}/{project.year}",
        metadata=(
            ("Auftraggeber", client.name),
            ("Standort", client.city),
            ("Projekt", project.name),
            ("Berichtszeitraum", f"Q{quarter}/{project.year}"),
            ("Erstellt durch", CONSULTANCY),
            ("Projektleitung", project.lead.name),
        ),
        sections=(
            Section(
                "Zusammenfassung des Berichtszeitraums",
                (
                    "Der vorliegende Statusbericht fasst den Projektfortschritt im "
                    "Berichtszeitraum zusammen und richtet sich an den Lenkungsausschuss. "
                    "Die Berichterstattung erfolgt gemäß der im Projekthandbuch "
                    "vereinbarten Struktur.",
                    f"Das Projekt verfolgt {project.goal}. Der Gesamtstatus wird zum "
                    f"Stichtag mit der Ampelfarbe {ampel} bewertet. Der Fortschritt "
                    f"beträgt {fortschritt} Prozent gegenüber einem Planwert von "
                    f"{plan} Prozent.",
                    "Wesentliche Abweichungen gegenüber der letzten Berichterstattung "
                    "werden in Abschnitt 'Risiken und Gegenmaßnahmen' erläutert. "
                    "Änderungen am Leistungsumfang wurden im Berichtszeitraum nicht "
                    "beantragt.",
                ),
            ),
            Section(
                "Statusampel und Fortschritt",
                (
                    f"Die Teilprojekte wurden nach der üblichen Systematik bewertet. "
                    f"Der Arbeitsstrang rund um {system} liegt im Plan, der Arbeitsstrang "
                    f"rund um {zweitsystem} weist eine Verzögerung von wenigen Wochen auf.",
                ),
                table=(
                    ("Teilprojekt", "Status", "Fortschritt", "Meilenstein"),
                    ("Fachkonzept", ampel, f"{fortschritt} %", f"M2 Q{quarter}/{project.year}"),
                    ("Umsetzung", "gelb", f"{max(fortschritt - 20, 5)} %", f"M3 Q{quarter}/{project.year}"),
                    ("Test und Abnahme", "offen", "0 %", f"M4 Q{min(quarter + 1, 4)}/{project.year}"),
                ),
            ),
            Section(
                "Arbeitsergebnisse im Berichtszeitraum",
                (
                    f"Im Berichtszeitraum wurde an {project.deliverable} gearbeitet. "
                    f"Die Abstimmung mit den Fachbereichen von {client.name} erfolgte in "
                    f"zwei Workshops.",
                    f"Die Anbindung von {system} wurde im Testsystem erprobt. "
                    f"Offene Punkte werden in der Maßnahmenliste geführt; zum Stichtag "
                    f"sind {offene_punkte} Punkte offen.",
                ),
            ),
            Section(
                "Risiken und Gegenmaßnahmen",
                (
                    f"Als wesentliches Risiko werden weiterhin {project.risk} geführt. "
                    f"Als Gegenmaßnahme wurde ein wöchentlicher Abstimmungstermin "
                    f"eingerichtet.",
                    "Weitere Risiken sind unveraendert und werden im Risikoregister "
                    "nachgehalten. Eine Eskalation an den Lenkungsausschuss war im "
                    "Berichtszeitraum nicht erforderlich.",
                ),
            ),
            Section(
                "Mittelabfluss und Aufwand",
                (
                    f"Das genehmigte Budget beträgt {budget} TEUR. Davon sind zum "
                    f"Stichtag {verbraucht} TEUR verbraucht. Im Berichtszeitraum wurden "
                    f"{personentage} Personentage geleistet.",
                    "Der Mittelabfluss entspricht im Wesentlichen der Planung. Eine "
                    "Nachsteuerung ist derzeit nicht vorgesehen.",
                ),
            ),
            Section(
                "Nächste Schritte",
                (
                    f"Im kommenden Berichtszeitraum steht die Fertigstellung von "
                    f"{project.deliverable} im Vordergrund. Anschließend beginnt die "
                    f"Abstimmung der Testfälle.",
                    _team_sentence(project),
                ),
            ),
        ),
    )


# The bridge. Two clients, one shared project name, and a single consultant who carried a
# reference architecture from the first engagement to the second. Neither text names the
# other client — the join exists only as a shared entity, which is exactly what the graph
# is supposed to recover and flat retrieval cannot.
BRIDGE_ORIGIN_CLIENT = "Nordlicht Energie AG"
BRIDGE_REUSE_CLIENT = "Brandt & Söhne Maschinenbau GmbH"
SECOND_BRIDGE_ORIGIN_CLIENT = "Sanitas Klinikverbund gGmbH"
SECOND_BRIDGE_REUSE_CLIENT = "Finanzkontor Weser AG"
SECOND_BRIDGE_ARTEFACT = "Prüfschema Berechtigungsrollen"


def _method_paragraphs(project: Project) -> tuple[str, ...]:
    """Methodology section — where the cross-client bridges live."""
    lead = project.lead.name
    if project.name == SHARED_PROJECT and project.client == BRIDGE_ORIGIN_CLIENT:
        return (
            f"Methodisch wurde das Vorgehen von {lead} verantwortet. Im Projektverlauf "
            f"entstand die {BRIDGE_ARTEFACT}, die den Ereignisstrom als "
            f"Integrationsrückgrat beschreibt und die Entkopplung der Bestandssysteme "
            f"über Apache Kafka regelt.",
            f"Die {BRIDGE_ARTEFACT} wurde nach der Abnahme in der Fassung 2 "
            f"verabschiedet und anschließend von {CONSULTANCY} als interner Standard "
            f"für vergleichbare Integrationsvorhaben übernommen. Die Musterbeschreibung "
            f"umfasst die Schnittstellenverträge, das Fehlerbehandlungsmuster und die "
            f"Namenskonvention der Ereignistypen.",
        )
    if project.name == SHARED_PROJECT and project.client == BRIDGE_REUSE_CLIENT:
        return (
            f"Als methodische Grundlage diente die {BRIDGE_ARTEFACT} in der Fassung 2. "
            f"{lead} brachte das Muster aus einem früheren Einsatz in der "
            f"Energiewirtschaft ein; die dort erprobten Schnittstellenverträge und die "
            f"Namenskonvention der Ereignistypen wurden unverändert übernommen.",
            "Angepasst wurden lediglich die Fachobjekte: an die Stelle der "
            "Zählpunktdaten treten Stücklisten und Baureihenstämme. Das "
            "Fehlerbehandlungsmuster über Apache Kafka blieb unverändert, sodass die "
            "Betriebsdokumentation weitgehend wiederverwendet werden konnte.",
        )
    if project.name == BRIDGE_PROJECT and project.client == SECOND_BRIDGE_ORIGIN_CLIENT:
        return (
            f"Die datenschutzrechtliche Bewertung verantwortete {SECOND_BRIDGE_CONSULTANT}. "
            f"Im Zuge der Rollenmodellierung entstand das {SECOND_BRIDGE_ARTEFACT}, mit "
            f"dem Berechtigungsrollen gegen den Zweckbindungsgrundsatz geprüft werden.",
            "Das Prüfschema unterscheidet zwischen Rollen mit Vollzugriff, Rollen mit "
            "fachlich begrenztem Zugriff und Leserollen für Auswertungszwecke.",
        )
    if project.name == BRIDGE_PROJECT and project.client == SECOND_BRIDGE_REUSE_CLIENT:
        return (
            f"{SECOND_BRIDGE_CONSULTANT} führte das {SECOND_BRIDGE_ARTEFACT} in das "
            f"Projekt ein. Die Kategorien des Schemas wurden auf die Rollenlandschaft "
            f"der Bestandssysteme übertragen und um aufsichtsrechtliche "
            f"Aufbewahrungsfristen ergänzt.",
            "Die Prüfung ergab, dass ein Teil der Altrollen fachlich nicht mehr "
            "begründbar ist und im Rahmen des Löschkonzepts zurückgebaut wird.",
        )
    return (
        f"Das Vorgehen folgte dem Standardmodell von {CONSULTANCY}: Aufnahme der "
        f"Ausgangslage, Zielbild, Umsetzungsplanung und Abnahme. {lead} verantwortete "
        f"die fachliche Steuerung.",
    )


def project_report(project: Project, client: Client, seed: int) -> Document:
    """The entity-dense report for a project: people, systems, deliverables, risks."""
    rng = _rng(project, "bericht", seed)
    aufwand = rng.randrange(180, 900, 10)
    dauer = rng.randint(6, 18)
    systems = ", ".join(project.systems)

    return Document(
        filename=f"projektbericht-{project.year}.pdf",
        title=f"Projektbericht {project.name}",
        subtitle=f"{client.name}, Geschäftsjahr {project.year}",
        metadata=(
            ("Auftraggeber", client.name),
            ("Branche", client.sector),
            ("Standort", client.city),
            ("Projekt", project.name),
            ("Geschäftsjahr", str(project.year)),
            ("Projektleitung", f"{project.lead.name}, {CONSULTANCY}"),
        ),
        sections=(
            Section(
                "Ausgangslage",
                (
                    f"{client.name} mit Sitz in {client.city} ist in der Branche "
                    f"{client.sector} tätig. Gegenstand des Projekts ist "
                    f"{project.topic}.",
                    f"Die bestehende Systemlandschaft ist über Jahre gewachsen. "
                    f"Schnittstellen wurden punktuell ergänzt und sind heute nur "
                    f"teilweise dokumentiert. Das Projekt verfolgt {project.goal}.",
                ),
            ),
            Section(
                "Vorgehen",
                (
                    f"Das Projekt wurde über {dauer} Monate durchgeführt. Der "
                    f"Gesamtaufwand beläuft sich auf {aufwand} TEUR.",
                    _team_sentence(project),
                ),
            ),
            Section("Methodisches Vorgehen", _method_paragraphs(project)),
            Section(
                "Eingesetzte Systeme und Schnittstellen",
                (
                    f"Im Projekt wurden folgende Systeme betrachtet: {systems}. Die "
                    f"Schnittstellen wurden in einem Schnittstellenverzeichnis erfasst "
                    f"und je Fachobjekt beschrieben.",
                ),
            ),
            Section(
                "Ergebnisse",
                (
                    f"Als zentrales Arbeitsergebnis liegt {project.deliverable} vor. "
                    f"Das Dokument wurde von {client.contact.name} "
                    f"({client.contact.role}) abgenommen.",
                    f"Als offenes Risiko bestehen weiterhin {project.risk}. Eine "
                    f"abschließende Bewertung erfolgt im Folgeprojekt.",
                ),
            ),
            Section(
                "Empfehlungen",
                (
                    f"{CONSULTANCY} empfiehlt, die im Projekt erarbeiteten Muster in "
                    f"den Regelbetrieb zu überführen und die Dokumentation jährlich zu "
                    f"überprüfen.",
                ),
            ),
        ),
    )
