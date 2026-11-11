"""The deterministic plan for the synthetic corpus.

This is the experimental apparatus, not set dressing: the thesis under test is that graph
RAG wins on nested, self-similar data, so the corpus is built to be hostile to flat
retrieval on purpose. Two properties matter more than realism:

1. A project name is reused by two unrelated clients, with genuinely different work behind
   it, so a naively phrased question retrieves the wrong client's report.
2. The only thing joining those two projects is a *shared entity* — a consultant and the
   reference architecture she carried across — and **no document names the other client**.
   A single chunk therefore never contains the connection; only the graph does. If the
   generator mentioned both clients in one passage, flat retrieval would answer the demo
   question and the demo would prove nothing.

Everything here is derived from a seed so a demo is reproducible and downstream tickets can
assert against known content.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

YEARS = (2022, 2023, 2024)

# The consultancy writing the reports. Appears in every document, so it is a useless
# discriminator — which is the point.
CONSULTANCY = "Vektor Consulting GmbH"

# The planted collision.
SHARED_PROJECT = "Projekt Atlas"
BRIDGE_CONSULTANT = "Dr. Katharina Vogt"
BRIDGE_ARTEFACT = "Atlas-Referenzarchitektur"
SECOND_BRIDGE_CONSULTANT = "Milan Horvat"


@dataclass(frozen=True)
class Person:
    name: str
    role: str


@dataclass(frozen=True)
class Project:
    name: str
    client: str
    year: int
    topic: str
    goal: str
    systems: tuple[str, ...]
    deliverable: str
    risk: str
    lead: Person
    team: tuple[Person, ...]
    quarters: tuple[int, ...]

    @property
    def people(self) -> tuple[Person, ...]:
        return (self.lead, *self.team)


@dataclass(frozen=True)
class Client:
    name: str
    # Folder names stay ASCII while the documents themselves are fully German. Corporate
    # file shares usually look like this, and it keeps the on-disk name deliberately
    # different from the legal name the text uses.
    folder: str
    sector: str
    city: str
    contact: Person


CLIENTS = (
    Client(
        name="Nordlicht Energie AG",
        folder="Nordlicht Energie AG",
        sector="Energieversorgung",
        city="Hamburg",
        contact=Person("Henrike Baumann", "Leiterin Digitalisierung"),
    ),
    Client(
        name="Brandt & Söhne Maschinenbau GmbH",
        folder="Brandt und Soehne Maschinenbau",
        sector="Maschinenbau",
        city="Bielefeld",
        contact=Person("Gerald Brandt", "Geschäftsführer"),
    ),
    Client(
        name="Sanitas Klinikverbund gGmbH",
        folder="Sanitas Klinikverbund",
        sector="Gesundheitswesen",
        city="Leipzig",
        contact=Person("Dr. med. Petra Ziegler", "Ärztliche Direktorin"),
    ),
    Client(
        name="Finanzkontor Weser AG",
        folder="Finanzkontor Weser AG",
        sector="Finanzdienstleistungen",
        city="Bremen",
        contact=Person("Ansgar Thiele", "Bereichsleiter Operations"),
    ),
)

CONSULTANTS = (
    Person(BRIDGE_CONSULTANT, "Principal Architektin"),
    Person(SECOND_BRIDGE_CONSULTANT, "Senior Berater Datenschutz"),
    Person("Sophie Lehmann", "Projektleiterin"),
    Person("Jens Petersen", "Senior Berater Prozesse"),
    Person("Dr. Annika Roth", "Fachexpertin Datenmanagement"),
    Person("Tobias Kern", "Berater Integration"),
    Person("Miriam Falk", "Projektleiterin"),
    Person("Luis Arndt", "Berater Testmanagement"),
)


@dataclass(frozen=True)
class ProjectTemplate:
    name: str
    topic: str
    goal: str
    systems: tuple[str, ...]
    deliverable: str
    risk: str


# Per-client project catalogues. Each client's work is plausible for its sector, so the
# documents differ in *topic* while remaining near-identical in *form*.
PROJECT_CATALOGUE: dict[str, tuple[ProjectTemplate, ...]] = {
    "Nordlicht Energie AG": (
        ProjectTemplate(
            name=SHARED_PROJECT,
            topic="Migration der Netzleittechnik in eine hybride Cloud-Umgebung",
            goal="die Ablösung der historisch gewachsenen Leitsystem-Schnittstellen durch "
            "eine ereignisgetriebene Integrationsschicht",
            systems=("SAP IS-U", "Apache Kafka", "Azure Data Factory"),
            deliverable="Zielarchitektur-Dokument Netzleittechnik",
            risk="Verzögerungen bei der Freigabe durch die Netzsicherheitsabteilung",
        ),
        ProjectTemplate(
            name="Projekt Windfeld",
            topic="Einführung eines Prognosemodells für die Einspeisung aus Windparks",
            goal="eine belastbare Tagesprognose je Netzgebiet",
            systems=("Snowflake", "Python", "Grafana"),
            deliverable="Prognosemodell mit Backtesting-Bericht",
            risk="unvollständige historische Messreihen aus den Umspannwerken",
        ),
        ProjectTemplate(
            name="Projekt Kompass",
            topic="Neuausrichtung des Kundenservice-Prozesses",
            goal="eine durchgängige Fallbearbeitung vom Erstkontakt bis zur Abrechnung",
            systems=("Salesforce Service Cloud", "SAP IS-U"),
            deliverable="Prozesshandbuch Kundenservice",
            risk="fehlende Kapazität der Fachbereiche für Abstimmungstermine",
        ),
        ProjectTemplate(
            name="Projekt Zaehlerwerk",
            topic="Rollout intelligenter Messsysteme im Netzgebiet Nord",
            goal="die Anbindung von 120.000 modernen Messeinrichtungen",
            systems=("SAP IS-U", "Gateway-Administration", "Apache Kafka"),
            deliverable="Rollout-Plan mit Mengengerüst",
            risk="Lieferengpässe bei den Smart-Meter-Gateways",
        ),
    ),
    "Brandt & Söhne Maschinenbau GmbH": (
        ProjectTemplate(
            name=SHARED_PROJECT,
            topic="Konsolidierung der Produktdaten aus vier Werken in ein zentrales PDM",
            goal="einen einheitlichen Produktdatenstamm über alle Baureihen hinweg",
            systems=("PTC Windchill", "SAP S/4HANA", "Apache Kafka"),
            deliverable="Datenmigrationskonzept Produktstamm",
            risk="abweichende Stücklistenlogik zwischen den Werken",
        ),
        ProjectTemplate(
            name="Projekt Taktzeit",
            topic="Optimierung der Montagelinie für die Baureihe BX",
            goal="eine Reduktion der Durchlaufzeit um fünfzehn Prozent",
            systems=("MES Hydra", "SAP S/4HANA"),
            deliverable="Maßnahmenkatalog Montagelinie",
            risk="Stillstandszeiten während der Umbauphase",
        ),
        ProjectTemplate(
            name="Projekt Serviceportal",
            topic="Aufbau eines Ersatzteilportals für Bestandskunden",
            goal="die Selbstbedienung bei Ersatzteilbestellungen",
            systems=("SAP S/4HANA", "Azure App Service"),
            deliverable="Fachkonzept Ersatzteilportal",
            risk="unklare Preisfindungslogik für Altanlagen",
        ),
    ),
    "Sanitas Klinikverbund gGmbH": (
        ProjectTemplate(
            name="Projekt Vitalis",
            topic="Harmonisierung der Patientendokumentation über sechs Standorte",
            goal="eine standortübergreifend einheitliche Fallakte",
            systems=("Agfa Orbis", "HL7 FHIR"),
            deliverable="Dokumentationsrichtlinie Fallakte",
            risk="divergierende Dokumentationsgewohnheiten der Fachabteilungen",
        ),
        ProjectTemplate(
            name="Projekt Triage",
            topic="Neugestaltung der Notaufnahme-Steuerung",
            goal="eine messbar verkürzte Wartezeit bis zur Ersteinschätzung",
            systems=("Agfa Orbis", "Dashboards"),
            deliverable="Steuerungskonzept Notaufnahme",
            risk="Personalengpässe im Pflegedienst",
        ),
        ProjectTemplate(
            name="Projekt Datenschild",
            topic="Umsetzung der Anforderungen aus dem Patientendatenschutz",
            goal="ein belastbares Berechtigungskonzept für die Fallakte",
            systems=("Agfa Orbis", "Active Directory"),
            deliverable="Berechtigungskonzept Patientendaten",
            risk="unklare Zuständigkeit für die Rollenpflege",
        ),
        ProjectTemplate(
            name="Projekt Logistik Plus",
            topic="Zentralisierung der Material- und Apothekenlogistik",
            goal="eine verbrauchsgesteuerte Versorgung der Stationen",
            systems=("SAP S/4HANA", "Lagerverwaltung"),
            deliverable="Logistikkonzept Zentrallager",
            risk="begrenzte Lagerfläche am Standort Leipzig",
        ),
    ),
    "Finanzkontor Weser AG": (
        ProjectTemplate(
            name="Projekt Kontrapunkt",
            topic="Ablösung der Altanwendung für die Kontoeröffnung",
            goal="eine vollständig digitale Kontoeröffnungsstrecke",
            systems=("Avaloq", "Azure App Service"),
            deliverable="Fachkonzept Kontoeröffnung",
            risk="Abhängigkeit von einem externen Identifizierungsdienst",
        ),
        ProjectTemplate(
            name="Projekt Pruefstand",
            topic="Automatisierung der aufsichtsrechtlichen Meldewege",
            goal="eine nachvollziehbare und fristgerechte Meldestrecke",
            systems=("Avaloq", "Snowflake"),
            deliverable="Meldekonzept Aufsichtsrecht",
            risk="kurzfristige Änderungen der aufsichtsrechtlichen Vorgaben",
        ),
        ProjectTemplate(
            name="Projekt Datenschild",
            topic="Überarbeitung des Löschkonzepts für Kundendaten",
            goal="ein revisionssicheres Löschkonzept über alle Bestandssysteme",
            systems=("Avaloq", "Active Directory"),
            deliverable="Löschkonzept Kundendaten",
            risk="fehlende Dokumentation der Altbestände",
        ),
    ),
}


BRIDGE_PROJECT = "Projekt Datenschild"

# Everyone who may be staffed freely. The two bridge consultants are excluded: they are
# placed deliberately, and only on the projects they bridge.
ORDINARY_CONSULTANTS = tuple(
    person
    for person in CONSULTANTS
    if person.name not in (BRIDGE_CONSULTANT, SECOND_BRIDGE_CONSULTANT)
)

# Quarters that get a status report. Two per project keeps the boilerplate collisions
# dense without doubling ingestion cost.
REPORTED_QUARTERS = (2, 4)


def _assign_people(
    rng: Random, template: ProjectTemplate, client: Client
) -> tuple[Person, tuple[Person, ...]]:
    """Staff a project, honouring the two planted cross-client bridges.

    The bridges are the only thing joining two same-named projects of different clients,
    so they are placed deliberately rather than left to the seed.
    """
    bridge = next(p for p in CONSULTANTS if p.name == BRIDGE_CONSULTANT)
    second_bridge = next(p for p in CONSULTANTS if p.name == SECOND_BRIDGE_CONSULTANT)

    if template.name == SHARED_PROJECT:
        lead = bridge
    elif template.name == BRIDGE_PROJECT:
        lead = second_bridge
    else:
        lead = rng.choice(list(ORDINARY_CONSULTANTS))

    # A bridge consultant must appear on their bridge project and nowhere else. Staffing
    # them onto unrelated projects would make them a hub touching every client, so
    # "which projects did she work on?" would answer with half the corpus and a traversal
    # through her would reach everywhere — destroying the very connection she exists to
    # carry.
    pool = [p for p in ORDINARY_CONSULTANTS if p.name != lead.name]
    team = tuple(rng.sample(pool, k=2)) + (client.contact,)
    return lead, team


def build_plan(seed: int) -> list[Project]:
    """Every project the corpus contains, in a stable order."""
    rng = Random(seed)
    projects = []
    for client in CLIENTS:
        catalogue = PROJECT_CATALOGUE[client.name]
        for year in YEARS:
            # 3 or 4 projects per client-year; the collision templates are never dropped.
            count = rng.choice((3, 4)) if len(catalogue) > 3 else 3
            selected = catalogue[: min(count, len(catalogue))]
            for template in selected:
                lead, team = _assign_people(rng, template, client)
                projects.append(
                    Project(
                        name=template.name,
                        client=client.name,
                        year=year,
                        topic=template.topic,
                        goal=template.goal,
                        systems=template.systems,
                        deliverable=template.deliverable,
                        risk=template.risk,
                        lead=lead,
                        team=team,
                        quarters=REPORTED_QUARTERS,
                    )
                )
    return projects


def client_by_name(name: str) -> Client:
    return next(client for client in CLIENTS if client.name == name)
