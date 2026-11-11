"""Generate the corpus tree and the note describing what was planted in it."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree

from pypdf import PdfReader

from graph_rag.corpus.machine_plan import (
    BRIDGE_ASSEMBLY,
    BRIDGE_ENGINEER,
    BRIDGE_ORIGIN,
    BRIDGE_REUSE,
    MANUFACTURERS,
    SECOND_BRIDGE_ASSEMBLY,
    SECOND_BRIDGE_ENGINEER,
    SECOND_BRIDGE_ORIGIN,
    SECOND_BRIDGE_REUSE,
    SHARED_MODEL_CODE,
    SHARED_OPTION_CODE,
    SHARED_TYPE_CODE,
    Machine,
    build_machine_plan,
    machine_by,
    manufacturer_by_name,
    option_by_code,
    variant_spec,
)
from graph_rag.corpus.machine_text import (
    data_sheet,
    maintenance_plan,
    variant_overview,
)
from graph_rag.corpus.pdf import render
from graph_rag.corpus.plan import (
    CLIENTS,
    BRIDGE_ARTEFACT,
    BRIDGE_CONSULTANT,
    SECOND_BRIDGE_CONSULTANT,
    SHARED_PROJECT,
    Project,
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

NOTE_FILENAME = "KOLLISIONEN.md"


@dataclass(frozen=True)
class CorpusReport:
    root: Path
    documents: list[Path]
    machine_documents: list[Path]
    note: Path
    verified_file: Path
    verified_characters: int

    @property
    def project_count(self) -> int:
        return len({path.parent for path in self.documents})

    @property
    def model_count(self) -> int:
        return len({path.parent for path in self.machine_documents})

    @property
    def all_documents(self) -> list[Path]:
        return [*self.documents, *self.machine_documents]


def _project_directory(root: Path, project: Project) -> Path:
    client = client_by_name(project.client)
    return root / client.folder / str(project.year) / project.name


def _documents(project: Project, seed: int):
    client = client_by_name(project.client)
    yield project_report(project, client, seed)
    for quarter in project.quarters:
        yield status_report(project, client, quarter, seed)


def _machine_directory(root: Path, machine: Machine) -> Path:
    """`<Hersteller>/<Maschinentyp>/<Modell>` — one level deeper than the client half,
    which is the point: the hierarchy is built from the walk, so nothing downstream may
    assume the three levels of `Kunde / Jahr / Projekt`."""
    manufacturer = manufacturer_by_name(machine.manufacturer)
    return root / manufacturer.folder / machine.type_code / machine.model.code


def _machine_documents(machine: Machine, seed: int):
    manufacturer = manufacturer_by_name(machine.manufacturer)
    yield data_sheet(machine, manufacturer, seed)
    yield variant_overview(machine, manufacturer, seed)
    yield maintenance_plan(machine, manufacturer, seed)


def _verify_text_layer(path: Path) -> int:
    """Confirm the PDF is born-digital. OCR is out of scope, so a missing text layer
    would break every downstream ticket silently rather than loudly."""
    reader = PdfReader(str(path))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if len(text.strip()) < 200:
        raise RuntimeError(
            f"{path} has no usable text layer ({len(text.strip())} characters extracted)"
        )
    return len(text)


class CorpusExists(RuntimeError):
    """Raised rather than silently writing into a tree that already holds a corpus."""


def _owned_paths(root: Path) -> list[Path]:
    """What a previous run of *this* generator would have written — never anything else,
    so pointing the generator at the wrong directory cannot delete a user's files."""
    return (
        [root / client.folder for client in CLIENTS]
        + [root / manufacturer.folder for manufacturer in MANUFACTURERS]
        + [root / NOTE_FILENAME]
    )


def _clear(root: Path) -> None:
    for path in _owned_paths(root):
        if path.is_dir():
            rmtree(path)
        elif path.exists():
            path.unlink()


def generate_corpus(root: Path, seed: int, force: bool = False) -> CorpusReport:
    """Write the full corpus tree under `root`. Deterministic for a given seed.

    A different seed selects a different set of projects, so writing over an existing
    corpus would leave orphaned folders behind — ingestion walks the directory, not the
    plan, and would happily ingest documents belonging to no corpus.
    """
    existing = [path for path in _owned_paths(root) if path.exists()]
    if existing and not force:
        raise CorpusExists(
            f"{root} already contains a generated corpus. Re-run with --force to replace it."
        )
    if existing:
        _clear(root)

    written = []
    for project in build_plan(seed):
        directory = _project_directory(root, project)
        for document in _documents(project, seed):
            written.append(render(document, directory / document.filename))

    machine_written = []
    for machine in build_machine_plan(seed):
        directory = _machine_directory(root, machine)
        for document in _machine_documents(machine, seed):
            machine_written.append(render(document, directory / document.filename))

    note = root / NOTE_FILENAME
    note.write_text(_note(seed), encoding="utf-8")

    sample = written[0]
    characters = _verify_text_layer(sample)
    return CorpusReport(
        root=root,
        documents=written,
        machine_documents=machine_written,
        note=note,
        verified_file=sample,
        verified_characters=characters,
    )


def machine_by_anchor(anchor: tuple[str, str, str], seed: int) -> Machine:
    return machine_by(*anchor, seed)


def _note(seed: int) -> str:
    origin = machine_by_anchor(BRIDGE_ORIGIN, seed)
    reuse = machine_by_anchor(BRIDGE_REUSE, seed)
    second_origin = machine_by_anchor(SECOND_BRIDGE_ORIGIN, seed)
    second_reuse = machine_by_anchor(SECOND_BRIDGE_REUSE, seed)
    colliding = machine_by_anchor(("Pleickner Werke KG", "ZM3K", SHARED_MODEL_CODE), seed)
    example = variant_spec(origin, ("FW", "OW"))

    BRIDGE_ORIGIN_MANUFACTURER = origin.manufacturer
    BRIDGE_REUSE_MANUFACTURER = reuse.manufacturer
    SECOND_BRIDGE_ORIGIN_MANUFACTURER = second_origin.manufacturer
    SECOND_BRIDGE_REUSE_MANUFACTURER = second_reuse.manufacturer
    BRIDGE_ORIGIN_KEY = origin.order_key
    BRIDGE_REUSE_KEY = reuse.order_key
    SECOND_BRIDGE_ORIGIN_KEY = second_origin.order_key
    SECOND_BRIDGE_REUSE_KEY = second_reuse.order_key
    ORIGIN_BASE_WEIGHT = origin.model.base_weight
    COLLIDING_BASE_WEIGHT = colliding.model.base_weight
    HK_HAGEDORN = option_by_code(origin.manufacturer, SHARED_OPTION_CODE).weight_delta
    HK_PLEICKNER = option_by_code(
        second_origin.manufacturer, SHARED_OPTION_CODE
    ).weight_delta
    HK_OSTMARK = option_by_code(reuse.manufacturer, SHARED_OPTION_CODE).weight_delta
    EXAMPLE_VARIANT = example.code
    EXAMPLE_BASE = origin.model.base_weight
    EXAMPLE_DELTA = example.weight - origin.model.base_weight
    EXAMPLE_WEIGHT = example.weight
    EXAMPLE_HEIGHT = example.height

    return f"""# Geplante Kollisionen im synthetischen Korpus

Dieses Verzeichnis wird von `graph-rag generate-corpus` erzeugt. Der Korpus ist absichtlich
feindlich gegenüber flacher Retrieval-Logik gebaut. Was geplant wurde und wo es liegt:

## 1. Gleicher Projektname bei zwei verschiedenen Kunden

`{SHARED_PROJECT}` existiert bei **{BRIDGE_ORIGIN_CLIENT}** und bei
**{BRIDGE_REUSE_CLIENT}** — in allen Jahren, mit völlig unterschiedlichem Inhalt
(Netzleittechnik gegenüber Produktdatenmanagement).

## 2. Die Brücke zwischen beiden

{BRIDGE_CONSULTANT} leitet beide Projekte. Die bei {BRIDGE_ORIGIN_CLIENT} entstandene
**{BRIDGE_ARTEFACT}** (Fassung 2) wird bei {BRIDGE_REUSE_CLIENT} als methodische Grundlage
wiederverwendet; übernommen wurden Schnittstellenverträge, Namenskonvention der
Ereignistypen und das Fehlerbehandlungsmuster über Apache Kafka.

**Kein Dokument nennt den jeweils anderen Kunden.** Die Verbindung existiert nur über die
gemeinsame Person und das gemeinsame Artefakt — also nur im Graphen. Ein einzelner Chunk
enthält sie nie. Genau das soll die Demo zeigen.

Zu finden in: `*/{{Jahr}}/{SHARED_PROJECT}/projektbericht-*.pdf`, Abschnitt
„Methodisches Vorgehen".

## 3. Zweite, schwächere Brücke

`Projekt Datenschild` existiert bei **{SECOND_BRIDGE_ORIGIN_CLIENT}** und
**{SECOND_BRIDGE_REUSE_CLIENT}**. {SECOND_BRIDGE_CONSULTANT} trägt das
**{SECOND_BRIDGE_ARTEFACT}** vom einen zum anderen. Nützlich, um zu zeigen, dass der erste
Treffer kein Einzelfall ist.

## 4. Nahezu identischer Boilerplate

Jeder Statusbericht benutzt dieselben Überschriften und dieselben Satzgerüste; es ändern
sich nur Namen und Zahlen. Zwei Statusberichte pro Projekt und Jahr. Dense Retrieval
verwechselt sie zuverlässig.

## Fragen für die Demo

- „Was verbindet das Projekt Atlas von {BRIDGE_ORIGIN_CLIENT} mit dem Projekt Atlas von
  {BRIDGE_REUSE_CLIENT}?" — die Zielfrage.
- „Woran hat {BRIDGE_CONSULTANT} gearbeitet?" — zeigt den kundenübergreifenden Knoten.
- „Wie war der Status von Projekt Atlas in Q2?" — naiv formuliert, trifft flach mit hoher
  Wahrscheinlichkeit den falschen Kunden.

---

# Zweiter Teilkorpus: Maschinen

Ablage: `<Hersteller>/<Maschinentyp>/<Modell>/` — eine Ebene tiefer als der Kundenteil.
Je Modell liegen drei Dokumente: `datenblatt-*.pdf`, `ausstattungsvarianten-*.pdf` und
`wartungsplan-*.pdf`.

Der Kundenteil ist feindlich, weil **Namen kollidieren**. Der Maschinenteil ist feindlich,
weil **Namen nichts bedeuten**: `{SHARED_TYPE_CODE}`, `{SHARED_MODEL_CODE}`, `HX-114` sind
undurchsichtige Codes. Die Einbettung einer Frage liegt nirgends in der Nähe des richtigen
Dokuments, und ein lexikalischer Treffer auf `{SHARED_MODEL_CODE}` ist wertlos, weil
derselbe Code zwei nicht verwandte Maschinen bezeichnet. Nur die Struktur — Hersteller,
Maschinentyp, Modell, Variante — macht den Code eindeutig.

## 5. Gleicher Modellcode unter zwei Maschinentypen

`{SHARED_MODEL_CODE}` ist ein Modell des {SHARED_TYPE_CODE} ({BRIDGE_ORIGIN_MANUFACTURER},
Vakuum-Mischtrockner, {ORIGIN_BASE_WEIGHT} kg Leergewicht) **und** ein Modell des ZM3K
(Pleickner Werke KG, Zahnkranz-Montagezelle, {COLLIDING_BASE_WEIGHT} kg). Eine Frage nach
„Modell {SHARED_MODEL_CODE}" ohne Baureihe ist nicht beantwortbar — die Differenz der
Leergewichte macht eine Antwort aus dem falschen Zweig sofort erkennbar.

## 6. Gleicher Maschinentyp bei zwei Herstellern

`{SHARED_TYPE_CODE}` ist bei {BRIDGE_ORIGIN_MANUFACTURER} ein Vakuum-Mischtrockner und bei
{BRIDGE_REUSE_MANUFACTURER} ein Rollenbahn-Umsetzer. Zwei Ordner gleichen Namens unter
verschiedenen Herstellern, inhaltlich ohne jede Berührung.

## 7. Gleicher Optionscode, andere Bedeutung

`{SHARED_OPTION_CODE}` heißt bei {BRIDGE_ORIGIN_MANUFACTURER} **Heizkreis**
(+{HK_HAGEDORN} kg) und bei {SECOND_BRIDGE_ORIGIN_MANUFACTURER} sowie
{BRIDGE_REUSE_MANUFACTURER} **Hubkonsole** (+{HK_PLEICKNER} kg bzw. +{HK_OSTMARK} kg).
Aufgelöst wird der Code nur im Dokument `ausstattungsvarianten-*.pdf` des jeweiligen
Modells.

## 8. Dritte Brücke: eine Baugruppe über zwei Hersteller

Das **{BRIDGE_ASSEMBLY}** steckt im {BRIDGE_ORIGIN_KEY} von {BRIDGE_ORIGIN_MANUFACTURER}
und im {BRIDGE_REUSE_KEY} von {BRIDGE_REUSE_MANUFACTURER}. {BRIDGE_ENGINEER} hat es
konstruiert und in die zweite Maschine übernommen. **Kein Dokument nennt den jeweils
anderen Hersteller** — die Verbindung existiert nur über Baugruppe und Person, also nur im
Graphen. Zu finden im Abschnitt „Konstruktive Herkunft der Baugruppen".

## 9. Vierte, schwächere Brücke

Der **{SECOND_BRIDGE_ASSEMBLY}** verbindet den {SECOND_BRIDGE_ORIGIN_KEY} von
{SECOND_BRIDGE_ORIGIN_MANUFACTURER} mit dem {SECOND_BRIDGE_REUSE_KEY} von
{SECOND_BRIDGE_REUSE_MANUFACTURER}; getragen von {SECOND_BRIDGE_ENGINEER}.

## 10. Varianten: die Antwort steht nur in einer Zeile

Leergewicht und Gesamthöhe einer Variante ergeben sich **additiv** aus der
Grundausführung und den Optionen. Beispiel: {EXAMPLE_VARIANT} wiegt
{EXAMPLE_BASE} kg + {EXAMPLE_DELTA} kg = **{EXAMPLE_WEIGHT} kg** bei einer Gesamthöhe von
{EXAMPLE_HEIGHT} mm. Dieser Wert steht in genau einer Tabellenzeile eines einzigen
Dokuments; jede Antwort aus dem Nachbarmodell oder aus der Grundausführung ist um einen
prüfbaren Betrag falsch.

## Fragen für die Demo (Maschinen)

- „Was wiegt ein {SHARED_TYPE_CODE} {SHARED_MODEL_CODE} mit Fahrwerk und
  Ölauffangwanne?" — verlangt Hersteller, Baureihe, Modell und Variante; flach
  beantwortet landet man im ZM3K.
- „Welche Maschinen verwenden das {BRIDGE_ASSEMBLY}?" — die herstellerübergreifende
  Brücke.
- „Was bedeutet die Option {SHARED_OPTION_CODE}?" — ohne Zweig nicht beantwortbar.
- „Welche Modelle gibt es zur Baureihe {SHARED_TYPE_CODE}?" — zwei Hersteller, zwei
  völlig verschiedene Antwortmengen.
"""
