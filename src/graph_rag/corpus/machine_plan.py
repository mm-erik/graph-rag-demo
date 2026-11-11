"""The deterministic plan for the machine half of the synthetic corpus.

The client half (`plan.py`) is hostile to flat retrieval because *names collide*: two
clients run a project of the same name. This half is hostile for a different and harder
reason: the names carry **no meaning at all**. `XTC27`, `B2BC`, `HX-114` are opaque
alphanumeric codes, so an embedding of the question sits nowhere near the embedding of
the right document, and a lexical hit on `B2BC` is worthless because the same code names
two unrelated machines. Only the structure — manufacturer, machine type, model, variant —
disambiguates, and structure is exactly what the graph has and a flat index does not.

Four properties are planted on purpose:

1. A *model* code is reused under two different machine types of two different
   manufacturers (`B2BC` is a Hagedorn mixer-dryer and a Pleickner assembly cell).
2. A *machine type* code is reused by two manufacturers (`XTC27` is a vacuum mixer-dryer
   at one and a roller-track transfer unit at the other).
3. An *option* code is reused with a different meaning per manufacturer (`HK`).
4. Two cross-manufacturer bridges — a shared assembly and the engineer who carried it —
   where **no document names the other manufacturer**, so the join exists only as a
   shared entity and never inside a single chunk.

The numbers are hand-set rather than sampled: a variant's empty weight is its model's
base weight plus the deltas of the options it carries, so "what does an XTC27 B2BC with
running gear and oil tray weigh?" has exactly one right answer, and a plausible-looking
wrong answer from a neighbouring datasheet is detectable.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

# The planted collisions, named so tests and the generated note cannot drift apart.
SHARED_MODEL_CODE = "B2BC"
SHARED_TYPE_CODE = "XTC27"
SHARED_OPTION_CODE = "HK"

BRIDGE_ENGINEER = "Dipl.-Ing. Reinhard Oswald"
BRIDGE_ASSEMBLY = "Planetengetriebe HX-114"
SECOND_BRIDGE_ENGINEER = "Beate Lindqvist"
SECOND_BRIDGE_ASSEMBLY = "Spannsatz KN-30"

# Which (manufacturer, type, model) each bridge is anchored to. The origin side invented
# the assembly; the reuse side took it over.
BRIDGE_ORIGIN = ("Hagedorn Verfahrenstechnik GmbH", "XTC27", "B2BC")
BRIDGE_REUSE = ("Ostmark Fördertechnik AG", "QH4L", "P6V")
SECOND_BRIDGE_ORIGIN = ("Pleickner Werke KG", "NB7X", "D1F")
SECOND_BRIDGE_REUSE = ("Hagedorn Verfahrenstechnik GmbH", "VR88D", "K4R")


@dataclass(frozen=True)
class Engineer:
    name: str
    role: str


@dataclass(frozen=True)
class Manufacturer:
    name: str
    #: ASCII folder name, deliberately unlike the legal name the documents use.
    folder: str
    city: str
    division: str
    contact: Engineer


@dataclass(frozen=True)
class Option:
    """An orderable option. The deltas are what makes a variant's data sheet row differ
    from its base model, and they are additive so a question about a two-option variant
    cannot be answered from the base row alone."""

    code: str
    name: str
    description: str
    weight_delta: int  # kg
    height_delta: int  # mm
    oil_delta: int  # Liter Auffangvolumen
    protection: str | None  # overrides the model's IP rating when set
    constraint: str


@dataclass(frozen=True)
class Model:
    code: str
    description: str
    base_weight: int  # kg, Leergewicht der Grundausführung
    base_height: int  # mm
    footprint: str
    capacity: str
    drive: str
    control: str
    protection: str
    noise: int  # dB(A)
    #: Option codes offered for this model, in catalogue order.
    options: tuple[str, ...]
    #: Each orderable variant as the option codes it carries. `()` is the base version.
    variants: tuple[tuple[str, ...], ...]
    assemblies: tuple[str, ...]


@dataclass(frozen=True)
class MachineType:
    code: str
    manufacturer: str
    name: str
    purpose: str
    models: tuple[Model, ...]


@dataclass(frozen=True)
class Machine:
    """One model of one machine type of one manufacturer — the unit a folder of
    documents describes."""

    manufacturer: str
    type_code: str
    type_name: str
    purpose: str
    model: Model
    designer: Engineer
    inspector: Engineer
    siblings: tuple[str, ...]  # the other model codes under the same machine type

    @property
    def order_key(self) -> str:
        return f"{self.type_code}-{self.model.code}"


MANUFACTURERS = (
    Manufacturer(
        name="Hagedorn Verfahrenstechnik GmbH",
        folder="Hagedorn Verfahrenstechnik",
        city="Krefeld",
        division="Verfahrenstechnik",
        contact=Engineer("Ulrike Hagedorn", "Leiterin Technischer Vertrieb"),
    ),
    Manufacturer(
        name="Pleickner Werke KG",
        folder="Pleickner Werke",
        city="Schweinfurt",
        division="Montage- und Antriebstechnik",
        contact=Engineer("Harald Pleickner", "Technischer Geschäftsführer"),
    ),
    Manufacturer(
        name="Ostmark Fördertechnik AG",
        folder="Ostmark Foerdertechnik",
        city="Magdeburg",
        division="Fördertechnik",
        contact=Engineer("Steffi Marwitz", "Leiterin Anwendungstechnik"),
    ),
)

# Option codes are namespaced per manufacturer, and `HK` deliberately means something
# different at each one: a lexical hit on the code says nothing without the branch above
# it.
OPTIONS: dict[str, tuple[Option, ...]] = {
    "Hagedorn Verfahrenstechnik GmbH": (
        Option(
            code="FW",
            name="Fahrwerk",
            description="vier lenkbare Schwerlastrollen mit Feststellbremse, "
            "Tragfähigkeit je Rolle 650 kg",
            weight_delta=85,
            height_delta=140,
            oil_delta=0,
            protection=None,
            constraint="Bei Ausführung mit Fahrwerk ist die Verankerung im Fundament "
            "entbehrlich; die zulässige Bodenlast steigt dadurch auf 1,8 kN je Rolle.",
        ),
        Option(
            code="OW",
            name="Ölauffangwanne",
            description="geschweißte Auffangwanne aus 1.4301 unter dem Getriebeblock",
            weight_delta=60,
            height_delta=55,
            oil_delta=42,
            protection=None,
            constraint="Die Wanne ist vor jeder Reinigung zu entleeren; das "
            "Auffangvolumen deckt die vollständige Getriebefüllung ab.",
        ),
        Option(
            code="EX",
            name="ATEX-Ausführung Zone 22",
            description="funkenarme Werkstoffpaarung, Potentialausgleich und "
            "Temperaturüberwachung am Getriebeflansch",
            weight_delta=45,
            height_delta=0,
            oil_delta=0,
            protection="IP66",
            constraint="Die ATEX-Ausführung schließt das Fahrwerk aus: in Zone 22 ist "
            "eine ortsfeste Aufstellung mit Potentialausgleich vorgeschrieben.",
        ),
        Option(
            code=SHARED_OPTION_CODE,
            name="Heizkreis",
            description="doppelwandiger Mantel mit Thermalölanschluss, "
            "Vorlauftemperatur bis 180 °C",
            weight_delta=130,
            height_delta=0,
            oil_delta=0,
            protection=None,
            constraint="Der Heizkreis erfordert bauseitig eine Thermalölstation; die "
            "Anschlussleistung erhöht sich um 6 kW.",
        ),
    ),
    "Pleickner Werke KG": (
        Option(
            code="FW",
            name="Fahrwerk",
            description="zwei Bockrollen und zwei Lenkrollen, Polyamidlauffläche",
            weight_delta=52,
            height_delta=95,
            oil_delta=0,
            protection=None,
            constraint="Im verfahrbaren Zustand ist die Pressenachse nicht freizugeben; "
            "die Verriegelung erfolgt über den Endschalter am Bremsbügel.",
        ),
        Option(
            code="OW",
            name="Ölauffangwanne",
            description="Blechwanne unter Hydraulikaggregat und Pressenständer",
            weight_delta=38,
            height_delta=40,
            oil_delta=25,
            protection=None,
            constraint="Das Auffangvolumen entspricht dem Inhalt des Hydraulik"
            "aggregats zuzüglich zehn Prozent Reserve.",
        ),
        Option(
            code=SHARED_OPTION_CODE,
            name="Hubkonsole",
            description="höhenverstellbare Werkstückkonsole mit Spindeltrieb, "
            "Verstellbereich 240 mm",
            weight_delta=74,
            height_delta=0,
            oil_delta=0,
            protection=None,
            constraint="Die Hubkonsole verringert die nutzbare Einbauhöhe um 240 mm, "
            "wenn sie in der oberen Raststellung steht.",
        ),
        Option(
            code="SK",
            name="Schallkapsel",
            description="zweischalige Kapsel mit Sichtfenster und Wartungsklappe",
            weight_delta=165,
            height_delta=210,
            oil_delta=0,
            protection=None,
            constraint="Mit Schallkapsel sinkt der Emissionsschalldruckpegel um "
            "9 dB(A); die Kapsel ist mit dem Fahrwerk nicht kombinierbar.",
        ),
    ),
    "Ostmark Fördertechnik AG": (
        Option(
            code="FW",
            name="Fahrwerk",
            description="Verfahrwagen auf Schiene S24 mit Energiekette",
            weight_delta=118,
            height_delta=160,
            oil_delta=0,
            protection=None,
            constraint="Der Verfahrwagen setzt eine durchgehende Schiene S24 voraus; "
            "die Schiene gehört nicht zum Lieferumfang.",
        ),
        Option(
            code=SHARED_OPTION_CODE,
            name="Hubkonsole",
            description="pneumatischer Hubtisch im Umsetzerrahmen, Hub 120 mm",
            weight_delta=96,
            height_delta=120,
            oil_delta=0,
            protection=None,
            constraint="Die Hubkonsole benötigt Druckluft von 6 bar und erhöht die "
            "Taktzeit um 0,8 Sekunden je Umsetzvorgang.",
        ),
        Option(
            code="OW",
            name="Ölauffangwanne",
            description="Tropfblech unter dem Kettentrieb mit Ablasshahn",
            weight_delta=22,
            height_delta=35,
            oil_delta=9,
            protection=None,
            constraint="Das Tropfblech ist bei Kettenschmierung mit Haftöl zwingend; "
            "bei Trockenlaufketten entfällt es.",
        ),
        Option(
            code="RS",
            name="Rücklaufsperre",
            description="Klemmrollenfreilauf an der Antriebswelle",
            weight_delta=14,
            height_delta=0,
            oil_delta=0,
            protection=None,
            constraint="Die Rücklaufsperre ist bei Steigungen über drei Grad "
            "vorgeschrieben und blockiert den Handbetrieb entgegen der Förderrichtung.",
        ),
    ),
}


MACHINE_TYPES = (
    MachineType(
        code=SHARED_TYPE_CODE,
        manufacturer="Hagedorn Verfahrenstechnik GmbH",
        name="Vakuum-Mischtrockner",
        purpose="die Trocknung und Homogenisierung rieselfähiger Feststoffe unter "
        "Vakuum",
        models=(
            Model(
                code=SHARED_MODEL_CODE,
                description="Baugröße 2, konischer Behälter, beheizbarer Mantel",
                base_weight=1480,
                base_height=2240,
                footprint="1.840 × 1.120 mm",
                capacity="Nutzvolumen 270 l, Gesamtvolumen 340 l",
                drive="Drehstrom-Getriebemotor 11 kW, 1.450 U/min",
                control="Steuerung SPX-9",
                protection="IP55",
                noise=74,
                options=("FW", "OW", "EX", SHARED_OPTION_CODE),
                variants=((), ("FW",), ("FW", "OW"), ("EX", "OW"), (SHARED_OPTION_CODE, "FW")),
                assemblies=(
                    BRIDGE_ASSEMBLY,
                    "Wellendichtung RK9-Doppellippe",
                    "Vakuumpumpe VP-40",
                ),
            ),
            Model(
                code="B2BD",
                description="Baugröße 2, zylindrischer Behälter, Doppelmantel",
                base_weight=1620,
                base_height=2240,
                footprint="1.840 × 1.240 mm",
                capacity="Nutzvolumen 290 l, Gesamtvolumen 365 l",
                drive="Drehstrom-Getriebemotor 15 kW, 1.450 U/min",
                control="Steuerung SPX-9",
                protection="IP55",
                noise=76,
                options=("FW", "OW", SHARED_OPTION_CODE),
                variants=((), ("OW",), ("FW", "OW"), (SHARED_OPTION_CODE,)),
                assemblies=(
                    "Stirnradgetriebe SG-82",
                    "Wellendichtung RK9-Doppellippe",
                    "Vakuumpumpe VP-40",
                ),
            ),
            Model(
                code="L7Q",
                description="Baugröße 4, konischer Behälter, Langsamläufer",
                base_weight=3150,
                base_height=2980,
                footprint="2.460 × 1.560 mm",
                capacity="Nutzvolumen 620 l, Gesamtvolumen 760 l",
                drive="Drehstrom-Getriebemotor 22 kW, 980 U/min",
                control="Steuerung SPX-11",
                protection="IP55",
                noise=79,
                options=("OW", "EX", SHARED_OPTION_CODE),
                variants=((), ("OW",), ("EX", "OW"), (SHARED_OPTION_CODE, "OW")),
                assemblies=(
                    "Stirnradgetriebe SG-120",
                    "Wellendichtung RK9-Doppellippe",
                    "Vakuumpumpe VP-63",
                ),
            ),
        ),
    ),
    MachineType(
        code="VR88D",
        manufacturer="Hagedorn Verfahrenstechnik GmbH",
        name="Doppelwellen-Zwangsmischer",
        purpose="das Zwangsmischen pastöser und feuchter Massen im Chargenbetrieb",
        models=(
            Model(
                code="K4R",
                description="Baugröße 4, Trogmischer, Standardwellen",
                base_weight=2240,
                base_height=1870,
                footprint="2.150 × 1.340 mm",
                capacity="Chargenvolumen 400 l",
                drive="Zwei Getriebemotoren je 18,5 kW",
                control="Steuerung SPX-11",
                protection="IP55",
                noise=82,
                options=("OW", "EX", SHARED_OPTION_CODE),
                variants=((), ("OW",), (SHARED_OPTION_CODE, "OW"), ("EX",)),
                assemblies=(
                    SECOND_BRIDGE_ASSEMBLY,
                    "Trogdichtung TD-55",
                    "Stirnradgetriebe SG-120",
                ),
            ),
            Model(
                code="K4RS",
                description="Baugröße 4, Trogmischer, Schwerlastwellen mit Panzerung",
                base_weight=2410,
                base_height=1870,
                footprint="2.150 × 1.340 mm",
                capacity="Chargenvolumen 400 l",
                drive="Zwei Getriebemotoren je 22 kW",
                control="Steuerung SPX-11",
                protection="IP55",
                noise=85,
                options=("OW", SHARED_OPTION_CODE),
                variants=((), ("OW",), (SHARED_OPTION_CODE,)),
                assemblies=(
                    "Trogdichtung TD-55",
                    "Stirnradgetriebe SG-120",
                    "Panzerleiste PL-18",
                ),
            ),
        ),
    ),
    MachineType(
        code="ZM3K",
        manufacturer="Pleickner Werke KG",
        name="Zahnkranz-Montagezelle",
        purpose="das lagerichtige Fügen von Zahnkränzen auf Radnaben",
        models=(
            # The planted model-code collision: same code as the Hagedorn mixer-dryer,
            # an entirely different machine, entirely different numbers.
            Model(
                code=SHARED_MODEL_CODE,
                description="Grundzelle mit einem Fügeplatz und Handbeschickung",
                base_weight=960,
                base_height=1980,
                footprint="1.420 × 1.420 mm",
                capacity="Fügekraft 160 kN, Taktzeit 42 s",
                drive="Servospindelpresse 7,5 kW",
                control="Steuerung PLC-4K",
                protection="IP54",
                noise=71,
                options=("FW", "OW", "SK"),
                variants=((), ("FW",), ("OW",), ("SK", "OW")),
                assemblies=(
                    "Servospindel SP-16",
                    "Kraftmessdose KMD-160",
                    "Zentrierdorn ZD-44",
                ),
            ),
            Model(
                code="G5T",
                description="Doppelzelle mit Drehtisch und zwei Fügeplätzen",
                base_weight=1740,
                base_height=2120,
                footprint="2.260 × 1.680 mm",
                capacity="Fügekraft 250 kN, Taktzeit 28 s",
                drive="Servospindelpresse 15 kW",
                control="Steuerung PLC-4K",
                protection="IP54",
                noise=75,
                options=("OW", "SK", SHARED_OPTION_CODE),
                variants=((), ("OW",), ("SK", "OW"), (SHARED_OPTION_CODE, "OW")),
                assemblies=(
                    "Servospindel SP-25",
                    "Kraftmessdose KMD-250",
                    "Rundschalttisch RT-80",
                ),
            ),
        ),
    ),
    MachineType(
        code="NB7X",
        manufacturer="Pleickner Werke KG",
        name="Nabenpressen-Automat",
        purpose="das Ein- und Auspressen von Lagersitzen in Radnaben",
        models=(
            Model(
                code="D1F",
                description="Einständerpresse mit festem Tisch",
                base_weight=1280,
                base_height=2340,
                footprint="1.180 × 980 mm",
                capacity="Presskraft 400 kN, Hub 320 mm",
                drive="Hydraulikaggregat 11 kW, 210 bar",
                control="Steuerung PLC-4K",
                protection="IP54",
                noise=78,
                options=("FW", "OW", SHARED_OPTION_CODE, "SK"),
                variants=((), ("OW",), ("FW", "OW"), (SHARED_OPTION_CODE, "OW"), ("SK", "OW")),
                assemblies=(
                    SECOND_BRIDGE_ASSEMBLY,
                    "Hydraulikaggregat HA-11",
                    "Presswerkzeugaufnahme PW-60",
                ),
            ),
            Model(
                code="D1FH",
                description="Einständerpresse mit Hubtisch und Werkstückerkennung",
                base_weight=1395,
                base_height=2340,
                footprint="1.180 × 1.060 mm",
                capacity="Presskraft 400 kN, Hub 320 mm",
                drive="Hydraulikaggregat 15 kW, 250 bar",
                control="Steuerung PLC-6K",
                protection="IP54",
                noise=80,
                options=("OW", SHARED_OPTION_CODE, "SK"),
                variants=((), ("OW",), (SHARED_OPTION_CODE, "OW"), ("SK",)),
                assemblies=(
                    "Hydraulikaggregat HA-15",
                    "Presswerkzeugaufnahme PW-60",
                    "Werkstücksensorik WS-3",
                ),
            ),
        ),
    ),
    MachineType(
        # The planted type-code collision: the same type code as the Hagedorn
        # mixer-dryer, on a machine that has nothing to do with it.
        code=SHARED_TYPE_CODE,
        manufacturer="Ostmark Fördertechnik AG",
        name="Rollenbahn-Umsetzer",
        purpose="das Umsetzen von Paletten zwischen zwei Rollenbahnsträngen",
        models=(
            Model(
                code="H3M",
                description="Umsetzer mit Kettenquertransport, Nutzlast 1.200 kg",
                base_weight=640,
                base_height=820,
                footprint="1.400 × 1.250 mm",
                capacity="Nutzlast 1.200 kg, Umsetzleistung 60 Paletten/h",
                drive="Getriebemotor 1,5 kW",
                control="Steuerung OMC-2",
                protection="IP54",
                noise=68,
                options=("FW", SHARED_OPTION_CODE, "OW", "RS"),
                variants=((), (SHARED_OPTION_CODE,), ("FW", "RS"), (SHARED_OPTION_CODE, "OW")),
                assemblies=(
                    "Kettentrieb KT-16",
                    "Tragrollensatz TR-50",
                    "Getriebemotor GM-15",
                ),
            ),
            Model(
                code="H3MZ",
                description="Umsetzer mit Zentriereinheit für Sonderladungsträger",
                base_weight=710,
                base_height=820,
                footprint="1.400 × 1.250 mm",
                capacity="Nutzlast 1.200 kg, Umsetzleistung 48 Paletten/h",
                drive="Getriebemotor 1,5 kW",
                control="Steuerung OMC-2",
                protection="IP54",
                noise=68,
                options=(SHARED_OPTION_CODE, "OW", "RS"),
                variants=((), (SHARED_OPTION_CODE,), ("OW", "RS")),
                assemblies=(
                    "Kettentrieb KT-16",
                    "Zentriereinheit ZE-9",
                    "Getriebemotor GM-15",
                ),
            ),
        ),
    ),
    MachineType(
        code="QH4L",
        manufacturer="Ostmark Fördertechnik AG",
        name="Kettenförderer-Modul",
        purpose="den Palettentransport über kurze Strecken im Schwerlastbereich",
        models=(
            Model(
                code="P6V",
                description="Dreistrang-Kettenförderer, Modullänge 2.400 mm",
                base_weight=890,
                base_height=760,
                footprint="2.400 × 1.250 mm",
                capacity="Nutzlast 1.500 kg, Fördergeschwindigkeit 0,2 m/s",
                drive="Drehstrom-Getriebemotor 2,2 kW",
                control="Steuerung OMC-2",
                protection="IP54",
                noise=70,
                options=("FW", "OW", "RS", SHARED_OPTION_CODE),
                variants=((), ("OW",), ("RS", "OW"), ("FW", "RS"), (SHARED_OPTION_CODE, "OW")),
                assemblies=(
                    BRIDGE_ASSEMBLY,
                    "Kettentrieb KT-24",
                    "Tragrollensatz TR-50",
                ),
            ),
            Model(
                code="P6VX",
                description="Dreistrang-Kettenförderer, Modullänge 3.600 mm, "
                "verstärkter Rahmen",
                base_weight=1210,
                base_height=760,
                footprint="3.600 × 1.250 mm",
                capacity="Nutzlast 2.000 kg, Fördergeschwindigkeit 0,2 m/s",
                drive="Drehstrom-Getriebemotor 3,0 kW",
                control="Steuerung OMC-2",
                protection="IP54",
                noise=72,
                options=("OW", "RS", SHARED_OPTION_CODE),
                variants=((), ("OW",), ("RS", "OW"), (SHARED_OPTION_CODE, "RS")),
                assemblies=(
                    "Kettentrieb KT-24",
                    "Tragrollensatz TR-50",
                    "Rahmenprofil RP-160",
                ),
            ),
            Model(
                code="T2L",
                description="Zweistrang-Kettenförderer für Gitterboxen, "
                "Modullänge 1.800 mm",
                base_weight=540,
                base_height=680,
                footprint="1.800 × 900 mm",
                capacity="Nutzlast 800 kg, Fördergeschwindigkeit 0,3 m/s",
                drive="Drehstrom-Getriebemotor 1,1 kW",
                control="Steuerung OMC-1",
                protection="IP54",
                noise=66,
                options=("OW", "RS"),
                variants=((), ("OW",), ("RS",)),
                assemblies=(
                    "Kettentrieb KT-12",
                    "Tragrollensatz TR-32",
                    "Getriebemotor GM-11",
                ),
            ),
        ),
    ),
)


ENGINEERS = (
    Engineer(BRIDGE_ENGINEER, "Konstruktionsleiter"),
    Engineer(SECOND_BRIDGE_ENGINEER, "Prüfingenieurin"),
    Engineer("Hartmut Zeller", "Konstrukteur Baugruppen"),
    Engineer("Ines Grabowski", "Konstrukteurin Antriebstechnik"),
    Engineer("Sven Kallweit", "Prüfingenieur"),
    Engineer("Marlene Dobrindt", "Konstrukteurin Steuerungstechnik"),
    Engineer("Pawel Strzelecki", "Prüfingenieur Abnahme"),
    Engineer("Dr. Ömer Yildiz", "Leiter Versuchsfeld"),
)

# Freely assignable. The two bridge engineers are excluded: each appears only on the two
# models their assembly joins, so a traversal through one of them reaches exactly the
# other side of the bridge instead of half the corpus.
ORDINARY_ENGINEERS = tuple(
    engineer
    for engineer in ENGINEERS
    if engineer.name not in (BRIDGE_ENGINEER, SECOND_BRIDGE_ENGINEER)
)


def manufacturer_by_name(name: str) -> Manufacturer:
    return next(m for m in MANUFACTURERS if m.name == name)


def option_by_code(manufacturer: str, code: str) -> Option:
    return next(option for option in OPTIONS[manufacturer] if option.code == code)


def options_for(machine: Machine) -> tuple[Option, ...]:
    return tuple(
        option_by_code(machine.manufacturer, code) for code in machine.model.options
    )


def _engineer_for(rng: Random, anchor: tuple[str, str, str], role_index: int) -> Engineer:
    """Staff one model, honouring the two planted bridges."""
    if anchor in (BRIDGE_ORIGIN, BRIDGE_REUSE) and role_index == 0:
        return next(e for e in ENGINEERS if e.name == BRIDGE_ENGINEER)
    if anchor in (SECOND_BRIDGE_ORIGIN, SECOND_BRIDGE_REUSE) and role_index == 1:
        return next(e for e in ENGINEERS if e.name == SECOND_BRIDGE_ENGINEER)
    return rng.choice(list(ORDINARY_ENGINEERS))


def build_machine_plan(seed: int) -> list[Machine]:
    """Every machine model the corpus contains, in a stable order."""
    rng = Random(f"{seed}:machines")
    machines: list[Machine] = []
    for machine_type in MACHINE_TYPES:
        siblings = tuple(model.code for model in machine_type.models)
        for model in machine_type.models:
            anchor = (machine_type.manufacturer, machine_type.code, model.code)
            designer = _engineer_for(rng, anchor, 0)
            inspector = _engineer_for(rng, anchor, 1)
            while inspector.name == designer.name:
                inspector = rng.choice(list(ORDINARY_ENGINEERS))
            machines.append(
                Machine(
                    manufacturer=machine_type.manufacturer,
                    type_code=machine_type.code,
                    type_name=machine_type.name,
                    purpose=machine_type.purpose,
                    model=model,
                    designer=designer,
                    inspector=inspector,
                    siblings=tuple(code for code in siblings if code != model.code),
                )
            )
    return machines


def machine_by(manufacturer: str, type_code: str, model_code: str, seed: int) -> Machine:
    return next(
        machine
        for machine in build_machine_plan(seed)
        if machine.manufacturer == manufacturer
        and machine.type_code == type_code
        and machine.model.code == model_code
    )


def variant_code(machine: Machine, variant: tuple[str, ...]) -> str:
    """The order key a customer would quote, e.g. `XTC27-B2BC-FW/OW`."""
    if not variant:
        return f"{machine.order_key}-00"
    return f"{machine.order_key}-{'/'.join(variant)}"


@dataclass(frozen=True)
class VariantSpec:
    code: str
    options: tuple[Option, ...]
    weight: int
    height: int
    oil_volume: int
    protection: str
    label: str


def variant_spec(machine: Machine, variant: tuple[str, ...]) -> VariantSpec:
    """A variant's data sheet row: the base model plus the deltas of its options.

    Additive on purpose. A two-option variant's weight appears nowhere but its own row,
    so an answer taken from the base model — or from the same model code under the other
    machine type — is wrong by a checkable amount.
    """
    model = machine.model
    options = tuple(option_by_code(machine.manufacturer, code) for code in variant)
    protection = model.protection
    for option in options:
        if option.protection:
            protection = option.protection
    label = (
        "Grundausführung"
        if not options
        else " + ".join(option.name for option in options)
    )
    return VariantSpec(
        code=variant_code(machine, variant),
        options=options,
        weight=model.base_weight + sum(option.weight_delta for option in options),
        height=model.base_height + sum(option.height_delta for option in options),
        oil_volume=sum(option.oil_delta for option in options),
        protection=protection,
        label=label,
    )


def variant_specs(machine: Machine) -> tuple[VariantSpec, ...]:
    return tuple(variant_spec(machine, variant) for variant in machine.model.variants)
