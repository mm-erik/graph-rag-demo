"""What each depth of the hierarchy *means*.

Ticket 04 models folders generically — one node type at every depth — which is what
makes a corpus of arbitrary depth work without configuration. The cost of that decision
is that the graph no longer records that depth 1 is a client and depth 2 a year. This
module recovers it: one model call per *depth*, inspecting the sibling folder names at
that depth, never one call per folder.

An explicit ordered list of level names bypasses the model entirely for the depths it
covers. That is the escape hatch for a bad inference during a live demo, so it must
genuinely skip the call rather than bias the prompt.

The model client is injected as a protocol; nothing here constructs one.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

#: Enough sibling names to characterise a level, few enough to keep the prompt cheap.
#: A level with 400 folders is not better understood from 400 names than from 40.
SAMPLE_LIMIT = 40

LevelSource = Literal["override", "inferred"]


@dataclass(frozen=True)
class LevelSample:
    """The sibling folder names at one depth, as read back from the graph."""

    depth: int
    folder_count: int
    distinct_count: int
    names: list[str]


@dataclass(frozen=True)
class Level:
    depth: int
    name: str
    confidence: float
    source: LevelSource
    #: Why this level is unnamed, when the call for it failed. Carried rather than
    #: raised: losing the levels that *did* resolve — and the calls already paid for —
    #: because the fourth one hit a rate limit is not an outcome worth having mid-demo.
    note: str | None = None


class ChatModel(Protocol):
    """The chat-tier model. Low volume, and its output is visible to the audience."""

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...


def parse_level_names(raw: str | None) -> list[str]:
    """The operator's `--levels` string as an ordered list of names.

    An empty segment is rejected rather than dropped. Silently skipping it would shift
    every later name up a level — `"Kunde,,Projekt"` would label the years "Projekt" —
    which is precisely the quiet mislabelling the override exists to rule out.
    """
    if not raw:
        return []
    names = [name.strip() for name in raw.split(",")]
    if not all(names):
        raise ValueError(
            f"empty level name in {raw!r}; name every level up to the deepest one you "
            "want to fix"
        )
    return names


def apply_overrides(
    overrides: Sequence[str], samples: Sequence[LevelSample]
) -> tuple[list[Level], list[LevelSample]]:
    """Split the levels into those the operator named and those left to infer.

    The nth override names the nth level *below the walked root*: the root is depth 0
    and is the corpus itself, not a level of it, so `Kunde,Jahr,Projekt` means depths
    1, 2 and 3. Overrides past the last real depth are dropped rather than inventing a
    level that does not exist.
    """
    ordered = sorted(samples, key=lambda sample: sample.depth)
    named = [
        Level(depth=sample.depth, name=name, confidence=1.0, source="override")
        for sample, name in zip(ordered, overrides)
    ]
    return named, ordered[len(named) :]


#: What an unusable reply becomes. German, like the corpus and every prompt here, and
#: paired with zero confidence so the operator sees a level they can distrust rather
#: than a plausible-looking guess.
UNKNOWN_LEVEL = "unbekannt"


def parse_inference(reply: str) -> tuple[str, float]:
    """The level name and confidence in a model reply.

    Total: any reply at all yields a usable pair. A model that returns prose, truncated
    JSON or an empty object must not abort an ingest halfway through, and must not be
    rewarded with a default confidence that reads as trustworthy.
    """
    try:
        payload = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return UNKNOWN_LEVEL, 0.0
    if not isinstance(payload, dict):
        return UNKNOWN_LEVEL, 0.0

    name = str(payload.get("name") or "").strip()
    if not name:
        return UNKNOWN_LEVEL, 0.0

    try:
        confidence = float(payload.get("confidence"))
    except (TypeError, ValueError):
        return name, 0.0
    return name, min(1.0, max(0.0, confidence))


# German, per the project convention: the corpus is German, every other prompt in this
# project is German, and these names are read out in German answers downstream ("aus dem
# Projekt Atlas des Kunden Nordlicht, 2023") rather than recited as a path.
_SYSTEM_PROMPT = (
    "Du analysierst die Ordnerstruktur eines Dokumentenarchivs. "
    "Auf jeder Ebene der Hierarchie stehen gleichartige Ordner nebeneinander. "
    "Benenne, was diese Ebene bedeutet — ein einzelnes Substantiv im Singular, "
    "zum Beispiel 'Kunde', 'Jahr', 'Projekt', 'Abteilung', 'Region' oder 'Phase'. "
    "Gib ausserdem an, wie sicher du dir bist. "
    'Antworte ausschliesslich als JSON: {"name": "<Substantiv>", "confidence": <0.0 bis 1.0>}'
)


def _user_prompt(sample: LevelSample) -> str:
    lines = [
        f"Ebene {sample.depth} der Hierarchie.",
        (
            f"{sample.folder_count} Ordner auf dieser Ebene, "
            f"{sample.distinct_count} verschiedene Namen."
        ),
        "Beispiele für Ordnernamen auf dieser Ebene:",
        *(f"- {name}" for name in sample.names),
    ]
    if sample.distinct_count > len(sample.names):
        lines.append(f"... und {sample.distinct_count - len(sample.names)} weitere.")
    return "\n".join(lines)


def infer_levels(
    samples: Sequence[LevelSample],
    model: ChatModel | None,
    overrides: Sequence[str] = (),
) -> list[Level]:
    """Name every level, by override where given and by one model call per depth else.

    Cost scales with the *depth* of the corpus, not with the number of folders in it:
    a four-level corpus of a thousand folders costs four calls, and fewer when the
    operator has supplied overrides. `model` may be None exactly when the overrides
    cover every level — the escape hatch has to work without a usable API key, which is
    most of the point of it.
    """
    named, remaining = apply_overrides(overrides, samples)
    if remaining and model is None:
        raise ValueError(
            f"no chat model given, but depths {[s.depth for s in remaining]} are not "
            "covered by the supplied level names"
        )
    for sample in remaining:
        note = None
        try:
            name, confidence = parse_inference(
                model.complete(_SYSTEM_PROMPT, _user_prompt(sample), json_object=True)
            )
        # Deliberately broad, as in `mistral_check`: a transport failure is a level we
        # cannot name, not an ingest we have to abandon.
        except Exception as error:  # noqa: BLE001
            name, confidence = UNKNOWN_LEVEL, 0.0
            note = str(error)
        named.append(
            Level(
                depth=sample.depth,
                name=name,
                confidence=confidence,
                source="inferred",
                note=note,
            )
        )
    return sorted(named, key=lambda level: level.depth)
