"""Deciding which extracted mentions are the same entity.

This is the highest-risk component in the system, and both of its failure modes are
silent:

- **Too loose** — distinct entities collapse together. Two clients' similarly-named
  projects become one node and the graph asserts a connection that does not exist.
- **Too strict** — the graph fragments into spelling variants of one organisation. A
  fragmented graph demonstrates the *opposite* of this project's thesis and looks worse
  than flat RAG.

Three escalating stages, cheapest first, behind a zeroth that only tidies up:

0. **One name, one type** — `unify_types`. Not a resolution stage so much as a repair
   of the per-chunk taxonomy, and it has to run first because the three stages below
   never compare across a type.
1. **Exact** match on the normalised identity. No embedding, no call.
2. **Embedding similarity** at or above `Thresholds.merge`. Catches the spellings
   normalisation cannot — `Aktiengesellschaft` against `AG`.
3. **Model adjudication**, for the band between `review` and `merge` only. It is the
   one expensive stage, so it is asked about the pairs the thresholds genuinely cannot
   decide and nothing else.

Comparison never crosses a type or a scope. A scope is the hierarchy's own verdict that
two identical names are different things, and no threshold or model call may overturn
it.

The similarity function and the adjudicator are injected; nothing here calls a model.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import sqrt
from typing import Protocol

from graph_rag.entities import EntityIdentity, normalise_name


@dataclass(frozen=True)
class Thresholds:
    """Where the three stages hand over to each other.

    Tuned empirically against the generated corpus rather than guessed — see the
    entity section of the README for the measurements these came from.
    """

    #: At or above this, two names are the same entity without asking anyone.
    merge: float
    #: Below this, they are different entities without asking anyone. Between the two
    #: lies the band the model adjudicates.
    review: float


@dataclass(frozen=True)
class Candidate:
    """One distinct identity extracted from the corpus, with how often it was seen."""

    identity: EntityIdentity
    #: The name as the documents actually write it, for display.
    surface: str
    embedding: list[float]
    mentions: int = 1


@dataclass
class ResolvedEntity:
    """One node, and every identity that folded into it."""

    identity: EntityIdentity
    surface: str
    embedding: list[float]
    mentions: int
    #: Surfaces merged away into this node, for reporting. The operator tuning the
    #: thresholds needs to see what stage two and stage three actually did.
    aliases: list[str] = field(default_factory=list)

    @property
    def type(self) -> str:
        return self.identity.type

    @property
    def scope(self) -> str:
        return self.identity.scope


@dataclass(frozen=True)
class Resolution:
    entities: list[ResolvedEntity]
    #: Every input identity mapped to the node it ended up on, including the ones that
    #: were merged away. The graph write attaches each mention's edges through this, so
    #: a missing key is a silently dropped edge.
    canonical: dict[EntityIdentity, EntityIdentity]
    #: Mentions folded in by stage one and by stage two respectively.
    exact: int = 0
    merged: int = 0
    #: Calls made in stage three, and how many of them said "same".
    adjudications: int = 0
    adjudicated_merges: int = 0


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, or 0.0 when either vector is missing or degenerate."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = sqrt(sum(a * a for a in left)) * sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


#: `(type, left surface, right surface) -> are these the same entity?`
Adjudicator = Callable[[str, str, str], bool]


def resolve_entities(
    candidates: Sequence[Candidate],
    thresholds: Thresholds,
    adjudicate: Adjudicator | None = None,
) -> Resolution:
    """Fold `candidates` into the entity nodes the graph should hold.

    Candidates are processed most-mentioned first, so the name the corpus actually uses
    seeds the node and the rarer spellings fold into it — rather than whichever
    variant happened to be extracted first. Ties break on the normalised name, so two
    runs over the same corpus resolve identically.

    A cluster keeps its seed's embedding rather than averaging in each new member: an
    average drifts towards whatever was merged, so a chain of near-misses could walk a
    node away from the name it is supposed to represent, one merge at a time.
    """
    aligned = unify_types(candidates)
    ordered = sorted(
        aligned, key=lambda pair: (-pair[1].mentions, pair[1].identity.normalized_name)
    )
    # Bucketed, so comparison can never cross a type or a scope — the two boundaries
    # that are decided before similarity is ever consulted.
    buckets: dict[tuple[str, str], list[ResolvedEntity]] = {}
    canonical: dict[EntityIdentity, EntityIdentity] = {}
    by_identity: dict[EntityIdentity, ResolvedEntity] = {}
    exact = merged = adjudications = adjudicated_merges = 0

    for original, candidate in ordered:
        adjudicated = False
        existing = by_identity.get(candidate.identity)
        if existing is not None:
            existing.mentions += candidate.mentions
            canonical[original] = existing.identity
            exact += candidate.mentions
            continue

        bucket = buckets.setdefault(
            (candidate.identity.type, candidate.identity.scope), []
        )
        match, score = _best_match(candidate, bucket)
        if score < thresholds.review:
            # Not close enough to be worth a call. This is the branch that keeps stage
            # three cheap: on a corpus of near-identical boilerplate almost every pair
            # lands here.
            match = None
        elif score < thresholds.merge:
            # The band. With no adjudicator — no API key, or the operator turned it off
            # — the pair is left unmerged: two nodes the operator can see and merge by
            # lowering the threshold, rather than one that quietly swallowed the other.
            if adjudicate is None:
                match = None
            else:
                adjudications += 1
                adjudicated = True
                if adjudicate(
                    candidate.identity.type, match.surface, candidate.surface
                ):
                    adjudicated_merges += 1
                else:
                    match = None

        if match is None:
            entity = ResolvedEntity(
                identity=candidate.identity,
                surface=candidate.surface,
                embedding=candidate.embedding,
                mentions=candidate.mentions,
            )
            bucket.append(entity)
            by_identity[candidate.identity] = entity
            canonical[original] = candidate.identity
        else:
            match.mentions += candidate.mentions
            match.aliases.append(candidate.surface)
            by_identity[candidate.identity] = match
            canonical[original] = match.identity
            # Stage two only. Counting the band's merges here as well would report
            # every adjudicated pair twice in the one place an operator looks to see
            # whether the thresholds have started collapsing distinct entities.
            if not adjudicated:
                merged += candidate.mentions

    return Resolution(
        entities=[entity for bucket in buckets.values() for entity in bucket],
        canonical=canonical,
        exact=exact,
        merged=merged,
        adjudications=adjudications,
        adjudicated_merges=adjudicated_merges,
    )


def unify_types(
    candidates: Sequence[Candidate],
) -> list[tuple[EntityIdentity, Candidate]]:
    """Give one name one type, pairing each candidate with the identity it came in as.

    Stage zero, and it exists because the taxonomy is applied per chunk. The extractor
    reads `Atlas-Referenzarchitektur` as a result in one chunk and as a concept in the
    next, and since comparison never crosses a type, the two readings would become two
    nodes that no threshold and no model call could ever bring back together. That is
    the quiet way to cut the one cross-client edge this demo exists to show.

    The type the corpus used most often wins, ties breaking on the type name so two
    runs agree. Scope is part of the key: it is the hierarchy's verdict and nothing
    here may cross it.
    """
    votes: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for candidate in candidates:
        votes[_name_key(candidate)][candidate.identity.type] += candidate.mentions

    aligned = []
    for candidate in candidates:
        tally = votes[_name_key(candidate)]
        if len(tally) == 1:
            aligned.append((candidate.identity, candidate))
            continue
        winner = max(tally.items(), key=lambda item: (item[1], item[0]))[0]
        aligned.append((candidate.identity, _retype(candidate, winner)))
    return aligned


def _name_key(candidate: Candidate) -> tuple[str, str]:
    """A type-independent identity for the surface name, plus its scope.

    Normalised as a concept rather than under the candidate's own type: the whole point
    is to compare names whose types disagree, and a type-aware normaliser would give
    `Nordlicht Energie AG` a different key for each of them.
    """
    return normalise_name(candidate.surface, "konzept"), candidate.identity.scope


def _retype(candidate: Candidate, entity_type: str) -> Candidate:
    """The same candidate under a different type — and so a different normalised name,
    because normalisation is type-aware. The scope is untouched: it is the hierarchy's
    verdict and has nothing to do with which type the extractor picked."""
    return replace(
        candidate,
        identity=EntityIdentity(
            type=entity_type,
            normalized_name=normalise_name(candidate.surface, entity_type),
            scope=candidate.identity.scope,
        ),
    )


def _best_match(
    candidate: Candidate, bucket: Sequence[ResolvedEntity]
) -> tuple[ResolvedEntity | None, float]:
    """The nearest entity in the bucket, and how near it is."""
    best: ResolvedEntity | None = None
    score = 0.0
    for entity in bucket:
        similarity = cosine(candidate.embedding, entity.embedding)
        if similarity > score:
            best, score = entity, similarity
    return best, score


# German, per the project convention. The prompt is written to *resist* merging: the
# band exists precisely where the embeddings cannot tell "two spellings of one thing"
# from "two things with similar names", and of the two mistakes only one is recoverable.
# A fragmented node pair is visible on screen and can be merged by lowering a threshold;
# a wrongly fused node silently asserts a connection that does not exist and nothing in
# the graph records that it was ever two things.
ADJUDICATION_SYSTEM_PROMPT = (
    "Du entscheidest, ob zwei Namen aus einem Projektarchiv dieselbe Entität "
    "bezeichnen.\n"
    "Dieselbe Entität sind sie bei Schreibweisen, Abkürzungen, Rechtsformen oder "
    "Titeln desselben Dings — etwa 'Nordlicht Energie AG' und 'Nordlicht Energie'.\n"
    "Dieselbe Entität sind sie auch, wenn einer der beiden Namen mitten im Wort "
    "abbricht oder verstümmelt ist und der andere ihn vollständig enthält — die "
    "Namen stammen aus Tabellenzellen eines PDF und werden dort abgeschnitten: "
    "'Vektor Consulti', 'Vektor Consulting Gmb' und 'Vekt' sind alle "
    "'Vektor Consulting GmbH'.\n"
    "Verschiedene Entitäten sind sie, wenn es sich um zwei Produkte desselben "
    "Herstellers, zwei Vorhaben mit ähnlichem Namen, zwei Dokumente derselben Gattung "
    "oder zwei Personen mit gleichem Nachnamen handelt.\n"
    "Im Zweifel lautet die Antwort false.\n"
    'Antworte ausschliesslich als JSON: {"same": true oder false}'
)


def build_adjudication_prompt(entity_type: str, left: str, right: str) -> str:
    return f"Typ: {entity_type}\nName A: {left}\nName B: {right}"


def parse_adjudication(reply: str) -> bool:
    """Whether the model said "same". Anything unusable means "not the same".

    Total, and biased the same way as the prompt: an unparseable reply must leave two
    nodes rather than fuse them on a coin toss.
    """
    try:
        payload = json.loads(reply)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("same") is True


class AdjudicationModel(Protocol):
    """The chat-tier model. Low volume by construction — only the band reaches it — and
    the stakes per call are high, so this is the one place in the resolution path worth
    the better model."""

    def complete(self, system: str, user: str, *, json_object: bool = False) -> str: ...


def model_adjudicator(model: AdjudicationModel) -> Adjudicator:
    """An `Adjudicator` backed by a model, with its failures read as "not the same".

    A transport failure must not fuse two nodes, and must not abort a pass over a
    corpus whose extraction has already been paid for.
    """

    def adjudicate(entity_type: str, left: str, right: str) -> bool:
        try:
            reply = model.complete(
                ADJUDICATION_SYSTEM_PROMPT,
                build_adjudication_prompt(entity_type, left, right),
                json_object=True,
            )
        # Deliberately broad, as elsewhere in this pipeline.
        except Exception:  # noqa: BLE001
            return False
        return parse_adjudication(reply)

    return adjudicate
