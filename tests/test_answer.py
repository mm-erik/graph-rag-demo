"""The silent failures of the join between the two legs.

Every defect guarded here produces a plausible answer. The merge is the last thing that
happens before a dozen chunks become one paragraph of German prose, and nothing
downstream of it can tell that a chunk was dropped, counted twice, or stripped of the
folder ancestry that says which client it belongs to. The answer still reads well. It is
simply about the wrong client, or about one client where the question asked about two —
which is the exact failure this whole project exists to disprove.

The fusion arithmetic itself is already guarded in `test_retrieval.py`; what is checked
here is what this layer does *with* it — the order the legs are offered in, the point at
which the cap is applied, and what survives the merge.
"""

from __future__ import annotations

from graph_rag.ancestry import Ancestor
from graph_rag.answer import (
    Answer,
    ContextChunk,
    Legs,
    Parameters,
    Source,
    build_user_prompt,
    cited_numbers,
    generate_answer,
    merge_context,
)
from graph_rag.entities import EntityIdentity
from graph_rag.retrieval import Chunk, LegResult, RetrievedChunk
from graph_rag.traversal import (
    GraphEntity,
    ReachedEntity,
    TraversalResult,
    TraversedChunk,
)


def _chunk(chunk_id: str, client: str = "Nordlicht Energie AG") -> Chunk:
    return Chunk(
        id=chunk_id,
        document_path=f"corpus/{client}/2023/bericht.pdf",
        filename="bericht.pdf",
        index=0,
        page=3,
        text=f"Text von {chunk_id}.",
        ancestry=[
            Ancestor(name=client, level="Kunde"),
            Ancestor(name="2023", level="Jahr"),
        ],
    )


def _reached(name: str, hop: int) -> ReachedEntity:
    identity = EntityIdentity(type="konzept", normalized_name=name)
    return ReachedEntity(
        entity=GraphEntity(identity=identity, name=name.title(), mentions=1), hop=hop
    )


def _search(*chunks: Chunk) -> LegResult:
    return LegResult(
        question="?",
        query="?",
        hits={},
        results=[
            RetrievedChunk(chunk=chunk, score=1.0, ranks={}) for chunk in chunks
        ],
    )


def _traversal(*chunks: Chunk, via: tuple[ReachedEntity, ...] = ()) -> TraversalResult:
    return TraversalResult(
        question="?",
        hops=2,
        entities=[],
        matches=[],
        reached=list(via),
        edges=[],
        results=[
            TraversedChunk(chunk=chunk, via=via, score=1.0) for chunk in chunks
        ],
    )


def _ids(context: list[ContextChunk]) -> list[str]:
    return [found.chunk.id for found in context]


def test_the_cap_is_applied_after_the_merge_and_not_to_each_leg():
    """The defect this is the whole guard against: capping per leg.

    With a cap of two and a leg each, a per-leg cap keeps `s1, s2` and `t1, t2` and
    then has four chunks for two slots — so whichever leg is merged first takes the
    context and the other client vanishes. Capping the *merged* order keeps one from
    each, which on a cross-client question is the answer.
    """
    context = merge_context(
        search=_search(_chunk("s1"), _chunk("s2"), _chunk("s3")),
        traversal=_traversal(_chunk("t1"), _chunk("t2"), _chunk("t3")),
        max_chunks=2,
    )

    assert _ids(context) == ["t1", "s1"]


def test_a_chunk_both_legs_found_appears_once_carrying_both_ranks():
    """Deduplication, and the evidence that it happened.

    A duplicate spends two of a dozen context slots on one passage and reads to the
    model as two documents agreeing. Collapsing it without recording both ranks is the
    opposite failure: the one chunk both legs independently ranked first is the
    strongest result in the system, and nothing would say so.
    """
    shared = _chunk("shared")
    context = merge_context(
        search=_search(_chunk("s1"), shared),
        traversal=_traversal(shared, _chunk("t1")),
    )

    assert _ids(context).count("shared") == 1
    found = next(entry for entry in context if entry.chunk.id == "shared")
    assert found.rank_in(Source.SEARCH) == 2
    assert found.rank_in(Source.TRAVERSAL) == 1
    # And it outranks both legs' lone favourites, which is why fusing is worth doing.
    assert _ids(context)[0] == "shared"


def test_ties_fall_to_the_traversal_leg():
    """Both legs' first place scores identically and the formula says nothing further.

    The order `Legs.sources` declares is what breaks it, and that it is declared rather
    than emergent is the point: left to dictionary order, the winner of a tie would
    change with an unrelated edit to the merge and nothing would report it.
    """
    context = merge_context(search=_search(_chunk("s1")), traversal=_traversal(_chunk("t1")))

    assert _ids(context) == ["t1", "s1"]


def test_the_entities_a_chunk_was_reached_through_survive_the_merge():
    """The traversal leg's provenance, which tickets 14 and 15 draw.

    A merge that kept only chunks would answer the question and lose the only evidence
    that the graph contributed anything — and the loss is invisible in the prose.
    """
    via = (_reached("projekt atlas", 0), _reached("taktzeitkonzept", 1))
    context = merge_context(search=None, traversal=_traversal(_chunk("t1"), via=via))

    assert context[0].via == via


def test_a_leg_that_was_not_run_contributes_nothing_even_if_a_result_is_passed():
    """Isolation has to hold at the merge, not only at the call.

    `--legs search` is how the demo shows what the graph adds by taking it away. A
    merge that read both result objects regardless would quietly keep answering from
    the entity leg, and the comparison would show no difference — the most convincing
    possible demonstration of the wrong conclusion.
    """
    context = merge_context(
        search=_search(_chunk("s1")),
        traversal=_traversal(_chunk("t1")),
        legs=Legs.SEARCH,
    )

    assert _ids(context) == ["s1"]


def test_every_context_chunk_carries_its_folder_ancestry_as_a_citable_line():
    """`Kunde: X › Jahr: 2023`, never a path.

    The structural advantage is legible in exactly one place, and an answer that can
    only recite `corpus/...` has thrown it away while still sounding cited.
    """
    context = merge_context(search=_search(_chunk("s1")), traversal=None)

    assert context[0].location == "Kunde: Nordlicht Energie AG › Jahr: 2023"


def test_the_prompt_numbers_its_sources_in_context_order():
    """The citation scheme. `[Quelle 2]` resolves by position into `context`, so a
    prompt numbered in any other order would attach every rendered citation to the
    wrong document — with both the number and the document existing, and nothing
    raising."""
    context = merge_context(
        search=_search(_chunk("s1", client="Brandt & Söhne")),
        traversal=_traversal(_chunk("t1")),
    )

    prompt = build_user_prompt("Was verbindet die Projekte?", context)

    assert prompt.index("--- Quelle 1 ---") < prompt.index("--- Quelle 2 ---")
    assert prompt.index("Kunde: Nordlicht Energie AG") < prompt.index("Kunde: Brandt & Söhne")
    assert "Was verbindet die Projekte?" in prompt


def test_an_empty_context_is_never_put_to_the_model():
    """Handed no sources and told to answer only from sources, the chat tier answers
    from its own knowledge of consulting reports instead — fluently, about a client
    that does not exist, with nothing on screen to check it against."""

    class Exploding:
        def complete(self, system: str, user: str) -> str:
            raise AssertionError("the model must not be called without sources")

    text = generate_answer("Was verbindet die Projekte?", [], Exploding())

    assert "nichts gefunden" in text


def test_a_citation_pointing_at_no_source_is_reported():
    """The one check on grounding this path can make without a human reading every
    chunk: a model citing `[Quelle 9]` against two sources has stopped quoting the
    context, and an uncaught invented citation looks exactly like a real one."""
    answer = Answer(
        question="?",
        text="Laut dem Bericht [Quelle 2] und ausserdem [Quelle 9].",
        context=merge_context(
            search=_search(_chunk("s1"), _chunk("s2")), traversal=None
        ),
        parameters=Parameters(),
    )

    assert answer.citations == [2, 9]
    assert answer.uncited == [9]


def test_citations_are_found_however_the_model_spaces_them():
    assert cited_numbers("a [Quelle 1] b [ quelle  2 ] c [Quelle 1]") == [1, 2]


def test_the_branches_an_answer_drew_on_are_counted_from_the_context():
    """Two clients in one context is the project's central claim, re-derived per
    question rather than asserted once."""
    answer = Answer(
        question="?",
        text="",
        context=merge_context(
            search=_search(_chunk("s1", client="Brandt & Söhne")),
            traversal=_traversal(_chunk("t1")),
        ),
        parameters=Parameters(),
    )

    assert answer.branches == ["Brandt & Söhne", "Nordlicht Energie AG"]


# The service's configured floor. `ANSWER_LEGS=traversal` is how a running application
# is turned into the graph-only case, so the two directions below are the feature: a
# bare question must take the configuration, and a question that names a knob must not.


def test_a_bare_request_takes_the_services_configured_legs():
    """A class-level default would silently overrule the configuration on every request
    that omitted the field — the service would keep answering from all three legs while
    reporting that it was graph-only."""
    from graph_rag.api import TuningRequest
    from graph_rag.config import AnswerSettings

    configured = AnswerSettings(
        legs="traversal", mode="hybrid", top_k=20, hops=2, max_chunks=12
    )

    parameters = TuningRequest().parameters(configured)

    assert parameters.legs is Legs.TRAVERSAL
    assert parameters.top_k == 20


def test_a_request_that_names_a_leg_still_overrides_the_configuration():
    """Isolation stays per question: the configuration moves the floor, it does not
    take the comparison away."""
    from graph_rag.api import TuningRequest
    from graph_rag.config import AnswerSettings

    configured = AnswerSettings(
        legs="traversal", mode="hybrid", top_k=20, hops=2, max_chunks=12
    )

    parameters = TuningRequest(legs=Legs.SEARCH).parameters(configured)

    assert parameters.legs is Legs.SEARCH
    assert parameters.top_k == 20  # the knobs it did not name are untouched
