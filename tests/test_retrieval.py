"""One silent failure guards this file.

Reciprocal rank fusion is the only thing standing between two independently ranked
lists and the single order a demo answer is built from. Fuse them wrongly — invert a
rank, drop the list that only one leg returned, let ties resolve by dictionary order —
and nothing raises. Retrieval simply answers from the second-best chunk, which on this
corpus means the second-best *client*, which is the exact failure the whole project
exists to disprove.

The expected orders below are worked by hand from the published formula
(`score = sum over legs of 1 / (k + rank)`, ranks 1-based), never recomputed the way
the implementation computes them.
"""

from graph_rag.retrieval import reciprocal_rank_fusion


def _order(rankings, **kwargs):
    return [fused.key for fused in reciprocal_rank_fusion(rankings, **kwargs)]


def test_a_chunk_ranked_well_by_both_legs_beats_one_ranked_first_by_only_one():
    """The property RRF exists for. With k=60, "b" scores 1/61 + 1/62 = 0.0325 against
    "a"'s single 1/61 = 0.0164, so agreement between the legs outranks a lone first
    place — the whole reason the lexical leg can rescue a question dense retrieval got
    wrong."""
    rankings = {"vector": ["a", "b", "c"], "fulltext": ["b", "d"]}

    assert _order(rankings) == ["b", "a", "d", "c"]


def test_a_chunk_only_one_leg_returned_is_still_in_the_result():
    """The lexical leg's whole job on this corpus is to surface the one document a
    distinctive literal token names and the embeddings all miss. A fusion that kept
    only the keys both legs agree on would discard exactly that chunk."""
    rankings = {"vector": ["a"], "fulltext": ["z"]}

    assert sorted(_order(rankings)) == ["a", "z"]


def test_tied_chunks_keep_the_order_the_legs_put_them_in():
    """"a" and "z" are each rank 1 in one leg and absent from the other, so their
    scores are identical. The order must then follow the legs — vector's first place
    ahead of full-text's — and never the keys' own ordering, or a re-run under a
    different chunk id would reshuffle the top of a demo."""
    assert _order({"vector": ["z"], "fulltext": ["a"]}) == ["z", "a"]
    assert _order({"vector": ["a"], "fulltext": ["z"]}) == ["a", "z"]


def test_each_result_records_the_rank_every_leg_gave_it():
    """What makes the leg inspectable: a chunk the vector index ranked eighth and the
    full-text index ranked first is the demonstrable case for running both."""
    fused = reciprocal_rank_fusion({"vector": ["a", "b"], "fulltext": ["b"]})

    assert {result.key: dict(result.ranks) for result in fused} == {
        "b": {"vector": 2, "fulltext": 1},
        "a": {"vector": 1},
    }


def test_one_leg_on_its_own_is_passed_through_in_its_own_order():
    """Each leg is runnable in isolation, so fusing a single list must not reorder it."""
    assert _order({"vector": ["a", "b", "c"]}) == ["a", "b", "c"]


def test_an_empty_leg_contributes_nothing_and_fuses_cleanly():
    """A question whose terms the German analyser strips entirely returns no lexical
    hits at all; the semantic leg must still answer."""
    assert _order({"vector": ["a", "b"], "fulltext": []}) == ["a", "b"]
    assert _order({"vector": [], "fulltext": []}) == []


def test_k_damps_how_far_a_single_first_place_outweighs_agreement():
    """`k` is the one tuning knob in the formula. At k=0 a lone first place scores 1.0
    and beats a chunk ranked second by both legs (1/2 + 1/2 = 1.0 — equal, so the leg
    order decides); the default k=60 reverses that. Pinned because a k read from the
    wrong place would change retrieval order and nothing else."""
    rankings = {"vector": ["a", "b"], "fulltext": ["c", "b"]}

    assert _order(rankings, k=0) == ["a", "b", "c"]
    assert _order(rankings) == ["b", "a", "c"]
