"""The schema is re-applied constantly during development; a statement that is not
idempotent, or a vector index that does not match the embedding width, fails silently
until a demo query returns nothing."""

from graph_rag.graph.schema import (
    CHUNK_FULLTEXT_INDEX,
    CHUNK_VECTOR_INDEX,
    ENTITY_VECTOR_INDEX,
    constraint_drift,
    index_drift,
    schema_statements,
)


def test_every_statement_is_idempotent():
    for statement in schema_statements(1024):
        assert "IF NOT EXISTS" in statement.cypher, statement.name


def test_vector_indexes_match_the_configured_embedding_width():
    statements = {s.name: s.cypher for s in schema_statements(1024)}
    for name in (CHUNK_VECTOR_INDEX, ENTITY_VECTOR_INDEX):
        assert "`vector.dimensions`: 1024" in statements[name]
        assert "'cosine'" in statements[name]


def test_the_lexical_leg_has_a_fulltext_index_over_chunk_text():
    statements = {s.name: s.cypher for s in schema_statements(1024)}
    assert "FOR (c:Chunk) ON EACH [c.text]" in statements[CHUNK_FULLTEXT_INDEX]


def _index(name, config):
    return {"name": name, "type": "VECTOR", "options": {"indexConfig": config}}


def test_drift_is_detected_for_the_configurations_if_not_exists_would_ignore():
    """`CREATE ... IF NOT EXISTS` silently keeps an existing index's options, so a
    wrong width or analyzer survives re-initialisation unnoticed."""
    wrong_width = _index(
        CHUNK_VECTOR_INDEX,
        {"vector.dimensions": 512, "vector.similarity_function": "COSINE"},
    )
    wrong_similarity = _index(
        CHUNK_VECTOR_INDEX,
        {"vector.dimensions": 1024, "vector.similarity_function": "EUCLIDEAN"},
    )
    wrong_analyzer = {
        "name": CHUNK_FULLTEXT_INDEX,
        "type": "FULLTEXT",
        "options": {"indexConfig": {"fulltext.analyzer": "standard"}},
    }
    assert index_drift(wrong_width, 1024) is not None
    assert index_drift(wrong_similarity, 1024) is not None
    assert index_drift(wrong_analyzer, 1024) is not None


def test_a_correctly_configured_index_is_not_dropped():
    correct = _index(
        CHUNK_VECTOR_INDEX,
        {"vector.dimensions": 1024, "vector.similarity_function": "COSINE"},
    )
    assert index_drift(correct, 1024) is None


def test_a_failed_index_counts_as_drift():
    """A FAILED index keeps its correct options, so only its state gives it away — and
    without that, `status` refuses to call the graph ready while `init` finds nothing to
    repair, looping the operator between the two."""
    failed = _index(
        CHUNK_VECTOR_INDEX,
        {"vector.dimensions": 1024, "vector.similarity_function": "COSINE"},
    ) | {"state": "FAILED"}
    assert index_drift(failed, 1024) is not None


def test_a_populating_index_is_not_drift():
    """It is rebuilding, not misconfigured; dropping it would restart the rebuild."""
    populating = _index(
        CHUNK_VECTOR_INDEX,
        {"vector.dimensions": 1024, "vector.similarity_function": "COSINE"},
    ) | {"state": "POPULATING"}
    assert index_drift(populating, 1024) is None


def test_a_constraint_over_the_wrong_properties_is_detected_as_drift():
    """`CREATE CONSTRAINT ... IF NOT EXISTS` matches on the name alone, so a constraint
    created before entity scope existed survives re-initialisation over two properties
    and then refuses the three-property writes it was widened to allow — a graph that
    silently loses one of two same-named projects."""
    stale = {
        "name": "entity_identity_unique",
        "properties": ["normalized_name", "type"],
    }
    current = {
        "name": "entity_identity_unique",
        "properties": ["type", "scope", "normalized_name"],
    }

    assert constraint_drift(stale) is not None
    assert constraint_drift(current) is None
