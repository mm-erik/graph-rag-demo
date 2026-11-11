"""Three silent failures guard this file.

The first is a *front-loaded excerpt*. Every report in this corpus opens with the same
boilerplate cover page, so a selection that takes the first N characters of a document
summarises the letterhead of all of them — a hundred documents whose summaries are
interchangeable, which is precisely the failure the summaries exist to prevent. Nothing
raises; the summaries simply stop distinguishing anything.

The second is an *unclean reply*. A property written once at ingestion and read back
only by the model is never looked at again, so a summary carrying a code fence or a
"Zusammenfassung:" preamble survives all the way into a generated answer.

The third is the *cache decision*. Too eager and a re-ingest re-summarises the whole
corpus; too lazy and a document whose bytes changed keeps the summary of its previous
version for good.
"""

from graph_rag.summaries import (
    DocumentState,
    clean_summary,
    needs_summary,
    select_excerpt,
)


def _chunks(count: int) -> list[str]:
    return [f"Abschnitt {index}" for index in range(count)]


def test_the_excerpt_spans_the_whole_document_rather_than_its_opening():
    """Every report shares a cover page; what distinguishes them is further in."""
    excerpt = select_excerpt(_chunks(40), max_chunks=4, max_characters=10_000)

    assert excerpt == ["Abschnitt 0", "Abschnitt 13", "Abschnitt 26", "Abschnitt 39"]


def test_a_document_with_fewer_chunks_than_the_cap_is_taken_whole():
    assert select_excerpt(_chunks(3), max_chunks=8, max_characters=10_000) == _chunks(3)


def test_the_budget_is_shared_out_rather_than_spent_on_the_first_chunk():
    """A single long opening section must not consume the excerpt and push every later
    section out of it — the spread is the point, so the budget is divided across the
    chosen sections rather than filled front to back."""
    chunks = ["A" * 400, "B" * 400, "C" * 400, "D" * 400]

    excerpt = select_excerpt(chunks, max_chunks=4, max_characters=400)

    assert len(excerpt) == 4
    assert [text[0] for text in excerpt] == ["A", "B", "C", "D"]
    assert sum(len(text) for text in excerpt) <= 400


def test_one_section_is_still_excerpted_when_only_one_is_asked_for():
    assert select_excerpt(_chunks(40), max_chunks=1, max_characters=10_000) == [
        "Abschnitt 0"
    ]


def test_a_fenced_or_prefaced_reply_is_stored_as_the_summary_alone():
    """The chat model is asked for prose and sometimes answers with a markdown fence or
    a "Zusammenfassung:" preamble. Both would be read back verbatim into a generated
    answer."""
    reply = "```\nZusammenfassung: Der Bericht beschreibt Projekt Atlas.\n```"

    assert clean_summary(reply, 500) == "Der Bericht beschreibt Projekt Atlas."


def test_an_unusable_reply_becomes_no_summary_at_all():
    """Better an absent property than one holding an apology — a stored non-summary is
    indistinguishable from a real one at query time."""
    assert clean_summary("", 500) == ""
    assert clean_summary("   \n  ", 500) == ""


def test_a_runaway_reply_is_capped_at_a_sentence_boundary():
    reply = "Erster Satz. Zweiter Satz. Dritter Satz, der viel zu lang geraten ist."

    assert clean_summary(reply, 30) == "Erster Satz. Zweiter Satz."


def _state(**overrides) -> DocumentState:
    base = {
        "path": "/corpus/Nordlicht/2023/Atlas/projektbericht-2023.pdf",
        "content_hash": "abc",
        "summary": "Der Bericht beschreibt Projekt Atlas.",
        "summary_hash": "abc",
    }
    return DocumentState(**{**base, **overrides})


def test_an_unchanged_document_is_not_summarised_again():
    """One chat call per document per ingest is the whole cost of this pass; a re-ingest
    of an untouched corpus must pay none of it."""
    assert needs_summary(_state(), refresh=False) is False


def test_a_document_whose_bytes_changed_is_summarised_again():
    """Keyed on the content hash already written by the structural pass: the summary
    records which version of the file it describes."""
    assert needs_summary(_state(content_hash="def"), refresh=True) is True
    assert needs_summary(_state(content_hash="def"), refresh=False) is True


def test_a_document_with_no_summary_is_summarised_whatever_the_hash_says():
    """A failed call leaves the hash of a summary that was never written; without this
    the document would stay unsummarised for the rest of the corpus's life."""
    assert needs_summary(_state(summary=None), refresh=False) is True
    assert needs_summary(_state(summary="  "), refresh=False) is True


def test_refresh_re_summarises_a_document_the_cache_would_have_skipped():
    """`--no-cache` is the escape hatch for a prompt change, which no hash can see."""
    assert needs_summary(_state(), refresh=True) is True
