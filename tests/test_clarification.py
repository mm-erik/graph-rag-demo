"""Asking back is only worth doing when the alternative is a confident wrong answer, so
both directions are guarded: the cases that must interrupt, and — more easily broken —
the cases that must not. A clarifier that fires on a question the corpus can answer is
worse than none, because every question reaches it before anything is retrieved."""

from graph_rag.clarification import Folder, clarify, is_enumerating, tokens

#: The machine half's shape: a model code reused under two machine types at two
#: manufacturers, and a machine type code reused by two manufacturers.
CORPUS = (
    Folder("Hagedorn", "c/Hagedorn", 1, ("Hagedorn",), ("XTC27", "VR88D")),
    Folder("XTC27", "c/Hagedorn/XTC27", 2, ("Hagedorn", "XTC27"), ("B2BC", "B2BD", "L7Q"), "Maschinentyp"),
    Folder("B2BC", "c/Hagedorn/XTC27/B2BC", 3, ("Hagedorn", "XTC27", "B2BC"), (), "Modell"),
    Folder("B2BD", "c/Hagedorn/XTC27/B2BD", 3, ("Hagedorn", "XTC27", "B2BD"), (), "Modell"),
    Folder("L7Q", "c/Hagedorn/XTC27/L7Q", 3, ("Hagedorn", "XTC27", "L7Q"), (), "Modell"),
    Folder("VR88D", "c/Hagedorn/VR88D", 2, ("Hagedorn", "VR88D"), ("K4R",), "Maschinentyp"),
    Folder("K4R", "c/Hagedorn/VR88D/K4R", 3, ("Hagedorn", "VR88D", "K4R"), (), "Modell"),
    Folder("Pleickner", "c/Pleickner", 1, ("Pleickner",), ("ZM3K",)),
    Folder("ZM3K", "c/Pleickner/ZM3K", 2, ("Pleickner", "ZM3K"), ("B2BC", "G5T"), "Maschinentyp"),
    Folder("B2BC", "c/Pleickner/ZM3K/B2BC", 3, ("Pleickner", "ZM3K", "B2BC"), (), "Modell"),
    Folder("G5T", "c/Pleickner/ZM3K/G5T", 3, ("Pleickner", "ZM3K", "G5T"), (), "Modell"),
    Folder("Ostmark", "c/Ostmark", 1, ("Ostmark",), ("XTC27", "QH4L")),
    Folder("XTC27", "c/Ostmark/XTC27", 2, ("Ostmark", "XTC27"), ("H3M",), "Maschinentyp"),
    Folder("H3M", "c/Ostmark/XTC27/H3M", 3, ("Ostmark", "XTC27", "H3M"), (), "Modell"),
    Folder("QH4L", "c/Ostmark/QH4L", 2, ("Ostmark", "QH4L"), ("P6V", "P6VX"), "Maschinentyp"),
    Folder("P6V", "c/Ostmark/QH4L/P6V", 3, ("Ostmark", "QH4L", "P6V"), (), "Modell"),
    Folder("P6VX", "c/Ostmark/QH4L/P6VX", 3, ("Ostmark", "QH4L", "P6VX"), (), "Modell"),
)


def test_a_model_code_belonging_to_two_machines_is_asked_about():
    """`B2BC` is a Hagedorn mixer-dryer and a Pleickner assembly cell. Answering picks
    one branch and sounds certain about it."""
    clarification = clarify("was weisst du zu B2BC?", CORPUS)

    assert clarification is not None
    assert clarification.kind == "ambiguous"
    assert clarification.named == "B2BC"
    assert set(clarification.paths) == {"c/Hagedorn/XTC27/B2BC", "c/Pleickner/ZM3K/B2BC"}
    # The context is what makes two identical names tellable apart on screen.
    assert {choice.context for choice in clarification.choices} == {
        "Hagedorn › XTC27",
        "Pleickner › ZM3K",
    }


def test_a_machine_type_without_a_model_offers_the_models():
    """The ask: a machine code with the model missing is answered with the models that
    exist — here from both manufacturers carrying the code, each labelled."""
    clarification = clarify("was weisst du zu ZM3K?", CORPUS)

    assert clarification is not None
    assert clarification.kind == "underspecified"
    assert [choice.name for choice in clarification.choices] == ["B2BC", "G5T"]
    assert all(choice.level == "Modell" for choice in clarification.choices)


def test_a_machine_type_and_a_model_together_resolve_each_other():
    """Both names are ambiguous alone — two XTC27s, two B2BCs — and neither is
    ambiguous in combination. Asking anyway would bounce a question that is already
    precise, which is the failure that gets a clarifier switched off."""
    assert clarify("was wiegt ein XTC27 B2BC mit Fahrwerk?", CORPUS) is None


def test_a_fully_specified_leaf_is_never_questioned():
    assert clarify("was sind die varianten von QH4L P6V?", CORPUS) is None


def test_a_longer_code_is_not_matched_by_a_shorter_one():
    """`P6V` is a prefix of `P6VX`. A substring test would answer a question about one
    model from the other model's documents — silently, and with citations."""
    clarification = clarify("was weisst du zu P6VX?", CORPUS)

    assert clarification is None
    assert tokens("QH4L-P6VX") == ("qh4l", "p6vx")


def test_a_question_naming_nothing_in_the_corpus_is_left_alone():
    assert clarify("Welche Risiken wurden gemeldet?", CORPUS) is None


def test_a_question_asking_for_the_list_is_answered_not_bounced():
    """`VR88D` has one model below it, so the question is underspecified by the rule —
    but "welche Modelle gibt es" is asking for exactly that list, and demanding one of
    its items first would be absurd."""
    assert clarify("Welche Modelle gibt es zu VR88D?", CORPUS) is None
    assert is_enumerating("Welche Modelle gibt es zu VR88D?")
    assert not is_enumerating("was wiegt ein VR88D K4R?")


def test_the_deepest_named_folder_decides_what_is_asked():
    """A question naming a manufacturer and a model is a question about the model.
    Clarifying the manufacturer would answer a question nobody asked."""
    assert clarify("was weisst du zu Hagedorn B2BD?", CORPUS) is None


def test_names_that_never_co_occur_still_offer_what_the_corpus_has():
    """`ZM3K` has no `L7Q` under it. Narrowing to nothing and reporting "no options"
    would be the one answer that is certainly useless."""
    clarification = clarify("was weisst du zu ZM3K L7Q?", CORPUS)

    assert clarification is None or clarification.choices
