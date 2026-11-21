import pytest

from moderation.models.lexicon import LexiconModerator


@pytest.fixture(scope="module")
def model():
    return LexiconModerator(threshold=0.5)


@pytest.mark.parametrize("text", ["sen tam bir SALAAAK sın", "s1kt1r git", "@mk ya", "gerizekali misin", "şerefsz herif"])
def test_offensive_messages_are_flagged(model, text):
    assert model.predict([text])[0] == 1


@pytest.mark.parametrize("text", ["geri dön lütfen", "götür bunu", "malzeme aldım", "bu gerçek bir yaprak", "sıktığım portakal"])
def test_clean_messages_are_not_flagged(model, text):
    assert model.predict([text])[0] == 0


def test_explain_reports_match_source(model):
    hits = model.explain("şerefsz salak")
    assert [(tok, src) for tok, _, src in hits] == [("şerefsz", "fuzzy"), ("salak", "seed")]
