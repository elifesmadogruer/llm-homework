import pytest

from moderation.preprocessing.normalize import (
    ascii_fold,
    collapse_repeats,
    decode_leet,
    join_spaced_letters,
    light_normalize,
    normalize,
    tokenize,
    turkish_lower,
)


def test_turkish_lower():
    assert turkish_lower("IŞIK İSTANBUL") == "ışık istanbul"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("s1kt1r", "siktir"),
        ("@mk", "amk"),
        ("g3r1zek4l1", "gerizekali"),
        ("$alak", "salak"),
        ("@q", "aq"),
        ("@ptal", "aptal"),
        ("salak!", "salak!"),
        ("yapmadık!yapacağınız", "yapmadık!yapacağınız"),
        ("2014te", "2014te"),
        ("100lerce", "100lerce"),
        ("saat 15", "saat 15"),
        ("3gün 4te got7 s400", "3gün 4te got7 s400"),
        ("5bingidamuhendisi", "5bingidamuhendisi"),
    ],
)
def test_decode_leet(raw, expected):
    assert decode_leet(raw) == expected


def test_collapse_repeats_keeps_double_letters():
    assert collapse_repeats("saaaçmaaaa") == "saçma"
    assert collapse_repeats("saat") == "saat"


def test_join_spaced_letters():
    assert join_spaced_letters("L Ü T F E N  B O K") == "LÜTFEN  BOK"
    assert join_spaced_letters("o ve ben") == "o ve ben"


def test_normalize_drops_handles_and_emails():
    assert normalize("@akparti @rt_erdogan mail at x.y@gmail.com @USER_en") == "mail at"


def test_ascii_fold():
    assert ascii_fold("çiğ ışık öşü") == "cig isik osu"


def test_normalize_full_pipeline():
    raw = "@USER @USER Ama terbiyemizi bozmuyoruz  L Ü T F E N  B O K https://t.co/x #Yenikapı"
    assert normalize(raw) == "ama terbiyemizi bozmuyoruz lütfen bok yenikapı"


def test_normalize_noisy_insult():
    assert normalize("SALAAAAK m1s1n s3n") == "salak misin sen"
    assert normalize("SALAAAAK m1s1n s3n", fold_ascii=True) == "salak misin sen"


def test_light_normalize_keeps_case_and_noise():
    assert light_normalize("@USER Saaaçma!! http://x.com") == "Saaaçma!!"


def test_tokenize():
    assert tokenize("hadi be, s*ktir git! 3'te gel") == ["hadi", "be", "s*ktir", "git", "3'te", "gel"]
