"""Regex normalization for noisy Turkish chat messages.

Pipeline (each step can be toggled):
    1. drop mentions, e-mails and URLs, strip '#' from hashtags
    2. join spaced-out letters        "L Ü T F E N"  -> "LÜTFEN"
    3. Turkish-aware lowercasing      "I" -> "ı", "İ" -> "i"
    4. leetspeak decoding             "s1kt1r", "@mk" -> "siktir", "amk"
    5. collapse letter repetitions    "saaaçmaaa"   -> "saçma"
    6. optional ASCII folding         "çğıöşü"     -> "cgiosu"
    7. whitespace cleanup
"""
import re

# anonymized @USER tokens, real handles (5+ chars) and e-mails; short "@mk", "@ptal" stay for leet
MENTION_RE = re.compile(r"\S+@\S+\.\S+|\S*@user\w*|@\w{5,}", re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+|\burl\b", re.IGNORECASE)
HASHTAG_RE = re.compile(r"#(\w+)")
# three or more single letters separated by exactly one space
SPACED_RE = re.compile(r"(?<![^\W\d_])(?:[^\W\d_] ){2,}[^\W\d_](?![^\W\d_])")
REPEAT_RE = re.compile(r"([^\W\d_])\1{2,}")
TOKEN_RE = re.compile(r"[^\W_]+(?:[*'][^\W_]+)*")
LEET_TOKEN_RE = re.compile(r"[\w@$|*]+")
WS_RE = re.compile(r"\s+")

LEET_MAP = {"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "|": "i"}
ASCII_MAP = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def ascii_fold(text: str) -> str:
    return text.translate(ASCII_MAP)


def join_spaced_letters(text: str) -> str:
    return SPACED_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def collapse_repeats(text: str) -> str:
    return REPEAT_RE.sub(r"\1", text)


def _decode_leet_token(token: str) -> str:
    if not any(c.isalpha() for c in token):
        return token
    n = len(token)
    alpha = [c.isalpha() for c in token]
    digit_run = [c.isdigit() for c in token]
    candidates = []  # (index, inside_word)
    for i, c in enumerate(token):
        if c not in LEET_MAP:
            continue
        # multi-digit numbers ("5000", "s400", "2014te") are never leetspeak
        if c.isdigit() and ((i > 0 and digit_run[i - 1]) or (i + 1 < n and digit_run[i + 1])):
            continue
        prev_alpha = i > 0 and alpha[i - 1]
        next_alpha = i + 1 < n and alpha[i + 1]
        if prev_alpha and next_alpha:
            candidates.append((i, True))
        elif prev_alpha or next_alpha:
            candidates.append((i, False))
    # '@'/'$' as a first letter is a strong signal ("@mk", "$alak"); a digit at a
    # word edge ("3gün", "got7") only counts when the word has in-word leet too.
    has_inner = any(inner for _, inner in candidates)
    chars = list(token)
    for i, inner in candidates:
        if inner or has_inner or token[i] in "@$":
            chars[i] = LEET_MAP[token[i]]
    return "".join(chars)


def decode_leet(text: str) -> str:
    return LEET_TOKEN_RE.sub(lambda m: _decode_leet_token(m.group(0)), text)


def normalize(
    text: str,
    *,
    lower: bool = True,
    leet: bool = True,
    repeats: bool = True,
    spaced: bool = True,
    fold_ascii: bool = False,
) -> str:
    text = MENTION_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)
    if spaced:
        text = join_spaced_letters(text)
    if lower:
        text = turkish_lower(text)
    if leet:
        text = decode_leet(text)
    if repeats:
        text = collapse_repeats(text)
    if fold_ascii:
        text = ascii_fold(text)
    return WS_RE.sub(" ", text).strip()


def light_normalize(text: str) -> str:
    """Minimal cleanup for subword models that handle casing and noise themselves."""
    return normalize(text, lower=False, leet=False, repeats=False, spaced=True)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)
