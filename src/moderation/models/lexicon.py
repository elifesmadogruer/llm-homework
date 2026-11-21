"""Rule baseline: edit-distance lexicon matcher over normalized tokens.

Two sources of offensive terms:
  * a hand-written seed list of roots (prefix match, plus edit distance 1 for
    roots of 7+ letters, so misspellings like "gerizakalı" or "şerefsz" still match);
  * terms mined from the training set: tokens with a high P(OFF | token)
    (exact match, scored with that probability).
A message's score is 1 - prod(1 - s_i) over all matched tokens.

Tokens written with Turkish letters are matched against the Turkish lexicon, so
"sıktığım" (squeezed) is not confused with "siktiğim"; ASCII-only tokens such as
"gerizekali" are matched against the ASCII-folded lexicon.
"""
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from moderation.models.base import Moderator
from moderation.preprocessing.normalize import ascii_fold, normalize, tokenize

SEED_PATH = Path(__file__).resolve().parents[1] / "resources" / "offensive_seed.txt"
SEED_SCORE = 0.9
FUZZY_SCORE = 0.7
MIN_FUZZY_LEN = 7


def load_seed(path: Path = SEED_PATH) -> tuple[set[str], set[str]]:
    prefix, exact = set(), set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        (exact if line.endswith("=") else prefix).add(line.rstrip("="))
    return prefix, exact


def preprocess(text: str) -> list[str]:
    return tokenize(normalize(text))


def mine_terms(texts, labels, min_count: int = 15, min_ratio: float = 0.6, smoothing: float = 1.0):
    """Tokens that appear mostly in offensive messages, and tokens that are clearly clean."""
    off, total = Counter(), Counter()
    for text, label in zip(texts, labels):
        for tok in set(preprocess(text)):
            total[tok] += 1
            off[tok] += int(label)
    prior = float(np.mean(labels))
    mined, clean = {}, set()
    for tok, n in total.items():
        # smoothed P(OFF | token), pulled toward the class prior for rare tokens
        ratio = (off[tok] + smoothing * prior) / (n + smoothing)
        if n >= min_count and ratio >= min_ratio and len(tok) > 1:
            mined[tok] = round(ratio, 3)
        elif n >= 20 and ratio < 0.3:
            clean.add(tok)
    return mined, clean


class _Lexicon:
    def __init__(self, prefix: set[str], exact: set[str]):
        self.prefix, self.exact = prefix, exact
        self.fuzzy_roots = [w for w in prefix if len(w) >= MIN_FUZZY_LEN]

    def exact_hit(self, tok: str) -> bool:
        return tok in self.exact or any(tok.startswith(w) for w in self.prefix)

    def fuzzy_hit(self, tok: str) -> bool:
        for w in self.fuzzy_roots:
            if tok[:2] != w[:2]:
                continue
            # compare with the token prefix of similar length (Turkish suffixes);
            # a token that is just the root cut short ("sikti" vs "siktir") is no match
            for k in (len(w) - 1, len(w), len(w) + 1):
                head = tok[:k]
                if len(tok) < k or (k < len(w) and head == w[:k]):
                    continue
                if Levenshtein.distance(head, w, score_cutoff=1) <= 1:
                    return True
        return False


class LexiconModerator(Moderator):
    name = "lexicon"

    def __init__(self, mined: dict[str, float] | None = None, clean: set[str] | None = None,
                 threshold: float = 0.5, path: Path | None = None):
        prefix, exact = load_seed()
        self.turkish = _Lexicon(prefix, exact)
        self.folded = _Lexicon({ascii_fold(w) for w in prefix}, {ascii_fold(w) for w in exact})
        self.mined = mined or {}
        # frequent clean words are never fuzzy-matched ("yaprak" is not "yarrak")
        self.clean = clean or set()
        self.threshold = threshold
        self.path = path
        self.token_score = lru_cache(maxsize=200_000)(self._token_score)

    def _token_score(self, tok: str) -> tuple[float, str]:
        lexicon = self.folded if tok.isascii() else self.turkish
        if lexicon.exact_hit(tok):
            return SEED_SCORE, "seed"
        if tok in self.mined:
            return self.mined[tok], "mined"
        if tok not in self.clean and lexicon.fuzzy_hit(tok):
            return FUZZY_SCORE, "fuzzy"
        return 0.0, ""

    def explain(self, text: str) -> list[tuple[str, float, str]]:
        """Matched tokens with their score and source; used to mask messages in the app."""
        hits = []
        for tok in preprocess(text):
            score, source = self.token_score(tok)
            if score > 0:
                hits.append((tok, score, source))
        return hits

    def score(self, text: str) -> float:
        keep = 1.0
        for _, s, _ in self.explain(text):
            keep *= 1.0 - s
        return 1.0 - keep

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.array([self.score(t) for t in texts])

    def artifact_paths(self) -> list[Path]:
        return [SEED_PATH] + ([self.path] if self.path else [])

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(
            {"threshold": self.threshold, "mined": self.mined, "clean": sorted(self.clean)},
            ensure_ascii=False,
        ))
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "LexiconModerator":
        data = json.loads(path.read_text())
        return cls(data["mined"], set(data["clean"]), data["threshold"], path)
