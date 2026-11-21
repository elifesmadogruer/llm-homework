"""Classical baseline: tf-idf (word + character n-grams) with logistic regression.

Character n-grams inside word boundaries make the model robust to Turkish
suffixes and to spelling variants ("salaksın", "salakkk", "slak").
"""
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from moderation.models.base import Moderator
from moderation.preprocessing.normalize import normalize, tokenize


def build_pipeline(C: float = 4.0, max_word: int = 50_000, max_char: int = 150_000) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            tokenizer=tokenize, token_pattern=None, lowercase=False,
            ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=3,
            max_features=max_char, sublinear_tf=True,
        )),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, solver="liblinear")
    return Pipeline([("features", features), ("clf", clf)])


class TfidfModerator(Moderator):
    name = "tfidf_lr"

    def __init__(self, pipeline: Pipeline, threshold: float = 0.5, path: Path | None = None):
        self.pipeline = pipeline
        self.threshold = threshold
        self.path = path

    @staticmethod
    def preprocess(texts) -> list[str]:
        return [normalize(t) for t in texts]

    def fit(self, texts, labels) -> "TfidfModerator":
        self.pipeline.fit(self.preprocess(texts), labels)
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(self.preprocess(texts))[:, 1]

    def top_features(self, k: int = 20) -> list[tuple[str, float]]:
        names = self.pipeline.named_steps["features"].get_feature_names_out()
        coef = self.pipeline.named_steps["clf"].coef_[0]
        order = np.argsort(coef)[::-1][:k]
        return [(names[i], round(float(coef[i]), 3)) for i in order]

    def artifact_paths(self) -> list[Path]:
        return [self.path] if self.path else []

    def save(self, path: Path) -> None:
        joblib.dump({"pipeline": self.pipeline, "threshold": self.threshold}, path, compress=3)
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "TfidfModerator":
        data = joblib.load(path)
        return cls(data["pipeline"], data["threshold"], path)
