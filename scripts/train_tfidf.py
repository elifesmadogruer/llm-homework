"""Train tf-idf + logistic regression, select C and threshold on val, evaluate on test.

Usage:
    python scripts/train_tfidf.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import classification_metrics, evaluate_model, tune_threshold
from moderation.models.tfidf import TfidfModerator, build_pipeline

C_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)


def main() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    best, search = None, {}
    for C in C_GRID:
        model = TfidfModerator(build_pipeline(C=C)).fit(train["text"], train["label"])
        proba = model.predict_proba(list(val["text"]))
        model.threshold = tune_threshold(val["label"], proba)
        metrics = classification_metrics(val["label"], (proba >= model.threshold).astype(int))
        search[C] = {"threshold": model.threshold, "macro_f1": metrics["macro_f1"]}
        print(f"C={C:<4} threshold={model.threshold} val macro-F1={metrics['macro_f1']}")
        if best is None or metrics["macro_f1"] > best[1]["macro_f1"]:
            best = (model, metrics, C)

    model, val_metrics, C = best
    print(f"selected C={C}")
    print("top OFF features:", model.top_features(15))

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "tfidf_lr.joblib")
    result = evaluate_model(
        model, test["text"], test["label"],
        extra={"val": val_metrics, "C": C, "c_search": search},
    )
    print(f"test macro-F1={result['test']['macro_f1']} size={result['size_mb']} MB "
          f"latency={result['latency']['mean_ms']} ms")


if __name__ == "__main__":
    main()
