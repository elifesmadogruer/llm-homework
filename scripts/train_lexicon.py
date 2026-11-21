"""Build the lexicon baseline, tune its threshold on val and evaluate on test.

Usage:
    python scripts/train_lexicon.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import classification_metrics, evaluate_model, tune_threshold
from moderation.models.lexicon import LexiconModerator, mine_terms


def main() -> None:
    train, val, test = load_split("train"), load_split("val"), load_split("test")

    seed_only = LexiconModerator()
    mined, clean = mine_terms(train["text"], train["label"])
    model = LexiconModerator(mined, clean)
    print(f"mined terms: {len(mined)}, clean guard words: {len(clean)}")

    val_scores = {}
    for name, m in [("seed only", seed_only), ("seed + mined", model)]:
        proba = m.predict_proba(list(val["text"]))
        m.threshold = tune_threshold(val["label"], proba)
        val_scores[name] = classification_metrics(val["label"], (proba >= m.threshold).astype(int))
        print(f"val [{name}] threshold={m.threshold} macro-F1={val_scores[name]['macro_f1']}")

    MODELS_DIR.mkdir(exist_ok=True)
    model.save(MODELS_DIR / "lexicon.json")
    result = evaluate_model(
        model, test["text"], test["label"],
        extra={"val": val_scores["seed + mined"], "val_seed_only": val_scores["seed only"],
               "n_mined": len(mined)},
    )
    print(f"test macro-F1={result['test']['macro_f1']} size={result['size_mb']} MB "
          f"latency={result['latency']['mean_ms']} ms")


if __name__ == "__main__":
    main()
