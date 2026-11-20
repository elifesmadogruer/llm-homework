"""Download OffensEval-TR 2020 and create fixed train/val/test splits.

The official test set is kept untouched as the shared held-out set for every
model. A stratified 10% validation split is carved out of the official train set.

Usage:
    python scripts/prepare_data.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"
SEED = 42
VAL_SIZE = 0.1
LABELS = {0: "NOT", 1: "OFF"}


def load_raw() -> dict[str, pd.DataFrame]:
    # The original repo ships a loading script, which recent `datasets`
    # versions refuse to run; the auto-converted parquet branch is identical.
    ds = load_dataset(
        "coltekin/offenseval2020_tr",
        revision="refs/convert/parquet",
        data_dir="offenseval2020-turkish",
    )
    frames = {}
    for split in ds:
        df = ds[split].to_pandas().rename(columns={"tweet": "text", "subtask_a": "label"})
        frames[split] = df[["id", "text", "label"]]
    return frames


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]
    return df.reset_index(drop=True)


def describe(df: pd.DataFrame) -> dict:
    lengths = df["text"].str.split().str.len()
    counts = df["label"].map(LABELS).value_counts()
    return {
        "n": int(len(df)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "off_ratio": round(float(df["label"].mean()), 4),
        "tokens_mean": round(float(lengths.mean()), 2),
        "tokens_median": float(lengths.median()),
        "tokens_p95": float(lengths.quantile(0.95)),
        "tokens_max": int(lengths.max()),
    }


def plot(splits: dict[str, pd.DataFrame], path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    names = list(splits)
    off = [splits[s]["label"].sum() for s in names]
    clean_ = [len(splits[s]) - o for s, o in zip(names, off)]
    axes[0].bar(names, clean_, label="NOT")
    axes[0].bar(names, off, bottom=clean_, label="OFF")
    axes[0].set_title("Class distribution per split")
    axes[0].legend()
    train = splits["train"]
    for label, name in LABELS.items():
        lengths = train.loc[train["label"] == label, "text"].str.split().str.len()
        axes[1].hist(lengths, bins=40, range=(0, 80), alpha=0.6, density=True, label=name)
    axes[1].set_title("Message length (tokens), train")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "figures").mkdir(parents=True, exist_ok=True)

    raw = load_raw()
    full_train, test = clean(raw["train"]), clean(raw["test"])
    train, val = train_test_split(
        full_train, test_size=VAL_SIZE, stratify=full_train["label"], random_state=SEED
    )
    splits = {
        "train": train.reset_index(drop=True),
        "val": val.reset_index(drop=True),
        "test": test,
    }

    stats = {}
    for name, df in splits.items():
        df.to_parquet(OUT_DIR / f"{name}.parquet", index=False)
        stats[name] = describe(df)
        print(f"{name:5s} {stats[name]}")

    overlap = set(splits["train"]["text"]) & set(splits["test"]["text"])
    stats["train_test_text_overlap"] = len(overlap)
    print(f"train/test identical texts: {len(overlap)}")

    (REPORT_DIR / "data_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    plot(splits, REPORT_DIR / "figures" / "data_overview.png")


if __name__ == "__main__":
    main()
