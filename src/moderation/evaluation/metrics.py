"""Task metrics, threshold tuning and CPU latency benchmarking."""
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from moderation.models.base import Moderator

ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = ROOT / "reports" / "results"


def classification_metrics(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1], zero_division=0)
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "off_precision": round(float(p[1]), 4),
        "off_recall": round(float(r[1]), 4),
        "off_f1": round(float(f[1]), 4),
        "not_f1": round(float(f[0]), 4),
        "confusion": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def tune_threshold(y_true, proba, grid=None) -> float:
    """Pick the decision threshold that maximizes macro-F1 on validation data."""
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    scores = [f1_score(y_true, (proba >= t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


def hardware() -> str:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        cpu = ""
    return f"{cpu or platform.processor()} ({platform.system()} {platform.machine()}, CPU only)"


def benchmark_latency(model: Moderator, texts: list[str], n_runs: int = 200, warmup: int = 10) -> dict:
    """Per-message latency: one message per call, as in the live chat."""
    for t in texts[:warmup]:
        model.predict_proba([t])
    timings = []
    for i in range(n_runs):
        msg = texts[i % len(texts)]
        start = time.perf_counter()
        model.predict_proba([msg])
        timings.append((time.perf_counter() - start) * 1000)
    timings = np.array(timings)
    return {
        "n_runs": n_runs,
        "mean_ms": round(float(timings.mean()), 3),
        "p50_ms": round(float(np.percentile(timings, 50)), 3),
        "p95_ms": round(float(np.percentile(timings, 95)), 3),
    }


def evaluate_model(model: Moderator, test_texts, test_labels, extra: dict | None = None) -> dict:
    proba = model.predict_proba(list(test_texts))
    preds = (proba >= model.threshold).astype(int)
    result = {
        "model": model.name,
        "threshold": model.threshold,
        "test": classification_metrics(test_labels, preds),
        "size_mb": round(model.size_mb(), 3),
        "latency": benchmark_latency(model, list(test_texts)),
        "hardware": hardware(),
        **(extra or {}),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{model.name}.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result
