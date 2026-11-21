"""Build the results table (R4): task metric, size and CPU latency for every model.

Macro-F1 comes from each model's evaluation on the shared test set
(reports/results/<model>.json). Latency is re-measured here for every model
under the same conditions: one message per call, 300 runs over test messages,
once with all CPU threads and once restricted to a single thread (a moderation
worker sharing the server with many chat channels).

Usage:
    python scripts/benchmark.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import RESULTS_DIR, benchmark_latency, hardware
from moderation.models.onnx_model import OnnxBertModerator
from moderation.registry import load_available

ROOT = Path(__file__).resolve().parents[1]
BUDGET_MS = 50.0
N_RUNS = 300


def single_thread_variant(model):
    if isinstance(model, OnnxBertModerator):
        return OnnxBertModerator(MODELS_DIR / "berturk", model.onnx_path.name, num_threads=1)
    return model


def main() -> None:
    texts = list(load_split("test")["text"])
    models, missing = load_available()
    if missing:
        print("missing models:", missing)
    default_threads = torch.get_num_threads()

    rows = []
    for display, model in models.items():
        result = json.loads((RESULTS_DIR / f"{model.name}.json").read_text())
        torch.set_num_threads(default_threads)
        multi = benchmark_latency(model, texts, n_runs=N_RUNS)
        torch.set_num_threads(1)
        single = benchmark_latency(single_thread_variant(model), texts, n_runs=N_RUNS)
        rows.append({
            "model": display,
            "macro_f1": result["test"]["macro_f1"],
            "off_f1": result["test"]["off_f1"],
            "size_mb": round(model.size_mb(), 1),
            "latency_ms": multi["mean_ms"],
            "latency_p95_ms": multi["p95_ms"],
            "latency_1thread_ms": single["mean_ms"],
            "latency_1thread_p95_ms": single["p95_ms"],
            "within_budget_1thread": single["mean_ms"] <= BUDGET_MS,
        })
        print(rows[-1])
    torch.set_num_threads(default_threads)

    out_csv = ROOT / "reports" / "results_table.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        f"Test set: OffensEval-TR 2020 official test split (n={len(texts)}). "
        f"Latency: mean over {N_RUNS} single-message calls, preprocessing included. "
        f"Hardware: {hardware()}, {default_threads} threads vs. 1 thread. "
        f"Budget: mean <= {BUDGET_MS:.0f} ms per message on 1 thread.",
        "",
        "Note: PyTorch on macOS runs matrix products through Apple Accelerate (AMX unit), which "
        "ignores torch's thread setting, so its 1-thread column is not truly single-threaded. "
        "The quantization effect is measured within one runtime: ONNX Runtime FP32 vs. INT8.",
        "",
        "| Model | Macro-F1 | OFF F1 | Size (MB) | Latency (ms) | p95 (ms) | 1-thread (ms) | 1-thread p95 (ms) | Budget |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['macro_f1']:.4f} | {r['off_f1']:.4f} | {r['size_mb']} | {r['latency_ms']:.2f} | "
            f"{r['latency_p95_ms']:.2f} | {r['latency_1thread_ms']:.2f} | {r['latency_1thread_p95_ms']:.2f} | "
            f"{'yes' if r['within_budget_1thread'] else 'no'} |"
        )
    (ROOT / "reports" / "results_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
