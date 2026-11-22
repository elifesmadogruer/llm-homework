"""Draw the architecture diagram (R6) and the technique map (R2) for the slides.

Outputs reports/figures/architecture.png and reports/figures/technique_map.png.

Usage:
    python scripts/make_figures.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

INK = "#1f2937"
MUTED = "#6b7280"
COLORS = {
    "ui": "#dbeafe",
    "pre": "#fef3c7",
    "model": "#e5e7eb",
    "default": "#bbf7d0",
    "post": "#fce7f3",
    "offline": "#f3f4f6",
}


def box(ax, x, y, w, h, title, body="", color="#ffffff", bold_edge=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.012",
                                fc=color, ec=INK, lw=2.0 if bold_edge else 1.0))
    ax.text(x + w / 2, y + h - 0.018, title, ha="center", va="top", fontsize=10.5,
            fontweight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.05, body, ha="center", va="top", fontsize=8.3,
                color=INK, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.1, color=INK))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.012, label, ha="center", fontsize=7.5, color=MUTED)


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(15, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.01, 0.985, "Runtime: Gradio app (app/app.py) on CPU · Apple M2 · 127.0.0.1:7860",
            fontsize=12, fontweight="bold", color=INK, va="top")

    # 1. input
    box(ax, 0.01, 0.62, 0.13, 0.25, "Chat input", "Gradio Textbox\n+ model dropdown\n\nnew message\n(typed live)",
        COLORS["ui"])

    # 2. preprocessing
    box(ax, 0.17, 0.70, 0.17, 0.20, "normalize()",
        "drop @mentions, URLs, e-mails\njoin spaced letters (S A L A K)\nTurkish lowercasing (I→ı, İ→i)\n"
        "leetspeak (s4l4k→salak)\ncollapse repeats (saaaçma)", COLORS["pre"])
    box(ax, 0.17, 0.44, 0.17, 0.17, "light_normalize()",
        "drop @mentions, URLs, e-mails\njoin spaced letters\ncase & noise kept\n(for BERTurk WordPiece)",
        COLORS["pre"])

    # 3. models
    mx, mw = 0.39, 0.25
    box(ax, mx, 0.83, mw, 0.12, "Lexicon + edit distance",
        "~90 seed roots + 75 mined terms · rapidfuzz\nLevenshtein ≤ 1 for roots ≥ 7 letters", COLORS["model"])
    box(ax, mx, 0.69, mw, 0.12, "tf-idf + logistic regression",
        "word 1–2 + char_wb 2–5 grams · scikit-learn\nFP64 · 2.5 MB", COLORS["model"])
    box(ax, mx, 0.55, mw, 0.12, "word2vec + 1D CNN",
        "own skip-gram (100-d) · kernels 3/4/5 · PyTorch\nFP32 · 12 MB", COLORS["model"])
    box(ax, mx, 0.36, mw, 0.17, "BERTurk · INT8 · ONNX Runtime",
        "dbmdz/bert-base-turkish-cased, fine-tuned\ndynamic INT8 (per-channel weights)\n"
        "CPUExecutionProvider · 107 MB\nDEFAULT MODEL", COLORS["default"], bold_edge=True)
    box(ax, mx, 0.22, mw, 0.12, "BERTurk · FP32 (reference)",
        "ONNX Runtime FP32 · PyTorch FP32\n423 MB each", COLORS["model"])

    # 4. post-processing
    box(ax, 0.69, 0.60, 0.14, 0.27, "Decision",
        "P(offensive)\n≥ threshold\n(tuned for macro-F1\non validation split)\n\nclean / offensive\n+ confidence",
        COLORS["post"])
    box(ax, 0.69, 0.30, 0.14, 0.24, "Masking",
        "if offensive:\nlexicon matcher finds\nwords → s∗∗∗∗∗\n\nno known word →\nhide whole message",
        COLORS["post"])

    # 5. output
    box(ax, 0.86, 0.60, 0.13, 0.27, "Chat panel", "label badge\n+ confidence\n+ masked text", COLORS["ui"])
    box(ax, 0.86, 0.30, 0.13, 0.24, "Side panel",
        "per-message latency\nlog (ms)\n\nmean / max per model\nvs. 50 ms budget\n\ncompare all models",
        COLORS["ui"])

    # arrows: input -> preprocessing
    arrow(ax, 0.14, 0.78, 0.17, 0.80)
    arrow(ax, 0.14, 0.70, 0.17, 0.53)
    # preprocessing -> models
    for y in (0.89, 0.75, 0.61):
        arrow(ax, 0.34, 0.80, mx, y)
    arrow(ax, 0.34, 0.52, mx, 0.445)
    arrow(ax, 0.34, 0.50, mx, 0.28)
    # models -> decision
    for y in (0.89, 0.75, 0.61, 0.445, 0.28):
        arrow(ax, mx + mw, y, 0.69, 0.735)
    arrow(ax, 0.76, 0.60, 0.76, 0.54)
    arrow(ax, 0.83, 0.735, 0.86, 0.735)
    arrow(ax, 0.83, 0.42, 0.86, 0.42)
    ax.text(0.515, 0.195, "one model per message, chosen in the dropdown; latency timed around predict_proba()",
            ha="center", fontsize=8, color=MUTED)

    # offline training strip
    ax.add_patch(FancyBboxPatch((0.01, 0.015), 0.98, 0.145, boxstyle="round,pad=0.005,rounding_size=0.01",
                                fc=COLORS["offline"], ec=MUTED, lw=0.8, ls="--"))
    ax.text(0.02, 0.15, "Offline (scripts/)", fontsize=10, fontweight="bold", color=INK, va="top")
    steps = [
        ("prepare_data.py", "OffensEval-TR 2020\ntrain 28,580 / val 3,176\n/ test 3,528"),
        ("train_lexicon.py\ntrain_tfidf.py", "mine terms, fit tf-idf,\ntune thresholds on val"),
        ("train_w2v_cnn.py", "gensim word2vec on train\n→ CNN, early stopping"),
        ("train_bert.py", "fine-tune on Apple MPS\nbest val epoch (1)"),
        ("quantize_bert.py", "torch.onnx export →\nquantize_dynamic INT8"),
        ("benchmark.py", "macro-F1, size, latency\n(300 runs, CPU)"),
    ]
    sx, sw = 0.03, 0.148
    for i, (name, body) in enumerate(steps):
        x = sx + i * (sw + 0.012)
        ax.text(x + sw / 2, 0.115, name, ha="center", va="top", fontsize=8.3, fontweight="bold", color=INK)
        ax.text(x + sw / 2, 0.068 if "\n" in name else 0.09, body, ha="center", va="top", fontsize=7.5,
                color=INK, linespacing=1.3)
        if i < len(steps) - 1:
            arrow(ax, x + sw, 0.075, x + sw + 0.012, 0.075)
    ax.text(0.98, 0.145, "weights: huggingface.co/elifesma/llm-homework", ha="right", va="top", fontsize=7.5,
            color=MUTED)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "architecture.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def technique_map() -> None:
    rows = [
        ("A · Neural networks", "own word2vec embeddings → 1D CNN (conv, max-pool, dropout), backprop + Adam",
         "src/moderation/models/w2v_cnn.py"),
        ("B · Classical NLP", "regex normalization: leetspeak, repeats, spaced letters, Turkish casing",
         "src/moderation/preprocessing/normalize.py"),
        ("B · Classical NLP", "edit-distance (Levenshtein) lexicon matcher, rule baseline + word masking",
         "src/moderation/models/lexicon.py"),
        ("B · Classical NLP", "tf-idf word + character n-grams with logistic regression",
         "src/moderation/models/tfidf.py"),
        ("B · Classical NLP", "word2vec (skip-gram) trained on the training messages",
         "src/moderation/models/w2v_cnn.py"),
        ("C · Transformers", "BERTurk fine-tuned with a classification head; WordPiece tokenizer",
         "src/moderation/models/bert.py"),
        ("D · Efficient models", "ONNX export + dynamic INT8 quantization, ONNX Runtime on CPU",
         "scripts/quantize_bert.py"),
        ("D · Efficient models", "latency budget 50 ms/message (1 thread); size/latency/accuracy benchmark",
         "scripts/benchmark.py"),
    ]
    fig, ax = plt.subplots(figsize=(15, 4.6))
    ax.axis("off")
    table = ax.table(cellText=[list(r) for r in rows], colLabels=["Area", "Technique / component", "Code"],
                     colWidths=[0.17, 0.53, 0.30], loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.75)
    area_colors = {"A": "#dbeafe", "B": "#fef3c7", "C": "#e9d5ff", "D": "#bbf7d0"}
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#d1d5db")
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_facecolor("#f3f4f6")
        elif c == 0:
            cell.set_facecolor(area_colors[rows[r - 1][0][0]])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "technique_map.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    technique_map()
    print(f"figures written to {FIG_DIR}")
