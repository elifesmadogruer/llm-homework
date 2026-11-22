# Real-Time Chat Moderation (Turkish) — SWE015 Use Case 4

A chat interface that labels every message live as **clean / offensive** with a
confidence score, masks offensive words, and logs the response time of every
message against a CPU latency budget. Four models of increasing sophistication
are compared on the same held-out test set; the application runs the
**INT8-quantized BERTurk** (ONNX Runtime) by default.

| Ladder step | Model | Technique area |
|---|---|---|
| Rule baseline | Regex normalization + edit-distance lexicon matcher | B · Classical NLP |
| Classical baseline | tf-idf (word 1–2 + char 2–5 grams) + logistic regression | B · Classical NLP |
| Neural network | Own word2vec (skip-gram, trained on the training messages) + 1D CNN | A · Neural networks |
| Transformer | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned with a classification head | C · Transformers |
| Efficient model | BERTurk exported to ONNX, dynamic INT8 quantization, 50 ms/message budget | D · Efficient models |

Data: [OffensEval-TR 2020](https://huggingface.co/datasets/coltekin/offenseval2020_tr)
(Çöltekin, 2020). Official test split (3,528 tweets) is the shared held-out set;
10% of the official train split (stratified, seed 42) is used for validation.

## Setup

Requires Python 3.10+ (tested with 3.13 on macOS, Apple M2).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the application

```bash
python scripts/download_models.py   # once: fetch trained models into models/
python app/app.py                   # then open http://127.0.0.1:7860
```

The app needs the trained models in `models/` (download them or train them, see below).
Models whose files are missing are skipped; the default is INT8 BERTurk.

### Option A — download the trained models

Trained weights are hosted on the Hugging Face Hub:
[elifesma/llm-homework](https://huggingface.co/elifesma/llm-homework).

```bash
python scripts/download_models.py              # all models (~1.1 GB)
python scripts/download_models.py --int8-only  # INT8 BERTurk + small models (~120 MB)
```

### Option B — reproduce everything

```bash
python scripts/prepare_data.py     # download data, create fixed splits
python scripts/train_lexicon.py    # rule baseline
python scripts/train_tfidf.py      # tf-idf + logistic regression
python scripts/train_w2v_cnn.py    # word2vec + 1D CNN
python scripts/train_bert.py       # BERTurk fine-tuning (GPU/MPS recommended)
python scripts/quantize_bert.py    # ONNX export + dynamic INT8 quantization
python scripts/benchmark.py        # results table (reports/results_table.md)
```

Tests: `python -m pytest`

## Results

Test set: OffensEval-TR 2020 official test split (n=3528). Latency: mean over 300 single-message calls, preprocessing included. Hardware: Apple M2 (Darwin arm64, CPU only), 4 threads vs. 1 thread. Budget: mean <= 50 ms per message on 1 thread.

Note: PyTorch on macOS runs matrix products through Apple Accelerate (AMX unit), which ignores torch's thread setting, so its 1-thread column is not truly single-threaded. The quantization effect is measured within one runtime: ONNX Runtime FP32 vs. INT8.

| Model | Macro-F1 | OFF F1 | Size (MB) | Latency (ms) | p95 (ms) | 1-thread (ms) | 1-thread p95 (ms) | Budget |
|---|---|---|---|---|---|---|---|---|
| BERTurk · INT8 · ONNX Runtime | 0.8141 | 0.6996 | 107.4 | 22.71 | 53.14 | 30.34 | 64.40 | yes |
| BERTurk · FP32 · ONNX Runtime | 0.8152 | 0.7022 | 423.0 | 66.93 | 135.70 | 88.69 | 189.50 | no |
| BERTurk · FP32 · PyTorch | 0.8152 | 0.7022 | 422.7 | 35.01 | 43.70 | 36.37 | 49.30 | yes |
| word2vec + 1D-CNN | 0.7482 | 0.5923 | 12.2 | 0.34 | 0.50 | 0.27 | 0.42 | yes |
| tf-idf + LogReg | 0.7715 | 0.6339 | 2.5 | 1.28 | 1.76 | 1.22 | 1.64 | yes |
| Lexicon + edit distance | 0.7142 | 0.5173 | 0.0 | 0.15 | 0.38 | 0.08 | 0.19 | yes |

Each model's decision threshold was tuned for macro-F1 on the validation split only.
Per-model details (confusion matrices, validation scores, training history) are in
`reports/results/`. BERTurk was trained for 1 epoch (3 planned; the run was stopped
during epoch 2 because of memory swapping on a 16 GB machine; epoch 1 was the best
validation checkpoint available).

## Repository layout

```
app/app.py                         Gradio chat interface
src/moderation/preprocessing/      regex normalization (leetspeak, repeats, spaced letters, Turkish casing)
src/moderation/models/             lexicon, tf-idf, word2vec+CNN, BERTurk (PyTorch / ONNX)
src/moderation/evaluation/         macro-F1, threshold tuning, latency benchmark
src/moderation/resources/          hand-written offensive seed lexicon
scripts/                           data preparation, training, quantization, benchmark
reports/                           per-model results (JSON), results table, figures
```

Model weights are not stored in this repository.
