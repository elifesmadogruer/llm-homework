# Presentation outline (12 slides) and demo video script

All numbers below come from `reports/results_table.md`, `reports/results/*.json`
and `reports/data_stats.json` (own runs, Apple M2, CPU). Figures are in `reports/figures/`.
Fill in the `<...>` fields before exporting to PDF.

---

## Slide 1 — Title

**Real-Time Chat Moderation for Turkish**
SWE015 – Introduction to Large Language Models · Use Case 4

- Name: `<your name>` · Student ID: `<id>`
- Repository: https://github.com/elifesmadogruer/llm-homework
- Demo video: `<YouTube unlisted / Drive link>`
- Model weights: https://huggingface.co/elifesma/llm-homework

---

## Introduction

### Slide 2 — Problem and success criteria

- A gaming platform must flag offensive chat messages in Turkish **live**, on a **CPU-only** server.
- Users: platform moderators and players; flagged messages are masked before others see them.
- Turkish chat is noisy: leetspeak (`s4l4k`), repeated letters (`saaalak`), spaced letters
  (`A P T A L`), missing Turkish characters (`aptalsin`), rich suffixes (`aptallarla`).
- Success =
  1. **macro-F1** on the held-out test set (classes are imbalanced, ~20% offensive)
  2. **latency budget: mean ≤ 50 ms per message on 1 CPU thread** (one moderation worker
     shares the server with many chat channels)
  3. a working chat app that runs the **quantized** Transformer

---

## Development

### Slide 3 — Data

- OffensEval-TR 2020 (Çöltekin, 2020), Turkish tweets, label NOT / OFF (subtask A).
  Source: huggingface.co/datasets/coltekin/offenseval2020_tr
- Official test split kept untouched = shared held-out set for all models.
- 10% of the official train split → validation (stratified, seed 42).

| Split | Messages | NOT | OFF | OFF ratio |
|---|---|---|---|---|
| train | 28,580 | 23,062 | 5,518 | 19.3% |
| val | 3,176 | 2,563 | 613 | 19.3% |
| test | 3,528 | 2,812 | 716 | 20.3% |

- Mean length 15 tokens, 95th percentile 36 → BERT `max_len = 64` (3.6% of training messages truncated).
- Preprocessing: regex normalization (mentions/URLs, spaced letters, Turkish lowercasing,
  leetspeak, repeated letters); light version for BERTurk.
- Figure: `reports/figures/data_overview.png`

### Slide 4 — Architecture

- Figure: `reports/figures/architecture.png`
- Say: message → normalization → selected model → threshold (tuned on val) → label +
  confidence → masking → chat panel + latency log. Everything runs on CPU in one Gradio process.

### Slide 5 — Technique map

- Figure: `reports/figures/technique_map.png`
- A: own word2vec + 1D CNN · B: regex normalization, edit-distance lexicon, tf-idf, word2vec ·
  C: BERTurk fine-tuning · D: ONNX + dynamic INT8 quantization, latency budget

### Slide 6 — Models 1 and 2: classical baselines

**Lexicon + edit distance (rule baseline)**
- ~90 hand-written offensive roots (prefix match) + 75 terms mined from train (P(OFF|token) ≥ 0.6, ≥ 15 occurrences).
- Levenshtein distance ≤ 1 only for roots of ≥ 7 letters: on val, 6-letter roots produced mostly false
  positives (`gerçek`/`gerzek`, `gözünü`/`götünü`).
- Turkish-letter tokens are matched against the Turkish lexicon, ASCII-only tokens against the folded one
  (keeps the ı / i difference).

**tf-idf + logistic regression**
- word 1–2-grams + character 2–5-grams (inside word boundaries), sublinear tf, class-balanced LR.
- C selected on val (C = 4); decision threshold tuned for macro-F1 on val.

### Slide 7 — Models 3 and 4: neural network and Transformer

**word2vec + 1D CNN (Area A)**
- Own skip-gram word2vec, 100-d, trained only on the training messages (29,298-word vocabulary).
- CNN: kernels 3/4/5 × 100 filters, ReLU, max-pool over time, dropout 0.5, class-weighted BCE, Adam,
  early stopping on val macro-F1 (best epoch 3).
- Ablation (val macro-F1): fine-tuned word2vec 0.747 · random embeddings 0.714 · frozen word2vec 0.709.

**BERTurk (Area C)**
- `dbmdz/bert-base-turkish-cased` + classification head, AdamW lr 3e-5, batch 32, linear warm-up,
  trained on Apple MPS.
- 3 epochs planned; run stopped during epoch 2 because the 16 GB machine started swapping;
  the epoch-1 checkpoint (val macro-F1 0.838) is used.

### Slide 8 — Quantization

- Export fine-tuned BERTurk to ONNX (opset 17, dynamic batch/sequence axes).
- ONNX Runtime **dynamic INT8 quantization**: MatMul weights stored as INT8 (per output channel),
  activations quantized at run time. No calibration data needed.
- Same runtime, FP32 → INT8:

| | Macro-F1 | Size | Latency, 1 thread (mean) |
|---|---|---|---|
| BERTurk FP32 · ONNX Runtime | 0.8152 | 423 MB | 88.7 ms ✗ |
| BERTurk INT8 · ONNX Runtime | 0.8141 | 107 MB | 30.3 ms ✓ |

- 3.9× smaller, 2.9× faster on 1 thread, −0.001 macro-F1; 99.2% identical decisions on the test set.

### Slide 9 — Results table (R4)

Test set n = 3,528 · latency = mean of 300 single-message calls incl. preprocessing ·
Apple M2, CPU only · budget: mean ≤ 50 ms on 1 thread

| Model | Macro-F1 | Size (MB) | Latency 4 thr (ms) | Latency 1 thr (ms) | Budget |
|---|---|---|---|---|---|
| Lexicon + edit distance | 0.7142 | 0.02 | 0.15 | 0.08 | ✓ |
| tf-idf + LogReg | 0.7715 | 2.5 | 1.28 | 1.22 | ✓ |
| word2vec + 1D CNN | 0.7482 | 12.2 | 0.34 | 0.27 | ✓ |
| BERTurk FP32 (PyTorch) | 0.8152 | 422.7 | 35.01 | 36.37* | ✓ |
| BERTurk FP32 (ONNX Runtime) | 0.8152 | 423.0 | 66.93 | 88.69 | ✗ |
| **BERTurk INT8 (ONNX Runtime)** — in the app | **0.8141** | **107.4** | 22.71 | 30.34 | ✓ |

\* PyTorch on macOS runs matrix products through Apple Accelerate (AMX unit), which ignores the
thread limit — not a true single-thread number and not available on a typical x86 server.
Latency varies with machine load: a first benchmark run measured INT8 at 11.5 ms (4 threads).

---

## Conclusion

### Slide 10 — What the comparison showed

- **BERTurk wins on quality** (INT8: +4.3 macro-F1 points over tf-idf): it catches insults without swear
  words. On the test set it is right on 256 messages where tf-idf is wrong (tf-idf: 149 the other way).
  Example caught only by BERT: *"beni mi hafife aldın yoksa zekan mı elvermedi"*.
- **INT8 quantization keeps the accuracy** (−0.001) and brings BERTurk inside the 50 ms budget on one thread.
- **tf-idf is the best cheap model**: 0.77 macro-F1 at ~1 ms and 2.5 MB; character n-grams handle
  suffixes and misspellings.
- **The CNN did not beat tf-idf**: 28k messages are few for a network trained from scratch;
  word-level word2vec maps unseen spellings to `<unk>`, character n-grams do not.
- **The lexicon is precise but blind** (OFF precision 0.74, recall 0.40): most offensive tweets
  contain no swear word. It is still useful for masking *which* words to hide.

### Slide 11 — Limitations, failure examples, next steps

Failure examples (INT8 BERTurk):
- Missed, test set (gold OFF, P(off) = 0.016): *"Paris gibi, şimdi hep çarşaflı dolu."* — offensive only
  through cultural/political context.
- Missed, new input (P(off) = 0.17): *"sen tam bir aptaaal oyuncusun"* — BERTurk gets the light
  normalization, so repeated letters break the word; tf-idf (0.96) and the lexicon catch it.
- The gold labels are noisy: some messages with clear insults are labeled NOT.

Limitations
- Tweets ≠ game chat; mined lexicon terms carry topic bias (`türbanlı`, `cumhuriyeti`).
- BERTurk trained for only 1 epoch (hardware); tail latency (1-thread p95 = 64 ms) exceeds 50 ms.
- Leetspeak rule still converts digits inside concatenated hashtags (`gthb5bin`).

Next steps
- Finish 3-epoch training on a GPU; distil BERTurk into a smaller student (e.g. 4–6 layers) for tail latency.
- Collect real in-game chat for fine-tuning; static INT8 with calibration; sentence-level masking.

### Slide 12 — AI usage (R8)

- **Claude Code (Anthropic, Claude Opus)** was used as a coding assistant for: project structure,
  writing and debugging the Python code (normalization, models, training/quantization scripts,
  Gradio app), the README, the figure scripts and a first draft of these slides.
- `<add any other tools you used, e.g. ChatGPT for …>`
- All design decisions were reviewed; every number was produced by running the code in this
  repository on my machine; I can explain each component.

---

## Demo video script (≤ 2:00, narrated)

**0:00–0:15 — title slide**
"Hi, I'm Elif Esma Doğruer. Gaming platforms have to hide offensive Turkish chat messages instantly,
on CPU-only servers. I built a live moderation app and compared four models, from a rule-based
baseline to a quantized BERTurk."

**0:15–1:15 — app (http://127.0.0.1:7860), model: BERTurk INT8**
"This is the app. It runs the INT8-quantized BERTurk on CPU by default."
1. `bu akşam maç kaç saatte başlıyor?` — "A normal message is labeled clean. On the right, the response
   time is logged: a few milliseconds, well inside our 50-millisecond budget."
2. `s4l4k misin sen, oyunu bilmiyorsun` — "Here is an insult written in leetspeak. It is flagged as
   offensive, and the insulting word is masked."
3. `senin gibi beyni yarım adamlarla oynamak zorunda kalıyoruz` — "Now a harder case: an insult with no
   offensive word at all. BERTurk still flags it, and since there is no single word to mask, the whole
   message is hidden."
4. Open "Compare all models", click "Run all models" — "Comparing all models on this message: the lexicon
   and the CNN miss it, tf-idf is borderline, and BERTurk is confident. Each model's latency is shown too."

**1:15–1:45 — architecture slide, then results slide**
"Each message is normalized for leetspeak, repeated letters and Turkish casing. The selected model gives
a probability, a threshold tuned on validation data decides, and a lexicon masks the words."
"On the official test set, BERTurk reaches 0.81 macro-F1 versus 0.77 for tf-idf. INT8 quantization makes
it four times smaller and three times faster on one thread, with almost no accuracy loss."

**1:45–2:00 — findings slide**
"So BERTurk is the most accurate model, and quantization is what makes it fit a real-time CPU budget.
tf-idf remains a strong, cheap fallback. Thanks for watching."
