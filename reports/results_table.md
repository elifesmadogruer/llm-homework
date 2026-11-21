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
