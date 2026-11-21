"""BERTurk served through ONNX Runtime on CPU, in FP32 or dynamic-INT8 precision."""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from moderation.models.base import Moderator
from moderation.models.bert import MAX_LEN, preprocess


class OnnxBertModerator(Moderator):
    def __init__(self, model_dir: Path, onnx_file: str = "model_int8.onnx", num_threads: int | None = None):
        self.path = Path(model_dir)
        self.onnx_path = self.path / onnx_file
        self.name = "berturk_int8_onnx" if "int8" in onnx_file else "berturk_fp32_onnx"
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.threshold = json.loads((self.path / "moderation.json").read_text())["threshold"]
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads:
            opts.intra_op_num_threads = num_threads
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.onnx_path), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer([preprocess(t) for t in texts[i : i + batch_size]], truncation=True,
                                 max_length=MAX_LEN, padding=True, return_tensors="np")
            feeds = {k: v.astype(np.int64) for k, v in enc.items() if k in self.input_names}
            logits = self.session.run(["logits"], feeds)[0]
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            out.append(exp[:, 1] / exp.sum(axis=1))
        return np.concatenate(out) if out else np.zeros(0)

    def artifact_paths(self) -> list[Path]:
        tokenizer = [p for p in self.path.iterdir() if p.name in {"vocab.txt", "tokenizer.json"}]
        return [self.onnx_path] + tokenizer
