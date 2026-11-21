"""Transformer model: BERTurk fine-tuned for offensive / clean classification.

Inference runs on CPU in FP32 here; the INT8 ONNX version lives in onnx_model.py.
"""
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from moderation.models.base import Moderator
from moderation.preprocessing.normalize import light_normalize

BASE_MODEL = "dbmdz/bert-base-turkish-cased"
MAX_LEN = 64


def preprocess(text: str) -> str:
    # BERTurk is cased and has its own WordPiece vocabulary, so only drop
    # mentions/URLs and join spaced-out letters.
    return light_normalize(text) or text


class BertModerator(Moderator):
    name = "berturk_fp32"

    def __init__(self, path: Path, threshold: float | None = None, num_threads: int | None = None):
        self.path = Path(path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path).eval()
        meta = json.loads((self.path / "moderation.json").read_text())
        self.threshold = meta["threshold"] if threshold is None else threshold
        if num_threads:
            torch.set_num_threads(num_threads)

    def encode(self, texts: list[str]):
        return self.tokenizer([preprocess(t) for t in texts], truncation=True,
                              max_length=MAX_LEN, padding=True, return_tensors="pt")

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            logits = self.model(**self.encode(texts[i : i + batch_size])).logits
            out.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def artifact_paths(self) -> list[Path]:
        weights = [p for p in self.path.iterdir() if p.suffix in {".safetensors", ".bin"}]
        tokenizer = [p for p in self.path.iterdir() if p.name in {"vocab.txt", "tokenizer.json"}]
        return weights + tokenizer + [self.path / "config.json"]
