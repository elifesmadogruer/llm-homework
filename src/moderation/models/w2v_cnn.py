"""Neural model: our own word2vec embeddings feeding a 1D CNN (Kim-style).

    tokens -> embedding (init from word2vec trained on the training messages, fine-tuned)
           -> dropout -> Conv1d with kernel sizes 3/4/5 -> ReLU -> max-pool over time
           -> concat -> dropout -> linear -> P(offensive)
"""
import json
from pathlib import Path

import numpy as np
import torch
from gensim.models import Word2Vec
from torch import nn

from moderation.models.base import Moderator
from moderation.preprocessing.normalize import normalize, tokenize

PAD, UNK = 0, 1
MAX_LEN = 64


def preprocess(text: str) -> list[str]:
    return tokenize(normalize(text))


def train_word2vec(token_lists, dim: int = 100, seed: int = 42) -> Word2Vec:
    return Word2Vec(
        sentences=token_lists, vector_size=dim, window=5, min_count=2,
        sg=1, negative=10, epochs=15, workers=4, seed=seed,
    )


def build_vocab(w2v: Word2Vec) -> tuple[dict[str, int], np.ndarray]:
    words = list(w2v.wv.index_to_key)
    vocab = {"<pad>": PAD, "<unk>": UNK, **{w: i + 2 for i, w in enumerate(words)}}
    dim = w2v.vector_size
    rng = np.random.default_rng(0)
    emb = np.zeros((len(vocab), dim), dtype=np.float32)
    emb[UNK] = rng.normal(0, 0.1, dim)
    emb[2:] = w2v.wv.vectors
    return vocab, emb


class TextCNN(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 100, n_filters: int = 100,
                 kernel_sizes=(3, 4, 5), dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.emb_dropout = nn.Dropout(0.25)
        self.convs = nn.ModuleList(nn.Conv1d(dim, n_filters, k) for k in kernel_sizes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(kernel_sizes), 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.emb_dropout(self.embedding(ids)).transpose(1, 2)  # (B, dim, T)
        pooled = [torch.relu(conv(x)).amax(dim=2) for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


def encode(token_lists, vocab: dict[str, int], max_len: int = MAX_LEN, min_len: int = 5) -> torch.Tensor:
    length = min(max_len, max(min_len, max((len(t) for t in token_lists), default=0)))
    ids = torch.full((len(token_lists), length), PAD, dtype=torch.long)
    for i, toks in enumerate(token_lists):
        row = [vocab.get(t, UNK) for t in toks[:length]]
        ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return ids


class W2VCNNModerator(Moderator):
    name = "w2v_cnn"

    def __init__(self, net: TextCNN, vocab: dict[str, int], threshold: float = 0.5,
                 config: dict | None = None, path: Path | None = None):
        self.net = net.eval()
        self.vocab = vocab
        self.threshold = threshold
        self.config = config or {}
        self.path = path

    @torch.inference_mode()
    def predict_proba(self, texts: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            ids = encode([preprocess(t) for t in texts[i : i + batch_size]], self.vocab)
            out.append(torch.sigmoid(self.net(ids)).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def artifact_paths(self) -> list[Path]:
        return [self.path / "model.pt", self.path / "vocab.json"] if self.path else []

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "config": self.config,
                    "threshold": self.threshold}, path / "model.pt")
        (path / "vocab.json").write_text(json.dumps(self.vocab, ensure_ascii=False))
        self.path = path

    @classmethod
    def load(cls, path: Path) -> "W2VCNNModerator":
        ckpt = torch.load(path / "model.pt", map_location="cpu")
        vocab = json.loads((path / "vocab.json").read_text())
        net = TextCNN(len(vocab), **ckpt["config"])
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, vocab, ckpt["threshold"], ckpt["config"], path)
