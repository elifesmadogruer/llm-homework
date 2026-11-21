"""Train word2vec on the training messages, then a 1D CNN on top; evaluate on test.

Usage:
    python scripts/train_w2v_cnn.py            # word2vec-initialized embeddings
    python scripts/train_w2v_cnn.py --random   # ablation: random embeddings
    python scripts/train_w2v_cnn.py --static   # ablation: frozen word2vec embeddings
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch import nn

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import RESULTS_DIR, classification_metrics, evaluate_model, tune_threshold
from moderation.models.w2v_cnn import TextCNN, W2VCNNModerator, build_vocab, encode, preprocess, train_word2vec

SEED = 42
CONFIG = {"dim": 100, "n_filters": 100, "kernel_sizes": (3, 4, 5), "dropout": 0.5}
EPOCHS, BATCH, LR, PATIENCE = 12, 64, 1e-3, 3


def run_epoch(net, opt, loss_fn, ids, labels, rng):
    net.train()
    order = rng.permutation(len(ids))
    total = 0.0
    for i in range(0, len(order), BATCH):
        idx = order[i : i + BATCH]
        opt.zero_grad()
        loss = loss_fn(net(ids[idx]), labels[idx])
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / len(ids)


@torch.inference_mode()
def val_proba(net, ids):
    net.eval()
    return torch.cat([torch.sigmoid(net(ids[i : i + 512])) for i in range(0, len(ids), 512)]).numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--random", action="store_true", help="random embedding init (ablation)")
    parser.add_argument("--static", action="store_true", help="freeze word2vec embeddings")
    parser.add_argument("--save-as", default=None, help="result/model name for ablation runs")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    train, val, test = load_split("train"), load_split("val"), load_split("test")
    train_tokens = [preprocess(t) for t in train["text"]]
    val_tokens = [preprocess(t) for t in val["text"]]

    start = time.time()
    w2v = train_word2vec(train_tokens, dim=CONFIG["dim"], seed=SEED)
    vocab, emb = build_vocab(w2v)
    print(f"word2vec: vocab={len(vocab)} dim={CONFIG['dim']} ({time.time() - start:.0f}s)")
    for probe in ("salak", "amk", "güzel"):
        if probe in w2v.wv:
            print(f"  nearest to '{probe}':", [w for w, _ in w2v.wv.most_similar(probe, topn=6)])

    net = TextCNN(len(vocab), **CONFIG)
    if not args.random:
        net.embedding.weight.data.copy_(torch.from_numpy(emb))
    net.embedding.weight.requires_grad = not args.static

    # sort-free fixed-length tensors are fine at this size (64 tokens max)
    train_ids, val_ids = encode(train_tokens, vocab), encode(val_tokens, vocab)
    y_train = torch.tensor(train["label"].values, dtype=torch.float32)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=LR)

    best_f1, best_state, best_epoch, history, wait = -1.0, None, 0, [], 0
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        loss = run_epoch(net, opt, loss_fn, train_ids, y_train, rng)
        proba = val_proba(net, val_ids)
        threshold = tune_threshold(val["label"], proba)
        f1 = classification_metrics(val["label"], (proba >= threshold).astype(int))["macro_f1"]
        history.append({"epoch": epoch, "train_loss": round(loss, 4), "val_macro_f1": f1})
        print(f"epoch {epoch:2d} loss={loss:.4f} val macro-F1={f1} (t={threshold}) {time.time() - t0:.0f}s")
        if f1 > best_f1:
            best_f1, best_epoch, wait = f1, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                print("early stopping")
                break

    net.load_state_dict(best_state)
    proba = val_proba(net, val_ids)
    threshold = tune_threshold(val["label"], proba)
    val_metrics = classification_metrics(val["label"], (proba >= threshold).astype(int))
    config = {**CONFIG, "kernel_sizes": list(CONFIG["kernel_sizes"])}
    model = W2VCNNModerator(net, vocab, threshold, config)
    extra = {"val": val_metrics, "best_epoch": best_epoch, "history": history,
             "embedding_init": "random" if args.random else "word2vec", "static": args.static}

    if args.save_as:
        model.name = args.save_as
        test_proba = model.predict_proba(list(test["text"]))
        result = {"model": model.name, "threshold": threshold, **extra,
                  "test": classification_metrics(test["label"], (test_proba >= threshold).astype(int))}
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / f"{model.name}.json").write_text(json.dumps(result, indent=2))
    else:
        w2v.save(str(MODELS_DIR / "word2vec.model"))
        model.save(MODELS_DIR / "w2v_cnn")
        result = evaluate_model(model, test["text"], test["label"], extra=extra)
        print(f"size={result['size_mb']} MB latency={result['latency']['mean_ms']} ms")
    print(f"test macro-F1={result['test']['macro_f1']} (best epoch {best_epoch}, threshold {threshold})")


if __name__ == "__main__":
    main()
