"""Fine-tune BERTurk with a classification head; select the best epoch on val.

Training uses Apple MPS (or CUDA) when available; evaluation runs on CPU.

Usage:
    python scripts/train_bert.py [--epochs 3] [--lr 3e-5] [--batch-size 32]
    python scripts/train_bert.py --eval-only   # evaluate the saved best checkpoint on CPU
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import classification_metrics, evaluate_model, tune_threshold
from moderation.models.bert import BASE_MODEL, MAX_LEN, BertModerator, preprocess

SEED = 42
OUT_DIR = MODELS_DIR / "berturk"


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def batches(enc, labels, order, size):
    for i in range(0, len(order), size):
        idx = order[i : i + size]
        # dynamic padding: trim to the longest sequence in the batch
        length = int(enc["attention_mask"][idx].sum(1).max())
        yield {k: v[idx, :length] for k, v in enc.items()}, labels[idx]


@torch.inference_mode()
def predict(model, enc, dev, size=128):
    model.eval()
    order = np.arange(len(enc["input_ids"]))
    dummy = torch.zeros(len(order), dtype=torch.long)
    out = []
    for batch, _ in batches(enc, dummy, order, size):
        logits = model(**{k: v.to(dev) for k, v in batch.items()}).logits
        out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()
    if args.eval_only:
        evaluate_checkpoint()
        return

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    dev = device()
    print(f"training on {dev}")

    train, val, test = load_split("train"), load_split("val"), load_split("test")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def tokenize(texts):
        return tokenizer([preprocess(t) for t in texts], truncation=True, max_length=MAX_LEN,
                         padding="max_length", return_tensors="pt")

    train_enc, val_enc = tokenize(train["text"]), tokenize(val["text"])
    y_train = torch.tensor(train["label"].values)
    truncated = int((train_enc["attention_mask"].sum(1) == MAX_LEN).sum())
    print(f"train messages hitting max_len={MAX_LEN}: {truncated}")

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1},
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, total, seen = time.time(), 0.0, 0
        order = rng.permutation(len(train))
        for step, (batch, labels) in enumerate(batches(train_enc, y_train, order, args.batch_size), 1):
            batch = {k: v.to(dev) for k, v in batch.items()}
            loss = model(**batch, labels=labels.to(dev)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            total += loss.item() * len(labels)
            seen += len(labels)
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step} loss={total / seen:.4f} {time.time() - t0:.0f}s", flush=True)

        proba = predict(model, val_enc, dev)
        threshold = tune_threshold(val["label"], proba)
        val_metrics = classification_metrics(val["label"], (proba >= threshold).astype(int))
        history.append({"epoch": epoch, "train_loss": round(total / seen, 4),
                        "val_macro_f1": val_metrics["macro_f1"], "threshold": threshold})
        print(f"epoch {epoch} loss={total / seen:.4f} val macro-F1={val_metrics['macro_f1']} "
              f"(t={threshold}) {time.time() - t0:.0f}s", flush=True)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            meta = {"threshold": threshold, "epoch": epoch, "val": val_metrics, "base_model": BASE_MODEL,
                    "max_len": MAX_LEN, "lr": args.lr, "batch_size": args.batch_size}
            (OUT_DIR / "moderation.json").write_text(json.dumps(meta, indent=2))
        meta = json.loads((OUT_DIR / "moderation.json").read_text())
        meta["history"] = history
        (OUT_DIR / "moderation.json").write_text(json.dumps(meta, indent=2))

    del model
    evaluate_checkpoint()


def evaluate_checkpoint() -> None:
    """Final evaluation on CPU, like the deployment target."""
    test = load_split("test")
    meta = json.loads((OUT_DIR / "moderation.json").read_text())
    cpu_model = BertModerator(OUT_DIR)
    result = evaluate_model(cpu_model, test["text"], test["label"],
                            extra={"val": meta["val"], "best_epoch": meta["epoch"],
                                   "history": meta.get("history", [])})
    print(f"test macro-F1={result['test']['macro_f1']} size={result['size_mb']} MB "
          f"latency={result['latency']['mean_ms']} ms (CPU)")


if __name__ == "__main__":
    main()
