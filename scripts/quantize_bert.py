"""Export fine-tuned BERTurk to ONNX and apply dynamic INT8 quantization.

    models/berturk/model_fp32.onnx  - FP32 graph (reference)
    models/berturk/model_int8.onnx  - INT8 weights, activations quantized on the fly

Both are evaluated on the test set and checked against the PyTorch model.

Usage:
    python scripts/quantize_bert.py [--skip-export]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import onnx
import onnxruntime.quantization.quantize as ort_quantize
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.quant_utils import add_infer_metadata

from moderation.data import MODELS_DIR, load_split
from moderation.evaluation.metrics import evaluate_model
from moderation.models.bert import BertModerator
from moderation.models.onnx_model import OnnxBertModerator

MODEL_DIR = MODELS_DIR / "berturk"


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).logits


def export_onnx(bert: BertModerator, path: Path) -> None:
    enc = bert.encode(["örnek bir mesaj", "ikinci, biraz daha uzun bir örnek mesaj"])
    args = (enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
    axes = {0: "batch", 1: "sequence"}
    torch.onnx.export(
        _LogitsOnly(bert.model), args, str(path),
        input_names=["input_ids", "attention_mask", "token_type_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": axes, "attention_mask": axes, "token_type_ids": axes, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def _infer_shapes_in_memory(model_path: Path) -> onnx.ModelProto:
    # ONNX Runtime writes a full "-inferred" copy of the model next to it before
    # quantizing; the model is < 2 GB, so the same shape inference runs in memory.
    model = onnx.shape_inference.infer_shapes(onnx.load(str(model_path)))
    add_infer_metadata(model)
    return model


ort_quantize.load_model_with_shape_infer = _infer_shapes_in_memory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-export", action="store_true", help="reuse an existing model_fp32.onnx")
    args = parser.parse_args()

    test = load_split("test")
    texts = list(test["text"])
    bert = BertModerator(MODEL_DIR)

    fp32_path, int8_path = MODEL_DIR / "model_fp32.onnx", MODEL_DIR / "model_int8.onnx"
    if not (args.skip_export and fp32_path.exists()):
        export_onnx(bert, fp32_path)
    # Weights of MatMul/Gemm (attention, feed-forward, classifier) become INT8 per output
    # channel; activations are quantized at run time. The shape-inference pre-pass is
    # skipped: symbolic inference fails on this graph and it is only an optimization.
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8, per_channel=True)

    reference = bert.predict_proba(texts)
    for onnx_file in ("model_fp32.onnx", "model_int8.onnx"):
        model = OnnxBertModerator(MODEL_DIR, onnx_file)
        proba = model.predict_proba(texts)
        agreement = float(((proba >= model.threshold) == (reference >= model.threshold)).mean())
        diff = float(np.abs(proba - reference).max())
        result = evaluate_model(model, test["text"], test["label"], extra={
            "agreement_with_pytorch": round(agreement, 4), "max_abs_prob_diff": round(diff, 4),
            "quantization": "dynamic INT8 (per-channel weights, ONNX Runtime)" if "int8" in onnx_file else "none",
        })
        print(f"{model.name}: test macro-F1={result['test']['macro_f1']} size={result['size_mb']} MB "
              f"latency={result['latency']['mean_ms']} ms agreement={agreement:.4f} max|dp|={diff:.4f}")


if __name__ == "__main__":
    main()
