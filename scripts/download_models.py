"""Download the trained models from the Hugging Face Hub into models/.

Usage:
    python scripts/download_models.py [--repo <user>/<repo>] [--int8-only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import snapshot_download

from moderation.data import MODELS_DIR

DEFAULT_REPO = "elifesma/llm-homework"
FP32_WEIGHTS = ["berturk/model.safetensors", "berturk/model_fp32.onnx"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--int8-only", action="store_true",
                        help="skip the FP32 BERTurk weights (~850 MB); the app then runs INT8 + small models")
    args = parser.parse_args()
    snapshot_download(
        repo_id=args.repo, repo_type="model", local_dir=str(MODELS_DIR),
        ignore_patterns=FP32_WEIGHTS + [".gitattributes", "README.md"] if args.int8_only
        else [".gitattributes", "README.md"],
    )
    print(f"models downloaded to {MODELS_DIR}")


if __name__ == "__main__":
    main()
