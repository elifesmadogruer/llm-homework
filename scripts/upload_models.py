"""Upload the trained models to a Hugging Face Hub model repository.

Requires `hf auth login` with a write token.

Usage:
    python scripts/upload_models.py <user>/<repo>
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from huggingface_hub import HfApi

from moderation.data import MODELS_DIR

# everything the application loads (see moderation/registry.py)
FILES = [
    "lexicon.json",
    "tfidf_lr.joblib",
    "w2v_cnn/model.pt",
    "w2v_cnn/vocab.json",
    "berturk/config.json",
    "berturk/model.safetensors",
    "berturk/tokenizer.json",
    "berturk/tokenizer_config.json",
    "berturk/moderation.json",
    "berturk/model_fp32.onnx",
    "berturk/model_int8.onnx",
]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    repo_id = sys.argv[1]
    missing = [f for f in FILES if not (MODELS_DIR / f).exists()]
    if missing:
        sys.exit(f"missing model files: {missing}")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", exist_ok=True)
    present = {s.rfilename: s.size for s in api.model_info(repo_id, files_metadata=True).siblings}
    # one commit per file, smallest first, with retries: large transfers over a
    # flaky connection fail as a whole otherwise
    for name in sorted(FILES, key=lambda f: (MODELS_DIR / f).stat().st_size):
        size = (MODELS_DIR / name).stat().st_size
        if present.get(name) == size:
            print(f"skip {name} (already uploaded)")
            continue
        for attempt in range(1, 4):
            try:
                api.upload_file(path_or_fileobj=str(MODELS_DIR / name), path_in_repo=name,
                                repo_id=repo_id, commit_message=f"Upload {name}")
                print(f"uploaded {name} ({size / 2**20:.1f} MB)", flush=True)
                break
            except Exception as exc:  # network errors surface as several exception types
                print(f"attempt {attempt} failed for {name}: {type(exc).__name__}", flush=True)
                if attempt == 3:
                    raise
                time.sleep(10 * attempt)
    print(f"uploaded to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
