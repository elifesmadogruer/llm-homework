"""Access to the fixed splits written by scripts/prepare_data.py."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


def load_split(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python scripts/prepare_data.py` first")
    return pd.read_parquet(path)
