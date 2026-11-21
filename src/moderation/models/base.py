"""Common interface shared by every model of the ladder."""
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class Moderator(ABC):
    name: str = "base"
    threshold: float = 0.5

    @abstractmethod
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Return P(offensive) for each message."""

    @abstractmethod
    def artifact_paths(self) -> list[Path]:
        """Files that make up the deployed model (used for size reporting)."""

    def predict(self, texts: list[str]) -> np.ndarray:
        return (self.predict_proba(texts) >= self.threshold).astype(int)

    def size_mb(self) -> float:
        total = 0
        for path in self.artifact_paths():
            files = path.rglob("*") if path.is_dir() else [path]
            total += sum(f.stat().st_size for f in files if f.is_file())
        return total / 2**20
