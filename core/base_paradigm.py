"""Abstract base class for all computational paradigms."""

from abc import ABC, abstractmethod
from core.metrics import ParadigmResult
import numpy as np


class BaseParadigm(ABC):
    """Every paradigm must implement run_attention and set name/short_name."""

    name: str = "Base"
    short_name: str = "base"

    @abstractmethod
    def run_attention(
        self,
        X: np.ndarray,
        W_q: np.ndarray,
        W_k: np.ndarray,
        W_v: np.ndarray,
        W_o: np.ndarray,
    ) -> ParadigmResult:
        """Execute full attention pipeline, return per-step metrics."""
        ...

    def is_available(self) -> bool:
        """Override to return False if hardware not present (e.g., no CUDA)."""
        return True

    def __repr__(self) -> str:
        return f"{self.name} ({'available' if self.is_available() else 'unavailable'})"
