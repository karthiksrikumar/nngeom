"""Shared utilities: parameter flattening, seeding, framework detection."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np

try:
    import torch

    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None
    HAS_TORCH = False

try:
    import matplotlib

    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    matplotlib = None
    HAS_MATPLOTLIB = False


def set_seed(seed: int) -> None:
    """Seed python, numpy, and torch (if available) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)


def flatten_dict(params: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[Tuple[str, tuple]]]:
    """Flatten a dict of arrays into a single 1-D vector plus a spec to invert it."""
    spec = [(name, np.asarray(arr).shape) for name, arr in params.items()]
    if not spec:
        return np.zeros(0), spec
    flat = np.concatenate([np.asarray(arr, dtype=np.float64).ravel() for arr in params.values()])
    return flat, spec


def unflatten_dict(flat: np.ndarray, spec: List[Tuple[str, tuple]]) -> Dict[str, np.ndarray]:
    """Invert :func:`flatten_dict`."""
    out: Dict[str, np.ndarray] = {}
    offset = 0
    for name, shape in spec:
        size = int(np.prod(shape)) if shape else 1
        out[name] = flat[offset : offset + size].reshape(shape)
        offset += size
    return out


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """Cosine similarity between two vectors."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def as_numpy(x) -> np.ndarray:
    """Convert torch tensors / array-likes to a numpy array."""
    if HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
