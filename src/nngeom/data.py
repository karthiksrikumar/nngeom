"""Data utilities for geometric analysis: sampling, stratification, probes."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def stratified_indices(labels: Sequence, per_class: int, seed: Optional[int] = None) -> np.ndarray:
    """Indices of up to ``per_class`` examples from each class, shuffled."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        chosen.append(idx[:per_class])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def representative_subset(X: np.ndarray, n: int, seed: Optional[int] = None) -> np.ndarray:
    """Greedy farthest-point subset of rows of X — a small, spread-out sample.

    Useful for probing geometry with few, diverse examples instead of a
    random (possibly clumped) batch.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = min(n, X.shape[0])
    chosen = [int(rng.integers(X.shape[0]))]
    dists = np.linalg.norm(X - X[chosen[0]], axis=1)
    for _ in range(n - 1):
        nxt = int(np.argmax(dists))
        chosen.append(nxt)
        dists = np.minimum(dists, np.linalg.norm(X - X[nxt], axis=1))
    return np.asarray(chosen)


def synthetic_probes(n: int, dim: int, kind: str = "gaussian", seed: Optional[int] = None) -> np.ndarray:
    """Synthetic probe inputs for stress-testing model geometry.

    kinds: 'gaussian' (isotropic noise), 'sphere' (unit-norm noise),
    'interpolation' (a line of points between two random anchors).
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((n, dim))
    if kind == "sphere":
        X = rng.standard_normal((n, dim))
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    if kind == "interpolation":
        a, b = rng.standard_normal(dim), rng.standard_normal(dim)
        t = np.linspace(0, 1, n)[:, None]
        return (1 - t) * a + t * b
    raise ValueError(f"Unknown probe kind: {kind!r}")


def batch_iterator(X: np.ndarray, y: Optional[np.ndarray] = None, batch_size: int = 64, seed: Optional[int] = None):
    """Yield shuffled minibatches (X_b,) or (X_b, y_b)."""
    X = np.asarray(X)
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    for start in range(0, X.shape[0], batch_size):
        idx = order[start : start + batch_size]
        if y is None:
            yield (X[idx],)
        else:
            yield (X[idx], np.asarray(y)[idx])
