"""Manifold structure of point clouds: dimension, rank, neighborhoods.

All functions take a plain (n_points, n_dims) numpy array — activations,
embeddings, or parameter snapshots — so they compose with any framework.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def pca(X: np.ndarray, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA via SVD. Returns (projected, components, explained_variance_ratio)."""
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2 / max(1, X.shape[0] - 1)
    ratio = var / var.sum() if var.sum() > 0 else var
    if k is not None:
        U, S, Vt, ratio = U[:, :k], S[:k], Vt[:k], ratio[:k]
    return U * S, Vt, ratio


def effective_rank(X: np.ndarray) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007) of the covariance.

    exp(H(p)) where p is the normalized singular-value distribution. Ranges
    from 1 (all variance in one direction) to min(n, d) (isotropic).
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > 1e-12]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(X: np.ndarray) -> float:
    """(sum lambda)^2 / sum lambda^2 of covariance eigenvalues.

    A standard neuroscience/ML dimensionality measure; ~number of dimensions
    that carry meaningful variance.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    lam = np.linalg.svd(Xc, compute_uv=False) ** 2
    denom = np.sum(lam**2)
    return float(np.sum(lam) ** 2 / denom) if denom > 1e-24 else 0.0


def intrinsic_dimension_twonn(X: np.ndarray, discard_fraction: float = 0.1) -> float:
    """TwoNN intrinsic dimension estimator (Facco et al., 2017).

    Uses the ratio of second- to first-nearest-neighbor distances; robust to
    curvature and density variation. Discards the largest ratios (noise tail).
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    if n < 3:
        raise ValueError("TwoNN needs at least 3 points.")
    d2 = _pairwise_sq_dists(X)
    np.fill_diagonal(d2, np.inf)
    part = np.partition(d2, 1, axis=1)[:, :2]
    r1 = np.sqrt(part[:, 0])
    r2 = np.sqrt(part[:, 1])
    ok = r1 > 1e-12
    mu = r2[ok] / r1[ok]
    mu = np.sort(mu)
    keep = int(np.floor(len(mu) * (1 - discard_fraction)))
    mu = mu[: max(keep, 2)]
    logs = np.log(mu[mu > 1.0])
    if logs.size == 0:
        return 0.0
    return float(logs.size / np.sum(logs))


def _pairwise_sq_dists(X: np.ndarray) -> np.ndarray:
    sq = np.sum(X**2, axis=1)
    d2 = sq[:, None] + sq[None, :] - 2 * X @ X.T
    np.maximum(d2, 0, out=d2)
    return d2


def knn_graph(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Indices of the k nearest neighbors of each row (excluding itself)."""
    X = np.asarray(X, dtype=np.float64)
    d2 = _pairwise_sq_dists(X)
    np.fill_diagonal(d2, np.inf)
    k = min(k, X.shape[0] - 1)
    return np.argpartition(d2, k - 1, axis=1)[:, :k]


def neighborhood_preservation(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Fraction of k-NN neighborhoods preserved between two spaces.

    E.g. compare a layer's input vs output geometry, or an embedding before
    and after fine-tuning. 1.0 = local structure fully preserved.
    """
    nx = knn_graph(X, k)
    ny = knn_graph(Y, k)
    overlaps = [len(set(a).intersection(b)) / k for a, b in zip(nx, ny)]
    return float(np.mean(overlaps))


def local_tangent_dimension(X: np.ndarray, k: int = 15, variance_threshold: float = 0.95) -> np.ndarray:
    """Per-point local dimension: PCA on each k-neighborhood, count components
    needed to reach ``variance_threshold`` of local variance."""
    X = np.asarray(X, dtype=np.float64)
    nbrs = knn_graph(X, k)
    dims = np.empty(X.shape[0])
    for i, idx in enumerate(nbrs):
        local = X[idx] - X[idx].mean(axis=0, keepdims=True)
        s = np.linalg.svd(local, compute_uv=False) ** 2
        total = s.sum()
        if total < 1e-24:
            dims[i] = 0
            continue
        cum = np.cumsum(s) / total
        dims[i] = int(np.searchsorted(cum, variance_threshold) + 1)
    return dims


def random_projection(X: np.ndarray, k: int, seed: Optional[int] = None) -> np.ndarray:
    """Johnson-Lindenstrauss random projection to k dimensions.

    Distances are approximately preserved, so downstream geometry metrics
    (kNN, intrinsic dimension, clustering) stay meaningful while huge
    activation matrices become tractable.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((X.shape[1], k)) / np.sqrt(k)
    return X @ R


def outlier_scores(X: np.ndarray, k: int = 10) -> np.ndarray:
    """Per-point outlier score: z-score of mean distance to k nearest neighbors.

    Points far from their local neighborhood (score >> 2) are geometric
    outliers: often mislabeled examples, domain-shifted inputs, or dead
    feature directions.
    """
    X = np.asarray(X, dtype=np.float64)
    d2 = _pairwise_sq_dists(X)
    np.fill_diagonal(d2, np.inf)
    k = min(k, X.shape[0] - 1)
    knn_d = np.sqrt(np.partition(d2, k - 1, axis=1)[:, :k]).mean(axis=1)
    mu, sigma = knn_d.mean(), knn_d.std()
    return (knn_d - mu) / sigma if sigma > 1e-12 else np.zeros_like(knn_d)


def class_manifold_dimensions(X: np.ndarray, labels) -> Dict[str, float]:
    """TwoNN intrinsic dimension of each class's own representation manifold.

    Classes living on much higher-dimensional manifolds than the rest are
    typically harder to learn and less robust.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    out: Dict[str, float] = {}
    for c in np.unique(labels):
        pts = X[labels == c]
        if pts.shape[0] >= 10:
            try:
                out[str(c)] = intrinsic_dimension_twonn(pts)
            except ValueError:
                pass
    return out


def manifold_summary(X: np.ndarray, k: int = 10) -> Dict[str, float]:
    """One-call summary of a point cloud's manifold structure."""
    X = np.asarray(X, dtype=np.float64)
    out = {
        "n_points": int(X.shape[0]),
        "ambient_dim": int(X.shape[1]),
        "effective_rank": effective_rank(X),
        "participation_ratio": participation_ratio(X),
        "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
    }
    if X.shape[0] >= 3:
        try:
            out["intrinsic_dim_twonn"] = intrinsic_dimension_twonn(X)
        except ValueError:
            pass
        dims = local_tangent_dimension(X, k=min(k, X.shape[0] - 1))
        out["local_dim_mean"] = float(dims.mean())
    return out
