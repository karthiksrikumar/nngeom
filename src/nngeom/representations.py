"""Representation geometry: hidden states and embeddings as point clouds.

Similarity (CKA), class structure, anisotropy, and collapse detection.
Activation extraction for torch models is provided via forward hooks; all
metrics themselves are pure numpy and work on any (n, d) matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .manifolds import effective_rank, participation_ratio
from .utils import HAS_TORCH, as_numpy

if HAS_TORCH:
    import torch


# ---------------------------------------------------------------------------
# Activation extraction (torch)
# ---------------------------------------------------------------------------

def extract_activations(module, inputs, layers: Sequence[str], flatten: bool = True) -> Dict[str, np.ndarray]:
    """Capture activations from named submodules of a torch model.

    Parameters
    ----------
    module : torch.nn.Module
    inputs : tensor or tuple of tensors passed to ``module(...)``
    layers : names from ``module.named_modules()`` to hook
    flatten : reshape each activation to (batch, -1)
    """
    if not HAS_TORCH:
        raise ImportError("extract_activations requires PyTorch. Install nngeom[torch].")
    acts: Dict[str, np.ndarray] = {}
    hooks = []
    available = dict(module.named_modules())
    for name in layers:
        if name not in available:
            raise KeyError(f"Layer {name!r} not found. Available: {sorted(available)[:20]}...")

        def make_hook(layer_name):
            def hook(_mod, _inp, out):
                if isinstance(out, tuple):
                    out = out[0]
                a = as_numpy(out)
                acts[layer_name] = a.reshape(a.shape[0], -1) if flatten else a
            return hook

        hooks.append(available[name].register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            if isinstance(inputs, (tuple, list)):
                module(*inputs)
            else:
                module(inputs)
    finally:
        for h in hooks:
            h.remove()
    return acts


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def _center_gram(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear Centered Kernel Alignment between two representations.

    X, Y are (n_examples, dim) with the same rows (same examples). 1.0 =
    identical geometry up to rotation/scale; near 0 = unrelated.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    normx = np.linalg.norm(Xc.T @ Xc, "fro")
    normy = np.linalg.norm(Yc.T @ Yc, "fro")
    if normx < 1e-12 or normy < 1e-12:
        return 0.0
    return float(hsic / (normx * normy))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma_fraction: float = 0.5) -> float:
    """RBF-kernel CKA; bandwidth set to ``sigma_fraction`` * median distance."""
    def gram(Z):
        Z = np.asarray(Z, dtype=np.float64)
        sq = np.sum(Z**2, axis=1)
        d2 = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
        med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
        sigma2 = sigma_fraction * med
        return np.exp(-d2 / (2 * max(sigma2, 1e-12)))

    Kx = _center_gram(gram(X))
    Ky = _center_gram(gram(Y))
    hsic = float(np.sum(Kx * Ky))
    denom = np.sqrt(np.sum(Kx * Kx) * np.sum(Ky * Ky))
    return float(hsic / denom) if denom > 1e-12 else 0.0


def cka_matrix(reps: Dict[str, np.ndarray], kernel: str = "linear") -> Dict:
    """Pairwise CKA between a set of named representations (e.g. layers)."""
    names = list(reps.keys())
    fn = linear_cka if kernel == "linear" else rbf_cka
    M = np.eye(len(names))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            M[i, j] = M[j, i] = fn(reps[names[i]], reps[names[j]])
    return {"names": names, "matrix": M}


def subspace_overlap(X: np.ndarray, Y: np.ndarray, k: int = 10) -> float:
    """Overlap of top-k principal subspaces of two representations.

    Mean squared cosine of principal angles; 1.0 = same subspace.
    """
    def top_components(Z, k):
        Zc = np.asarray(Z, dtype=np.float64)
        Zc = Zc - Zc.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        return Vt[:k].T

    k = min(k, min(X.shape) - 1, min(Y.shape) - 1)
    U = top_components(X, k)
    V = top_components(Y, k)
    s = np.linalg.svd(U.T @ V, compute_uv=False)
    return float(np.mean(s**2))


def svcca(X: np.ndarray, Y: np.ndarray, variance_kept: float = 0.99) -> float:
    """SVCCA similarity (Raghu et al., 2017): SVD denoising then mean CCA.

    Keeps the top singular directions covering ``variance_kept`` of each
    representation, then reports the mean canonical correlation. 1.0 means
    the informative subspaces are linearly identical.
    """
    def svd_reduce(Z):
        Z = np.asarray(Z, dtype=np.float64)
        Zc = Z - Z.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(Zc, full_matrices=False)
        var = S**2
        cum = np.cumsum(var) / var.sum() if var.sum() > 0 else np.ones_like(var)
        k = int(np.searchsorted(cum, variance_kept) + 1)
        return U[:, :k] * S[:k]

    A = svd_reduce(X)
    B = svd_reduce(Y)
    # CCA via QR: canonical correlations are the singular values of Qa^T Qb
    Qa, _ = np.linalg.qr(A - A.mean(axis=0, keepdims=True))
    Qb, _ = np.linalg.qr(B - B.mean(axis=0, keepdims=True))
    corrs = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    k = min(A.shape[1], B.shape[1])
    return float(np.clip(corrs[:k], 0, 1).mean())


def procrustes_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes distance between two same-shape representations.

    Finds the best rotation aligning X to Y and reports the residual as a
    fraction of Y's norm, in [0, ~1]. 0 means X and Y differ only by rotation.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.shape != Y.shape:
        raise ValueError("Procrustes needs same-shape matrices (same examples, same dim).")
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    nx = np.linalg.norm(Xc)
    ny = np.linalg.norm(Yc)
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    Xc /= nx
    Yc /= ny
    U, S, Vt = np.linalg.svd(Xc.T @ Yc)
    residual = 1.0 - 2.0 * S.sum() + 1.0  # ||Xc R - Yc||_F^2 with optimal R
    return float(np.sqrt(max(residual, 0.0)))


def uniformity(X: np.ndarray, t: float = 2.0, n_pairs: int = 2000, seed: Optional[int] = None) -> float:
    """Uniformity of normalized representations (Wang & Isola, 2020).

    log E[exp(-t ||xi - xj||^2)] over random pairs on the unit sphere.
    More negative is more uniform (features spread out); values near 0 mean
    everything is bunched together.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Z = X / np.maximum(norms, 1e-12)
    rng = np.random.default_rng(seed)
    n = Z.shape[0]
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    d2 = np.sum((Z[i[keep]] - Z[j[keep]]) ** 2, axis=1)
    return float(np.log(np.mean(np.exp(-t * d2))))


def linear_probe_score(X: np.ndarray, labels, l2: float = 1e-3, train_fraction: float = 0.8,
                       seed: Optional[int] = None) -> Dict[str, float]:
    """Closed-form linear probe: how linearly decodable are the labels?

    Fits ridge regression to one-hot targets on a train split and reports
    train/test accuracy. High probe accuracy with low downstream accuracy
    means the information is present but the head is not using it.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    Yhot = (labels[:, None] == classes[None, :]).astype(np.float64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(X.shape[0])
    n_train = max(int(train_fraction * X.shape[0]), len(classes))
    tr, te = order[:n_train], order[n_train:]
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1))])
    W = np.linalg.solve(Xtr.T @ Xtr + l2 * np.eye(Xtr.shape[1]), Xtr.T @ Yhot[tr])

    def acc(idx):
        if len(idx) == 0:
            return float("nan")
        Xb = np.hstack([X[idx], np.ones((len(idx), 1))])
        pred = classes[np.argmax(Xb @ W, axis=1)]
        return float(np.mean(pred == labels[idx]))

    return {"train_accuracy": acc(tr), "test_accuracy": acc(te),
            "n_classes": int(len(classes)), "n_train": int(len(tr)), "n_test": int(len(te))}


def neuron_alignment(X: np.ndarray, Y: np.ndarray) -> Dict:
    """Match individual neurons between two representations by correlation.

    Greedy one-to-one matching on the absolute correlation matrix. Useful for
    comparing seeds ('did the same features emerge?') and for permutation-
    aligning models before interpolation.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    sx = Xc.std(axis=0)
    sy = Yc.std(axis=0)
    C = (Xc.T @ Yc) / X.shape[0]
    C /= np.maximum(np.outer(sx, sy), 1e-12)
    A = np.abs(C.copy())
    pairs = []
    for _ in range(min(A.shape)):
        i, j = np.unravel_index(np.argmax(A), A.shape)
        if A[i, j] <= 0:
            break
        pairs.append((int(i), int(j), float(C[i, j])))
        A[i, :] = -1
        A[:, j] = -1
    corrs = np.asarray([abs(c) for _, _, c in pairs])
    return {
        "pairs": pairs,
        "mean_matched_correlation": float(corrs.mean()) if corrs.size else 0.0,
        "fraction_well_matched": float(np.mean(corrs > 0.5)) if corrs.size else 0.0,
    }


def representation_drift(before: Dict[str, np.ndarray], after: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Per-layer geometric drift between two snapshots of the same layers.

    Feed activations for the SAME inputs before and after an intervention
    (fine-tuning, pruning, quantization). Reports CKA, Procrustes distance,
    and norm change per layer, so you can see where behavior actually moved.
    """
    out: Dict[str, Dict[str, float]] = {}
    for name in before:
        if name not in after:
            continue
        B, A = np.asarray(before[name], dtype=np.float64), np.asarray(after[name], dtype=np.float64)
        entry = {
            "cka": linear_cka(B, A),
            "norm_ratio": float(np.linalg.norm(A) / max(np.linalg.norm(B), 1e-12)),
        }
        if B.shape == A.shape:
            entry["procrustes_distance"] = procrustes_distance(B, A)
        out[name] = entry
    return out


# ---------------------------------------------------------------------------
# Structure metrics
# ---------------------------------------------------------------------------

def anisotropy(X: np.ndarray, n_pairs: int = 2000, seed: Optional[int] = None) -> float:
    """Expected cosine similarity between random pairs of representations.

    High anisotropy (common in transformer embeddings) means all vectors
    point in a narrow cone; 0 means directions are spread isotropically.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    a, b = X[i[keep]], X[j[keep]]
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    ok = (na > 1e-12) & (nb > 1e-12)
    cos = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return float(cos.mean()) if cos.size else 0.0


def collapse_score(X: np.ndarray) -> float:
    """How collapsed a representation is, in [0, 1].

    1 - participation_ratio / min(n, d): 0 = variance spread across all
    available dimensions, ->1 = everything mapped to (nearly) one direction.
    """
    X = np.asarray(X, dtype=np.float64)
    max_dim = min(X.shape[0], X.shape[1])
    if max_dim == 0:
        return 0.0
    return float(np.clip(1.0 - participation_ratio(X) / max_dim, 0.0, 1.0))


def class_geometry(X: np.ndarray, labels: Sequence) -> Dict:
    """Class-cluster structure of a labeled representation.

    Returns centroids, per-class radius (mean distance to centroid),
    between/within separation ratio (Fisher-style), and centroid distances.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = np.unique(labels)
    centroids = {}
    radii = {}
    within = []
    for c in classes:
        pts = X[labels == c]
        mu = pts.mean(axis=0)
        centroids[c] = mu
        d = np.linalg.norm(pts - mu, axis=1)
        radii[c] = float(d.mean())
        within.append(float((d**2).mean()))
    global_mu = X.mean(axis=0)
    between = float(np.mean([np.sum((centroids[c] - global_mu) ** 2) for c in classes]))
    within_mean = float(np.mean(within))
    C = np.stack([centroids[c] for c in classes])
    diff = C[:, None, :] - C[None, :, :]
    centroid_dists = np.linalg.norm(diff, axis=2)
    return {
        "classes": classes.tolist(),
        "centroids": C,
        "class_radius": {str(c): radii[c] for c in classes},
        "within_scatter": within_mean,
        "between_scatter": between,
        "separation_ratio": between / within_mean if within_mean > 1e-24 else float("inf"),
        "centroid_distances": centroid_dists,
        "min_centroid_distance": float(centroid_dists[centroid_dists > 0].min()) if len(classes) > 1 else 0.0,
    }


def representation_summary(X: np.ndarray, labels: Optional[Sequence] = None) -> Dict:
    """One-call geometric profile of a representation matrix."""
    X = np.asarray(X, dtype=np.float64)
    out = {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "mean_norm": float(np.linalg.norm(X, axis=1).mean()),
        "effective_rank": effective_rank(X),
        "participation_ratio": participation_ratio(X),
        "anisotropy": anisotropy(X, seed=0),
        "collapse_score": collapse_score(X),
    }
    if labels is not None:
        cg = class_geometry(X, labels)
        out["separation_ratio"] = cg["separation_ratio"]
        out["min_centroid_distance"] = cg["min_centroid_distance"]
    return out
