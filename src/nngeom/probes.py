"""Direction and probe generation for geometric experiments."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .models import Model


def random_direction(model: Model, seed: Optional[int] = None, normalize: bool = True) -> np.ndarray:
    """A random Gaussian direction in the model's parameter space."""
    rng = np.random.default_rng(seed)
    d = rng.standard_normal(model.num_parameters)
    if normalize:
        d /= np.linalg.norm(d)
    return d


def filter_normalize(direction: np.ndarray, model: Model) -> np.ndarray:
    """Filter-wise normalization (Li et al., 2018).

    Rescales the direction layer-by-layer so each layer's slice has the same
    norm as the corresponding weights. This removes scale-invariance artifacts
    that make raw random-direction landscapes misleading.
    """
    theta = model.parameters_flat()
    out = direction.astype(np.float64).copy()
    offset = 0
    for _, shape in model.layer_shapes().items():
        size = int(np.prod(shape)) if shape else 1
        sl = slice(offset, offset + size)
        d_norm = np.linalg.norm(out[sl])
        w_norm = np.linalg.norm(theta[sl])
        if d_norm > 1e-12:
            out[sl] *= w_norm / d_norm
        offset += size
    return out


def orthogonalize(directions: List[np.ndarray]) -> List[np.ndarray]:
    """Gram-Schmidt orthonormalization of a list of directions."""
    basis: List[np.ndarray] = []
    for d in directions:
        v = np.asarray(d, dtype=np.float64).copy()
        for b in basis:
            v -= np.dot(v, b) * b
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            basis.append(v / norm)
    return basis


def concept_direction(positives: np.ndarray, negatives: np.ndarray, normalize: bool = True) -> np.ndarray:
    """Difference-of-means concept direction in an activation space.

    ``positives``/``negatives`` are (n, d) activation matrices for examples
    with and without the concept.
    """
    d = np.asarray(positives).mean(axis=0) - np.asarray(negatives).mean(axis=0)
    if normalize:
        n = np.linalg.norm(d)
        if n > 1e-12:
            d = d / n
    return d


def class_separating_axis(X: np.ndarray, labels: np.ndarray, class_a, class_b) -> np.ndarray:
    """Direction separating two classes in representation space (mean difference)."""
    X = np.asarray(X)
    labels = np.asarray(labels)
    return concept_direction(X[labels == class_a], X[labels == class_b])
