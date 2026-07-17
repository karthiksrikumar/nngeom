"""Optimization dynamics: gradient statistics, SAM sharpness, local linearity.

These tools answer "how is training behaving right now?" questions:
noisy gradients, conflicting batches, sharpness-aware loss rises, and how
linear the loss is in a neighborhood (trust-region quality).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .models import Model
from .utils import cosine_similarity


def batch_gradients(model: Model, batches: Sequence) -> np.ndarray:
    """Stack per-batch gradients into a (n_batches, n_params) matrix."""
    return np.stack([model.gradient(b) for b in batches])


def gradient_alignment(model: Model, batches: Sequence) -> Dict[str, float]:
    """How consistently do different batches pull the model the same way?

    Returns the mean pairwise cosine between batch gradients. Near 1 the
    batches agree (clean signal); near 0 the gradient is mostly noise;
    negative values mean batches actively conflict.
    """
    G = batch_gradients(model, batches)
    n = G.shape[0]
    cosines = []
    for i in range(n):
        for j in range(i + 1, n):
            cosines.append(cosine_similarity(G[i], G[j]))
    cosines = np.asarray(cosines)
    return {
        "mean_cosine": float(cosines.mean()) if cosines.size else 1.0,
        "min_cosine": float(cosines.min()) if cosines.size else 1.0,
        "n_batches": n,
    }


def gradient_noise_scale(model: Model, batches: Sequence) -> float:
    """Simple gradient noise scale: tr(Cov(g)) / ||E[g]||^2.

    Large values mean minibatch gradients are dominated by noise, which
    suggests a larger batch size (or lower learning rate) would help.
    """
    G = batch_gradients(model, batches)
    mean_g = G.mean(axis=0)
    trace_cov = float(G.var(axis=0, ddof=1).sum()) if G.shape[0] > 1 else 0.0
    denom = float(np.dot(mean_g, mean_g))
    return trace_cov / denom if denom > 1e-24 else float("inf")


def fisher_diagonal(model: Model, batches: Sequence) -> np.ndarray:
    """Empirical Fisher diagonal: mean of squared per-batch gradients.

    A cheap curvature/importance proxy per parameter, used in EWC-style
    regularization and importance pruning.
    """
    G = batch_gradients(model, batches)
    return (G**2).mean(axis=0)


def fisher_layer_importance(model: Model, batches: Sequence) -> Dict[str, float]:
    """Sum of the empirical Fisher diagonal within each layer block."""
    diag = fisher_diagonal(model, batches)
    out: Dict[str, float] = {}
    offset = 0
    for name, shape in model.layer_shapes().items():
        size = int(np.prod(shape)) if shape else 1
        out[name] = float(diag[offset : offset + size].sum())
        offset += size
    return out


def sam_sharpness(model: Model, batch, rho: float = 0.05, relative: bool = True) -> Dict[str, float]:
    """Sharpness-aware (SAM-style) worst-case loss rise.

    Takes one ascent step of size rho along the normalized gradient (the
    first-order approximation of the worst perturbation) and reports the
    loss rise. More targeted than random-perturbation sharpness.
    """
    base = model.parameters_flat()
    base_loss = model.loss(batch)
    g = model.gradient(batch)
    g_norm = np.linalg.norm(g)
    scale = rho * (max(np.linalg.norm(base), 1.0) if relative else 1.0)
    if g_norm < 1e-12:
        return {"base_loss": float(base_loss), "adversarial_rise": 0.0, "rho": float(scale)}
    try:
        model.set_parameters_flat(base + scale * g / g_norm)
        adv_loss = model.loss(batch)
    finally:
        model.set_parameters_flat(base)
    return {
        "base_loss": float(base_loss),
        "adversarial_loss": float(adv_loss),
        "adversarial_rise": float(adv_loss - base_loss),
        "rho": float(scale),
        "gradient_norm": float(g_norm),
    }


def local_linearity(model: Model, batch, radius: float = 0.01, n_samples: int = 10,
                    seed: Optional[int] = None) -> Dict[str, float]:
    """How well a first-order (linear) model predicts nearby losses.

    Samples points at a fixed radius, predicts their loss with
    loss + g . d, and reports the relative residual. Near 0 the region is
    locally linear (safe for large steps); large values mean strong
    curvature or non-smoothness.
    """
    base = model.parameters_flat()
    base_loss = model.loss(batch)
    g = model.gradient(batch)
    rng = np.random.default_rng(seed)
    scale = radius * max(1.0, np.linalg.norm(base))
    residuals, actual_changes = [], []
    try:
        for _ in range(n_samples):
            d = rng.standard_normal(base.size)
            d *= scale / np.linalg.norm(d)
            model.set_parameters_flat(base + d)
            actual = model.loss(batch)
            predicted = base_loss + float(np.dot(g, d))
            residuals.append(abs(actual - predicted))
            actual_changes.append(abs(actual - base_loss))
    finally:
        model.set_parameters_flat(base)
    residuals = np.asarray(residuals)
    denom = max(float(np.mean(actual_changes)), 1e-12)
    return {
        "linearity_error": float(residuals.mean() / denom),
        "mean_residual": float(residuals.mean()),
        "radius": float(scale),
        "n_samples": n_samples,
    }
