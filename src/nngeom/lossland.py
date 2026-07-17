"""Loss landscape analysis: 1-D slices, 2-D planes, barriers, sharpness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .models import Model
from .probes import filter_normalize, orthogonalize, random_direction


@dataclass
class LossCurve:
    """A 1-D slice of the loss surface."""

    alphas: np.ndarray
    losses: np.ndarray
    kind: str = "interpolation"
    metadata: Dict = field(default_factory=dict)

    def barrier_height(self) -> float:
        """Max loss along the path minus the larger endpoint loss.

        > 0 means the two endpoints are separated by an energy barrier;
        ~<= 0 means they are (linearly) mode-connected.
        """
        endpoints = max(self.losses[0], self.losses[-1])
        return float(self.losses.max() - endpoints)

    def is_monotonic(self, tol: float = 1e-8) -> bool:
        d = np.diff(self.losses)
        return bool(np.all(d >= -tol) or np.all(d <= tol))

    def plot(self, ax=None, **kwargs):
        from .visualize import plot_curve

        return plot_curve(self, ax=ax, **kwargs)

    def to_dict(self) -> Dict:
        return {
            "kind": self.kind,
            "alphas": self.alphas.tolist(),
            "losses": self.losses.tolist(),
            "barrier_height": self.barrier_height(),
            **self.metadata,
        }


@dataclass
class LossSurface:
    """A 2-D slice of the loss surface over a plane in parameter space."""

    alphas: np.ndarray
    betas: np.ndarray
    losses: np.ndarray  # shape (len(betas), len(alphas))
    metadata: Dict = field(default_factory=dict)

    def center_loss(self) -> float:
        i = len(self.betas) // 2
        j = len(self.alphas) // 2
        return float(self.losses[i, j])

    def plot(self, ax=None, **kwargs):
        from .visualize import plot_surface

        return plot_surface(self, ax=ax, **kwargs)

    def to_dict(self) -> Dict:
        return {
            "alphas": self.alphas.tolist(),
            "betas": self.betas.tolist(),
            "losses": self.losses.tolist(),
            **self.metadata,
        }


def _sweep(model: Model, batch, base: np.ndarray, direction: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    losses = np.empty(len(alphas))
    try:
        for i, a in enumerate(alphas):
            model.set_parameters_flat(base + a * direction)
            losses[i] = model.loss(batch)
    finally:
        model.set_parameters_flat(base)
    return losses


def linear_interpolation(model_a: Model, model_b: Model, batch, steps: int = 25, extend: float = 0.0) -> LossCurve:
    """Loss along the straight line between two models' parameters.

    ``extend`` widens the range beyond [0, 1] on both sides (e.g. 0.25 gives
    [-0.25, 1.25]) to see the basin walls around each endpoint.
    """
    theta_a = model_a.parameters_flat()
    theta_b = model_b.parameters_flat()
    if theta_a.shape != theta_b.shape:
        raise ValueError("Models have different parameter counts; cannot interpolate.")
    alphas = np.linspace(-extend, 1.0 + extend, steps)
    losses = _sweep(model_a, batch, theta_a, theta_b - theta_a, alphas)
    return LossCurve(alphas=alphas, losses=losses, kind="interpolation",
                     metadata={"model_a": model_a.name, "model_b": model_b.name})


def random_slice_1d(model: Model, batch, span: float = 1.0, steps: int = 25,
                    seed: Optional[int] = None, normalize: str = "filter") -> LossCurve:
    """Loss along a random direction through the current parameters."""
    d = random_direction(model, seed=seed)
    if normalize == "filter":
        d = filter_normalize(d, model)
    alphas = np.linspace(-span, span, steps)
    losses = _sweep(model, batch, model.parameters_flat(), d, alphas)
    return LossCurve(alphas=alphas, losses=losses, kind="random_slice",
                     metadata={"normalize": normalize, "seed": seed})


def plane_slice_2d(model: Model, batch, span: float = 1.0, steps: int = 15,
                   directions: Optional[List[np.ndarray]] = None,
                   seed: Optional[int] = None, normalize: str = "filter") -> LossSurface:
    """Loss over a 2-D plane spanned by two directions around the model.

    If ``directions`` is None, two random orthogonal (filter-normalized)
    directions are drawn — the classic Li et al. visualization.
    """
    base = model.parameters_flat()
    if directions is None:
        d1 = random_direction(model, seed=seed)
        d2 = random_direction(model, seed=None if seed is None else seed + 1)
        d1, d2 = orthogonalize([d1, d2])
        if normalize == "filter":
            d1, d2 = filter_normalize(d1, model), filter_normalize(d2, model)
    else:
        d1, d2 = directions
    alphas = np.linspace(-span, span, steps)
    betas = np.linspace(-span, span, steps)
    losses = np.empty((steps, steps))
    try:
        for i, b in enumerate(betas):
            for j, a in enumerate(alphas):
                model.set_parameters_flat(base + a * d1 + b * d2)
                losses[i, j] = model.loss(batch)
    finally:
        model.set_parameters_flat(base)
    return LossSurface(alphas=alphas, betas=betas, losses=losses,
                       metadata={"normalize": normalize, "seed": seed, "span": span})


def sharpness(model: Model, batch, radius: float = 0.05, n_samples: int = 10,
              seed: Optional[int] = None, relative: bool = True) -> Dict[str, float]:
    """Random-perturbation sharpness of the current minimum.

    Samples random directions at a fixed (relative) radius and reports how
    much the loss rises. High values = sharp basin; low = flat.
    """
    base = model.parameters_flat()
    base_loss = model.loss(batch)
    scale = radius * (max(np.linalg.norm(base), 1.0) if relative else 1.0)
    rng = np.random.default_rng(seed)
    rises = []
    try:
        for _ in range(n_samples):
            d = rng.standard_normal(base.size)
            d *= scale / np.linalg.norm(d)
            model.set_parameters_flat(base + d)
            rises.append(model.loss(batch) - base_loss)
    finally:
        model.set_parameters_flat(base)
    rises = np.asarray(rises)
    return {
        "base_loss": float(base_loss),
        "mean_rise": float(rises.mean()),
        "max_rise": float(rises.max()),
        "std_rise": float(rises.std()),
        "radius": float(scale),
        "n_samples": n_samples,
    }


def interpolation_smoothness(curve: LossCurve) -> float:
    """Smoothness of a 1-D loss slice: normalized second-difference energy.

    0 for a straight line; larger values mean a bumpier, less predictable
    path between the endpoints.
    """
    y = np.asarray(curve.losses, dtype=np.float64)
    if y.size < 3:
        return 0.0
    d2 = np.diff(y, n=2)
    scale = max(float(y.max() - y.min()), 1e-12)
    return float(np.sqrt(np.mean(d2**2)) / scale)


def loss_barrier_matrix(models: List[Model], batch, steps: int = 11) -> Dict:
    """Pairwise linear-interpolation barrier heights between many checkpoints.

    Entry (i, j) is the barrier along the straight line between model i and
    model j. Near-zero entries mean the pair is (linearly) mode-connected.
    """
    n = len(models)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            curve = linear_interpolation(models[i], models[j], batch, steps=steps)
            M[i, j] = M[j, i] = curve.barrier_height()
    return {"names": [m.name for m in models], "matrix": M}


def bezier_path(model_a: Model, model_b: Model, control: np.ndarray, batch, steps: int = 25) -> LossCurve:
    """Loss along a quadratic Bezier path between two models through a control point."""
    ta = model_a.parameters_flat()
    tb = model_b.parameters_flat()
    alphas = np.linspace(0.0, 1.0, steps)
    losses = np.empty(steps)
    try:
        for i, t in enumerate(alphas):
            point = (1 - t) ** 2 * ta + 2 * (1 - t) * t * control + t**2 * tb
            model_a.set_parameters_flat(point)
            losses[i] = model_a.loss(batch)
    finally:
        model_a.set_parameters_flat(ta)
    return LossCurve(alphas=alphas, losses=losses, kind="bezier",
                     metadata={"model_a": model_a.name, "model_b": model_b.name})


def find_low_loss_path(model_a: Model, model_b: Model, batch, steps: int = 15,
                       iters: int = 30, step_scale: float = 0.1,
                       seed: Optional[int] = None) -> Dict:
    """Search for a low-loss curved path between two models (mode connectivity).

    Optimizes the control point of a quadratic Bezier curve by random-search
    hill climbing to minimize the maximum loss along the path. Returns both
    the straight-line barrier and the barrier of the best curved path; a big
    drop means the modes are connected by a curved valley.
    """
    ta = model_a.parameters_flat()
    tb = model_b.parameters_flat()
    rng = np.random.default_rng(seed)
    straight = linear_interpolation(model_a, model_b, batch, steps=steps)
    control = 0.5 * (ta + tb)
    best_curve = bezier_path(model_a, model_b, control, batch, steps=steps)
    best_max = float(best_curve.losses.max())
    scale = step_scale * max(np.linalg.norm(tb - ta), 1e-12)
    for _ in range(iters):
        candidate = control + scale * rng.standard_normal(ta.size)
        curve = bezier_path(model_a, model_b, candidate, batch, steps=steps)
        cand_max = float(curve.losses.max())
        if cand_max < best_max:
            control, best_max, best_curve = candidate, cand_max, curve
    return {
        "straight_barrier": straight.barrier_height(),
        "curved_barrier": best_curve.barrier_height(),
        "curve": best_curve,
        "control_point": control,
        "connected": best_curve.barrier_height() < 0.1 * max(abs(straight.barrier_height()), 1e-12)
        or best_curve.barrier_height() <= 1e-6,
    }


def basin_width(model: Model, batch, threshold: float = 0.1, max_span: float = 2.0,
                steps: int = 41, n_directions: int = 4, seed: Optional[int] = None) -> Dict[str, float]:
    """Estimate basin width: how far one can move before loss rises by ``threshold``.

    Averages over several filter-normalized random directions. Reported width
    is in units of alpha (relative to layerwise weight norms).
    """
    widths = []
    for k in range(n_directions):
        curve = random_slice_1d(model, batch, span=max_span, steps=steps,
                                seed=None if seed is None else seed + k)
        center = steps // 2
        base_loss = curve.losses[center]
        above = np.flatnonzero(curve.losses > base_loss + threshold)
        left = above[above < center]
        right = above[above > center]
        lo = curve.alphas[left.max()] if left.size else -max_span
        hi = curve.alphas[right.min()] if right.size else max_span
        widths.append(hi - lo)
    widths = np.asarray(widths)
    return {"mean_width": float(widths.mean()), "min_width": float(widths.min()),
            "max_width": float(widths.max()), "threshold": threshold, "n_directions": n_directions}
