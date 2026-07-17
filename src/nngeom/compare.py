"""Model and checkpoint comparison: distances, drift, task vectors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .models import Model
from .utils import cosine_similarity


@dataclass
class CompareReport:
    """Geometric delta between two models with the same architecture."""

    model_a: str
    model_b: str
    l2_distance: float
    relative_distance: float
    cosine: float
    layer_drift: Dict[str, Dict[str, float]]
    metadata: Dict = field(default_factory=dict)

    def top_changed_layers(self, n: int = 5) -> List[str]:
        return sorted(self.layer_drift, key=lambda k: self.layer_drift[k]["relative_drift"], reverse=True)[:n]

    def summary(self) -> str:
        lines = [
            f"Compare: {self.model_a} vs {self.model_b}",
            f"  L2 distance:        {self.l2_distance:.6g}",
            f"  relative distance:  {self.relative_distance:.4f} (fraction of base norm)",
            f"  parameter cosine:   {self.cosine:.4f}",
            "  most-changed layers (relative drift):",
        ]
        for name in self.top_changed_layers():
            d = self.layer_drift[name]
            lines.append(f"    {name}: {d['relative_drift']:.4f} (L2 {d['l2']:.4g})")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "l2_distance": self.l2_distance,
            "relative_distance": self.relative_distance,
            "cosine": self.cosine,
            "layer_drift": self.layer_drift,
            **self.metadata,
        }


def compare(model_a: Model, model_b: Model) -> CompareReport:
    """Compare two models' positions in parameter space, globally and per layer."""
    ta = model_a.parameters_flat()
    tb = model_b.parameters_flat()
    if ta.shape != tb.shape:
        raise ValueError(
            f"Parameter counts differ ({ta.size} vs {tb.size}); models must share an architecture."
        )
    delta = tb - ta
    base_norm = np.linalg.norm(ta)
    layer_drift: Dict[str, Dict[str, float]] = {}
    offset = 0
    for name, shape in model_a.layer_shapes().items():
        size = int(np.prod(shape)) if shape else 1
        d = delta[offset : offset + size]
        w = ta[offset : offset + size]
        wn = np.linalg.norm(w)
        dn = float(np.linalg.norm(d))
        if wn > 1e-12:
            rel = dn / wn
        else:
            rel = 0.0 if dn < 1e-12 else float("inf")
        layer_drift[name] = {
            "l2": dn,
            "relative_drift": rel,
            "cosine": cosine_similarity(w, tb[offset : offset + size]),
        }
        offset += size
    return CompareReport(
        model_a=model_a.name,
        model_b=model_b.name,
        l2_distance=float(np.linalg.norm(delta)),
        relative_distance=float(np.linalg.norm(delta) / base_norm) if base_norm > 1e-12 else float("inf"),
        cosine=cosine_similarity(ta, tb),
        layer_drift=layer_drift,
    )


def task_vector(base: Model, tuned: Model) -> np.ndarray:
    """tuned - base in parameter space (Ilharco et al., 2023 task arithmetic)."""
    return tuned.parameters_flat() - base.parameters_flat()


def apply_task_vector(model: Model, vector: np.ndarray, scale: float = 1.0) -> None:
    """Move a model along a task vector in place: theta <- theta + scale * v."""
    model.set_parameters_flat(model.parameters_flat() + scale * np.asarray(vector))


def task_vector_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine between two task vectors: do two fine-tunes push the model
    the same way (>0), independently (~0), or in conflict (<0)?"""
    return cosine_similarity(v1, v2)


def model_soup(models: List[Model], weights: Optional[List[float]] = None) -> np.ndarray:
    """Uniform (or weighted) parameter average of several checkpoints.

    The 'model soup' / SWA recipe: averaging fine-tunes that share a basin
    often lands in a flatter, better-generalizing point. Returns the averaged
    flat parameter vector; load it with ``model.set_parameters_flat(...)``.
    Check basin sharing first with ``lossland.loss_barrier_matrix``.
    """
    if not models:
        raise ValueError("model_soup needs at least one model.")
    P = np.stack([m.parameters_flat() for m in models])
    if weights is None:
        return P.mean(axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if w.shape[0] != P.shape[0]:
        raise ValueError("weights must match the number of models.")
    w = w / w.sum()
    return (w[:, None] * P).sum(axis=0)


def eigenvector_overlap(spec_a, spec_b, k: Optional[int] = None) -> float:
    """Overlap between the top eigenspaces of two Hessian spectra.

    Pass two ``Spectrum`` objects that carry eigenvectors (power_iteration
    provides them). Mean squared cosine of principal angles between the two
    top-k eigenspaces; 1.0 means curvature directions are unchanged.
    """
    if spec_a.eigenvectors is None or spec_b.eigenvectors is None:
        raise ValueError("Both spectra need eigenvectors (use power_iteration).")
    Ua = spec_a.eigenvectors
    Ub = spec_b.eigenvectors
    k = k or min(Ua.shape[1], Ub.shape[1])
    s = np.linalg.svd(Ua[:, :k].T @ Ub[:, :k], compute_uv=False)
    return float(np.mean(s**2))
