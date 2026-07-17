"""Central metric registry.

Built-in metrics operate on (n, d) representation matrices. Users can add
their own with :func:`register_metric` and compute batches of metrics with
:func:`compute`.

    @ng.register_metric("my_metric")
    def my_metric(X):
        return float(...)

    ng.metrics.compute(X, ["effective_rank", "my_metric"])
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional

import numpy as np

from .manifolds import effective_rank, intrinsic_dimension_twonn, participation_ratio
from .representations import anisotropy, collapse_score

_REGISTRY: Dict[str, Callable] = {}


def register_metric(name: str) -> Callable:
    """Decorator registering a metric function fn(X, **kwargs) -> float."""

    def decorator(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"Metric {name!r} is already registered.")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_metric(name: str) -> Callable:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown metric {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def list_metrics() -> list:
    return sorted(_REGISTRY)


def compute(X: np.ndarray, names: Optional[Iterable[str]] = None, **kwargs) -> Dict[str, float]:
    """Compute several registered metrics on a representation matrix."""
    names = list(names) if names is not None else list_metrics()
    return {name: float(get_metric(name)(X, **kwargs.get(name, {}))) for name in names}


# -- built-ins ---------------------------------------------------------------

register_metric("effective_rank")(effective_rank)
register_metric("participation_ratio")(participation_ratio)
register_metric("anisotropy")(anisotropy)
register_metric("collapse_score")(collapse_score)


@register_metric("mean_norm")
def _mean_norm(X: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1).mean())


@register_metric("norm_variance")
def _norm_variance(X: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(X, dtype=np.float64), axis=1).var())


@register_metric("intrinsic_dimension")
def _intrinsic_dim(X: np.ndarray) -> float:
    return intrinsic_dimension_twonn(X)


@register_metric("uniformity")
def _uniformity(X: np.ndarray) -> float:
    from .representations import uniformity

    return uniformity(X, seed=0)


@register_metric("stable_rank")
def _stable_rank(X: np.ndarray) -> float:
    X = np.asarray(X, dtype=np.float64)
    s = np.linalg.svd(X - X.mean(axis=0, keepdims=True), compute_uv=False)
    return float(np.sum(s**2) / s[0] ** 2) if s.size and s[0] > 1e-12 else 0.0
