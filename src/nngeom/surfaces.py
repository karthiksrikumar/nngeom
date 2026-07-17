"""Generic scalar-field sampling over parameter-space grids.

``lossland`` uses these primitives with loss as the field; you can sample any
scalar (accuracy, entropy, gradient norm, a custom metric) the same way.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from .lossland import LossCurve, LossSurface
from .models import Model


def scalar_field_1d(model: Model, field_fn: Callable[[Model], float], direction: np.ndarray,
                    span: float = 1.0, steps: int = 25) -> LossCurve:
    """Sample field_fn(model) along a direction through the current point."""
    base = model.parameters_flat()
    alphas = np.linspace(-span, span, steps)
    values = np.empty(steps)
    try:
        for i, a in enumerate(alphas):
            model.set_parameters_flat(base + a * direction)
            values[i] = float(field_fn(model))
    finally:
        model.set_parameters_flat(base)
    return LossCurve(alphas=alphas, losses=values, kind="scalar_field")


def scalar_field_2d(model: Model, field_fn: Callable[[Model], float],
                    d1: np.ndarray, d2: np.ndarray,
                    span: float = 1.0, steps: int = 15) -> LossSurface:
    """Sample field_fn(model) over the plane spanned by d1, d2."""
    base = model.parameters_flat()
    alphas = np.linspace(-span, span, steps)
    betas = np.linspace(-span, span, steps)
    values = np.empty((steps, steps))
    try:
        for i, b in enumerate(betas):
            for j, a in enumerate(alphas):
                model.set_parameters_flat(base + a * d1 + b * d2)
                values[i, j] = float(field_fn(model))
    finally:
        model.set_parameters_flat(base)
    return LossSurface(alphas=alphas, betas=betas, losses=values,
                       metadata={"field": getattr(field_fn, "__name__", "custom")})
