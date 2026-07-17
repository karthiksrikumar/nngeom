"""The automated geometry report: ``ng.analyze(model, batch, ...)``.

One call runs a standard battery of geometric measurements, collects them
into sections, flags suspicious geometry (collapse, saddle points, sharp
basins, erratic trajectories), and can export everything as JSON or a
self-contained HTML report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import curvature as _curvature
from . import lossland as _lossland
from .compare import compare as _compare
from .export import report_html, to_json
from .models import Model
from .representations import representation_summary
from .trajectories import Trajectory
from .utils import HAS_MATPLOTLIB


@dataclass
class GeometryReport:
    """Structured result of :func:`analyze`."""

    title: str
    sections: Dict[str, Dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    figures: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable text summary."""
        lines = [f"=== {self.title} ==="]
        for name, content in self.sections.items():
            lines.append(f"\n[{name}]")
            if isinstance(content, dict):
                for k, v in content.items():
                    if isinstance(v, dict):
                        lines.append(f"  {k}:")
                        for k2, v2 in v.items():
                            if not isinstance(v2, (dict, list, np.ndarray)):
                                lines.append(f"    {k2}: {_fmt(v2)}")
                    elif not isinstance(v, (list, np.ndarray)):
                        lines.append(f"  {k}: {_fmt(v)}")
        if self.warnings:
            lines.append("\n[warnings]")
            lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_json(self, path: str) -> str:
        return to_json({"title": self.title, "sections": self.sections, "warnings": self.warnings}, path)

    def to_html(self, path: str) -> str:
        return report_html(self, path)

    def __repr__(self) -> str:
        return f"GeometryReport(sections={list(self.sections)}, warnings={len(self.warnings)})"


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def analyze(
    model: Model,
    batch=None,
    checkpoints: Optional[Sequence[Model]] = None,
    representations: Optional[Dict[str, np.ndarray]] = None,
    labels: Optional[np.ndarray] = None,
    depth: str = "standard",
    seed: int = 0,
    make_figures: Optional[bool] = None,
) -> GeometryReport:
    """Run an automated geometric profile of a model.

    Parameters
    ----------
    model : nngeom.Model
    batch : data batch for loss/curvature analyses (optional but recommended)
    checkpoints : sequence of Models over training for trajectory analysis
    representations : dict of layer_name -> (n, d) activation matrices
    labels : labels aligned with the representation rows (for class geometry)
    depth : 'quick' | 'standard' | 'full' — controls sample counts and extras
    make_figures : attach matplotlib figures (default: if matplotlib installed)
    """
    if depth not in ("quick", "standard", "full"):
        raise ValueError("depth must be 'quick', 'standard', or 'full'")
    budget = {"quick": 0, "standard": 1, "full": 2}[depth]
    if make_figures is None:
        make_figures = HAS_MATPLOTLIB
    report = GeometryReport(title=f"Geometry report: {model.name}")

    # -- model / parameter space ------------------------------------------------
    norms = model.layer_norms()
    flat = model.parameters_flat()
    report.sections["model"] = {
        "name": model.name,
        "n_parameters": model.num_parameters,
        "n_layers": len(norms),
        "global_norm": float(np.linalg.norm(flat)),
        "layer_norms": norms,
    }

    # -- loss geometry ------------------------------------------------------------
    if batch is not None:
        sharp = _lossland.sharpness(model, batch, n_samples=[5, 10, 25][budget], seed=seed)
        loss_sec: Dict = {"sharpness": sharp}
        curve = _lossland.random_slice_1d(model, batch, span=0.5,
                                          steps=[11, 21, 41][budget], seed=seed)
        loss_sec["random_slice_min_alpha"] = float(curve.alphas[np.argmin(curve.losses)])
        if budget >= 1:
            loss_sec["basin"] = _lossland.basin_width(model, batch, n_directions=[2, 3, 6][budget], seed=seed)
        report.sections["loss_landscape"] = loss_sec
        if sharp["mean_rise"] > 10 * max(abs(sharp["base_loss"]), 1e-3):
            report.warnings.append(
                "Very sharp basin: random perturbations raise loss by "
                f"{sharp['mean_rise']:.3g} (base loss {sharp['base_loss']:.3g})."
            )
        if abs(loss_sec["random_slice_min_alpha"]) > 0.1:
            report.warnings.append(
                "Model is not at a local minimum along a random slice "
                f"(min at alpha={loss_sec['random_slice_min_alpha']:.2f})."
            )
        if make_figures:
            report.figures["random_slice"] = curve.plot().figure

        # -- curvature ---------------------------------------------------------
        k = [3, 5, 10][budget]
        spectrum = _curvature.hessian_spectrum(model, batch, k=k, method="lanczos",
                                               trace=budget >= 1, seed=seed)
        report.sections["curvature"] = {
            "method": spectrum.method,
            "top_eigenvalue": spectrum.top_eigenvalue,
            "eigenvalues": spectrum.eigenvalues[:k].tolist(),
            "trace_estimate": spectrum.trace_estimate,
            "negative_fraction": spectrum.negative_fraction(),
        }
        if budget >= 2:
            report.sections["curvature"]["layerwise_trace"] = _curvature.layerwise_curvature(
                model, batch, seed=seed
            )
        if spectrum.negative_fraction() > 0.25:
            report.warnings.append(
                f"{spectrum.negative_fraction():.0%} of estimated Hessian eigenvalues are negative -- "
                "the model may sit near a saddle point rather than a minimum."
            )
        if make_figures:
            report.figures["hessian_spectrum"] = spectrum.plot().figure

    # -- representations ---------------------------------------------------------
    if representations:
        rep_sec: Dict = {}
        for name, X in representations.items():
            summ = representation_summary(X, labels=labels)
            rep_sec[name] = summ
            if summ["collapse_score"] > 0.9:
                report.warnings.append(
                    f"Representation {name!r} looks collapsed "
                    f"(collapse score {summ['collapse_score']:.2f})."
                )
            if summ["anisotropy"] > 0.7:
                report.warnings.append(
                    f"Representation {name!r} is highly anisotropic "
                    f"(mean random-pair cosine {summ['anisotropy']:.2f})."
                )
        report.sections["representations"] = rep_sec
        if len(representations) > 1:
            from .representations import cka_matrix

            cka = cka_matrix(representations)
            report.sections["representation_similarity"] = {
                "layers": cka["names"],
                "cka_matrix": cka["matrix"].tolist(),
            }
            if make_figures:
                from .visualize import plot_cka_matrix

                report.figures["cka_matrix"] = plot_cka_matrix(cka).figure

    # -- trajectory ---------------------------------------------------------------
    if checkpoints is not None and len(checkpoints) >= 2:
        traj = Trajectory.from_models(list(checkpoints))
        report.sections["trajectory"] = traj.summary()
        if traj.directional_persistence() < 0:
            report.warnings.append(
                "Consecutive training steps point in opposing directions on average -- "
                "possible oscillation (learning rate too high?)."
            )
        final_compare = _compare(checkpoints[0], checkpoints[-1])
        report.sections["trajectory"]["top_changed_layers"] = final_compare.top_changed_layers()
        if make_figures:
            report.figures["trajectory"] = traj.plot().figure

    # -- suggestions ----------------------------------------------------------------
    suggestions = []
    if batch is not None and "curvature" in report.sections:
        suggestions.append("Compare the Hessian spectrum across checkpoints to track sharpening.")
    if representations:
        suggestions.append("Run linear_interpolation between checkpoints to test mode connectivity.")
    if checkpoints is None:
        suggestions.append("Pass checkpoints=[...] to add training-trajectory geometry.")
    report.sections["suggested_next_steps"] = {str(i + 1): s for i, s in enumerate(suggestions)}

    return report
