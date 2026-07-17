"""Matplotlib visualization of nngeom objects (optional dependency)."""

from __future__ import annotations

import numpy as np

from .utils import HAS_MATPLOTLIB


def _require_mpl():
    if not HAS_MATPLOTLIB:
        raise ImportError("Plotting requires matplotlib. Install nngeom[viz].")
    import matplotlib.pyplot as plt

    return plt


def _get_ax(ax):
    plt = _require_mpl()
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    return ax


def plot_curve(curve, ax=None, label=None, **kwargs):
    """Plot a 1-D loss slice / interpolation curve."""
    ax = _get_ax(ax)
    ax.plot(curve.alphas, curve.losses, label=label, **kwargs)
    ax.set_xlabel("alpha")
    ax.set_ylabel("loss")
    ax.set_title(f"Loss slice ({curve.kind})")
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_surface(surface, ax=None, kind: str = "contour", levels: int = 20, **kwargs):
    """Plot a 2-D loss surface as contour lines or a heatmap."""
    ax = _get_ax(ax)
    A, B = np.meshgrid(surface.alphas, surface.betas)
    if kind == "contour":
        cs = ax.contour(A, B, surface.losses, levels=levels, **kwargs)
        ax.clabel(cs, inline=True, fontsize=7)
    elif kind == "heatmap":
        im = ax.pcolormesh(A, B, surface.losses, shading="auto", **kwargs)
        ax.figure.colorbar(im, ax=ax, label="loss")
    else:
        raise ValueError("kind must be 'contour' or 'heatmap'")
    ax.plot(0, 0, "r*", markersize=12, label="model")
    ax.set_xlabel("direction 1")
    ax.set_ylabel("direction 2")
    ax.set_title("Loss landscape")
    ax.legend()
    return ax


def plot_spectrum(spectrum, ax=None, **kwargs):
    """Plot a Hessian eigenvalue spectrum."""
    ax = _get_ax(ax)
    vals = spectrum.eigenvalues
    ax.stem(np.arange(len(vals)), vals, **kwargs)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    title = f"Hessian spectrum ({spectrum.method})"
    if spectrum.trace_estimate is not None:
        title += f", tr ~= {spectrum.trace_estimate:.3g}"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_trajectory(trajectory, ax=None, annotate: bool = True, **kwargs):
    """Plot a trajectory in its top-2 PCA plane."""
    ax = _get_ax(ax)
    proj, ratio = trajectory.pca_projection(k=2)
    ax.plot(proj[:, 0], proj[:, 1], "-o", markersize=4, **kwargs)
    ax.plot(proj[0, 0], proj[0, 1], "g^", markersize=10, label="start")
    ax.plot(proj[-1, 0], proj[-1, 1], "r*", markersize=12, label="end")
    if annotate and len(trajectory.labels) == len(proj):
        for i in range(0, len(proj), max(1, len(proj) // 8)):
            ax.annotate(trajectory.labels[i], proj[i], fontsize=7, alpha=0.7)
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%} var)" if len(ratio) > 1 else "PC2")
    ax.set_title("Training trajectory (PCA of parameter space)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_cka_matrix(cka_result, ax=None, **kwargs):
    """Heatmap of a pairwise CKA matrix from representations.cka_matrix()."""
    ax = _get_ax(ax)
    names, M = cka_result["names"], cka_result["matrix"]
    im = ax.imshow(M, vmin=0, vmax=1, cmap=kwargs.pop("cmap", "viridis"), **kwargs)
    ax.figure.colorbar(im, ax=ax, label="CKA")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_title("Representation similarity (CKA)")
    return ax


def plot_layer_bars(values: dict, ax=None, title: str = "Layerwise profile", ylabel: str = "value", **kwargs):
    """Bar chart over named layers (norms, drift, curvature...)."""
    ax = _get_ax(ax)
    names = list(values.keys())
    ax.bar(range(len(names)), [values[n] for n in names], **kwargs)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def plot_embedding(X, labels=None, ax=None, **kwargs):
    """2-D PCA scatter of a representation matrix, colored by label."""
    from .manifolds import pca

    ax = _get_ax(ax)
    proj, _, ratio = pca(np.asarray(X, dtype=np.float64), k=2)
    if labels is not None:
        labels = np.asarray(labels)
        for c in np.unique(labels):
            pts = proj[labels == c]
            ax.scatter(pts[:, 0], pts[:, 1], s=12, label=str(c), alpha=0.7, **kwargs)
        ax.legend(fontsize=8)
    else:
        ax.scatter(proj[:, 0], proj[:, 1], s=12, alpha=0.7, **kwargs)
    ax.set_xlabel(f"PC1 ({ratio[0]:.0%})")
    ax.set_ylabel(f"PC2 ({ratio[1]:.0%})" if len(ratio) > 1 else "PC2")
    ax.set_title("Representation (PCA)")
    return ax
