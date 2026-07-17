"""Second-order geometry: Hessian spectra via matrix-free methods.

Everything here only needs a Hessian-vector product (``model.hvp``), so it
scales to models where the full Hessian is intractable. Torch models get
exact double-backward HVPs; functional models fall back to finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .models import Model


@dataclass
class Spectrum:
    """Result of a Hessian spectrum estimate."""

    eigenvalues: np.ndarray          # descending by value
    eigenvectors: Optional[np.ndarray] = None  # columns, matching eigenvalues
    trace_estimate: Optional[float] = None
    method: str = "power_iteration"
    metadata: Dict = field(default_factory=dict)

    @property
    def top_eigenvalue(self) -> float:
        return float(self.eigenvalues[0])

    def condition_ratio(self) -> float:
        """lambda_max / lambda_k — spread of the computed spectrum."""
        lo = np.abs(self.eigenvalues[-1])
        return float(np.abs(self.eigenvalues[0]) / lo) if lo > 1e-12 else float("inf")

    def negative_fraction(self) -> float:
        """Fraction of computed eigenvalues that are negative (saddle signal)."""
        return float(np.mean(self.eigenvalues < 0))

    def plot(self, ax=None, **kwargs):
        from .visualize import plot_spectrum

        return plot_spectrum(self, ax=ax, **kwargs)

    def to_dict(self) -> Dict:
        return {
            "eigenvalues": self.eigenvalues.tolist(),
            "trace_estimate": self.trace_estimate,
            "method": self.method,
            "top_eigenvalue": self.top_eigenvalue,
            "negative_fraction": self.negative_fraction(),
            **self.metadata,
        }


def power_iteration(model: Model, batch, k: int = 1, iters: int = 50, tol: float = 1e-6,
                    seed: Optional[int] = None) -> Spectrum:
    """Top-k Hessian eigenpairs (by |lambda|) via power iteration with deflation."""
    n = model.num_parameters
    rng = np.random.default_rng(seed)
    eigvals: List[float] = []
    eigvecs: List[np.ndarray] = []

    def deflated_hvp(v: np.ndarray) -> np.ndarray:
        hv = model.hvp(batch, v)
        for lam, u in zip(eigvals, eigvecs):
            hv -= lam * np.dot(u, v) * u
        return hv

    for _ in range(k):
        v = rng.standard_normal(n)
        for u in eigvecs:
            v -= np.dot(u, v) * u
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(iters):
            hv = deflated_hvp(v)
            norm = np.linalg.norm(hv)
            if norm < 1e-12:
                break
            v_new = hv / norm
            lam_new = float(np.dot(v_new, deflated_hvp(v_new)))
            if abs(lam_new - lam) < tol * max(1.0, abs(lam_new)):
                v, lam = v_new, lam_new
                break
            v, lam = v_new, lam_new
        eigvals.append(lam)
        eigvecs.append(v)

    order = np.argsort(eigvals)[::-1]
    vals = np.asarray(eigvals)[order]
    vecs = np.stack([eigvecs[i] for i in order], axis=1)
    return Spectrum(eigenvalues=vals, eigenvectors=vecs, method="power_iteration",
                    metadata={"k": k, "iters": iters})


def lanczos_spectrum(model: Model, batch, m: int = 20, seed: Optional[int] = None) -> Spectrum:
    """Lanczos tridiagonalization: approximates the extreme Hessian spectrum.

    Returns the Ritz values (eigenvalues of the m x m tridiagonal matrix),
    which converge to the largest/smallest true eigenvalues. Uses full
    reorthogonalization for numerical stability.
    """
    n = model.num_parameters
    m = min(m, n)
    rng = np.random.default_rng(seed)
    q = rng.standard_normal(n)
    q /= np.linalg.norm(q)
    Q = [q]
    alphas: List[float] = []
    betas: List[float] = []
    for j in range(m):
        w = model.hvp(batch, Q[j])
        a = float(np.dot(w, Q[j]))
        alphas.append(a)
        w = w - a * Q[j] - (betas[-1] * Q[j - 1] if j > 0 else 0.0)
        # full reorthogonalization
        for qi in Q:
            w -= np.dot(w, qi) * qi
        b = float(np.linalg.norm(w))
        if b < 1e-10 or j == m - 1:
            break
        betas.append(b)
        Q.append(w / b)
    k = len(alphas)
    T = np.diag(alphas)
    for i, b in enumerate(betas[: k - 1]):
        T[i, i + 1] = T[i + 1, i] = b
    ritz = np.linalg.eigvalsh(T)[::-1]
    return Spectrum(eigenvalues=ritz, method="lanczos", metadata={"m": k})


def hutchinson_trace(model: Model, batch, n_samples: int = 20, seed: Optional[int] = None) -> float:
    """Hutchinson estimator of tr(H) using Rademacher probes.

    tr(H)/n is the average curvature; a cheap global sharpness proxy.
    """
    n = model.num_parameters
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_samples):
        v = rng.choice([-1.0, 1.0], size=n)
        total += float(np.dot(v, model.hvp(batch, v)))
    return total / n_samples


def hessian_spectrum(model: Model, batch, k: int = 5, method: str = "lanczos",
                     trace: bool = True, seed: Optional[int] = None) -> Spectrum:
    """One-call curvature summary: top eigenvalues + trace estimate."""
    if method == "lanczos":
        spec = lanczos_spectrum(model, batch, m=max(k, 10), seed=seed)
        spec.eigenvalues = spec.eigenvalues[: max(k, len(spec.eigenvalues))]
    elif method == "power":
        spec = power_iteration(model, batch, k=k, seed=seed)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'lanczos' or 'power'.")
    if trace:
        spec.trace_estimate = hutchinson_trace(model, batch, seed=seed)
    return spec


def curvature_along(model: Model, batch, direction: np.ndarray) -> float:
    """Second directional derivative d^T H d / ||d||^2 along a direction."""
    d = np.asarray(direction, dtype=np.float64)
    norm2 = float(np.dot(d, d))
    if norm2 < 1e-24:
        return 0.0
    return float(np.dot(d, model.hvp(batch, d)) / norm2)


def layerwise_curvature(model: Model, batch, n_samples: int = 5, seed: Optional[int] = None) -> Dict[str, float]:
    """Hutchinson trace estimate restricted to each layer's parameter block.

    Reveals which layers carry the most curvature (often where training is
    unstable or where fine-tuning concentrates).
    """
    rng = np.random.default_rng(seed)
    n = model.num_parameters
    out: Dict[str, float] = {}
    offset = 0
    for name, shape in model.layer_shapes().items():
        size = int(np.prod(shape)) if shape else 1
        est = 0.0
        for _ in range(n_samples):
            v = np.zeros(n)
            v[offset : offset + size] = rng.choice([-1.0, 1.0], size=size)
            hv = model.hvp(batch, v)
            est += float(np.dot(v[offset : offset + size], hv[offset : offset + size]))
        out[name] = est / n_samples
        offset += size
    return out
