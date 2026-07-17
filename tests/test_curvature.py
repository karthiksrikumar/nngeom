import numpy as np

import nngeom as ng
from tests.conftest import EIGENVALUES


def test_power_iteration_top_eigenvalue(quad_model):
    spec = ng.power_iteration(quad_model, None, k=2, iters=200, seed=0)
    assert np.isclose(spec.top_eigenvalue, EIGENVALUES[0], rtol=1e-2)
    assert np.isclose(spec.eigenvalues[1], EIGENVALUES[1], rtol=5e-2)


def test_lanczos_recovers_full_spectrum(quad_model):
    spec = ng.lanczos_spectrum(quad_model, None, m=8, seed=0)
    assert np.allclose(np.sort(spec.eigenvalues), np.sort(EIGENVALUES), rtol=1e-2, atol=1e-2)


def test_hutchinson_trace(quad_model):
    tr = ng.hutchinson_trace(quad_model, None, n_samples=200, seed=0)
    assert np.isclose(tr, EIGENVALUES.sum(), rtol=0.25)


def test_curvature_along_direction(quad_model):
    spec = ng.power_iteration(quad_model, None, k=1, iters=200, seed=0)
    top_vec = spec.eigenvectors[:, 0]
    c = ng.curvature_along(quad_model, None, top_vec)
    assert np.isclose(c, EIGENVALUES[0], rtol=1e-2)


def test_spectrum_diagnostics(quad_model):
    spec = ng.hessian_spectrum(quad_model, None, k=5, seed=0)
    assert spec.negative_fraction() == 0.0  # PSD quadratic
    assert spec.trace_estimate is not None
    d = spec.to_dict()
    assert d["top_eigenvalue"] > 0


def test_layerwise_curvature_sums_to_trace(quad_model):
    lw = ng.layerwise_curvature(quad_model, None, n_samples=100, seed=0)
    assert set(lw) == {"w", "b"}
    assert np.isclose(sum(lw.values()), EIGENVALUES.sum(), rtol=0.35)
