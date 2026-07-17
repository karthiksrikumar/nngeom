import numpy as np
import pytest

import nngeom as ng
from tests.conftest import EIGENVALUES, HESS, MU, quad_grad, quad_loss


def test_parameters_roundtrip(quad_model):
    flat = quad_model.parameters_flat()
    assert flat.shape == (8,)
    new = np.arange(8, dtype=np.float64)
    quad_model.set_parameters_flat(new)
    assert np.allclose(quad_model.parameters_flat(), new)


def test_layer_names_and_norms(quad_model):
    assert quad_model.layer_names() == ["w", "b"]
    quad_model.set_parameters_flat(np.ones(8))
    norms = quad_model.layer_norms()
    assert np.isclose(norms["w"], np.sqrt(6))
    assert np.isclose(norms["b"], np.sqrt(2))


def test_loss_and_gradient(quad_model):
    theta = quad_model.parameters_flat()
    expected = 0.5 * (theta - MU) @ HESS @ (theta - MU)
    assert np.isclose(quad_model.loss(None), expected)
    assert np.allclose(quad_model.gradient(None), HESS @ (theta - MU))


def test_finite_difference_gradient_matches_analytic():
    params = {"w": np.ones((2, 3)), "b": np.zeros(2)}
    fd_model = ng.Model.from_function(params, quad_loss)  # no grad_fn -> finite diff
    an_model = ng.Model.from_function(params, quad_loss, grad_fn=quad_grad)
    assert np.allclose(fd_model.gradient(None), an_model.gradient(None), atol=1e-4)


def test_hvp_matches_true_hessian(quad_model):
    v = np.random.default_rng(0).standard_normal(8)
    hv = quad_model.hvp(None, v)
    assert np.allclose(hv, HESS @ v, atol=1e-3)


def test_gradient_zero_at_minimum(quad_model_at_minimum):
    assert np.allclose(quad_model_at_minimum.gradient(None), 0, atol=1e-8)
    assert np.isclose(quad_model_at_minimum.loss(None), 0, atol=1e-12)


def test_model_rejects_bad_input():
    with pytest.raises(TypeError):
        ng.Model(42)
    with pytest.raises(ValueError):
        ng.Model({"w": np.ones(3)})  # missing loss_fn
