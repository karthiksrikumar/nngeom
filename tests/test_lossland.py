import numpy as np

import nngeom as ng
from tests.conftest import MU, quad_grad, quad_loss


def _model_at(theta, name="m"):
    params = {"w": theta[:6].reshape(2, 3).copy(), "b": theta[6:].copy()}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name=name)


def test_interpolation_endpoint_losses(quad_model, quad_model_at_minimum):
    curve = ng.linear_interpolation(quad_model, quad_model_at_minimum, None, steps=11)
    assert np.isclose(curve.losses[0], quad_model.loss(None))
    assert np.isclose(curve.losses[-1], 0.0, atol=1e-10)
    # a quadratic has no barrier between any two points
    assert curve.barrier_height() <= 1e-10


def test_interpolation_restores_parameters(quad_model, quad_model_at_minimum):
    before = quad_model.parameters_flat().copy()
    ng.linear_interpolation(quad_model, quad_model_at_minimum, None, steps=5)
    assert np.allclose(quad_model.parameters_flat(), before)


def test_random_slice_minimum_at_center(quad_model_at_minimum):
    curve = ng.random_slice_1d(quad_model_at_minimum, None, span=1.0, steps=21, seed=0)
    assert np.argmin(curve.losses) == 10  # center of 21 points


def test_plane_slice_shape_and_center(quad_model_at_minimum):
    surf = ng.plane_slice_2d(quad_model_at_minimum, None, span=0.5, steps=7, seed=0)
    assert surf.losses.shape == (7, 7)
    assert np.isclose(surf.center_loss(), 0.0, atol=1e-10)
    assert np.isclose(surf.losses.min(), surf.center_loss(), atol=1e-10)


def test_sharpness_positive_at_minimum(quad_model_at_minimum):
    s = ng.sharpness(quad_model_at_minimum, None, radius=0.1, n_samples=10, seed=0, relative=False)
    assert s["base_loss"] < 1e-10
    assert s["mean_rise"] > 0


def test_basin_width_wider_for_flatter_model():
    flat_scale, sharp_scale = 0.1, 10.0

    def make(scale):
        def loss(params, batch):
            t = np.concatenate([params["w"].ravel(), params["b"].ravel()])
            return scale * float(t @ t)

        return ng.Model.from_function({"w": np.ones((2, 3)) * 0.5, "b": np.ones(2) * 0.5}, loss)

    wide = ng.basin_width(make(flat_scale), None, threshold=0.5, n_directions=2, seed=1)
    narrow = ng.basin_width(make(sharp_scale), None, threshold=0.5, n_directions=2, seed=1)
    assert wide["mean_width"] > narrow["mean_width"]
