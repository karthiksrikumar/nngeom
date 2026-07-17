import numpy as np

import nngeom as ng
from tests.conftest import HESS, MU


def _batched_quad_model():
    """Quadratic model whose loss depends on the batch: batch shifts the minimum."""

    def loss(params, batch):
        t = np.concatenate([params["w"].ravel(), params["b"].ravel()])
        d = t - (MU + batch)
        return 0.5 * d @ HESS @ d

    def grad(params, batch):
        t = np.concatenate([params["w"].ravel(), params["b"].ravel()])
        g = HESS @ (t - (MU + batch))
        return {"w": g[:6].reshape(2, 3), "b": g[6:]}

    return ng.Model.from_function({"w": np.zeros((2, 3)), "b": np.zeros(2)}, loss, grad_fn=grad)


def test_gradient_alignment_identical_batches():
    model = _batched_quad_model()
    batches = [np.zeros(8)] * 3
    out = ng.gradient_alignment(model, batches)
    assert np.isclose(out["mean_cosine"], 1.0)


def test_gradient_noise_scale_zero_for_identical_batches():
    model = _batched_quad_model()
    assert ng.gradient_noise_scale(model, [np.zeros(8)] * 4) == 0.0


def test_gradient_noise_scale_positive_for_noisy_batches():
    model = _batched_quad_model()
    rng = np.random.default_rng(0)
    batches = [rng.standard_normal(8) * 0.5 for _ in range(6)]
    assert ng.gradient_noise_scale(model, batches) > 0


def test_fisher_diagonal_and_layer_importance():
    model = _batched_quad_model()
    rng = np.random.default_rng(1)
    batches = [rng.standard_normal(8) for _ in range(4)]
    diag = ng.fisher_diagonal(model, batches)
    assert diag.shape == (8,)
    assert np.all(diag >= 0)
    imp = ng.fisher_layer_importance(model, batches)
    assert set(imp) == {"w", "b"}
    assert np.isclose(sum(imp.values()), diag.sum())


def test_sam_sharpness_positive_away_from_minimum(quad_model):
    out = ng.sam_sharpness(quad_model, None, rho=0.01)
    assert out["adversarial_rise"] > 0  # moving up the gradient raises loss
    # parameters restored
    assert np.allclose(quad_model.parameters_flat(), np.zeros(8))


def test_sam_sharpness_zero_gradient(quad_model_at_minimum):
    out = ng.sam_sharpness(quad_model_at_minimum, None, rho=0.05)
    assert out["adversarial_rise"] == 0.0


def test_local_linearity_low_for_gentle_quadratic(quad_model):
    out = ng.local_linearity(quad_model, None, radius=0.001, n_samples=8, seed=0)
    # at tiny radius a smooth quadratic is nearly linear
    assert out["linearity_error"] < 0.5
    assert np.allclose(quad_model.parameters_flat(), np.zeros(8))
