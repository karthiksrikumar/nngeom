import numpy as np
import pytest

import nngeom as ng

# A quadratic model with a KNOWN Hessian lets us verify curvature and
# landscape math against ground truth.
#
# loss(theta) = 0.5 * (theta - mu)^T H (theta - mu)
# grad        = H (theta - mu)
# Hessian     = H  (constant)

DIM = 8
EIGENVALUES = np.array([10.0, 5.0, 3.0, 2.0, 1.0, 0.5, 0.2, 0.1])


def _make_hessian(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
    return Q @ np.diag(EIGENVALUES) @ Q.T


HESS = _make_hessian()
MU = np.arange(DIM, dtype=np.float64) * 0.1


def quad_loss(params, batch):
    theta = np.concatenate([params["w"].ravel(), params["b"].ravel()])
    d = theta - MU
    return 0.5 * d @ HESS @ d


def quad_grad(params, batch):
    theta = np.concatenate([params["w"].ravel(), params["b"].ravel()])
    g = HESS @ (theta - MU)
    return {"w": g[:6].reshape(2, 3), "b": g[6:]}


@pytest.fixture
def quad_model():
    params = {"w": np.zeros((2, 3)), "b": np.zeros(2)}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name="quad")


@pytest.fixture
def quad_model_at_minimum():
    params = {"w": MU[:6].reshape(2, 3).copy(), "b": MU[6:].copy()}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name="quad_min")


@pytest.fixture
def clustered_reps():
    """Three well-separated Gaussian clusters in 20 dims."""
    rng = np.random.default_rng(42)
    centers = rng.standard_normal((3, 20)) * 10
    X = np.concatenate([centers[i] + rng.standard_normal((30, 20)) for i in range(3)])
    labels = np.repeat([0, 1, 2], 30)
    return X, labels
