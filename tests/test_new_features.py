import numpy as np
import pytest

import nngeom as ng
from tests.conftest import quad_grad, quad_loss


def _model_at(theta, name="m"):
    params = {"w": np.asarray(theta[:6], dtype=float).reshape(2, 3).copy(),
              "b": np.asarray(theta[6:], dtype=float).copy()}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name=name)


# -- lossland extensions ------------------------------------------------------

def test_interpolation_smoothness_zero_for_line():
    curve = ng.LossCurve(alphas=np.linspace(0, 1, 10), losses=np.linspace(1, 2, 10))
    assert ng.interpolation_smoothness(curve) < 1e-12
    bumpy = ng.LossCurve(alphas=np.linspace(0, 1, 10),
                         losses=np.sin(np.linspace(0, 10, 10)))
    assert ng.interpolation_smoothness(bumpy) > 0.1


def test_loss_barrier_matrix_symmetric():
    models = [_model_at(np.full(8, float(t)), name=f"m{t}") for t in range(3)]
    out = ng.loss_barrier_matrix(models, None, steps=7)
    M = out["matrix"]
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 0)
    assert np.all(M <= 1e-8)  # quadratic: no barriers anywhere


def test_find_low_loss_path_on_quadratic():
    a = _model_at(np.zeros(8), "a")
    b = _model_at(np.ones(8), "b")
    out = ng.find_low_loss_path(a, b, None, steps=9, iters=5, seed=0)
    assert out["connected"]
    assert out["curved_barrier"] <= out["straight_barrier"] + 1e-9
    assert np.allclose(a.parameters_flat(), np.zeros(8))  # restored


# -- compare extensions ----------------------------------------------------------

def test_model_soup_average():
    models = [_model_at(np.full(8, v)) for v in (0.0, 2.0, 4.0)]
    soup = ng.model_soup(models)
    assert np.allclose(soup, 2.0)
    weighted = ng.model_soup(models, weights=[1, 0, 0])
    assert np.allclose(weighted, 0.0)
    with pytest.raises(ValueError):
        ng.model_soup([])


def test_eigenvector_overlap_same_model(quad_model):
    s1 = ng.power_iteration(quad_model, None, k=3, iters=200, seed=0)
    s2 = ng.power_iteration(quad_model, None, k=3, iters=200, seed=99)
    assert ng.eigenvector_overlap(s1, s2) > 0.95


# -- manifolds extensions ---------------------------------------------------------

def test_random_projection_preserves_distances():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 200))
    Y = ng.random_projection(X, k=60, seed=0)
    assert Y.shape == (50, 60)
    d_orig = np.linalg.norm(X[0] - X[1])
    d_proj = np.linalg.norm(Y[0] - Y[1])
    assert 0.5 * d_orig < d_proj < 1.5 * d_orig


def test_outlier_scores_flag_planted_outlier():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 8))
    X[0] += 50  # planted outlier
    scores = ng.outlier_scores(X, k=5)
    assert np.argmax(scores) == 0
    assert scores[0] > 3


def test_class_manifold_dimensions(clustered_reps):
    X, labels = clustered_reps
    dims = ng.class_manifold_dimensions(X, labels)
    assert set(dims) == {"0", "1", "2"}
    assert all(d > 0 for d in dims.values())


# -- model spectral properties -------------------------------------------------------

def test_spectral_norms_and_stable_ranks():
    theta = np.zeros(8)
    theta[:6] = np.array([3, 0, 0, 0, 0, 0])  # w = [[3,0,0],[0,0,0]] rank 1
    m = _model_at(theta)
    sn = m.spectral_norms()
    assert set(sn) == {"w"}  # b is 1-D, excluded
    assert np.isclose(sn["w"], 3.0)
    sr = m.stable_ranks()
    assert np.isclose(sr["w"], 1.0)  # rank-1 matrix has stable rank 1


# -- trajectory extensions --------------------------------------------------------------

def test_norm_growth_and_phase_changes():
    pts = [np.full(8, 0.0), np.full(8, 1.0), np.full(8, 2.0), np.full(8, 1.0)]
    traj = ng.Trajectory.from_vectors(pts)
    growth = traj.norm_growth()
    assert growth.shape == (4,)
    assert growth[2] > growth[1]
    phases = traj.phase_changes()
    assert phases == [2]  # course reverses at checkpoint 2


# -- representation extensions --------------------------------------------------------------

def test_svcca_identity_and_rotation():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((80, 10))
    assert ng.svcca(X, X) > 0.99
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    assert ng.svcca(X, X @ Q) > 0.99


def test_procrustes_distance_rotation_invariance():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((40, 6))
    Q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
    assert ng.procrustes_distance(X, X @ Q) < 1e-8
    Y = rng.standard_normal((40, 6))
    assert ng.procrustes_distance(X, Y) > 0.5


def test_uniformity_ordering():
    rng = np.random.default_rng(4)
    spread = rng.standard_normal((150, 16))
    bunched = np.ones((150, 16)) + 0.01 * rng.standard_normal((150, 16))
    assert ng.uniformity(spread, seed=0) < ng.uniformity(bunched, seed=0)


def test_linear_probe_score_separable(clustered_reps):
    X, labels = clustered_reps
    out = ng.linear_probe_score(X, labels, seed=0)
    assert out["train_accuracy"] > 0.95
    assert out["test_accuracy"] > 0.8


def test_neuron_alignment_permuted_copy():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((100, 8))
    perm = rng.permutation(8)
    out = ng.neuron_alignment(X, X[:, perm])
    assert out["mean_matched_correlation"] > 0.99
    # every matched pair should recover the permutation
    for i, j, _ in out["pairs"]:
        assert perm[j] == i


def test_representation_drift():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((50, 12))
    before = {"layer1": X}
    after = {"layer1": 2.0 * X}  # pure rescale: same geometry
    out = ng.representation_drift(before, after)
    assert np.isclose(out["layer1"]["cka"], 1.0)
    assert np.isclose(out["layer1"]["norm_ratio"], 2.0)
    assert out["layer1"]["procrustes_distance"] < 1e-8


def test_new_metrics_registered():
    for name in ("uniformity", "stable_rank"):
        assert name in ng.list_metrics()
    X = np.random.default_rng(7).standard_normal((40, 6))
    out = ng.compute_metrics(X, ["uniformity", "stable_rank"])
    assert out["stable_rank"] > 1
