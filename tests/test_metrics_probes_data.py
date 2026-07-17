import numpy as np
import pytest

import nngeom as ng


def test_metric_registry_compute():
    X = np.random.default_rng(0).standard_normal((50, 10))
    out = ng.compute_metrics(X, ["effective_rank", "mean_norm", "collapse_score"])
    assert set(out) == {"effective_rank", "mean_norm", "collapse_score"}


def test_register_custom_metric():
    @ng.register_metric("test_dim_count")
    def dim_count(X):
        return float(X.shape[1])

    X = np.zeros((5, 7))
    assert ng.compute_metrics(X, ["test_dim_count"])["test_dim_count"] == 7.0
    with pytest.raises(ValueError):
        ng.register_metric("test_dim_count")(dim_count)


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        ng.compute_metrics(np.zeros((3, 3)), ["nope"])


def test_random_direction_and_filter_normalize(quad_model):
    quad_model.set_parameters_flat(np.concatenate([np.full(6, 2.0), np.full(2, 0.5)]))
    d = ng.random_direction(quad_model, seed=0)
    assert np.isclose(np.linalg.norm(d), 1.0)
    fn = ng.filter_normalize(d, quad_model)
    # per-layer slices now match the weights' norms
    assert np.isclose(np.linalg.norm(fn[:6]), np.linalg.norm(np.full(6, 2.0)))
    assert np.isclose(np.linalg.norm(fn[6:]), np.linalg.norm(np.full(2, 0.5)))


def test_orthogonalize():
    rng = np.random.default_rng(1)
    basis = ng.orthogonalize([rng.standard_normal(10) for _ in range(4)])
    G = np.array([[np.dot(a, b) for b in basis] for a in basis])
    assert np.allclose(G, np.eye(4), atol=1e-10)


def test_concept_direction(clustered_reps):
    X, labels = clustered_reps
    d = ng.concept_direction(X[labels == 0], X[labels == 1])
    assert np.isclose(np.linalg.norm(d), 1.0)
    # projections should separate the two classes
    p0 = X[labels == 0] @ d
    p1 = X[labels == 1] @ d
    assert p0.mean() > p1.mean()


def test_stratified_indices():
    labels = np.repeat([0, 1, 2], 50)
    idx = ng.data.stratified_indices(labels, per_class=10, seed=0)
    assert len(idx) == 30
    _, counts = np.unique(labels[idx], return_counts=True)
    assert np.all(counts == 10)


def test_representative_subset_spread():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((100, 5))
    idx = ng.data.representative_subset(X, n=10, seed=0)
    assert len(np.unique(idx)) == 10


def test_synthetic_probes():
    assert ng.data.synthetic_probes(10, 4, "gaussian", seed=0).shape == (10, 4)
    sphere = ng.data.synthetic_probes(10, 4, "sphere", seed=0)
    assert np.allclose(np.linalg.norm(sphere, axis=1), 1.0)
    line = ng.data.synthetic_probes(10, 4, "interpolation", seed=0)
    diffs = np.diff(line, axis=0)
    assert np.allclose(diffs, diffs[0])  # collinear
