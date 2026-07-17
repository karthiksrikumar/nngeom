import numpy as np

import nngeom as ng


def test_cka_identity_and_invariance():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 10))
    assert np.isclose(ng.linear_cka(X, X), 1.0)
    # invariant to rotation and scale
    Q, _ = np.linalg.qr(rng.standard_normal((10, 10)))
    assert np.isclose(ng.linear_cka(X, 3.0 * X @ Q), 1.0, atol=1e-8)
    # unrelated representations score low
    Y = rng.standard_normal((50, 10))
    assert ng.linear_cka(X, Y) < 0.5


def test_rbf_cka_self_similarity():
    X = np.random.default_rng(1).standard_normal((40, 5))
    assert np.isclose(ng.rbf_cka(X, X), 1.0, atol=1e-8)


def test_cka_matrix_symmetric():
    rng = np.random.default_rng(2)
    reps = {f"layer{i}": rng.standard_normal((30, 8)) for i in range(3)}
    out = ng.cka_matrix(reps)
    M = out["matrix"]
    assert np.allclose(M, M.T)
    assert np.allclose(np.diag(M), 1.0)


def test_subspace_overlap():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 12))
    assert np.isclose(ng.subspace_overlap(X, X, k=5), 1.0, atol=1e-8)


def test_class_geometry_separated_clusters(clustered_reps):
    X, labels = clustered_reps
    cg = ng.class_geometry(X, labels)
    assert cg["separation_ratio"] > 1.0
    assert cg["min_centroid_distance"] > 5.0
    assert len(cg["classes"]) == 3


def test_collapse_score_detects_collapse():
    rng = np.random.default_rng(4)
    healthy = rng.standard_normal((100, 10))
    direction = rng.standard_normal(10)
    collapsed = np.outer(rng.standard_normal(100), direction)  # rank 1
    assert ng.collapse_score(collapsed) > 0.85
    assert ng.collapse_score(healthy) < 0.3


def test_anisotropy_extremes():
    rng = np.random.default_rng(5)
    isotropic = rng.standard_normal((200, 20))
    cone = np.abs(rng.standard_normal((200, 1))) @ np.ones((1, 20)) + 0.01 * rng.standard_normal((200, 20))
    assert abs(ng.anisotropy(isotropic, seed=0)) < 0.15
    assert ng.anisotropy(cone, seed=0) > 0.9


def test_representation_summary_keys(clustered_reps):
    X, labels = clustered_reps
    s = ng.representation_summary(X, labels=labels)
    for key in ("effective_rank", "anisotropy", "collapse_score", "separation_ratio"):
        assert key in s
