import numpy as np

import nngeom as ng


def test_effective_rank_isotropic_vs_rank1():
    rng = np.random.default_rng(0)
    iso = rng.standard_normal((200, 10))
    rank1 = np.outer(rng.standard_normal(200), rng.standard_normal(10))
    assert ng.effective_rank(iso) > 8
    assert ng.effective_rank(rank1) < 1.5


def test_participation_ratio_bounds():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 15))
    pr = ng.participation_ratio(X)
    assert 1.0 <= pr <= 15.0
    assert pr > 10  # isotropic should use most dimensions


def test_twonn_recovers_planted_dimension():
    rng = np.random.default_rng(2)
    # 3-dimensional manifold embedded in 20 dims via a random linear map
    latent = rng.standard_normal((500, 3))
    embed = latent @ rng.standard_normal((3, 20))
    d = ng.intrinsic_dimension_twonn(embed)
    assert 2.0 < d < 4.5


def test_knn_graph_shape():
    X = np.random.default_rng(3).standard_normal((30, 5))
    nbrs = ng.knn_graph(X, k=4)
    assert nbrs.shape == (30, 4)
    assert all(i not in nbrs[i] for i in range(30))


def test_neighborhood_preservation_identity():
    X = np.random.default_rng(4).standard_normal((50, 8))
    assert np.isclose(ng.neighborhood_preservation(X, X, k=5), 1.0)


def test_manifold_summary(clustered_reps):
    X, _ = clustered_reps
    s = ng.manifold_summary(X)
    assert s["n_points"] == 90
    assert s["ambient_dim"] == 20
    assert "intrinsic_dim_twonn" in s


def test_pca_explained_variance_sums_to_one():
    X = np.random.default_rng(5).standard_normal((100, 6))
    _, _, ratio = ng.pca(X)
    assert np.isclose(ratio.sum(), 1.0)
