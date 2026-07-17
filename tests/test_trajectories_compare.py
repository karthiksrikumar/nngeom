import numpy as np

import nngeom as ng
from tests.conftest import quad_grad, quad_loss


def _model_at(theta, name="m"):
    params = {"w": np.asarray(theta[:6], dtype=float).reshape(2, 3).copy(),
              "b": np.asarray(theta[6:], dtype=float).copy()}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name=name)


def test_straight_trajectory_geometry():
    pts = [np.linspace(0, 1, 8) * t for t in range(5)]
    traj = ng.Trajectory.from_vectors(pts)
    assert np.isclose(traj.tortuosity(), 1.0)
    assert np.isclose(traj.directional_persistence(), 1.0)
    assert np.isclose(traj.path_length(), traj.net_displacement())


def test_oscillating_trajectory_negative_persistence():
    base = np.zeros(8)
    step = np.ones(8)
    pts = [base, base + step, base, base + step, base]
    traj = ng.Trajectory.from_vectors(pts)
    assert traj.directional_persistence() < -0.9
    assert traj.tortuosity() > 100 or traj.tortuosity() == float("inf")


def test_trajectory_from_models_layerwise_drift():
    models = [_model_at(np.full(8, float(t)), name=f"ckpt{t}") for t in range(4)]
    traj = ng.Trajectory.from_models(models)
    drift = traj.layerwise_drift()
    assert set(drift) == {"w", "b"}
    assert drift["w"][0] == 0.0
    assert np.all(np.diff(drift["w"]) > 0)  # monotone drift away from start


def test_trajectory_pca_projection():
    rng = np.random.default_rng(0)
    pts = [rng.standard_normal(8) for _ in range(6)]
    traj = ng.Trajectory.from_vectors(pts)
    proj, ratio = traj.pca_projection(k=2)
    assert proj.shape == (6, 2)
    assert ratio[0] >= ratio[1]


def test_compare_models():
    a = _model_at(np.zeros(8), "a")
    b = _model_at(np.concatenate([np.zeros(6), np.ones(2)]), "b")
    rep = ng.compare_models(a, b)
    assert np.isclose(rep.l2_distance, np.sqrt(2))
    assert rep.layer_drift["w"]["l2"] == 0.0
    assert rep.layer_drift["b"]["l2"] > 0
    assert rep.top_changed_layers(1) == ["b"]
    assert "b" in rep.summary()


def test_task_vector_roundtrip():
    base = _model_at(np.zeros(8), "base")
    tuned = _model_at(np.ones(8), "tuned")
    v = ng.task_vector(base, tuned)
    assert np.allclose(v, 1.0)
    ng.apply_task_vector(base, v, scale=1.0)
    assert np.allclose(base.parameters_flat(), tuned.parameters_flat())
    assert np.isclose(ng.task_vector_alignment(v, v), 1.0)
    assert np.isclose(ng.task_vector_alignment(v, -v), -1.0)


def test_loss_along_trajectory():
    models = [_model_at(np.full(8, float(t))) for t in range(3)]
    losses = ng.loss_along_trajectory(models, None)
    assert losses.shape == (3,)
