import json
import os

import numpy as np
import pytest

import nngeom as ng
from tests.conftest import quad_grad, quad_loss


def _model_at(theta, name="m"):
    params = {"w": np.asarray(theta[:6], dtype=float).reshape(2, 3).copy(),
              "b": np.asarray(theta[6:], dtype=float).copy()}
    return ng.Model.from_function(params, quad_loss, grad_fn=quad_grad, name=name)


def test_analyze_end_to_end(quad_model_at_minimum, clustered_reps, tmp_path):
    X, labels = clustered_reps
    checkpoints = [_model_at(np.full(8, 1.0 - t / 3), name=f"ckpt{t}") for t in range(4)]
    report = ng.analyze(
        quad_model_at_minimum,
        batch=0,  # the quadratic loss ignores the batch, but analyze needs a non-None batch
        checkpoints=checkpoints,
        representations={"layer1": X},
        labels=labels,
        depth="quick",
        make_figures=False,
    )
    assert "model" in report.sections
    assert "loss_landscape" in report.sections
    assert "curvature" in report.sections
    assert "representations" in report.sections
    assert "trajectory" in report.sections
    assert report.sections["curvature"]["top_eigenvalue"] > 0

    text = report.summary()
    assert "Geometry report" in text

    json_path = report.to_json(str(tmp_path / "report.json"))
    with open(json_path) as f:
        data = json.load(f)
    assert data["sections"]["model"]["n_parameters"] == 8

    html_path = report.to_html(str(tmp_path / "report.html"))
    html = open(html_path, encoding="utf-8").read()
    assert "Curvature" in html


def test_analyze_flags_collapse(quad_model_at_minimum):
    rng = np.random.default_rng(0)
    collapsed = np.outer(rng.standard_normal(50), rng.standard_normal(16))
    report = ng.analyze(quad_model_at_minimum, representations={"bad_layer": collapsed},
                        depth="quick", make_figures=False)
    assert any("collapsed" in w for w in report.warnings)


def test_analyze_without_batch(quad_model):
    report = ng.analyze(quad_model, depth="quick", make_figures=False)
    assert "model" in report.sections
    assert "loss_landscape" not in report.sections


def test_analyze_bad_depth(quad_model):
    with pytest.raises(ValueError):
        ng.analyze(quad_model, depth="ultra")


def test_export_csv(tmp_path):
    table = {"layer1": {"l2": 1.0, "cosine": 0.9}, "layer2": {"l2": 2.0, "cosine": 0.5}}
    path = ng.export.to_csv(table, str(tmp_path / "drift.csv"))
    content = open(path).read()
    assert "layer1" in content and "cosine" in content


def test_to_json_handles_numpy(tmp_path):
    obj = {"arr": np.arange(3), "val": np.float64(1.5), "nested": {"x": np.int32(2)}}
    path = ng.export.to_json(obj, str(tmp_path / "x.json"))
    data = json.load(open(path))
    assert data["arr"] == [0, 1, 2]
    assert data["nested"]["x"] == 2
