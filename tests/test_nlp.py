import numpy as np
import pytest

import nngeom as ng


def _fake_hidden_states(n_layers=6, n_tokens=10, dim=16, seed=0):
    """Hidden states that smoothly morph from an 'input' to an 'output' basis."""
    rng = np.random.default_rng(seed)
    start = rng.standard_normal((n_tokens, dim))
    end = rng.standard_normal((n_tokens, dim))
    layers = [(1 - t) * start + t * end + 0.05 * rng.standard_normal((n_tokens, dim))
              for t in np.linspace(0, 1, n_layers)]
    return np.stack(layers)


def test_token_trajectory():
    hs = _fake_hidden_states()
    out = ng.token_trajectory(hs, token_index=3)
    assert out["path"].shape == (6, 16)
    assert out["path_length"] >= out["net_displacement"]
    assert 0 <= out["busiest_layer"] < 5
    with pytest.raises(ValueError):
        ng.token_trajectory(np.zeros((3, 4)), 0)  # wrong ndim


def test_contextualization_score():
    rng = np.random.default_rng(1)
    static = rng.standard_normal(16)
    tight = static + 0.01 * rng.standard_normal((20, 16))
    wild = rng.standard_normal((20, 16)) * 5
    t = ng.contextualization_score(static, tight)
    w = ng.contextualization_score(static, wild)
    assert t["mean_context_shift"] < w["mean_context_shift"]
    assert t["self_similarity"] > w["self_similarity"]


def test_prompt_divergence():
    rng = np.random.default_rng(2)
    hs_a = _fake_hidden_states(seed=3)
    hs_b = hs_a.copy()
    hs_b[3:] += 2.0 * rng.standard_normal(hs_b[3:].shape)  # diverge from layer 3
    out = ng.prompt_divergence(hs_a, hs_b)
    assert out["divergence_layer"] >= 3
    assert out["per_layer_distance"][2] < out["per_layer_distance"][4]
    identical = ng.prompt_divergence(hs_a, hs_a)
    assert np.allclose(identical["per_layer_distance"], 0)


def test_sentence_curvature_straight_vs_twisty():
    t = np.linspace(0, 1, 12)[:, None]
    straight = t @ np.ones((1, 8))
    rng = np.random.default_rng(4)
    twisty = rng.standard_normal((12, 8))
    s = ng.sentence_curvature(straight)
    w = ng.sentence_curvature(twisty)
    assert s["mean_turning_angle"] < 0.1
    assert w["mean_turning_angle"] > 0.5
    assert np.isclose(s["tortuosity"], 1.0)


def test_polysemy_score_two_senses():
    rng = np.random.default_rng(5)
    sense_a = np.zeros(12) + rng.standard_normal((25, 12)) * 0.3
    sense_b = np.full(12, 10.0) + rng.standard_normal((25, 12)) * 0.3
    two = ng.polysemy_score(np.concatenate([sense_a, sense_b]), seed=0)
    one = ng.polysemy_score(sense_a, seed=0)
    assert two["n_senses"] >= 2
    assert one["n_senses"] == 1


def test_layer_transition_profile():
    hs = _fake_hidden_states()
    out = ng.layer_transition_profile(hs)
    assert out["cka_to_input"][0] > 0.99
    assert out["cka_to_output"][-1] > 0.99
    assert 0 < out["transition_layer"] < 6


def test_isotropy_correction_reduces_anisotropy():
    rng = np.random.default_rng(6)
    common = rng.standard_normal(20)
    X = 5 * np.abs(rng.standard_normal((100, 1))) @ common[None, :] + rng.standard_normal((100, 20))
    out = ng.isotropy_correction(X, n_components=2)
    assert out["anisotropy_after"] < out["anisotropy_before"]
    assert out["corrected"].shape == X.shape


def test_semantic_axis_and_projection():
    rng = np.random.default_rng(7)
    axis_true = np.zeros(10)
    axis_true[0] = 1.0
    pos = rng.standard_normal((30, 10)) * 0.2 + 3 * axis_true
    neg = rng.standard_normal((30, 10)) * 0.2 - 3 * axis_true
    axis = ng.semantic_axis(pos, neg)
    assert abs(axis[0]) > 0.9
    scores = ng.semantic_projection(np.concatenate([pos, neg]), axis)
    assert scores[:30].mean() > scores[30:].mean()


def test_repetition_attractor_score():
    rng = np.random.default_rng(8)
    healthy = rng.standard_normal((40, 12))
    loop = np.tile(rng.standard_normal((4, 12)), (10, 1)) + 0.01 * rng.standard_normal((40, 12))
    assert (ng.repetition_attractor_score(loop)["recurrence_rate"]
            > ng.repetition_attractor_score(healthy)["recurrence_rate"])


def test_token_novelty_detects_topic_shift():
    rng = np.random.default_rng(9)
    basis_a = rng.standard_normal((3, 16))
    on_topic = rng.standard_normal((10, 3)) @ basis_a  # spanned by basis_a
    shift = rng.standard_normal(16) * 3  # new direction
    seq = np.vstack([on_topic, shift[None, :]])
    scores = ng.token_novelty(seq)
    assert scores[0] == 1.0
    assert scores[-1] > scores[4:-1].max()


def test_nlp_summary():
    hs = _fake_hidden_states()
    out = ng.nlp_summary(hs)
    assert out["n_layers"] == 6
    assert out["n_tokens"] == 10
    assert "transition_layer" in out and "repetition" in out
