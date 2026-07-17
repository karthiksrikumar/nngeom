"""Token geometry for language models: all 10 NLP features on one script.

Uses synthetic hidden states shaped exactly like what you get from a real
transformer (e.g. HuggingFace `output_hidden_states=True` gives a tuple of
(n_layers+1) tensors of shape (batch, n_tokens, dim); stack one sequence into
an array (n_layers, n_tokens, dim) and everything below applies unchanged).

Run:  python examples/nlp_geometry.py
"""

import numpy as np

import nngeom as ng

rng = np.random.default_rng(0)
L, T, D = 8, 14, 24  # layers, tokens, dim

# synthetic residual stream: smooth morph from input to output space
start = rng.standard_normal((T, D))
end = rng.standard_normal((T, D))
hidden = np.stack([(1 - t) * start + t * end + 0.05 * rng.standard_normal((T, D))
                   for t in np.linspace(0, 1, L)])

# 1. token trajectory: how far does token 5 travel through the layers?
traj = ng.token_trajectory(hidden, token_index=5)
print("token 5 path length:", f"{traj['path_length']:.2f}",
      " busiest layer:", traj["busiest_layer"],
      " tortuosity:", f"{traj['tortuosity']:.2f}")

# 2. contextualization: how much does context reshape a token?
static = rng.standard_normal(D)
contexts = static + 0.8 * rng.standard_normal((25, D))
print("contextualization:", ng.contextualization_score(static, contexts))

# 3. prompt divergence: where do two prompts split inside the model?
hidden_b = hidden.copy()
hidden_b[4:] += 1.5 * rng.standard_normal(hidden_b[4:].shape)
div = ng.prompt_divergence(hidden, hidden_b)
print("prompts diverge at layer:", div["divergence_layer"])

# 4. sentence curvature: shape of the token path in the last layer
print("sentence curvature:", {k: round(v, 3) for k, v in ng.sentence_curvature(hidden[-1]).items()})

# 5. polysemy: does this token show multiple sense clusters?
sense_a = rng.standard_normal((20, D)) * 0.4
sense_b = np.full(D, 6.0) + rng.standard_normal((20, D)) * 0.4
senses = ng.polysemy_score(np.concatenate([sense_a, sense_b]), seed=0)
print("estimated senses:", senses["n_senses"])

# 6. layer transition: where does input-processing hand off to output-building?
profile = ng.layer_transition_profile(hidden)
print("transition layer:", profile["transition_layer"], "of", profile["n_layers"])

# 7. isotropy correction: remove the dominant cone directions
cone = np.abs(rng.standard_normal((100, 1))) @ rng.standard_normal((1, D)) * 4 \
    + rng.standard_normal((100, D))
iso = ng.isotropy_correction(cone, n_components=2)
print("anisotropy before:", f"{iso['anisotropy_before']:.3f}",
      " after:", f"{iso['anisotropy_after']:.3f}")

# 8. semantic axis: build a formality-style probe and score sentences
formal = rng.standard_normal((15, D)) + 3
casual = rng.standard_normal((15, D)) - 3
axis = ng.semantic_axis(formal, casual)
scores = ng.semantic_projection(np.concatenate([formal[:3], casual[:3]]), axis)
print("axis scores (3 formal, 3 casual):", np.round(scores, 2))

# 9. repetition attractor: detect degeneration loops
loop = np.tile(rng.standard_normal((3, D)), (8, 1)) + 0.02 * rng.standard_normal((24, D))
print("recurrence healthy:", f"{ng.repetition_attractor_score(hidden[-1])['recurrence_rate']:.3f}",
      " looping:", f"{ng.repetition_attractor_score(loop)['recurrence_rate']:.3f}")

# 10. token novelty: find the topic shift
basis = rng.standard_normal((3, D))
seq = np.vstack([rng.standard_normal((10, 3)) @ basis, rng.standard_normal(D) * 3])
novelty = ng.token_novelty(seq)
print("novelty spike at token:", int(np.argmax(novelty[1:]) + 1), "of", len(seq) - 1)

# one-call profile
print("\nnlp_summary:", {k: v for k, v in ng.nlp_summary(hidden).items()
                          if not isinstance(v, dict)})
