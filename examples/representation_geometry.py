"""Representation geometry: similarity, probes, collapse, and outliers.

Builds synthetic 'layer activations' with known structure and shows how each
tool exposes it: CKA vs SVCCA vs Procrustes, linear probes, collapse and
anisotropy detection, outlier hunting, and neuron matching across seeds.

Run:  python examples/representation_geometry.py
"""

import numpy as np

import nngeom as ng

rng = np.random.default_rng(0)

# three classes, 40 examples each, in 32 dims
centers = rng.standard_normal((3, 32)) * 6
X = np.concatenate([c + rng.standard_normal((40, 32)) for c in centers])
labels = np.repeat([0, 1, 2], 40)

# a 'later layer': rotated + slightly compressed version of the same geometry
Q, _ = np.linalg.qr(rng.standard_normal((32, 32)))
Y = (X @ Q) * 0.7 + 0.5 * rng.standard_normal(X.shape)

# 1. similarity: three lenses on the same pair of layers
print("linear CKA:         ", f"{ng.linear_cka(X, Y):.3f}")
print("SVCCA:              ", f"{ng.svcca(X, Y):.3f}")
print("Procrustes distance:", f"{ng.procrustes_distance(X, Y):.3f}")
print("subspace overlap:   ", f"{ng.subspace_overlap(X, Y, k=5):.3f}")

# 2. is the label linearly decodable from each layer?
print("\nprobe on X:", ng.linear_probe_score(X, labels, seed=0))
print("probe on Y:", ng.linear_probe_score(Y, labels, seed=0))

# 3. class and manifold structure
cg = ng.class_geometry(X, labels)
print("\nseparation ratio:", f"{cg['separation_ratio']:.2f}")
print("class manifold dims:", ng.class_manifold_dimensions(X, labels))
print("manifold summary:", {k: round(v, 2) for k, v in ng.manifold_summary(X).items()})

# 4. collapse and anisotropy health checks
collapsed = np.outer(rng.standard_normal(120), rng.standard_normal(32))
print("\ncollapse score healthy:", f"{ng.collapse_score(X):.3f}",
      " collapsed:", f"{ng.collapse_score(collapsed):.3f}")
print("uniformity healthy:", f"{ng.uniformity(X, seed=0):.3f}",
      " collapsed:", f"{ng.uniformity(collapsed, seed=0):.3f}")

# 5. find geometric outliers (planted mislabeled-like point)
X_bad = X.copy()
X_bad[7] += 40
scores = ng.outlier_scores(X_bad, k=8)
print("\nplanted outlier index:", int(np.argmax(scores)), " z-score:", f"{scores[7]:.1f}")

# 6. did two 'seeds' learn the same neurons?
perm = rng.permutation(32)
seed2 = X[:, perm] + 0.1 * rng.standard_normal(X.shape)
match = ng.neuron_alignment(X, seed2)
print("\nneuron match quality:", f"{match['mean_matched_correlation']:.3f}",
      " well-matched fraction:", f"{match['fraction_well_matched']:.2f}")

# 7. drift report: same inputs before/after an intervention
drift = ng.representation_drift({"layer1": X}, {"layer1": Y})
print("\ndrift layer1:", {k: round(v, 3) for k, v in drift["layer1"].items()})
