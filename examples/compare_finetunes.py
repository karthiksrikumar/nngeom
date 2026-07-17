"""Comparing checkpoints: task vectors, model soups, and mode connectivity.

Simulates a base model and two fine-tunes, then answers:
- what did each fine-tune change, and where?
- do the two fine-tunes push the model the same way?
- are they in the same basin (can we average them into a soup)?

Run:  python examples/compare_finetunes.py
"""

import numpy as np

import nngeom as ng

ng.set_seed(0)
rng = np.random.default_rng(0)

DIM = 16
H = np.diag(np.geomspace(8, 0.1, DIM))
MU = rng.standard_normal(DIM)


def loss_fn(params, batch):
    t = np.concatenate([p.ravel() for p in params.values()])
    d = t - MU
    return 0.5 * d @ H @ d


def grad_fn(params, batch):
    t = np.concatenate([p.ravel() for p in params.values()])
    g = H @ (t - MU)
    return {"backbone": g[:12].reshape(3, 4), "head": g[12:]}


def model_at(theta, name):
    return ng.Model.from_function(
        {"backbone": theta[:12].reshape(3, 4).copy(), "head": theta[12:].copy()},
        loss_fn, grad_fn=grad_fn, name=name)


base = model_at(MU + 0.5 * rng.standard_normal(DIM), "base")
tune_a = model_at(base.parameters_flat() + 0.3 * rng.standard_normal(DIM), "tune_a")
tune_b = model_at(base.parameters_flat() + 0.3 * rng.standard_normal(DIM), "tune_b")

# 1. what changed, layer by layer
print(ng.compare_models(base, tune_a).summary())

# 2. do the fine-tunes agree?
va = ng.task_vector(base, tune_a)
vb = ng.task_vector(base, tune_b)
print("\ntask vector alignment:", f"{ng.task_vector_alignment(va, vb):.3f}")

# 3. same basin? check pairwise barriers, then soup them
barriers = ng.loss_barrier_matrix([base, tune_a, tune_b], None, steps=11)
print("barrier matrix:\n", np.round(barriers["matrix"], 4))

soup = ng.model_soup([tune_a, tune_b])
soup_model = model_at(soup, "soup")
print("\nlosses:  tune_a", f"{tune_a.loss(None):.4f}",
      " tune_b", f"{tune_b.loss(None):.4f}",
      " soup", f"{soup_model.loss(None):.4f}")

# 4. if the straight path had a barrier, search for a curved one
path = ng.find_low_loss_path(tune_a, tune_b, None, steps=11, iters=10, seed=0)
print("straight barrier:", f"{path['straight_barrier']:.5f}",
      " curved barrier:", f"{path['curved_barrier']:.5f}",
      " connected:", path["connected"])
