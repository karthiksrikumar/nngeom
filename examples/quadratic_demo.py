"""nngeom end-to-end demo on a model with known geometry (no torch needed).

We build a quadratic 'model' whose Hessian we control exactly, then run the
full nngeom battery: loss slices, spectrum estimation, sharpness, trajectory
analysis, model comparison, representation metrics, and the automated report.

Run:  python examples/quadratic_demo.py
"""

import numpy as np

import nngeom as ng

ng.set_seed(0)

# --- define a model with a known Hessian ------------------------------------
DIM = 12
rng = np.random.default_rng(0)
Q, _ = np.linalg.qr(rng.standard_normal((DIM, DIM)))
EIGS = np.geomspace(20.0, 0.05, DIM)
H = Q @ np.diag(EIGS) @ Q.T
MU = rng.standard_normal(DIM)  # location of the true minimum


def loss_fn(params, batch):
    theta = np.concatenate([p.ravel() for p in params.values()])
    d = theta - MU
    return 0.5 * d @ H @ d


def grad_fn(params, batch):
    theta = np.concatenate([p.ravel() for p in params.values()])
    g = H @ (theta - MU)
    return {"encoder": g[:8].reshape(2, 4), "head": g[8:]}


def model_at(theta, name):
    params = {"encoder": theta[:8].reshape(2, 4).copy(), "head": theta[8:].copy()}
    return ng.Model.from_function(params, loss_fn, grad_fn=grad_fn, name=name)


model = model_at(MU.copy(), "trained")          # at the minimum
init = model_at(np.zeros(DIM), "init")          # far away

# --- curvature: recover the spectrum we planted ------------------------------
spec = ng.hessian_spectrum(model, None, k=5)
print("top eigenvalue (true 20.0):", f"{spec.top_eigenvalue:.3f}")
print("trace estimate  (true {:.2f}):".format(EIGS.sum()), f"{spec.trace_estimate:.2f}")

# --- loss landscape -----------------------------------------------------------
curve = ng.linear_interpolation(init, model, None, steps=25, extend=0.25)
print("barrier between init and trained:", f"{curve.barrier_height():.4f}")
print("sharpness:", ng.sharpness(model, None, seed=0))

# --- a fake training trajectory ------------------------------------------------
checkpoints = [model_at(t * MU + 0.3 * (1 - t) * rng.standard_normal(DIM), f"step{i}")
               for i, t in enumerate(np.linspace(0, 1, 8))]
traj = ng.Trajectory.from_models(checkpoints)
print("trajectory:", traj.summary())

# --- compare first and last checkpoint ----------------------------------------
print(ng.compare_models(checkpoints[0], checkpoints[-1]).summary())

# --- representation geometry on synthetic activations ---------------------------
centers = rng.standard_normal((3, 16)) * 8
X = np.concatenate([c + rng.standard_normal((40, 16)) for c in centers])
labels = np.repeat([0, 1, 2], 40)
print("representation:", ng.representation_summary(X, labels=labels))

# --- the automated report -------------------------------------------------------
report = ng.analyze(model, batch=0, checkpoints=checkpoints,
                    representations={"layer1": X}, labels=labels,
                    depth="standard", make_figures=False)
print()
print(report.summary())
report.to_json("geometry_report.json")
print("\nwrote geometry_report.json")
