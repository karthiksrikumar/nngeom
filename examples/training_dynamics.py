"""Training dynamics: run real gradient descent and watch its geometry.

Trains a small quadratic model with plain SGD at two learning rates and
compares the geometry of the two runs: path shape, phase changes, gradient
noise, and SAM sharpness of the endpoints.

Run:  python examples/training_dynamics.py
"""

import numpy as np

import nngeom as ng

ng.set_seed(0)
rng = np.random.default_rng(0)

DIM = 10
H = np.diag(np.geomspace(10, 0.2, DIM))
MU = rng.standard_normal(DIM)


def loss_fn(params, batch):
    t = np.concatenate([p.ravel() for p in params.values()])
    d = t - (MU + (batch if batch is not None else 0))
    return 0.5 * d @ H @ d


def grad_fn(params, batch):
    t = np.concatenate([p.ravel() for p in params.values()])
    g = H @ (t - (MU + (batch if batch is not None else 0)))
    return {"w": g[:8].reshape(2, 4), "b": g[8:]}


def fresh_model(name):
    return ng.Model.from_function({"w": np.zeros((2, 4)), "b": np.zeros(2)},
                                  loss_fn, grad_fn=grad_fn, name=name)


def train(lr, steps=25, noise=0.2):
    """Plain SGD with noisy batches; returns the checkpoint list."""
    model = fresh_model(f"lr={lr}")
    checkpoints = [model.parameters_flat().copy()]
    for _ in range(steps):
        batch = noise * rng.standard_normal(DIM)
        theta = model.parameters_flat() - lr * model.gradient(batch)
        model.set_parameters_flat(theta)
        checkpoints.append(theta.copy())
    return model, checkpoints


for lr in (0.05, 0.19):  # 0.19 is just under 2/lambda_max = 0.2: barely stable
    model, ckpts = train(lr)
    traj = ng.Trajectory.from_vectors(ckpts)
    print(f"\n--- learning rate {lr} ---")
    print("final loss:            ", f"{model.loss(None):.5f}")
    s = traj.summary()
    print("tortuosity:            ", f"{s['tortuosity']:.2f}")
    print("directional persistence:", f"{s['directional_persistence']:.3f}")
    print("phase changes at steps: ", traj.phase_changes()[:8])

    batches = [0.2 * rng.standard_normal(DIM) for _ in range(6)]
    print("gradient noise scale:   ", f"{ng.gradient_noise_scale(model, batches):.3f}")
    print("gradient alignment:     ", f"{ng.gradient_alignment(model, batches)['mean_cosine']:.3f}")
    print("SAM sharpness rise:     ", f"{ng.sam_sharpness(model, None, rho=0.02)['adversarial_rise']:.5f}")
    print("local linearity error:  ", f"{ng.local_linearity(model, None, seed=0)['linearity_error']:.3f}")
