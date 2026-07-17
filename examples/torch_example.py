"""nngeom with a real PyTorch model (requires: pip install nngeom[torch]).

Trains a tiny MLP on synthetic 2-class data, then runs the geometry battery:
autograd Hessian spectrum, loss slices, activation extraction with hooks,
CKA across layers, and the automated report.

Run:  python examples/torch_example.py
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise SystemExit("This example needs PyTorch: pip install nngeom[torch]")

import numpy as np

import nngeom as ng

torch.manual_seed(0)
ng.set_seed(0)

# data: two Gaussian blobs
n = 400
X = torch.randn(n, 10)
y = (X[:, 0] + X[:, 1] > 0).long()
batch = (X, y)

net = nn.Sequential(
    nn.Linear(10, 32), nn.ReLU(),
    nn.Linear(32, 32), nn.ReLU(),
    nn.Linear(32, 2),
)

def loss_fn(module, b):
    return F.cross_entropy(module(b[0]), b[1])

model = ng.Model(net, loss_fn=loss_fn, name="mlp")
print("initial loss:", f"{model.loss(batch):.4f}")

# train, keeping checkpoints for trajectory analysis
opt = torch.optim.SGD(net.parameters(), lr=0.1)
snapshots = [model.parameters_flat().copy()]
for step in range(200):
    opt.zero_grad()
    loss_fn(net, batch).backward()
    opt.step()
    if step % 25 == 24:
        snapshots.append(model.parameters_flat().copy())
print("final loss:  ", f"{model.loss(batch):.4f}")

# curvature via exact autograd HVPs
spec = ng.hessian_spectrum(model, batch, k=5)
print("top Hessian eigenvalues:", np.round(spec.eigenvalues[:5], 3))

# loss landscape
print("sharpness:", ng.sharpness(model, batch, seed=0))

# activations + CKA between layers (named_modules indices of Linear layers)
acts = ng.extract_activations(net, X, layers=["0", "2", "4"])
cka = ng.cka_matrix(acts)
print("layer CKA matrix:\n", np.round(cka["matrix"], 3))
print("collapse scores:", {k: round(ng.collapse_score(v), 3) for k, v in acts.items()})

# trajectory of training
traj = ng.Trajectory.from_vectors(snapshots)
print("trajectory:", {k: round(v, 3) for k, v in traj.summary().items()})

# full automated report
report = ng.analyze(model, batch, representations=acts,
                    labels=y.numpy(), depth="standard", make_figures=False)
print()
print(report.summary())
report.to_html("torch_geometry_report.html")
print("\nwrote torch_geometry_report.html")
