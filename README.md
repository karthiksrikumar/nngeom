# nngeom

**Neural networks as geometric objects.**

`nngeom` is a Python library for measuring, visualizing, and reasoning about the geometry of neural networks: loss landscapes, curvature, representation structure, training trajectories, token geometry for language models, and model comparison. It treats a trained model not just as a function, but as a point (and a path) in a high-dimensional space with a shape you can inspect.

**Documentation:** open [docs/index.html](docs/index.html) in a browser for the full guide, [docs/api.html](docs/api.html) for the API reference, and [docs/nlp.html](docs/nlp.html) for the NLP geometry tools.

## Installation

```bash
pip install nngeom            # core (numpy only)
pip install nngeom[torch]     # + PyTorch adapter (autograd HVPs, activation hooks)
pip install nngeom[viz]       # + matplotlib plotting
pip install nngeom[all]       # everything
```

The core library has **no framework dependency**: every analysis works on plain numpy models via `Model.from_function`, and PyTorch models get exact autograd-based gradients and Hessian-vector products when torch is installed.

## Quick start

```python
import nngeom as ng

# Wrap a PyTorch model (any nn.Module + a loss closure)
model = ng.Model(net, loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]))

# One-call automated geometry report
report = ng.analyze(model, batch, depth="standard")
print(report.summary())
report.to_html("geometry_report.html")   # self-contained, shareable
```

## What it can do

| Area | Highlights |
|---|---|
| **Loss landscapes** | interpolation barriers, 2D slices, sharpness, basin width, curved low-loss path search (`find_low_loss_path`), pairwise barrier matrices |
| **Curvature** | Hessian spectra (Lanczos / power iteration), Hutchinson trace, per-layer curvature, all matrix-free |
| **Representations** | CKA, SVCCA, Procrustes, subspace overlap, linear probes, neuron matching, collapse / anisotropy / uniformity, drift reports |
| **Manifolds** | TwoNN intrinsic dimension, effective rank, kNN graphs, outlier scores, random projections, per-class manifold dimensions |
| **Trajectories** | path length, tortuosity, directional persistence, phase-change detection, norm growth, layerwise drift |
| **Optimization dynamics** | gradient noise scale, batch gradient alignment, empirical Fisher, SAM sharpness, local linearity |
| **Comparison** | per-layer drift, task vectors and alignment, model soups, Hessian eigenspace overlap |
| **NLP geometry** | token trajectories, contextualization, prompt divergence, sentence curvature, polysemy, layer transition, isotropy correction, semantic axes, repetition attractors, token novelty |
| **Infrastructure** | metric registry, scalar-field surfaces, matplotlib plots, JSON / CSV / HTML export, automated warnings |

## Examples

```python
# Is my fine-tune in the same basin as the base model?
ng.linear_interpolation(base, tuned, batch).barrier_height()

# If the straight path has a barrier, is there a curved valley?
ng.find_low_loss_path(base, tuned, batch)["connected"]

# Can I average my fine-tunes into a soup?
soup = ng.model_soup([tune_a, tune_b])

# Are my minibatch gradients mostly noise?
ng.gradient_noise_scale(model, batches)

# Where do two prompts diverge inside my language model?
ng.prompt_divergence(hidden_a, hidden_b)["divergence_layer"]

# Is this generation stuck in a repetition loop?
ng.repetition_attractor_score(last_layer_states)["recurrence_rate"]
```

Runnable walkthroughs live in [examples/](examples/): a ground-truth quadratic demo, fine-tune comparison, training dynamics under two learning rates, representation health checks, all ten NLP tools, and a full PyTorch workflow.

## Testing

```bash
pip install nngeom[dev]
pytest
```

The suite verifies curvature and landscape math against a quadratic model with a known Hessian, and every representation / NLP metric against planted structure.

## Design

- **Geometry first.** Models are points, checkpoints are trajectories, losses are surfaces, activations are manifolds.
- **Model-agnostic.** A thin `Model` adapter layer; torch is optional, other frameworks pluggable.
- **Matrix-free.** Curvature uses only Hessian-vector products, so it scales past toy models.
- **Reproducible.** Every stochastic routine takes a `seed`; results carry metadata and export cleanly.
- **Honest.** The automated report surfaces warnings (saddle points, collapse, sharp basins) instead of just pretty plots.

## License

MIT
