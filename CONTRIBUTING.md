# Contributing to nngeom

Thanks for considering a contribution. nngeom is a small, focused library, so the bar for
new functions is that they measure something genuinely useful about a model's geometry,
not just wrap an existing metric under a new name.

## Setup

```bash
git clone https://github.com/karthiksrikumar/nngeom.git
cd nngeom
pip install -e .[dev]
```

## Running tests

```bash
pytest
```

Curvature and loss-landscape tests are checked against a quadratic model with a known,
planted Hessian. Representation and NLP metrics are checked against planted structure
(known clusters, known intrinsic dimension, known divergence layers, and so on). If you
add a new metric, add a test that verifies it against a case where the right answer is
known analytically or by construction, not just that the function runs without error.

## Adding a new function

- Put it in the module that matches its object: `lossland.py` for loss-surface tools,
  `representations.py` for activation-matrix tools, `nlp.py` for token/hidden-state tools,
  and so on.
- Accept plain numpy arrays wherever possible, so the function works regardless of which
  framework produced the data.
- Export it from `src/nngeom/__init__.py` and add it to `__all__`.
- Document it in `docs/api.html` (or `docs/nlp.html` for NLP tools) using the existing table
  format.

## Style

- No em dashes in docs or docstrings.
- Every stochastic routine takes a `seed` argument.
- Any routine that moves a model through parameter space must restore the original
  parameters before returning, including on error (use `try`/`finally`).

## Reporting issues

Open an issue at https://github.com/karthiksrikumar/nngeom/issues with a minimal
reproduction. If it is a numerical discrepancy, include what value you expected and why.
