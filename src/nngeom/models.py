"""Model wrappers: a uniform geometric interface over frameworks.

The rest of nngeom talks to a :class:`Model`, which exposes the network as a
point in parameter space:

- ``parameters_flat()``   -> 1-D numpy vector (the "point")
- ``set_parameters_flat`` -> move the model to another point
- ``loss(batch)``         -> scalar field evaluated at the current point
- ``gradient(batch)``     -> flat gradient vector

Two adapters are provided:

- :class:`TorchAdapter` wraps any ``torch.nn.Module`` + loss function and uses
  autograd for gradients and Hessian-vector products.
- :class:`FunctionalAdapter` wraps a plain dict of numpy parameter arrays plus
  a loss callable, using an optional analytic gradient or central finite
  differences. This keeps the entire library usable without any deep learning
  framework installed (and makes testing easy).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from .utils import HAS_TORCH, flatten_dict, unflatten_dict

if HAS_TORCH:
    import torch


class BaseAdapter:
    """Interface every framework adapter implements."""

    def parameters_flat(self) -> np.ndarray:
        raise NotImplementedError

    def set_parameters_flat(self, flat: np.ndarray) -> None:
        raise NotImplementedError

    def layer_names(self) -> List[str]:
        raise NotImplementedError

    def layer_shapes(self) -> Dict[str, tuple]:
        raise NotImplementedError

    def loss(self, batch) -> float:
        raise NotImplementedError

    def gradient(self, batch) -> np.ndarray:
        raise NotImplementedError

    def hvp(self, batch, vec: np.ndarray) -> np.ndarray:
        """Hessian-vector product. Default: finite difference of the gradient."""
        flat0 = self.parameters_flat()
        norm = np.linalg.norm(vec)
        if norm == 0:
            return np.zeros_like(vec)
        eps = 1e-4 / norm * max(1.0, np.linalg.norm(flat0))
        try:
            self.set_parameters_flat(flat0 + eps * vec)
            g_plus = self.gradient(batch)
            self.set_parameters_flat(flat0 - eps * vec)
            g_minus = self.gradient(batch)
        finally:
            self.set_parameters_flat(flat0)
        return (g_plus - g_minus) / (2 * eps)


class FunctionalAdapter(BaseAdapter):
    """Adapter for numpy parameter dicts + a loss callable.

    Parameters
    ----------
    params : dict[str, np.ndarray]
        Named parameter arrays; each key is treated as a "layer".
    loss_fn : callable(params_dict, batch) -> float
    grad_fn : optional callable(params_dict, batch) -> dict[str, np.ndarray]
        Analytic gradient. If omitted, central finite differences are used
        (fine for small models / tests, O(2n) loss evaluations).
    """

    def __init__(self, params: Dict[str, np.ndarray], loss_fn: Callable, grad_fn: Optional[Callable] = None):
        self._params = {k: np.asarray(v, dtype=np.float64).copy() for k, v in params.items()}
        self._loss_fn = loss_fn
        self._grad_fn = grad_fn
        _, self._spec = flatten_dict(self._params)

    def parameters_flat(self) -> np.ndarray:
        flat, _ = flatten_dict(self._params)
        return flat

    def set_parameters_flat(self, flat: np.ndarray) -> None:
        self._params = {k: v.copy() for k, v in unflatten_dict(np.asarray(flat, dtype=np.float64), self._spec).items()}

    def layer_names(self) -> List[str]:
        return list(self._params.keys())

    def layer_shapes(self) -> Dict[str, tuple]:
        return {k: tuple(np.asarray(v).shape) for k, v in self._params.items()}

    def loss(self, batch) -> float:
        return float(self._loss_fn(self._params, batch))

    def gradient(self, batch) -> np.ndarray:
        if self._grad_fn is not None:
            g = self._grad_fn(self._params, batch)
            flat, _ = flatten_dict(g)
            return flat
        # central finite differences
        flat0 = self.parameters_flat()
        grad = np.zeros_like(flat0)
        eps = 1e-6 * max(1.0, np.linalg.norm(flat0))
        try:
            for i in range(flat0.size):
                step = np.zeros_like(flat0)
                step[i] = eps
                self.set_parameters_flat(flat0 + step)
                lp = self.loss(batch)
                self.set_parameters_flat(flat0 - step)
                lm = self.loss(batch)
                grad[i] = (lp - lm) / (2 * eps)
        finally:
            self.set_parameters_flat(flat0)
        return grad


class TorchAdapter(BaseAdapter):
    """Adapter for ``torch.nn.Module`` models.

    Parameters
    ----------
    module : torch.nn.Module
    loss_fn : callable(module, batch) -> torch.Tensor (scalar)
        E.g. ``lambda m, (x, y): F.cross_entropy(m(x), y)``.
    """

    def __init__(self, module, loss_fn: Callable):
        if not HAS_TORCH:  # pragma: no cover
            raise ImportError("PyTorch is not installed; install nngeom[torch].")
        self.module = module
        self._loss_fn = loss_fn

    def _named_params(self):
        return [(n, p) for n, p in self.module.named_parameters() if p.requires_grad]

    def parameters_flat(self) -> np.ndarray:
        return np.concatenate(
            [p.detach().cpu().numpy().astype(np.float64).ravel() for _, p in self._named_params()]
        )

    def set_parameters_flat(self, flat: np.ndarray) -> None:
        offset = 0
        with torch.no_grad():
            for _, p in self._named_params():
                size = p.numel()
                chunk = np.asarray(flat[offset : offset + size]).reshape(tuple(p.shape))
                p.copy_(torch.as_tensor(chunk, dtype=p.dtype, device=p.device))
                offset += size

    def layer_names(self) -> List[str]:
        return [n for n, _ in self._named_params()]

    def layer_shapes(self) -> Dict[str, tuple]:
        return {n: tuple(p.shape) for n, p in self._named_params()}

    def loss(self, batch) -> float:
        was_training = self.module.training
        self.module.eval()
        try:
            with torch.no_grad():
                return float(self._loss_fn(self.module, batch).item())
        finally:
            self.module.train(was_training)

    def _loss_tensor(self, batch):
        return self._loss_fn(self.module, batch)

    def gradient(self, batch) -> np.ndarray:
        self.module.zero_grad(set_to_none=True)
        loss = self._loss_tensor(batch)
        params = [p for _, p in self._named_params()]
        grads = torch.autograd.grad(loss, params, create_graph=False)
        return np.concatenate([g.detach().cpu().numpy().astype(np.float64).ravel() for g in grads])

    def hvp(self, batch, vec: np.ndarray) -> np.ndarray:
        """Exact Hessian-vector product via double backward."""
        params = [p for _, p in self._named_params()]
        loss = self._loss_tensor(batch)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        offset = 0
        vecs = []
        for p in params:
            size = p.numel()
            chunk = np.asarray(vec[offset : offset + size]).reshape(tuple(p.shape))
            vecs.append(torch.as_tensor(chunk, dtype=p.dtype, device=p.device))
            offset += size
        dot = sum((g * v).sum() for g, v in zip(grads, vecs))
        hv = torch.autograd.grad(dot, params, retain_graph=False)
        return np.concatenate([h.detach().cpu().numpy().astype(np.float64).ravel() for h in hv])


class Model:
    """A neural network viewed as a point in parameter space.

    Construct from a torch module::

        model = Model(torch_module, loss_fn=lambda m, b: F.cross_entropy(m(b[0]), b[1]))

    or from plain numpy parameters (no framework needed)::

        model = Model.from_function(params_dict, loss_fn, grad_fn=None)
    """

    def __init__(self, obj, loss_fn: Optional[Callable] = None, adapter: Optional[BaseAdapter] = None, name: str = "model"):
        self.name = name
        if adapter is not None:
            self.adapter = adapter
        elif HAS_TORCH and isinstance(obj, torch.nn.Module):
            if loss_fn is None:
                raise ValueError("A torch model requires loss_fn(module, batch) -> scalar tensor.")
            self.adapter = TorchAdapter(obj, loss_fn)
        elif isinstance(obj, dict):
            if loss_fn is None:
                raise ValueError("A functional model requires loss_fn(params, batch) -> float.")
            self.adapter = FunctionalAdapter(obj, loss_fn)
        else:
            raise TypeError(
                f"Cannot wrap object of type {type(obj)!r}. Pass a torch.nn.Module or a dict of numpy arrays."
            )

    # -- construction helpers ------------------------------------------------
    @classmethod
    def from_function(cls, params: Dict[str, np.ndarray], loss_fn: Callable, grad_fn: Optional[Callable] = None, name: str = "model") -> "Model":
        return cls(None, adapter=FunctionalAdapter(params, loss_fn, grad_fn), name=name)

    @classmethod
    def load(cls, path: str, module=None, loss_fn: Optional[Callable] = None, name: Optional[str] = None) -> "Model":
        """Load a torch checkpoint (state_dict) into ``module`` and wrap it."""
        if not HAS_TORCH:
            raise ImportError("Model.load requires PyTorch. Install nngeom[torch].")
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if module is None:
            raise ValueError("Pass module= (an instance of the architecture) to load a state_dict into.")
        module.load_state_dict(state)
        return cls(module, loss_fn=loss_fn, name=name or path)

    # -- geometric interface -------------------------------------------------
    def parameters_flat(self) -> np.ndarray:
        return self.adapter.parameters_flat()

    def set_parameters_flat(self, flat: np.ndarray) -> None:
        self.adapter.set_parameters_flat(flat)

    def layer_names(self) -> List[str]:
        return self.adapter.layer_names()

    def layer_shapes(self) -> Dict[str, tuple]:
        return self.adapter.layer_shapes()

    @property
    def num_parameters(self) -> int:
        return int(self.parameters_flat().size)

    def loss(self, batch) -> float:
        return self.adapter.loss(batch)

    def gradient(self, batch) -> np.ndarray:
        return self.adapter.gradient(batch)

    def hvp(self, batch, vec: np.ndarray) -> np.ndarray:
        return self.adapter.hvp(batch, vec)

    def clone_at(self, flat: np.ndarray) -> "Model":
        """Return a lightweight view of this model shifted to another point.

        Note: for torch models this mutates and restores shared weights during
        analyses; nngeom routines always restore the original point.
        """
        import copy

        new = copy.copy(self)
        new.set_parameters_flat(flat)
        return new

    def layer_norms(self) -> Dict[str, float]:
        """L2 norm of each named parameter tensor."""
        flat = self.parameters_flat()
        norms: Dict[str, float] = {}
        offset = 0
        for name, shape in self.layer_shapes().items():
            size = int(np.prod(shape)) if shape else 1
            norms[name] = float(np.linalg.norm(flat[offset : offset + size]))
            offset += size
        return norms

    def _layer_blocks(self):
        """Yield (name, shape, flat_slice) for each layer block."""
        flat = self.parameters_flat()
        offset = 0
        for name, shape in self.layer_shapes().items():
            size = int(np.prod(shape)) if shape else 1
            yield name, shape, flat[offset : offset + size]
            offset += size

    def spectral_norms(self) -> Dict[str, float]:
        """Largest singular value of each weight matrix (>=2-D layers only).

        The product of spectral norms bounds the network's Lipschitz constant;
        individual spikes flag layers that amplify perturbations.
        """
        out: Dict[str, float] = {}
        for name, shape, block in self._layer_blocks():
            if len(shape) >= 2:
                W = block.reshape(shape[0], -1)
                out[name] = float(np.linalg.svd(W, compute_uv=False)[0])
        return out

    def stable_ranks(self) -> Dict[str, float]:
        """Stable rank ||W||_F^2 / ||W||_2^2 of each weight matrix.

        A robust effective-rank proxy: low stable rank means the layer's
        transformation is dominated by a few directions.
        """
        out: Dict[str, float] = {}
        for name, shape, block in self._layer_blocks():
            if len(shape) >= 2:
                W = block.reshape(shape[0], -1)
                s = np.linalg.svd(W, compute_uv=False)
                if s[0] > 1e-12:
                    out[name] = float(np.sum(s**2) / s[0] ** 2)
        return out

    def __repr__(self) -> str:
        return f"Model(name={self.name!r}, n_params={self.num_parameters})"
