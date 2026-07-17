"""Training trajectories: a model as a path through parameter space."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .manifolds import pca
from .models import Model


@dataclass
class Trajectory:
    """A sequence of parameter-space points (e.g. training checkpoints)."""

    points: np.ndarray  # (n_checkpoints, n_params)
    labels: List[str] = field(default_factory=list)
    layer_shapes: Optional[Dict[str, tuple]] = None

    @classmethod
    def from_models(cls, models: Sequence[Model], labels: Optional[List[str]] = None) -> "Trajectory":
        pts = np.stack([m.parameters_flat() for m in models])
        labels = labels or [m.name for m in models]
        return cls(points=pts, labels=labels, layer_shapes=models[0].layer_shapes())

    @classmethod
    def from_vectors(cls, vectors: Sequence[np.ndarray], labels: Optional[List[str]] = None) -> "Trajectory":
        pts = np.stack([np.asarray(v, dtype=np.float64).ravel() for v in vectors])
        labels = labels or [f"step_{i}" for i in range(len(vectors))]
        return cls(points=pts, labels=labels)

    def __len__(self) -> int:
        return self.points.shape[0]

    # -- path geometry -------------------------------------------------------
    def steps(self) -> np.ndarray:
        """Displacement vectors between consecutive checkpoints."""
        return np.diff(self.points, axis=0)

    def step_norms(self) -> np.ndarray:
        return np.linalg.norm(self.steps(), axis=1)

    def path_length(self) -> float:
        """Total distance traveled along the path."""
        return float(self.step_norms().sum())

    def net_displacement(self) -> float:
        """Straight-line distance from start to end."""
        return float(np.linalg.norm(self.points[-1] - self.points[0]))

    def tortuosity(self) -> float:
        """path_length / net_displacement. 1.0 = perfectly straight path;
        large values = the optimizer wandered."""
        net = self.net_displacement()
        return float(self.path_length() / net) if net > 1e-12 else float("inf")

    def directional_persistence(self) -> float:
        """Mean cosine similarity of consecutive steps.

        Near 1: steady directed motion. Near 0: diffusion-like wandering.
        Negative: oscillation (e.g. too-high learning rate).
        """
        s = self.steps()
        if len(s) < 2:
            return 1.0
        cosines = []
        for a, b in zip(s[:-1], s[1:]):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 1e-12:
                cosines.append(float(np.dot(a, b) / denom))
        return float(np.mean(cosines)) if cosines else 0.0

    def distance_from_start(self) -> np.ndarray:
        return np.linalg.norm(self.points - self.points[0], axis=1)

    def norm_growth(self) -> np.ndarray:
        """Global parameter norm at each checkpoint.

        Steady growth is normal; sudden jumps often coincide with instability
        or a learning-rate change.
        """
        return np.linalg.norm(self.points, axis=1)

    def step_cosines(self) -> np.ndarray:
        """Cosine similarity between each pair of consecutive steps."""
        s = self.steps()
        out = []
        for a, b in zip(s[:-1], s[1:]):
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            out.append(float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0)
        return np.asarray(out)

    def phase_changes(self, threshold: float = 0.0) -> List[int]:
        """Checkpoints where the optimization direction sharply changed.

        Returns indices where the cosine between consecutive steps drops
        below ``threshold`` (default 0: the step actually reversed course).
        These often mark learning-rate drops, warmup ends, or instability.
        """
        cosines = self.step_cosines()
        return [int(i) + 1 for i in np.flatnonzero(cosines < threshold)]

    # -- layerwise view --------------------------------------------------------
    def layerwise_drift(self) -> Dict[str, np.ndarray]:
        """Per-layer distance from the first checkpoint, over time.

        Requires the trajectory to have been built from Models (layer shapes
        known). Shows which layers move most during training/fine-tuning.
        """
        if self.layer_shapes is None:
            raise ValueError("Layer shapes unknown; build the trajectory with from_models().")
        out: Dict[str, np.ndarray] = {}
        offset = 0
        for name, shape in self.layer_shapes.items():
            size = int(np.prod(shape)) if shape else 1
            block = self.points[:, offset : offset + size]
            out[name] = np.linalg.norm(block - block[0], axis=1)
            offset += size
        return out

    # -- projection ------------------------------------------------------------
    def pca_projection(self, k: int = 2):
        """Project the trajectory into its top-k principal components.

        Returns (projected (n, k), explained_variance_ratio (k,)).
        """
        proj, _, ratio = pca(self.points, k=k)
        return proj, ratio

    def summary(self) -> Dict:
        return {
            "n_checkpoints": len(self),
            "path_length": self.path_length(),
            "net_displacement": self.net_displacement(),
            "tortuosity": self.tortuosity(),
            "directional_persistence": self.directional_persistence(),
            "mean_step_norm": float(self.step_norms().mean()) if len(self) > 1 else 0.0,
            "max_step_norm": float(self.step_norms().max()) if len(self) > 1 else 0.0,
        }

    def plot(self, ax=None, **kwargs):
        from .visualize import plot_trajectory

        return plot_trajectory(self, ax=ax, **kwargs)


def loss_along_trajectory(models: Sequence[Model], batch) -> np.ndarray:
    """Loss at each checkpoint (convenience for plotting geometry vs loss)."""
    return np.asarray([m.loss(batch) for m in models])
