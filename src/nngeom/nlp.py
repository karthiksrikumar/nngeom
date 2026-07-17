"""Token and prompt geometry for language models.

Every function here works on plain numpy arrays of hidden states, so it is
framework-agnostic: extract states from any transformer (HF ``output_hidden_states=True``,
a torch hook, a JAX model) and analyze them here.

Conventions
-----------
- ``hidden_states``: array of shape (n_layers, n_tokens, dim), the residual
  stream of one sequence across layers (embedding layer first).
- ``embeddings``: array of shape (n_tokens, dim), one layer of one sequence.
- ``contexts``: array of shape (n_contexts, dim), the SAME token's vector
  collected from many different sentences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .manifolds import pca
from .representations import anisotropy, linear_cka


def _check_hs(hidden_states) -> np.ndarray:
    hs = np.asarray(hidden_states, dtype=np.float64)
    if hs.ndim != 3:
        raise ValueError("hidden_states must be (n_layers, n_tokens, dim).")
    return hs


# 1 -------------------------------------------------------------------------

def token_trajectory(hidden_states, token_index: int) -> Dict:
    """The path of one token through the network's layers.

    Treats the token's residual-stream states as a curve through
    representation space and measures its length, straightness, and where
    the biggest transformations happen. Tokens whose meaning the model works
    hard to resolve (rare words, ambiguous words) travel farther and turn more.
    """
    hs = _check_hs(hidden_states)
    path = hs[:, token_index, :]
    steps = np.diff(path, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    total = float(step_norms.sum())
    net = float(np.linalg.norm(path[-1] - path[0]))
    turns = []
    for a, b in zip(steps[:-1], steps[1:]):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            turns.append(float(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))))
    return {
        "path": path,
        "layer_step_norms": step_norms,
        "path_length": total,
        "net_displacement": net,
        "tortuosity": total / net if net > 1e-12 else float("inf"),
        "mean_turning_angle": float(np.mean(turns)) if turns else 0.0,
        "busiest_layer": int(np.argmax(step_norms)) if len(step_norms) else 0,
    }


# 2 -------------------------------------------------------------------------

def contextualization_score(static_embedding: np.ndarray, contexts: np.ndarray) -> Dict[str, float]:
    """How much context reshapes a token, and how varied the reshaping is.

    Compares a token's static (layer-0) embedding against its contextual
    vectors from many sentences. Returns the mean shift away from the static
    point and the self-similarity across contexts. Low self-similarity means
    the model treats the token very differently per context.
    """
    s = np.asarray(static_embedding, dtype=np.float64)
    C = np.asarray(contexts, dtype=np.float64)
    shifts = np.linalg.norm(C - s, axis=1) / max(np.linalg.norm(s), 1e-12)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    sims = Cn @ Cn.T
    n = C.shape[0]
    off_diag = sims[~np.eye(n, dtype=bool)]
    return {
        "mean_context_shift": float(shifts.mean()),
        "max_context_shift": float(shifts.max()),
        "self_similarity": float(off_diag.mean()) if off_diag.size else 1.0,
        "n_contexts": int(n),
    }


# 3 -------------------------------------------------------------------------

def prompt_divergence(hidden_a, hidden_b) -> Dict:
    """Where two prompts' representations split inside the model.

    Give the hidden states of two prompts that share a prefix (e.g. the same
    question phrased two ways, or with/without a system instruction). Reports
    the per-layer distance between the final-token states and the first layer
    where they meaningfully diverge: the depth at which the model starts
    treating the prompts differently.
    """
    A = _check_hs(hidden_a)
    B = _check_hs(hidden_b)
    L = min(A.shape[0], B.shape[0])
    a_last = A[:L, -1, :]
    b_last = B[:L, -1, :]
    dists = np.linalg.norm(a_last - b_last, axis=1)
    scale = 0.5 * (np.linalg.norm(a_last, axis=1) + np.linalg.norm(b_last, axis=1))
    rel = dists / np.maximum(scale, 1e-12)
    threshold = 0.5 * rel.max() if rel.max() > 0 else 0.0
    diverged = np.flatnonzero(rel >= max(threshold, 1e-12))
    return {
        "per_layer_distance": dists,
        "per_layer_relative": rel,
        "divergence_layer": int(diverged[0]) if diverged.size else L - 1,
        "final_relative_distance": float(rel[-1]),
    }


# 4 -------------------------------------------------------------------------

def sentence_curvature(embeddings: np.ndarray) -> Dict[str, float]:
    """The shape of a sentence as a curve of token embeddings.

    Walks the token sequence in order within one layer and measures the
    turning angles and tortuosity of the resulting path. Coherent prose forms
    smoother paths; syntactically chaotic or adversarial inputs turn sharply.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    steps = np.diff(E, axis=0)
    norms = np.linalg.norm(steps, axis=1)
    angles = []
    for a, b in zip(steps[:-1], steps[1:]):
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom > 1e-12:
            angles.append(float(np.arccos(np.clip(np.dot(a, b) / denom, -1, 1))))
    total = float(norms.sum())
    net = float(np.linalg.norm(E[-1] - E[0]))
    return {
        "mean_turning_angle": float(np.mean(angles)) if angles else 0.0,
        "max_turning_angle": float(np.max(angles)) if angles else 0.0,
        "tortuosity": total / net if net > 1e-12 else float("inf"),
        "mean_step_norm": float(norms.mean()) if norms.size else 0.0,
    }


# 5 -------------------------------------------------------------------------

def polysemy_score(contexts: np.ndarray, max_senses: int = 5, seed: Optional[int] = None) -> Dict:
    """Estimate how many distinct senses a token exhibits geometrically.

    Clusters the token's contextual embeddings (k-means over k=1..max_senses)
    and picks the sense count by the elbow of within-cluster dispersion.
    A data-driven word-sense-induction signal with no lexicon needed.
    """
    C = np.asarray(contexts, dtype=np.float64)
    n = C.shape[0]
    max_senses = min(max_senses, n)
    rng = np.random.default_rng(seed)

    def kmeans(k):
        centers = C[rng.choice(n, size=k, replace=False)].copy()
        assign = np.zeros(n, dtype=int)
        for _ in range(25):
            d = np.linalg.norm(C[:, None, :] - centers[None, :, :], axis=2)
            new_assign = d.argmin(axis=1)
            if np.array_equal(new_assign, assign) and _ > 0:
                break
            assign = new_assign
            for j in range(k):
                pts = C[assign == j]
                if len(pts):
                    centers[j] = pts.mean(axis=0)
        inertia = sum(float(np.sum((C[assign == j] - centers[j]) ** 2)) for j in range(k))
        return inertia, assign

    inertias = []
    assigns = {}
    for k in range(1, max_senses + 1):
        inertia, assign = kmeans(k)
        inertias.append(inertia)
        assigns[k] = assign
    inertias = np.asarray(inertias)
    # elbow: last k whose relative improvement over k-1 exceeds 20%
    n_senses = 1
    for k in range(1, len(inertias)):
        prev = max(inertias[k - 1], 1e-12)
        if (prev - inertias[k]) / prev > 0.2:
            n_senses = k + 1
    return {
        "n_senses": int(n_senses),
        "inertias": inertias.tolist(),
        "assignments": assigns[n_senses],
        "spread": float(np.sqrt(inertias[0] / max(n, 1))),
    }


# 6 -------------------------------------------------------------------------

def layer_transition_profile(hidden_states) -> Dict:
    """Where the model stops encoding the input and starts building the output.

    Computes CKA of every layer against the embedding layer (input-likeness)
    and against the final layer (output-likeness), using token vectors as
    examples. The crossing point is the network's processing 'waist', a
    layer-resolution map of the syntax-to-semantics transition.
    """
    hs = _check_hs(hidden_states)
    L = hs.shape[0]
    first, last = hs[0], hs[-1]
    to_input = np.array([linear_cka(hs[l], first) for l in range(L)])
    to_output = np.array([linear_cka(hs[l], last) for l in range(L)])
    crossing = int(np.flatnonzero(to_output >= to_input)[0]) if np.any(to_output >= to_input) else L - 1
    return {
        "cka_to_input": to_input,
        "cka_to_output": to_output,
        "transition_layer": crossing,
        "n_layers": L,
    }


# 7 -------------------------------------------------------------------------

def isotropy_correction(embeddings: np.ndarray, n_components: int = 3) -> Dict:
    """Remove the dominant common directions from an embedding space.

    Implements all-but-the-top: subtract the mean and the top principal
    components that make token embeddings anisotropic (bunched in a cone).
    Returns the corrected embeddings and the anisotropy before/after, so you
    can quantify how much usable geometry the correction unlocks.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    before = anisotropy(X, seed=0)
    Xc = X - X.mean(axis=0, keepdims=True)
    _, components, _ = pca(Xc, k=min(n_components, min(Xc.shape) - 1))
    corrected = Xc - (Xc @ components.T) @ components
    after = anisotropy(corrected, seed=0)
    return {
        "corrected": corrected,
        "anisotropy_before": before,
        "anisotropy_after": after,
        "removed_components": components,
    }


# 8 -------------------------------------------------------------------------

def semantic_axis(positive: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """A unit direction from a contrast set (e.g. formal vs casual sentences)."""
    d = np.asarray(positive, dtype=np.float64).mean(axis=0) - np.asarray(negative, dtype=np.float64).mean(axis=0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else d


def semantic_projection(embeddings: np.ndarray, axis: np.ndarray, center: bool = True) -> np.ndarray:
    """Score every token/sentence along a semantic axis.

    Combine with :func:`semantic_axis` to build custom geometric probes:
    sentiment axes, formality axes, toxicity axes, topic axes. The scores are
    the signed projections, so ordering and magnitude are both meaningful.
    """
    X = np.asarray(embeddings, dtype=np.float64)
    if center:
        X = X - X.mean(axis=0, keepdims=True)
    return X @ np.asarray(axis, dtype=np.float64)


# 9 -------------------------------------------------------------------------

def repetition_attractor_score(embeddings: np.ndarray, min_separation: int = 3,
                               percentile: float = 5.0) -> Dict[str, float]:
    """Detect degeneration loops: does generation orbit an attractor?

    Measures the recurrence rate of a token-embedding sequence: the fraction
    of well-separated token pairs whose states are unusually close. Repetitive
    or looping generations revisit the same region of representation space and
    score high; healthy text keeps exploring.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = E.shape[0]
    if n < min_separation + 2:
        return {"recurrence_rate": 0.0, "n_tokens": int(n)}
    sq = np.sum(E**2, axis=1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * E @ E.T, 0))
    ii, jj = np.triu_indices(n, k=1)
    all_d = d[ii, jj]
    eps = np.percentile(all_d, percentile)
    far_pairs = np.abs(ii - jj) >= min_separation
    recur = np.mean(all_d[far_pairs] <= eps) if far_pairs.any() else 0.0
    return {
        "recurrence_rate": float(recur),
        "epsilon": float(eps),
        "n_tokens": int(n),
    }


# 10 ------------------------------------------------------------------------

def token_novelty(embeddings: np.ndarray, window: Optional[int] = None) -> np.ndarray:
    """Per-token geometric surprise relative to everything said so far.

    For each token, computes the residual of projecting its embedding onto
    the subspace spanned by the previous tokens (optionally a sliding window),
    normalized by its norm. High-novelty tokens introduce genuinely new
    directions: topic shifts, new entities, or hallucination onsets.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = E.shape[0]
    scores = np.zeros(n)
    scores[0] = 1.0
    for i in range(1, n):
        start = max(0, i - window) if window else 0
        prev = E[start:i]
        v = E[i]
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        # least-squares projection of v onto rowspace of prev
        coef, *_ = np.linalg.lstsq(prev.T, v, rcond=None)
        residual = v - prev.T @ coef
        scores[i] = float(np.linalg.norm(residual) / vn)
    return scores


def nlp_summary(hidden_states) -> Dict:
    """One-call geometric profile of a single sequence's hidden states."""
    hs = _check_hs(hidden_states)
    last = hs[-1]
    profile = layer_transition_profile(hs)
    return {
        "n_layers": int(hs.shape[0]),
        "n_tokens": int(hs.shape[1]),
        "dim": int(hs.shape[2]),
        "transition_layer": profile["transition_layer"],
        "final_layer_anisotropy": anisotropy(last, seed=0),
        "sentence_curvature": sentence_curvature(last),
        "repetition": repetition_attractor_score(last),
        "mean_token_novelty": float(token_novelty(last).mean()),
    }
