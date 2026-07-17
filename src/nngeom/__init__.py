"""nngeom — neural networks as geometric objects.

Measure, visualize, and reason about loss landscapes, curvature,
representation geometry, training trajectories, and model differences.

Quick start::

    import nngeom as ng

    model = ng.Model(torch_module, loss_fn=my_loss)   # or Model.from_function(...)
    report = ng.analyze(model, batch)
    print(report.summary())
    report.to_html("geometry_report.html")
"""

from . import (
    compare,
    curvature,
    data,
    dynamics,
    export,
    lossland,
    manifolds,
    metrics,
    nlp,
    probes,
    representations,
    surfaces,
    trajectories,
    utils,
    visualize,
)
from .compare import (
    CompareReport,
    apply_task_vector,
    eigenvector_overlap,
    model_soup,
    task_vector,
    task_vector_alignment,
)
from .compare import compare as compare_models
from .dynamics import (
    fisher_diagonal,
    fisher_layer_importance,
    gradient_alignment,
    gradient_noise_scale,
    local_linearity,
    sam_sharpness,
)
from .curvature import (
    Spectrum,
    curvature_along,
    hessian_spectrum,
    hutchinson_trace,
    lanczos_spectrum,
    layerwise_curvature,
    power_iteration,
)
from .lossland import (
    LossCurve,
    LossSurface,
    basin_width,
    bezier_path,
    find_low_loss_path,
    interpolation_smoothness,
    linear_interpolation,
    loss_barrier_matrix,
    plane_slice_2d,
    random_slice_1d,
    sharpness,
)
from .manifolds import (
    class_manifold_dimensions,
    effective_rank,
    intrinsic_dimension_twonn,
    knn_graph,
    manifold_summary,
    neighborhood_preservation,
    outlier_scores,
    participation_ratio,
    pca,
    random_projection,
)
from .nlp import (
    contextualization_score,
    isotropy_correction,
    layer_transition_profile,
    nlp_summary,
    polysemy_score,
    prompt_divergence,
    repetition_attractor_score,
    semantic_axis,
    semantic_projection,
    sentence_curvature,
    token_novelty,
    token_trajectory,
)
from .metrics import compute as compute_metrics
from .metrics import list_metrics, register_metric
from .models import Model
from .probes import concept_direction, filter_normalize, orthogonalize, random_direction
from .report import GeometryReport, analyze
from .representations import (
    anisotropy,
    cka_matrix,
    class_geometry,
    collapse_score,
    extract_activations,
    linear_cka,
    linear_probe_score,
    neuron_alignment,
    procrustes_distance,
    rbf_cka,
    representation_drift,
    representation_summary,
    subspace_overlap,
    svcca,
    uniformity,
)
from .trajectories import Trajectory, loss_along_trajectory
from .utils import set_seed

__version__ = "0.2.0"

__all__ = [
    "Model",
    "analyze",
    "GeometryReport",
    "set_seed",
    # lossland
    "LossCurve",
    "LossSurface",
    "linear_interpolation",
    "random_slice_1d",
    "plane_slice_2d",
    "sharpness",
    "basin_width",
    "interpolation_smoothness",
    "loss_barrier_matrix",
    "bezier_path",
    "find_low_loss_path",
    # dynamics
    "gradient_alignment",
    "gradient_noise_scale",
    "fisher_diagonal",
    "fisher_layer_importance",
    "sam_sharpness",
    "local_linearity",
    # nlp
    "token_trajectory",
    "contextualization_score",
    "prompt_divergence",
    "sentence_curvature",
    "polysemy_score",
    "layer_transition_profile",
    "isotropy_correction",
    "semantic_axis",
    "semantic_projection",
    "repetition_attractor_score",
    "token_novelty",
    "nlp_summary",
    # curvature
    "Spectrum",
    "hessian_spectrum",
    "power_iteration",
    "lanczos_spectrum",
    "hutchinson_trace",
    "curvature_along",
    "layerwise_curvature",
    # representations
    "linear_cka",
    "rbf_cka",
    "cka_matrix",
    "subspace_overlap",
    "svcca",
    "procrustes_distance",
    "uniformity",
    "linear_probe_score",
    "neuron_alignment",
    "representation_drift",
    "anisotropy",
    "collapse_score",
    "class_geometry",
    "representation_summary",
    "extract_activations",
    # manifolds
    "pca",
    "effective_rank",
    "participation_ratio",
    "intrinsic_dimension_twonn",
    "knn_graph",
    "neighborhood_preservation",
    "manifold_summary",
    "random_projection",
    "outlier_scores",
    "class_manifold_dimensions",
    # trajectories
    "Trajectory",
    "loss_along_trajectory",
    # compare
    "compare_models",
    "CompareReport",
    "task_vector",
    "apply_task_vector",
    "task_vector_alignment",
    "model_soup",
    "eigenvector_overlap",
    # probes
    "random_direction",
    "filter_normalize",
    "orthogonalize",
    "concept_direction",
    # metrics
    "register_metric",
    "compute_metrics",
    "list_metrics",
]
