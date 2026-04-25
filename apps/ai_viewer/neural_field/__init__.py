"""Neural-field / Gaussian-splatting renderer for the AI Viewer.

Phase 24 contribution: a CPU-only Gaussian splatting prototype that
turns discrete galaxy or density-grid samples into a continuous
visual field. Pure NumPy. No GPU, no Vulkan, no OpenGL — those come
in a later phase.
"""

from ai_viewer.neural_field.field_builder import (
    build_gaussian_field_from_density,
    build_gaussian_field_from_galaxy_batch,
)
from ai_viewer.neural_field.field_warp import (
    warp_gaussian_field,
    warp_gaussian_point,
)
from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gpu import (
    CPUSplatFallback,
    GPUDevice,
    GPUGaussianBuffer,
    GaussianSplatPipeline,
)
from ai_viewer.neural_field.gr import (
    BlackHole,
    apply_black_hole_to_points,
    apply_lensing,
    apply_lensing_to_points,
    compute_deflection_angle,
)
from ai_viewer.neural_field.splat_renderer import GaussianSplatRenderer

__all__ = [
    "BlackHole",
    "CPUSplatFallback",
    "GPUDevice",
    "GPUGaussianBuffer",
    "GaussianPoint",
    "GaussianSplatPipeline",
    "GaussianSplatRenderer",
    "apply_black_hole_to_points",
    "apply_lensing",
    "apply_lensing_to_points",
    "build_gaussian_field_from_density",
    "build_gaussian_field_from_galaxy_batch",
    "compute_deflection_angle",
    "warp_gaussian_field",
    "warp_gaussian_point",
]
