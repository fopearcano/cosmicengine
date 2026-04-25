"""AI Viewer client for CosmicEngine.

A minimal standalone client that consumes the runtime server's
newline-delimited JSON stream and renders SceneState messages and
PPM frames locally. No GUI, no GPU, no Unreal. Designed to be
replaced later by a Vulkan / WebGPU / neural renderer.
"""

from ai_viewer.client import RuntimeClient
from ai_viewer.config import AIViewerConfig
from ai_viewer.frame_buffer import FrameBuffer
from ai_viewer.neural_field import (
    BlackHole,
    CPUSplatFallback,
    GPUDevice,
    GPUGaussianBuffer,
    GaussianPoint,
    GaussianSplatPipeline,
    GaussianSplatRenderer,
    GeodesicRayMarcher,
    apply_black_hole_to_points,
    apply_lensing,
    apply_lensing_to_points,
    build_gaussian_field_from_density,
    build_gaussian_field_from_galaxy_batch,
    compute_deflection_angle,
    integrate_geodesic_step,
    schwarzschild_acceleration,
    trace_points_through_geodesic,
    warp_gaussian_field,
    warp_gaussian_point,
)
from ai_viewer.neural_postprocess import ONNXFrameProcessor
from ai_viewer.neural_warp_viewer import NeuralWarpViewer
from ai_viewer.postprocess import (
    BrightnessProcessor,
    ColorShiftProcessor,
    CompositeProcessor,
    ContrastBoostProcessor,
    FramePostProcessor,
    SafeProcessor,
    build_postprocessor_from_config,
)
from ai_viewer.viewer import AIViewer
from ai_viewer.window import ViewerWindow

__all__ = [
    "AIViewer",
    "AIViewerConfig",
    "BlackHole",
    "BrightnessProcessor",
    "ColorShiftProcessor",
    "CompositeProcessor",
    "ContrastBoostProcessor",
    "CPUSplatFallback",
    "FrameBuffer",
    "FramePostProcessor",
    "GPUDevice",
    "GPUGaussianBuffer",
    "GaussianPoint",
    "GaussianSplatPipeline",
    "GaussianSplatRenderer",
    "GeodesicRayMarcher",
    "NeuralWarpViewer",
    "ONNXFrameProcessor",
    "RuntimeClient",
    "SafeProcessor",
    "ViewerWindow",
    "apply_black_hole_to_points",
    "apply_lensing",
    "apply_lensing_to_points",
    "build_gaussian_field_from_density",
    "build_gaussian_field_from_galaxy_batch",
    "build_postprocessor_from_config",
    "compute_deflection_angle",
    "integrate_geodesic_step",
    "schwarzschild_acceleration",
    "trace_points_through_geodesic",
    "warp_gaussian_field",
    "warp_gaussian_point",
]
