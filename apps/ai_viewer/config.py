"""AI Viewer configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIViewerConfig:
    """Top-level dial for the standalone AI Viewer client."""

    server_host: str = "127.0.0.1"
    server_port: int = 8765
    width: int = 512
    height: int = 512
    enable_ai_postprocess: bool = False
    output_directory: str = "outputs/viewer"

    enable_window: bool = True
    enable_postprocess: bool = True
    max_fps: float = 30.0

    neural_model_path: str | None = None
    neural_input_width: int | None = None
    neural_input_height: int | None = None
    neural_normalize: bool = True

    use_photon_warp: bool = False
    photon_warp_model_path: str | None = None

    render_mode: str = "ppm"
    gaussian_sigma_scale: float = 1.0
    max_gaussian_points: int = 50_000

    enable_field_warp: bool = False
    field_warp_mode: str = "deterministic"

    use_gpu_pipeline: bool = False

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is invalid."""
        if not self.server_host:
            raise ValueError("server_host must be a non-empty string")
        if not (0 < self.server_port < 65_536):
            raise ValueError(
                f"server_port must be in (0, 65536); got {self.server_port}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not self.output_directory:
            raise ValueError("output_directory must be a non-empty string")
        if self.max_fps <= 0.0:
            raise ValueError("max_fps must be positive")
        # Neural input size: both dimensions or neither.
        w_set = self.neural_input_width is not None
        h_set = self.neural_input_height is not None
        if w_set != h_set:
            raise ValueError(
                "neural_input_width and neural_input_height must both "
                "be set or both be None"
            )
        if self.neural_input_width is not None and self.neural_input_width <= 0:
            raise ValueError("neural_input_width must be positive")
        if self.neural_input_height is not None and self.neural_input_height <= 0:
            raise ValueError("neural_input_height must be positive")
        if self.render_mode not in ("ppm", "gaussian"):
            raise ValueError(
                f"render_mode must be 'ppm' or 'gaussian'; "
                f"got {self.render_mode!r}"
            )
        if self.gaussian_sigma_scale <= 0.0:
            raise ValueError("gaussian_sigma_scale must be positive")
        if self.max_gaussian_points <= 0:
            raise ValueError("max_gaussian_points must be positive")
        if self.field_warp_mode not in ("none", "deterministic", "ai"):
            raise ValueError(
                f"field_warp_mode must be 'none', 'deterministic', or "
                f"'ai'; got {self.field_warp_mode!r}"
            )
