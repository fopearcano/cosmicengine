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
