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
