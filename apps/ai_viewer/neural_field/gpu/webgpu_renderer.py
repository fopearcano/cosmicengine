"""WebGPU-backed Gaussian splat renderer.

Renders to an off-screen RGBA texture and reads it back into a NumPy
array, so the renderer works without a window. Falls back by raising
``RuntimeError`` when the device is not real WebGPU; the calling
:class:`GaussianSplatPipeline` catches that and routes through the
CPU fallback.

The relativistic / hyper warp is applied **inside the shader**: the
WGSL stage computes the warped direction + Doppler tint per point so
no CPU pre-pass is required.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ai_viewer.neural_field.gaussian import GaussianPoint
from ai_viewer.neural_field.gpu.webgpu_buffer import WebGPUGaussianBuffer
from ai_viewer.neural_field.gpu.webgpu_device import WebGPUDevice
from cosmic_engine.perception.observer import ObserverState
from cosmic_engine.rendering.simple_camera import SimpleCamera


_SHADER_PATH = Path(__file__).parent / "shaders" / "splat.wgsl"
_GR_SHADER_PATH = Path(__file__).parent / "shaders" / "gr_splat.wgsl"
# Enough room for the GR shader's 7 vec4 uniform block, padded to 256.
_UNIFORM_SIZE_BYTES: int = 256


class WebGPUSplatRenderer:
    """Splat a list of :class:`GaussianPoint` on a real WebGPU device."""

    def __init__(
        self,
        device: WebGPUDevice,
        width: int,
        height: int,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        self.device = device
        self.width = width
        self.height = height
        self._pipeline = None
        self._uniform_buffer = None
        self._color_texture = None
        self._readback_buffer = None
        self._shader_module = None
        self._pipeline_initialized = False
        # GR state (Phase 28)
        self._black_hole = None
        self._enable_lensing: bool = False
        self._use_gr_shader: bool = False

    def enable_gr_effects(
        self,
        black_hole=None,
        enable_lensing: bool = True,
    ) -> None:
        """Wire a black hole + lensing toggle into the render pass.

        Pass ``black_hole=None`` and ``enable_lensing=False`` to revert
        to the plain Phase 27 splat shader. Reconfiguring requires
        re-initializing the pipeline, so the flag is captured before
        the next ``render`` call.
        """
        self._black_hole = black_hole
        self._enable_lensing = bool(enable_lensing)
        self._use_gr_shader = (
            black_hole is not None or self._enable_lensing
        )
        # Force pipeline recreation so the right shader is selected.
        self._pipeline_initialized = False
        self._shader_module = None

    # --- pipeline setup --------------------------------------------------

    def initialize_pipeline(self) -> None:
        """Compile the shader and create persistent GPU resources."""
        if self._pipeline_initialized:
            return
        if self.device.device is None:
            raise RuntimeError("WebGPUSplatRenderer: device not initialized")
        import wgpu

        wgpu_dev = self.device.device
        shader_path = _GR_SHADER_PATH if self._use_gr_shader else _SHADER_PATH
        shader_source = shader_path.read_text(encoding="utf-8")
        self._shader_module = wgpu_dev.create_shader_module(code=shader_source)

        self._uniform_buffer = wgpu_dev.create_buffer(
            size=_UNIFORM_SIZE_BYTES,
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Will be (re)created when render() is first called or when size
        # changes; created here so initialize is enough on its own.
        self._color_texture = wgpu_dev.create_texture(
            size=(self.width, self.height, 1),
            format=wgpu.TextureFormat.rgba8unorm,
            usage=(
                wgpu.TextureUsage.RENDER_ATTACHMENT
                | wgpu.TextureUsage.COPY_SRC
            ),
        )
        # 4 bytes per pixel; pad each row to 256 bytes (WebGPU mandate).
        bytes_per_row = self._padded_bytes_per_row()
        self._readback_buffer = wgpu_dev.create_buffer(
            size=bytes_per_row * self.height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        self._pipeline_initialized = True

    def _padded_bytes_per_row(self) -> int:
        raw = self.width * 4
        align = 256
        return ((raw + align - 1) // align) * align

    # --- public API -------------------------------------------------------

    def render(
        self,
        points: list[GaussianPoint],
        camera: SimpleCamera,
        observer: ObserverState | None = None,
    ) -> np.ndarray:
        """Render ``points`` to a ``(H, W, 3)`` uint8 image.

        Raises :class:`RuntimeError` if the WebGPU device isn't real.
        """
        if self.device.get_backend() != "webgpu" or self.device.device is None:
            raise RuntimeError(
                "WebGPUSplatRenderer: WebGPU device unavailable; "
                "use the CPU fallback instead"
            )
        camera.validate()
        if not self._pipeline_initialized:
            self.initialize_pipeline()

        n = len(points)
        if n == 0:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Upload Gaussian points + uniforms.
        gaussian_buf = WebGPUGaussianBuffer()
        gaussian_buf.upload(points, self.device)
        if gaussian_buf.gpu_buffer is None:
            raise RuntimeError(
                "WebGPUSplatRenderer: Gaussian buffer upload failed"
            )
        self._write_uniforms(camera, observer)

        try:
            self._run_render_pass(gaussian_buf, n)
            image = self._read_back_color()
        finally:
            gaussian_buf.release()
        return image

    # --- internals --------------------------------------------------------

    def _write_uniforms(
        self,
        camera: SimpleCamera,
        observer: ObserverState | None,
    ) -> None:
        forward = np.array(
            [camera.forward.x, camera.forward.y, camera.forward.z],
            dtype=np.float32,
        )
        forward /= np.linalg.norm(forward)
        up_raw = np.array(
            [camera.up.x, camera.up.y, camera.up.z], dtype=np.float32
        )
        right = np.cross(forward, up_raw)
        right_norm = np.linalg.norm(right)
        if right_norm == 0.0:
            raise ValueError("camera forward and up are parallel")
        right = right / right_norm
        up = np.cross(right, forward)

        fov_scale = float(math.tan(math.radians(camera.fov_degrees) / 2.0))
        aspect = float(self.height) / float(self.width)

        cam_pos = np.array(
            [
                camera.position_m.x,
                camera.position_m.y,
                camera.position_m.z,
                fov_scale,
            ],
            dtype=np.float32,
        )
        forward_v4 = np.array([*forward, aspect], dtype=np.float32)
        up_v4 = np.array([*up, float(self.width)], dtype=np.float32)
        right_v4 = np.array([*right, float(self.height)], dtype=np.float32)

        if observer is not None:
            beta = float(observer.beta())
            warp = float(observer.warp_factor)
            enable_warp = 1.0 if warp > 0.0 else 0.0
        else:
            beta = 0.0
            warp = 1.0
            enable_warp = 0.0
        params = np.array(
            [beta, warp, enable_warp, enable_warp], dtype=np.float32
        )

        # GR uniforms (zero-filled when GR is off so the GR shader still
        # works as a no-op).
        from cosmic_engine.physics.nbody import GRAVITATIONAL_CONSTANT
        from cosmic_engine.core.units import SPEED_OF_LIGHT_M_S

        if self._black_hole is not None:
            bh_pos_v4 = np.array(
                [
                    self._black_hole.position[0],
                    self._black_hole.position[1],
                    self._black_hole.position[2],
                    self._black_hole.mass_kg,
                ],
                dtype=np.float32,
            )
            r_s = self._black_hole.schwarzschild_radius()
        else:
            bh_pos_v4 = np.zeros(4, dtype=np.float32)
            r_s = 0.0
        deflection_scale = (
            4.0 * GRAVITATIONAL_CONSTANT / (SPEED_OF_LIGHT_M_S * SPEED_OF_LIGHT_M_S)
        )
        bh_params_v4 = np.array(
            [
                1.0 if self._enable_lensing else 0.0,
                float(r_s),
                float(deflection_scale),
                0.0,
            ],
            dtype=np.float32,
        )

        payload = np.concatenate(
            [cam_pos, forward_v4, up_v4, right_v4, params, bh_pos_v4, bh_params_v4]
        ).astype(np.float32)
        # Pad to the uniform buffer size.
        padded = np.zeros(_UNIFORM_SIZE_BYTES // 4, dtype=np.float32)
        padded[: payload.size] = payload
        self.device.queue.write_buffer(
            self._uniform_buffer, 0, padded.tobytes()
        )

    def _run_render_pass(
        self,
        gaussian_buf: WebGPUGaussianBuffer,
        instance_count: int,
    ) -> None:
        import wgpu

        wgpu_dev = self.device.device
        bind_group_layout = wgpu_dev.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX
                    | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.VERTEX,
                    "buffer": {"type": wgpu.BufferBindingType.read_only_storage},
                },
            ]
        )
        pipeline_layout = wgpu_dev.create_pipeline_layout(
            bind_group_layouts=[bind_group_layout]
        )
        pipeline = wgpu_dev.create_render_pipeline(
            layout=pipeline_layout,
            vertex={
                "module": self._shader_module,
                "entry_point": "vs_main",
            },
            fragment={
                "module": self._shader_module,
                "entry_point": "fs_main",
                "targets": [
                    {
                        "format": wgpu.TextureFormat.rgba8unorm,
                        "blend": {
                            "color": {
                                "operation": wgpu.BlendOperation.add,
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one,
                            },
                            "alpha": {
                                "operation": wgpu.BlendOperation.add,
                                "src_factor": wgpu.BlendFactor.one,
                                "dst_factor": wgpu.BlendFactor.one,
                            },
                        },
                    }
                ],
            },
            primitive={"topology": wgpu.PrimitiveTopology.triangle_list},
        )

        bind_group = wgpu_dev.create_bind_group(
            layout=bind_group_layout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self._uniform_buffer,
                        "offset": 0,
                        "size": _UNIFORM_SIZE_BYTES,
                    },
                },
                {
                    "binding": 1,
                    "resource": {
                        "buffer": gaussian_buf.gpu_buffer,
                        "offset": 0,
                        "size": gaussian_buf.gpu_buffer.size,
                    },
                },
            ],
        )

        encoder = wgpu_dev.create_command_encoder()
        view = self._color_texture.create_view()
        render_pass = encoder.begin_render_pass(
            color_attachments=[
                {
                    "view": view,
                    "load_op": wgpu.LoadOp.clear,
                    "store_op": wgpu.StoreOp.store,
                    "clear_value": (0.0, 0.0, 0.0, 1.0),
                }
            ]
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.draw(6, instance_count, 0, 0)
        render_pass.end()

        bytes_per_row = self._padded_bytes_per_row()
        encoder.copy_texture_to_buffer(
            {
                "texture": self._color_texture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": self._readback_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self.height,
            },
            (self.width, self.height, 1),
        )
        wgpu_dev.queue.submit([encoder.finish()])

    def _read_back_color(self) -> np.ndarray:
        import wgpu

        bytes_per_row = self._padded_bytes_per_row()
        self._readback_buffer.map_sync(wgpu.MapMode.READ)
        try:
            view = self._readback_buffer.read_mapped()
            arr = np.frombuffer(view, dtype=np.uint8).copy()
        finally:
            self._readback_buffer.unmap()
        # Strip per-row padding.
        arr = arr.reshape(self.height, bytes_per_row)[:, : self.width * 4]
        rgba = arr.reshape(self.height, self.width, 4)
        return rgba[..., :3].copy()

    def release(self) -> None:
        """Release persistent GPU resources. Idempotent."""
        for resource in (
            self._uniform_buffer,
            self._color_texture,
            self._readback_buffer,
        ):
            if resource is not None:
                try:
                    resource.destroy()
                except Exception:
                    pass
        self._uniform_buffer = None
        self._color_texture = None
        self._readback_buffer = None
        self._pipeline_initialized = False
