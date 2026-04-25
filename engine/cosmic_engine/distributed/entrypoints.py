"""Per-role service entrypoints for Lambda Cloud deployment.

Each function reads a :class:`DistributedConfig` from the environment,
prints a human-readable banner, brings up the requested role, and
blocks until terminated. They're imported as functions (never as a
side effect of importing this module) so unit tests can call them
without spinning up real services.

A tiny ``main()`` dispatcher at the bottom makes
``python -m cosmic_engine.distributed.entrypoints`` work from any
container entrypoint script.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from cosmic_engine.distributed.config import DistributedConfig
from cosmic_engine.distributed.health import health_report


_LOG = logging.getLogger(__name__)


# --- helpers ----------------------------------------------------------


def _print_banner(config: DistributedConfig, role: str) -> None:
    """Emit a single multi-line summary so logs always show what booted."""
    banner = json.dumps(
        {
            "cosmic_engine": "starting",
            "role": role,
            "config": config.summary(),
        },
        indent=2,
        sort_keys=True,
    )
    print("=" * 72)
    print(banner)
    print("=" * 72, flush=True)


def _maybe_init_ray(config: DistributedConfig) -> bool:
    """Try to bring up Ray if a cluster address is configured.

    Returns ``True`` on success, ``False`` if Ray isn't installed or
    the address is empty. Failures don't abort the entrypoint —
    the caller decides whether the role *requires* Ray.
    """
    if not config.ray_address:
        return False
    try:
        from cosmic_engine.distributed.ray_runtime import (
            init_ray,
            is_ray_available,
        )

        if not is_ray_available():
            _LOG.warning(
                "ray not installed; continuing without distributed mode"
            )
            return False
        init_ray(config)
        return True
    except Exception as e:  # pragma: no cover - defensive
        _LOG.warning("ray init failed: %s", e)
        return False


def _block_until_signal(label: str) -> None:
    """Idle in a sleep loop until SIGTERM / KeyboardInterrupt.

    Used by long-running roles (head, worker, training) that don't
    have their own foreground loop.
    """
    print(f"[{label}] entering blocking idle loop", flush=True)
    try:
        while True:
            time.sleep(60.0)
    except KeyboardInterrupt:
        print(f"[{label}] interrupted, shutting down", flush=True)


# --- role entrypoints ------------------------------------------------


def run_head_node(config: DistributedConfig | None = None) -> dict[str, Any]:
    """Start the Ray head process (or print instructions if Ray missing).

    Returns a dict describing what was started — handy for tests that
    invoke the function in a dry-run mode.
    """
    cfg = config or DistributedConfig.from_env()
    cfg.role = "head"
    cfg.validate()
    _print_banner(cfg, "head")
    started = _maybe_init_ray(cfg)
    info = {"role": "head", "ray_started": started}
    if not started:
        print(
            "[head] Ray is not initialized. Start it manually with:\n"
            f"    ray start --head --port={cfg.head_port} "
            f"--dashboard-host=0.0.0.0",
            flush=True,
        )
        return info
    return info


def run_worker_node(config: DistributedConfig | None = None) -> dict[str, Any]:
    """Connect this process to a Ray head and idle as a Ray worker."""
    cfg = config or DistributedConfig.from_env()
    cfg.role = "worker"
    cfg.validate()
    _print_banner(cfg, "worker")

    if not cfg.ray_address and cfg.head_ip:
        cfg.ray_address = f"{cfg.head_ip}:{cfg.head_port}"
    if not cfg.ray_address:
        msg = (
            "worker requires COSMIC_RAY_ADDRESS or COSMIC_HEAD_IP "
            "to know where to attach"
        )
        print(f"[worker] error: {msg}", flush=True)
        return {"role": "worker", "ray_started": False, "error": msg}

    started = _maybe_init_ray(cfg)
    return {"role": "worker", "ray_started": started}


def run_runtime_service(
    config: DistributedConfig | None = None,
    *,
    block: bool = True,
) -> dict[str, Any]:
    """Start the :class:`CosmicRuntime` TCP server.

    ``block=False`` returns the started :class:`RuntimeServer`
    instead of blocking — used by tests that want to inspect state
    without entering the idle loop.
    """
    cfg = config or DistributedConfig.from_env()
    cfg.role = "runtime"
    cfg.validate()
    _print_banner(cfg, "runtime")
    _maybe_init_ray(cfg)

    # Lazy: keep the entrypoint import cheap.
    from cosmic_engine.runtime import (
        CosmicRuntime,
        RuntimeConfig,
        RuntimeServer,
    )

    runtime = CosmicRuntime(
        config=RuntimeConfig(
            enable_physics=False,
            enable_perception=False,
            output_directory=cfg.output_dir,
        )
    )
    # Best-effort: try the bundled sample data so the server has
    # something to serve out of the box.
    try:
        runtime.load_sample_data()
    except Exception as e:  # pragma: no cover - defensive
        print(f"[runtime] load_sample_data skipped: {e}", flush=True)

    server = RuntimeServer(
        runtime,
        host=cfg.runtime_host,
        port=cfg.runtime_port,
    )
    server.start()
    print(
        f"[runtime] listening on {cfg.runtime_host}:{server.port}",
        flush=True,
    )
    if not block:
        return {"role": "runtime", "server": server, "port": server.port}
    try:
        _block_until_signal("runtime")
    finally:
        server.stop()
    return {"role": "runtime", "port": server.port}


def run_viewer_service(config: DistributedConfig | None = None) -> dict[str, Any]:
    """Run the AI Viewer pulling frames from a runtime server.

    The viewer shuts down cleanly when the runtime closes the
    connection — there's no separate stop signal to plumb in.
    """
    cfg = config or DistributedConfig.from_env()
    cfg.role = "viewer"
    cfg.validate()
    _print_banner(cfg, "viewer")

    from ai_viewer import AIViewer, AIViewerConfig
    from ai_viewer.client import RuntimeClient

    target_host = cfg.head_ip or cfg.runtime_host
    if target_host == "0.0.0.0":
        # 0.0.0.0 is a server bind address; clients can't connect to it.
        target_host = "127.0.0.1"
    client = RuntimeClient(host=target_host, port=cfg.runtime_port)
    viewer_config = AIViewerConfig(
        server_host=target_host,
        server_port=cfg.runtime_port,
        output_directory=cfg.output_dir,
        enable_window=False,
    )
    viewer = AIViewer(viewer_config, client)
    print(
        f"[viewer] connecting to {target_host}:{cfg.runtime_port}",
        flush=True,
    )
    processed = viewer.run_loop(max_frames=None)
    return {"role": "viewer", "processed": processed}


def run_training_service(config: DistributedConfig | None = None) -> dict[str, Any]:
    """Print available training entrypoints. Bring up Ray if requested.

    Training is an optional, opt-in node role. We don't auto-launch a
    training run from a long-lived service — just surface the
    documented commands so an operator can shell in and run them.
    """
    cfg = config or DistributedConfig.from_env()
    cfg.role = "training"
    cfg.validate()
    _print_banner(cfg, "training")
    _maybe_init_ray(cfg)
    print(
        "[training] available commands:\n"
        "  python examples/train_spacetime_model.py\n"
        "  python examples/test_trained_spacetime_model.py",
        flush=True,
    )
    _block_until_signal("training")
    return {"role": "training"}


# --- CLI dispatch ----------------------------------------------------


_ENTRYPOINTS = {
    "head": run_head_node,
    "worker": run_worker_node,
    "runtime": run_runtime_service,
    "viewer": run_viewer_service,
    "training": run_training_service,
}


def main(argv: list[str] | None = None) -> int:
    """``python -m cosmic_engine.distributed.entrypoints [role]``.

    With no argv, the role comes from ``COSMIC_ROLE``. The special
    role ``health`` prints a JSON health report and exits.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    cfg = DistributedConfig.from_env()
    role = args[0] if args else cfg.role

    if role == "health":
        print(json.dumps(health_report(cfg), indent=2, sort_keys=True))
        return 0

    fn = _ENTRYPOINTS.get(role)
    if fn is None:
        print(
            f"unknown role {role!r}; expected one of "
            f"{list(_ENTRYPOINTS) + ['health']}",
            file=sys.stderr,
        )
        return 2
    fn(cfg)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual entrypoint
    sys.exit(main())
