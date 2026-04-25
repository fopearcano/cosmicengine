"""Tests for the RunPod deployment scaffold.

The cosmic_engine.distributed package is exercised by
test_lambda_deployment_config.py (the engine layer is cloud-
agnostic). These tests just verify the RunPod-specific files exist,
are well-formed, and use the same env-var contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmic_engine.distributed import DistributedConfig
from cosmic_engine.distributed.entrypoints import main as entrypoints_main


def _runpod_root() -> Path:
    return Path(__file__).resolve().parent.parent / "deploy" / "runpod"


# --- file presence ---------------------------------------------------


def test_runpod_dir_exists():
    assert _runpod_root().is_dir()


def test_runpod_dockerfiles_exist():
    root = _runpod_root() / "docker"
    for name in (
        "Dockerfile.runtime",
        "Dockerfile.worker",
        "Dockerfile.viewer",
        "Dockerfile.training",
    ):
        assert (root / name).is_file(), f"missing {name}"


def test_runpod_start_scripts_exist_and_are_executable():
    root = _runpod_root() / "scripts"
    for name in (
        "start_runtime.sh",
        "start_worker.sh",
        "start_viewer.sh",
        "start_training.sh",
        "healthcheck.sh",
    ):
        path = root / name
        assert path.is_file(), f"missing {name}"
        assert path.stat().st_mode & 0o100, f"{name} not executable"


def test_runpod_env_example_files_exist():
    root = _runpod_root() / "config"
    for name in (
        "runtime.env.example",
        "worker.env.example",
        "ray_cluster.yaml.example",
    ):
        assert (root / name).is_file(), f"missing {name}"


def test_runpod_docker_compose_exists():
    assert (_runpod_root() / "docker-compose.runpod.yml").is_file()


def test_runpod_readme_exists():
    assert (_runpod_root() / "README.md").is_file()


# --- env example contracts ------------------------------------------


def _parse_env_example(path: Path) -> dict[str, str]:
    """Tiny .env-style parser that ignores comments and blank lines."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def test_runpod_runtime_env_contains_required_vars():
    env = _parse_env_example(
        _runpod_root() / "config" / "runtime.env.example"
    )
    for key in (
        "COSMIC_ROLE",
        "COSMIC_RAY_ADDRESS",
        "COSMIC_RUNTIME_PORT",
        "COSMIC_DATA_DIR",
        "COSMIC_MODEL_DIR",
        "COSMIC_OUTPUT_DIR",
        "COSMIC_ENABLE_GPU",
    ):
        assert key in env, f"runtime env missing {key}"
    assert env["COSMIC_ROLE"] == "runtime"


def test_runpod_worker_env_contains_required_vars():
    env = _parse_env_example(
        _runpod_root() / "config" / "worker.env.example"
    )
    for key in (
        "COSMIC_ROLE",
        "COSMIC_HEAD_IP",
        "COSMIC_HEAD_PORT",
        "COSMIC_RAY_ADDRESS",
        "COSMIC_DATA_DIR",
        "COSMIC_MODEL_DIR",
        "COSMIC_OUTPUT_DIR",
        "COSMIC_ENABLE_GPU",
    ):
        assert key in env, f"worker env missing {key}"
    assert env["COSMIC_ROLE"] == "worker"


def test_runpod_runtime_env_loads_through_distributed_config():
    """The runtime.env.example must produce a valid DistributedConfig."""
    env_path = _runpod_root() / "config" / "runtime.env.example"
    env = _parse_env_example(env_path)
    cfg = DistributedConfig.from_env(env=env)
    cfg.validate()
    assert cfg.role == "runtime"
    assert cfg.runtime_port == 8765


def test_runpod_worker_env_role_is_worker():
    env = _parse_env_example(
        _runpod_root() / "config" / "worker.env.example"
    )
    cfg = DistributedConfig.from_env(env=env)
    # Worker validates fine — we don't *require* a real address in
    # the example file; placeholders are documented inline.
    cfg.validate()
    assert cfg.role == "worker"


# --- dockerfiles content sanity -------------------------------------


def _read(path: Path) -> str:
    return path.read_text()


def test_runtime_dockerfile_uses_runtime_script():
    text = _read(_runpod_root() / "docker" / "Dockerfile.runtime")
    assert "start_runtime.sh" in text
    assert "EXPOSE 8765" in text


def test_worker_dockerfile_uses_worker_script_and_cuda():
    text = _read(_runpod_root() / "docker" / "Dockerfile.worker")
    assert "start_worker.sh" in text
    assert "nvidia/cuda" in text


def test_viewer_dockerfile_uses_viewer_script():
    text = _read(_runpod_root() / "docker" / "Dockerfile.viewer")
    assert "start_viewer.sh" in text


def test_training_dockerfile_uses_devel_cuda():
    text = _read(_runpod_root() / "docker" / "Dockerfile.training")
    assert "start_training.sh" in text
    assert "devel" in text  # CUDA devel base, not just runtime


def test_dockerfiles_install_distributed_extra():
    for name in (
        "Dockerfile.runtime",
        "Dockerfile.worker",
        "Dockerfile.training",
    ):
        text = _read(_runpod_root() / "docker" / name)
        assert "[distributed]" in text, (
            f"{name} should install the distributed extra"
        )


# --- docker-compose sanity ------------------------------------------


def test_compose_declares_expected_services():
    text = _read(_runpod_root() / "docker-compose.runpod.yml")
    for service in (
        "ray-head:",
        "worker-ai:",
        "worker-render:",
        "runtime:",
        "viewer:",
    ):
        assert service in text, f"compose missing service {service}"


def test_compose_uses_runpod_dockerfiles():
    text = _read(_runpod_root() / "docker-compose.runpod.yml")
    for df in (
        "deploy/runpod/docker/Dockerfile.runtime",
        "deploy/runpod/docker/Dockerfile.worker",
        "deploy/runpod/docker/Dockerfile.viewer",
    ):
        assert df in text, f"compose should reference {df}"


def test_compose_mounts_workspace_volumes():
    text = _read(_runpod_root() / "docker-compose.runpod.yml")
    for path in (
        "/workspace/data",
        "/workspace/models",
        "/workspace/outputs",
    ):
        assert path in text, f"compose missing {path} mount"


# --- start scripts dispatch through the entrypoints ------------------


def test_start_scripts_invoke_the_distributed_entrypoint():
    """Every start script should hand off to
    ``python -m cosmic_engine.distributed.entrypoints``."""
    for name in (
        "start_runtime.sh",
        "start_worker.sh",
        "start_viewer.sh",
        "start_training.sh",
        "healthcheck.sh",
    ):
        text = _read(_runpod_root() / "scripts" / name)
        assert "cosmic_engine.distributed.entrypoints" in text, (
            f"{name} should dispatch to the entrypoints module"
        )


# --- README / docs / pyproject coherence -----------------------------


def test_readme_mentions_runpod_deployment_dir():
    """The top-level README points to the RunPod deployment files."""
    readme = (
        Path(__file__).resolve().parent.parent / "README.md"
    ).read_text()
    assert "deploy/runpod" in readme or "RunPod" in readme


def test_pyproject_still_declares_distributed_extra():
    pyproject = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text()
    assert "distributed" in pyproject and "ray" in pyproject


# --- entrypoints health round-trip works without GPU/Ray ------------


def test_health_entrypoint_returns_zero(capsys):
    rc = entrypoints_main(["health"])
    assert rc == 0
    parsed = json.loads(capsys.readouterr().out)
    for k in ("config", "cuda", "ray", "directories"):
        assert k in parsed
