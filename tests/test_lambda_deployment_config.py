"""Tests for the Phase 39 Lambda deployment / distributed package.

Nothing in this file requires Ray, CUDA, Docker, or actual Lambda
Cloud — the whole layer is designed to import cleanly on a barebones
machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cosmic_engine.distributed import (
    DistributedConfig,
    check_cuda_available,
    check_data_dirs,
    check_ray_status,
    health_report,
    is_ray_available,
)
from cosmic_engine.distributed.tasks import (
    process_catalog_chunk,
    render_gaussian_frame,
    run_ai_warp_batch,
    run_density_reconstruction,
    run_physics_step,
)


# --- DistributedConfig defaults / from_env ----------------------------


def test_config_defaults_validate():
    cfg = DistributedConfig()
    cfg.validate()
    assert cfg.role == "runtime"
    assert cfg.runtime_host == "0.0.0.0"
    assert cfg.runtime_port == 8765
    assert cfg.head_port == 6379
    assert cfg.data_dir == "/workspace/data"
    assert cfg.model_dir == "/workspace/models"
    assert cfg.output_dir == "/workspace/outputs"


def test_config_from_env_uses_defaults_for_empty_env():
    cfg = DistributedConfig.from_env(env={})
    cfg.validate()
    assert cfg.role == "runtime"
    assert cfg.ray_address is None
    assert cfg.enable_gpu is False


def test_config_from_env_reads_known_vars():
    env = {
        "COSMIC_ROLE": "worker",
        "COSMIC_RAY_ADDRESS": "ray-head:6379",
        "COSMIC_HEAD_IP": "10.0.0.1",
        "COSMIC_HEAD_PORT": "6379",
        "COSMIC_RUNTIME_PORT": "9000",
        "COSMIC_DATA_DIR": "/data",
        "COSMIC_MODEL_DIR": "/models",
        "COSMIC_OUTPUT_DIR": "/out",
        "COSMIC_ENABLE_GPU": "true",
        "COSMIC_ENABLE_VIEWER": "false",
        "COSMIC_ENABLE_TRAINING": "yes",
        "COSMIC_NODE_NAME": "lambda-1",
        "COSMIC_LAMBDA_INSTANCE_TYPE": "gpu_1x_a10",
        "COSMIC_NUM_GPUS": "2",
        "COSMIC_NUM_CPUS": "16",
        "COSMIC_OBJECT_STORE_GB": "8.5",
    }
    cfg = DistributedConfig.from_env(env=env)
    cfg.validate()
    assert cfg.role == "worker"
    assert cfg.ray_address == "ray-head:6379"
    assert cfg.head_ip == "10.0.0.1"
    assert cfg.runtime_port == 9000
    assert cfg.enable_gpu is True
    assert cfg.enable_viewer is False
    assert cfg.enable_training is True
    assert cfg.num_gpus == 2.0
    assert cfg.num_cpus == 16
    assert cfg.object_store_memory_gb == pytest.approx(8.5)
    assert cfg.instance_type == "gpu_1x_a10"
    assert cfg.node_name == "lambda-1"


@pytest.mark.parametrize("role", ["head", "worker", "runtime", "viewer", "training"])
def test_config_accepts_lambda_roles(role):
    cfg = DistributedConfig(role=role)
    cfg.validate()


def test_config_rejects_unknown_role():
    cfg = DistributedConfig(role="overseer")
    with pytest.raises(ValueError):
        cfg.validate()


def test_config_rejects_invalid_ports():
    with pytest.raises(ValueError):
        DistributedConfig(head_port=0).validate()
    with pytest.raises(ValueError):
        DistributedConfig(runtime_port=70_000).validate()


def test_config_rejects_negative_resources():
    with pytest.raises(ValueError):
        DistributedConfig(num_gpus=-1.0).validate()
    with pytest.raises(ValueError):
        DistributedConfig(num_cpus=-1).validate()
    with pytest.raises(ValueError):
        DistributedConfig(object_store_memory_gb=-0.5).validate()


def test_config_summary_is_json_serializable():
    cfg = DistributedConfig(role="worker", node_name="n1")
    out = cfg.summary()
    json.dumps(out)  # would raise if non-serializable
    assert out["role"] == "worker"
    assert out["node_name"] == "n1"


# --- ray_runtime / health ---------------------------------------------


def test_is_ray_available_returns_bool():
    assert isinstance(is_ray_available(), bool)


def test_check_cuda_available_does_not_raise():
    info = check_cuda_available()
    assert "available" in info
    assert isinstance(info["available"], bool)


def test_check_ray_status_returns_dict_without_ray():
    info = check_ray_status()
    assert isinstance(info, dict)
    assert "available" in info
    assert "initialized" in info
    if not is_ray_available():
        assert info["available"] is False
        assert info["initialized"] is False


def test_check_data_dirs_reports_each_dir(tmp_path: Path):
    # Use tmp_path so this works in any CI environment.
    cfg = DistributedConfig(
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        output_dir=str(tmp_path / "outputs"),
    )
    info = check_data_dirs(cfg)
    assert set(info.keys()) == {"data_dir", "model_dir", "output_dir"}
    for key in info:
        assert "exists" in info[key]
        assert "writable" in info[key]


def test_health_report_returns_dict_without_gpu_or_ray():
    cfg = DistributedConfig.from_env(env={})
    report = health_report(cfg)
    assert isinstance(report, dict)
    for k in (
        "timestamp",
        "hostname",
        "python_version",
        "platform",
        "config",
        "cuda",
        "ray",
        "directories",
    ):
        assert k in report
    json.dumps(report)  # round-trip


# --- task fallback contracts ------------------------------------------


def test_process_catalog_chunk_missing_path_returns_error():
    out = process_catalog_chunk("/no/such/file.csv")
    assert out["ok"] is False
    assert "error" in out


def test_process_catalog_chunk_empty_arg_returns_error():
    out = process_catalog_chunk("")
    assert out["ok"] is False


def test_run_physics_step_handles_bad_payload():
    # No keys at all — must not raise; must return error dict.
    out = run_physics_step({})
    assert out["ok"] is False
    assert "error" in out


def test_run_ai_warp_batch_requires_model_path():
    out = run_ai_warp_batch({"directions": [[1, 0, 0]], "brightnesses": [1.0],
                             "observer_velocity": [0.0, 0.0, 0.0]})
    assert out["ok"] is False
    assert "model_path" in out["error"] or "error" in out


def test_run_density_reconstruction_requires_model_path():
    out = run_density_reconstruction({"positions": [[0.0, 0.0, 0.0]],
                                       "luminosities": [1.0]})
    assert out["ok"] is False


def test_render_gaussian_frame_handles_bad_payload():
    out = render_gaussian_frame({})
    assert out["ok"] is False
    assert "error" in out


def test_task_results_are_json_serializable():
    # Each error path must still produce JSON-safe output.
    for fn, payload in [
        (process_catalog_chunk, ""),
        (run_physics_step, {}),
        (run_ai_warp_batch, {}),
        (run_density_reconstruction, {}),
        (render_gaussian_frame, {}),
    ]:
        out = fn(payload)
        json.dumps(out)


# --- entrypoints: importable + dispatch -------------------------------


def test_entrypoints_importable():
    """Importing the module must never start a service."""
    from cosmic_engine.distributed import entrypoints

    for name in (
        "run_head_node",
        "run_worker_node",
        "run_runtime_service",
        "run_viewer_service",
        "run_training_service",
        "main",
    ):
        assert hasattr(entrypoints, name)


def test_entrypoints_main_health_role(capsys):
    from cosmic_engine.distributed.entrypoints import main

    rc = main(["health"])
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert "config" in parsed
    assert "cuda" in parsed
    assert "ray" in parsed


def test_entrypoints_main_unknown_role_returns_error(capsys):
    from cosmic_engine.distributed.entrypoints import main

    rc = main(["nonsense_role"])
    assert rc == 2


def test_run_head_node_without_ray_is_safe():
    """run_head_node must not raise on a barebones machine.

    When Ray isn't installed, it should print instructions and
    return a dict with ``ray_started=False``.
    """
    from cosmic_engine.distributed.entrypoints import run_head_node

    cfg = DistributedConfig(role="head")
    out = run_head_node(cfg)
    assert out["role"] == "head"
    assert isinstance(out["ray_started"], bool)


def test_run_worker_node_without_address_returns_error():
    from cosmic_engine.distributed.entrypoints import run_worker_node

    cfg = DistributedConfig(role="worker", ray_address=None, head_ip=None)
    out = run_worker_node(cfg)
    assert out["ray_started"] is False
    assert "error" in out


# --- deployment file presence -----------------------------------------


def _deploy_root() -> Path:
    return Path(__file__).resolve().parent.parent / "deploy" / "lambda"


def test_dockerfiles_exist():
    root = _deploy_root() / "docker"
    for name in (
        "Dockerfile.runtime",
        "Dockerfile.worker",
        "Dockerfile.viewer",
        "Dockerfile.training",
    ):
        assert (root / name).is_file(), f"missing {name}"


def test_start_scripts_exist_and_are_executable():
    root = _deploy_root() / "scripts"
    for name in (
        "start_runtime.sh",
        "start_worker.sh",
        "start_viewer.sh",
        "start_training.sh",
        "healthcheck.sh",
    ):
        path = root / name
        assert path.is_file(), f"missing {name}"
        # Executable bit on owner.
        assert path.stat().st_mode & 0o100, f"{name} not executable"


def test_env_example_files_exist():
    root = _deploy_root() / "config"
    for name in (
        "runtime.env.example",
        "worker.env.example",
        "lambda_cluster.env.example",
        "ray_cluster.yaml.example",
    ):
        assert (root / name).is_file(), f"missing {name}"


def test_docker_compose_exists():
    assert (_deploy_root() / "docker-compose.lambda.yml").is_file()


def test_lambda_readme_exists():
    assert (_deploy_root() / "README.md").is_file()


def test_pyproject_declares_distributed_extra():
    pyproject = (
        Path(__file__).resolve().parent.parent / "pyproject.toml"
    ).read_text()
    assert "distributed" in pyproject
    assert "ray" in pyproject
