from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path("scripts/run_official_offline_client.sh").resolve()


def _run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "TEAM_SORTING_CLIENT_GPUS",
        "TEAM_SORTING_COLCON_CACHE_VOLUME",
        "TEAM_SORTING_CLEAN_BUILD",
    ):
        env.pop(name, None)
    env.update({"DRY_RUN": "1", **overrides})
    return subprocess.run(
        ["bash", str(SCRIPT)], env=env, text=True, capture_output=True, check=False
    )


def _official_root(tmp_path: Path) -> Path:
    root = tmp_path / "official"
    server = root / "examples/material_sorting/material_sorting_server.py"
    server.parent.mkdir(parents=True)
    server.write_text("# official marker\n", encoding="utf-8")
    return root


def test_official_root_is_required_before_docker(tmp_path):
    result = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT="")
    assert result.returncode != 0
    assert "MATERIAL_SORTING_OFFICIAL_ROOT为必填" in result.stderr
    assert "docker" not in result.stdout


def test_missing_or_incomplete_official_root_fails_before_docker(tmp_path):
    missing = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT=str(tmp_path / "missing"))
    assert missing.returncode != 0 and "目录不存在" in missing.stderr
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    result = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT=str(incomplete))
    assert result.returncode != 0 and "缺少关键文件" in result.stderr


def test_default_dry_run_includes_all_gpus_and_observe_only(tmp_path):
    result = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)))
    assert result.returncode == 0
    assert "GPU behavior: --gpus all" in result.stdout
    assert "--gpus all" in result.stdout
    assert "observe_only=true" in result.stdout
    assert "team_sorting_offline_client_colcon_v1:/opt/team_sorting_colcon" in result.stdout
    assert "/opt/team_sorting_colcon/build" in result.stdout
    assert "/opt/team_sorting_colcon/install" in result.stdout
    assert "/opt/team_sorting_colcon/log" in result.stdout
    assert "--symlink-install" in result.stdout
    assert "Clean build: 0" in result.stdout


def test_none_or_empty_gpu_setting_omits_docker_gpus_argument(tmp_path):
    root = str(_official_root(tmp_path))
    for value in ("none", ""):
        result = _run(
            tmp_path,
            MATERIAL_SORTING_OFFICIAL_ROOT=root,
            TEAM_SORTING_CLIENT_GPUS=value,
        )
        assert result.returncode == 0
        assert "GPU behavior: disabled (no --gpus argument)" in result.stdout
        command_line = next(line for line in result.stdout.splitlines() if line.startswith("Command:"))
        assert "--gpus" not in command_line


def test_custom_gpu_value_is_passed_as_one_docker_argument(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_CLIENT_GPUS="device=0,1",
    )
    assert result.returncode == 0
    assert "GPU behavior: --gpus device=0,1" in result.stdout
    assert "--gpus device=0\\,1" in result.stdout


def test_host_keeps_nounset_but_container_ros_setup_does_not_use_it():
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    container = source[source.index("'set -eo pipefail") :]
    ros_source = container.index("source /opt/ros/humble/setup.bash")
    assert "set -eo pipefail" in container[:ros_source]
    assert "set -u" not in container[:ros_source]
    assert "set -euo pipefail" not in container[:ros_source]
    assert "source /opt/team_sorting_colcon/install/setup.bash" in container


def test_clean_build_and_custom_cache_volume_are_transmitted(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_CLEAN_BUILD="1",
        TEAM_SORTING_COLCON_CACHE_VOLUME="custom_colcon_cache",
    )
    assert result.returncode == 0
    assert "Colcon cache volume: custom_colcon_cache -> /opt/team_sorting_colcon" in result.stdout
    assert "Clean build: 1" in result.stdout
    assert "custom_colcon_cache:/opt/team_sorting_colcon" in result.stdout
    assert "TEAM_SORTING_CLEAN_BUILD=1" in result.stdout
    assert "rm -rf /opt/team_sorting_colcon/build" in result.stdout


def test_invalid_clean_build_value_fails_before_docker(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_CLEAN_BUILD="yes",
    )
    assert result.returncode != 0
    assert "只能是0或1" in result.stderr
