from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPT = Path("scripts/run_official_offline_client.sh").resolve()


def _run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
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
