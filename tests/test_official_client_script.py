from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


SCRIPT = Path("scripts/run_official_offline_client.sh").resolve()


def _run(tmp_path: Path, **overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in (
        "TEAM_SORTING_CLIENT_GPUS",
        "TEAM_SORTING_RUNTIME_VOLUME",
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


def _fingerprint(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def test_default_dry_run_is_offline_runtime_install_and_observe_only(tmp_path):
    result = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)))
    output = result.stdout
    assert result.returncode == 0
    assert "Client image: material_sorting:offline-client" in output
    assert "Client image ID: <not-inspected:dry-run>" in output
    assert "GPU behavior: --gpus all" in output and "--gpus all" in output
    assert "team_sorting_offline_client_runtime_v1:/opt/team_sorting_runtime" in output
    assert "/workspace/baseline:ro" in output
    assert "/workspace/official:ro" in output
    assert "ROS prefix: /opt/team_sorting_runtime/prefix/local" in output
    assert "Pip cache: /opt/team_sorting_runtime/pip-cache" in output
    assert "--no-index" in output
    assert "--no-deps" in output
    assert "--no-build-isolation" in output
    assert "observe_only=true" in output
    assert "colcon build" not in output


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


def test_clean_build_and_custom_runtime_volume_are_transmitted(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_CLEAN_BUILD="1",
        TEAM_SORTING_RUNTIME_VOLUME="custom_runtime",
    )
    assert result.returncode == 0
    assert "Runtime volume: custom_runtime -> /opt/team_sorting_runtime" in result.stdout
    assert "Clean build: 1" in result.stdout
    assert "custom_runtime:/opt/team_sorting_runtime" in result.stdout
    assert "TEAM_SORTING_CLEAN_BUILD=1" in result.stdout
    for path in ("src", "prefix", "pip-cache", "source.sha256"):
        assert f"/opt/team_sorting_runtime/{path}" in result.stdout
    assert "/workspace/baseline:ro" in result.stdout


def test_invalid_clean_build_value_fails_before_docker(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_CLEAN_BUILD="yes",
    )
    assert result.returncode != 0
    assert "只能是0或1" in result.stderr


def test_script_has_no_colcon_build_and_uses_only_offline_pip():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "colcon build" not in source
    assert "command -v colcon" not in source
    pip_start = source.index("python3 -m pip install")
    launch = source.index("ros2 launch team_sorting team.launch.xml", pip_start)
    pip_block = source[pip_start:launch]
    for option in ("--no-index", "--no-deps", "--no-build-isolation"):
        assert option in pip_block
    assert "--prefix \"${TEAM_SORTING_PREFIX_ROOT}\"" in pip_block


def test_required_installed_copy_paths_are_fail_closed():
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "share/ament_index/resource_index/packages/team_sorting",
        "lib/python3.10/dist-packages/team_sorting",
        "bin/perception_node",
        "bin/team_client_node",
        "bin/dataset_recorder_node",
        "share/team_sorting/launch/team.launch.xml",
        "share/team_sorting/config/config.yaml",
    )
    for relative in required:
        assert f'${{TEAM_SORTING_ROS_PREFIX}}/{relative}' in source
    for executable in ("perception_node", "team_client_node", "dataset_recorder_node"):
        assert f'${{TEAM_SORTING_LIBEXEC_DIR}}/{executable}' in source


def test_prefix_environment_precedes_filtered_host_source_paths():
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'export AMENT_PREFIX_PATH="${TEAM_SORTING_ROS_PREFIX}:${AMENT_PREFIX_PATH:-}"' in source
    assert 'export PATH="${TEAM_SORTING_ROS_PREFIX}/bin:${PATH}"' in source
    py_export = (
        'export PYTHONPATH="${TEAM_SORTING_ROS_PREFIX}/lib/python3.10/dist-packages'
        '${FILTERED_PYTHONPATH:+:${FILTERED_PYTHONPATH}}"'
    )
    assert py_export in source
    assert 'blocked = {"/workspace/baseline", "/workspace/baseline/team_sorting"}' in source


def test_import_ros_prefix_and_node_authenticity_checks_precede_launch():
    source = SCRIPT.read_text(encoding="utf-8")
    launch = source.index("ros2 launch team_sorting team.launch.xml")
    assert source.index("cd /tmp") < launch
    assert source.index("import team_sorting; print(team_sorting.__file__)") < launch
    assert source.index('"${TEAM_SORTING_ROS_PREFIX}"/*') < launch
    assert source.index("ros2 pkg prefix team_sorting") < launch
    assert source.index("ros2 pkg executables team_sorting") < launch


def test_fingerprint_is_content_based_and_cache_paths_cover_required_sources(tmp_path):
    source = SCRIPT.read_text(encoding="utf-8")
    for path in ("setup.py", "setup.cfg", "package.xml", "resource", "team_sorting", "launch", "config"):
        assert f'Path("{path}")' in source
    assert "path.read_bytes()" in source
    assert "source.sha256" in source
    assert "runtime cache hit: source unchanged; skipping pip install" in source
    assert "runtime cache miss/source changed" in source
    assert 'if [[ "${CACHED_SHA256}" == "${SOURCE_SHA256}" ]] && pip_installation_complete' in source

    tree = tmp_path / "source"
    tree.mkdir()
    (tree / "setup.py").write_text("one", encoding="utf-8")
    first = _fingerprint(tree)
    assert _fingerprint(tree) == first
    (tree / "setup.py").write_text("two", encoding="utf-8")
    assert _fingerprint(tree) != first


def test_first_install_records_fingerprint_only_after_pip_and_validation():
    source = SCRIPT.read_text(encoding="utf-8")
    miss = source.index("runtime cache miss/source changed")
    pip_install = source.index("python3 -m pip install", miss)
    validate = source.index("ensure_ros_libexec_layout", pip_install)
    save = source.index('> "${FINGERPRINT_FILE}"', validate)
    assert miss < pip_install < validate < save


def test_ros_libexec_layout_uses_validated_symlinks():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "TEAM_SORTING_LIBEXEC_DIR=${TEAM_SORTING_ROS_PREFIX}/lib/team_sorting" in source
    function_start = source.index("ensure_ros_libexec_layout()")
    function_end = source.index("\n}\n", function_start)
    function = source[function_start:function_end]
    assert 'mkdir -p "${TEAM_SORTING_LIBEXEC_DIR}"' in function
    assert 'source_entry="${TEAM_SORTING_ROS_PREFIX}/bin/${executable}"' in function
    assert 'if [[ ! -x "${source_entry}" ]]' in function
    assert 'ln -sfn "${source_entry}" "${libexec_entry}"' in function
    assert 'if [[ ! -x "${libexec_entry}" ]]' in function


def test_cache_hit_repairs_and_validates_libexec_before_skipping_pip():
    source = SCRIPT.read_text(encoding="utf-8")
    cache_condition = source.index(
        'if [[ "${CACHED_SHA256}" == "${SOURCE_SHA256}" ]] && pip_installation_complete'
    )
    ensure = source.index("ensure_ros_libexec_layout", cache_condition)
    complete = source.index("installation_complete", ensure)
    hit = source.index("runtime cache hit: source unchanged; skipping pip install", complete)
    else_branch = source.index("\nelse\n", hit)
    assert cache_condition < ensure < complete < hit < else_branch


def test_launch_uses_the_three_standard_ros_package_executables():
    root = ET.parse("launch/team.launch.xml").getroot()
    nodes = {
        node.attrib["exec"]
        for node in root.findall("node")
        if node.attrib.get("pkg") == "team_sorting"
    }
    assert nodes == {"perception_node", "team_client_node", "dataset_recorder_node"}


def test_legacy_colcon_volume_variable_maps_to_runtime_with_warning(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_COLCON_CACHE_VOLUME="legacy_volume",
    )
    assert result.returncode == 0
    assert "已弃用" in result.stderr
    assert "Runtime volume: legacy_volume -> /opt/team_sorting_runtime" in result.stdout
    assert "legacy_volume:/opt/team_sorting_runtime" in result.stdout


def test_new_runtime_volume_takes_precedence_over_legacy_variable(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_RUNTIME_VOLUME="new_volume",
        TEAM_SORTING_COLCON_CACHE_VOLUME="legacy_volume",
    )
    assert result.returncode == 0
    assert "已弃用" not in result.stderr
    assert "Runtime volume: new_volume -> /opt/team_sorting_runtime" in result.stdout


def test_empty_runtime_volume_is_rejected(tmp_path):
    result = _run(
        tmp_path,
        MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)),
        TEAM_SORTING_RUNTIME_VOLUME="",
    )
    assert result.returncode != 0
    assert "TEAM_SORTING_RUNTIME_VOLUME不能为空" in result.stderr


def test_dry_run_does_not_inspect_or_execute_docker(tmp_path):
    result = _run(tmp_path, MATERIAL_SORTING_OFFICIAL_ROOT=str(_official_root(tmp_path)))
    assert result.returncode == 0
    assert "Client image ID: <not-inspected:dry-run>" in result.stdout
    source = SCRIPT.read_text(encoding="utf-8")
    dry_branch = source.index('if [[ "${DRY_RUN}" == "1" ]]; then\n    IMAGE_ID=')
    inspect = source.index("docker image inspect", dry_branch)
    exec_docker = source.index('exec "${command[@]}"', inspect)
    assert dry_branch < inspect < exec_docker
