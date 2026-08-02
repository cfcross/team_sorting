#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${TEAM_SORTING_CLIENT_IMAGE:-material_sorting:offline-client}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"
OFFICIAL_ROOT="${MATERIAL_SORTING_OFFICIAL_ROOT:-}"
CLIENT_GPUS="${TEAM_SORTING_CLIENT_GPUS-all}"
CLEAN_BUILD="${TEAM_SORTING_CLEAN_BUILD:-0}"
YOLO_CHECKPOINT="${MATERIAL_SORTING_YOLO_CHECKPOINT:-}"
MJCF_PATH="${TEAM_SORTING_MJCF:-}"

if [[ -n "${TEAM_SORTING_RUNTIME_VOLUME+x}" ]]; then
    RUNTIME_VOLUME="${TEAM_SORTING_RUNTIME_VOLUME}"
elif [[ -n "${TEAM_SORTING_COLCON_CACHE_VOLUME:-}" ]]; then
    RUNTIME_VOLUME="${TEAM_SORTING_COLCON_CACHE_VOLUME}"
    printf '警告：TEAM_SORTING_COLCON_CACHE_VOLUME已弃用；请改用TEAM_SORTING_RUNTIME_VOLUME。\n' >&2
else
    RUNTIME_VOLUME="team_sorting_offline_client_runtime_v1"
fi

if [[ -z "${RUNTIME_VOLUME}" ]]; then
    echo "错误：TEAM_SORTING_RUNTIME_VOLUME不能为空" >&2
    exit 1
fi
if [[ "${CLEAN_BUILD}" != "0" && "${CLEAN_BUILD}" != "1" ]]; then
    echo "错误：TEAM_SORTING_CLEAN_BUILD只能是0或1" >&2
    exit 1
fi

validate_optional_path() {
    local label="$1" value="$2"
    if [[ -n "${value}" && ! -e "${value}" ]]; then
        echo "错误：${label}不存在：${value}" >&2
        exit 1
    fi
}
if [[ -z "${OFFICIAL_ROOT}" ]]; then
    echo "错误：MATERIAL_SORTING_OFFICIAL_ROOT为必填宿主机目录。" >&2
    echo "示例：MATERIAL_SORTING_OFFICIAL_ROOT=/path/to/material_sorting_official_offline DRY_RUN=1 $0" >&2
    exit 1
fi
if [[ ! -d "${OFFICIAL_ROOT}" ]]; then
    echo "错误：MATERIAL_SORTING_OFFICIAL_ROOT目录不存在：${OFFICIAL_ROOT}" >&2
    exit 1
fi
OFFICIAL_SERVER_REL="examples/material_sorting/material_sorting_server.py"
if [[ ! -f "${OFFICIAL_ROOT}/${OFFICIAL_SERVER_REL}" ]]; then
    echo "错误：官方源码根目录缺少关键文件：${OFFICIAL_ROOT}/${OFFICIAL_SERVER_REL}" >&2
    exit 1
fi
validate_optional_path MATERIAL_SORTING_YOLO_CHECKPOINT "${YOLO_CHECKPOINT}"
validate_optional_path TEAM_SORTING_MJCF "${MJCF_PATH}"

command=(docker run --rm -it --network host
    -e ROS_DOMAIN_ID=99
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    -e MATERIAL_SORTING_OFFICIAL_ROOT=/workspace/official
    -e "TEAM_SORTING_CLEAN_BUILD=${CLEAN_BUILD}"
    -v "${PROJECT_ROOT}:/workspace/baseline:ro"
    -v "${OFFICIAL_ROOT}:/workspace/official:ro"
    -v "${RUNTIME_VOLUME}:/opt/team_sorting_runtime")
case "${CLIENT_GPUS}" in
    ""|none)
        GPU_BEHAVIOR="disabled (no --gpus argument)"
        ;;
    *)
        GPU_BEHAVIOR="--gpus ${CLIENT_GPUS}"
        command+=(--gpus "${CLIENT_GPUS}")
        ;;
esac
if [[ -n "${YOLO_CHECKPOINT}" ]]; then
    command+=(-e MATERIAL_SORTING_YOLO_CHECKPOINT=/workspace/runtime/yolo.pt
              -v "${YOLO_CHECKPOINT}:/workspace/runtime/yolo.pt:ro")
fi
if [[ -n "${MJCF_PATH}" ]]; then
    command+=(-e TEAM_SORTING_MJCF=/workspace/runtime/scene.xml
              -v "${MJCF_PATH}:/workspace/runtime/scene.xml:ro")
fi
command+=("${IMAGE_NAME}" bash -lc
    'set -eo pipefail
source /opt/ros/humble/setup.bash
test -d /workspace/baseline/team_sorting
test -d "${MATERIAL_SORTING_OFFICIAL_ROOT}"
if [[ -n "${MATERIAL_SORTING_YOLO_CHECKPOINT:-}" ]]; then test -f "${MATERIAL_SORTING_YOLO_CHECKPOINT}"; fi
if [[ -n "${TEAM_SORTING_MJCF:-}" ]]; then test -f "${TEAM_SORTING_MJCF}"; fi

TEAM_SORTING_RUNTIME_ROOT=/opt/team_sorting_runtime
TEAM_SORTING_RUNTIME_SRC=${TEAM_SORTING_RUNTIME_ROOT}/src
TEAM_SORTING_PREFIX_ROOT=${TEAM_SORTING_RUNTIME_ROOT}/prefix
TEAM_SORTING_ROS_PREFIX=${TEAM_SORTING_PREFIX_ROOT}/local
TEAM_SORTING_LIBEXEC_DIR=${TEAM_SORTING_ROS_PREFIX}/lib/team_sorting
PIP_CACHE_DIR=${TEAM_SORTING_RUNTIME_ROOT}/pip-cache
FINGERPRINT_FILE=${TEAM_SORTING_RUNTIME_ROOT}/source.sha256
export TEAM_SORTING_PREFIX_ROOT TEAM_SORTING_ROS_PREFIX TEAM_SORTING_LIBEXEC_DIR PIP_CACHE_DIR

if [[ "${TEAM_SORTING_CLEAN_BUILD}" == "1" ]]; then
  echo "runtime clean build: removing cached src/prefix/pip-cache/fingerprint"
  rm -rf /opt/team_sorting_runtime/src \
         /opt/team_sorting_runtime/prefix \
         /opt/team_sorting_runtime/pip-cache \
         /opt/team_sorting_runtime/source.sha256
fi
mkdir -p "${TEAM_SORTING_RUNTIME_ROOT}" "${PIP_CACHE_DIR}"

SOURCE_SHA256="$(python3 - <<'"'"'PY'"'"'
from hashlib import sha256
from pathlib import Path

root = Path("/workspace/baseline")
required = (
    Path("setup.py"), Path("setup.cfg"), Path("package.xml"),
    Path("resource"), Path("team_sorting"), Path("launch"), Path("config"),
)
ignored_names = {"__pycache__", ".pytest_cache"}

files = []
for item in required:
    path = root / item
    if not path.exists():
        raise SystemExit(f"required source path missing: {path}")
    if path.is_file():
        files.append(path)
    else:
        files.extend(
            candidate for candidate in path.rglob("*")
            if candidate.is_file()
            and not any(part in ignored_names for part in candidate.relative_to(root).parts)
            and candidate.suffix != ".pyc"
        )

digest = sha256()
for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
    relative = path.relative_to(root).as_posix().encode("utf-8")
    digest.update(relative)
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"
echo "source fingerprint: ${SOURCE_SHA256}"

pip_installation_complete() {
  test -f "${TEAM_SORTING_ROS_PREFIX}/share/ament_index/resource_index/packages/team_sorting" &&
  test -d "${TEAM_SORTING_ROS_PREFIX}/lib/python3.10/dist-packages/team_sorting" &&
  test -x "${TEAM_SORTING_ROS_PREFIX}/bin/perception_node" &&
  test -x "${TEAM_SORTING_ROS_PREFIX}/bin/team_client_node" &&
  test -x "${TEAM_SORTING_ROS_PREFIX}/bin/dataset_recorder_node" &&
  test -f "${TEAM_SORTING_ROS_PREFIX}/share/team_sorting/launch/team.launch.xml" &&
  test -f "${TEAM_SORTING_ROS_PREFIX}/share/team_sorting/config/config.yaml"
}

ensure_ros_libexec_layout() {
  mkdir -p "${TEAM_SORTING_LIBEXEC_DIR}"
  for executable in perception_node team_client_node dataset_recorder_node; do
    source_entry="${TEAM_SORTING_ROS_PREFIX}/bin/${executable}"
    libexec_entry="${TEAM_SORTING_LIBEXEC_DIR}/${executable}"
    if [[ ! -x "${source_entry}" ]]; then
      echo "错误：pip安装副本缺少可执行入口：${source_entry}" >&2
      return 1
    fi
    ln -sfn "${source_entry}" "${libexec_entry}"
    if [[ ! -x "${libexec_entry}" ]]; then
      echo "错误：ROS libexec入口创建失败：${libexec_entry}" >&2
      return 1
    fi
  done
}

installation_complete() {
  pip_installation_complete &&
  test -x "${TEAM_SORTING_LIBEXEC_DIR}/perception_node" &&
  test -x "${TEAM_SORTING_LIBEXEC_DIR}/team_client_node" &&
  test -x "${TEAM_SORTING_LIBEXEC_DIR}/dataset_recorder_node"
}

CACHED_SHA256=""
if [[ -f "${FINGERPRINT_FILE}" ]]; then
  CACHED_SHA256="$(<"${FINGERPRINT_FILE}")"
fi
if [[ "${CACHED_SHA256}" == "${SOURCE_SHA256}" ]] && pip_installation_complete; then
  ensure_ros_libexec_layout
  installation_complete
  echo "runtime cache hit: source unchanged; skipping pip install"
else
  echo "runtime cache miss/source changed: refreshing writable source and offline install"
  rm -rf /opt/team_sorting_runtime/src /opt/team_sorting_runtime/prefix
  SOURCE_ROOT=/workspace/baseline RUNTIME_SRC="${TEAM_SORTING_RUNTIME_SRC}" python3 - <<'"'"'PY'"'"'
import os
from pathlib import Path
import shutil

source = Path(os.environ["SOURCE_ROOT"])
target = Path(os.environ["RUNTIME_SRC"])
ignored = shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc")
target.mkdir(parents=True, exist_ok=False)
for name in ("setup.py", "setup.cfg", "package.xml"):
    shutil.copy2(source / name, target / name)
for name in ("resource", "team_sorting", "launch", "config"):
    shutil.copytree(source / name, target / name, ignore=ignored)
PY
  python3 -m pip install \
    --no-index \
    --no-deps \
    --no-build-isolation \
    --upgrade \
    --force-reinstall \
    --prefix "${TEAM_SORTING_PREFIX_ROOT}" \
    "${TEAM_SORTING_RUNTIME_SRC}"
  pip_installation_complete
  ensure_ros_libexec_layout
  installation_complete
  printf "%s\n" "${SOURCE_SHA256}" > "${FINGERPRINT_FILE}"
fi

FILTERED_PYTHONPATH="$(python3 - <<'"'"'PY'"'"'
import os

blocked = {"/workspace/baseline", "/workspace/baseline/team_sorting"}
parts = []
for entry in os.environ.get("PYTHONPATH", "").split(":"):
    if entry and os.path.normpath(entry) not in blocked:
        parts.append(entry)
print(":".join(parts))
PY
)"
export AMENT_PREFIX_PATH="${TEAM_SORTING_ROS_PREFIX}:${AMENT_PREFIX_PATH:-}"
export PATH="${TEAM_SORTING_ROS_PREFIX}/bin:${PATH}"
export PYTHONPATH="${TEAM_SORTING_ROS_PREFIX}/lib/python3.10/dist-packages${FILTERED_PYTHONPATH:+:${FILTERED_PYTHONPATH}}"

cd /tmp
IMPORTED_PATH="$(python3 -c '"'"'import team_sorting; print(team_sorting.__file__)'"'"')"
echo "team_sorting import: ${IMPORTED_PATH}"
case "${IMPORTED_PATH}" in
  "${TEAM_SORTING_ROS_PREFIX}"/*) ;;
  *) echo "错误：team_sorting未从持久安装prefix导入：${IMPORTED_PATH}" >&2; exit 1 ;;
esac
ROS_PACKAGE_PREFIX="$(ros2 pkg prefix team_sorting)"
echo "ros2 pkg prefix team_sorting: ${ROS_PACKAGE_PREFIX}"
if [[ "${ROS_PACKAGE_PREFIX}" != "${TEAM_SORTING_ROS_PREFIX}" ]]; then
  echo "错误：ROS包前缀不匹配：${ROS_PACKAGE_PREFIX}" >&2
  exit 1
fi
EXECUTABLES="$(ros2 pkg executables team_sorting)"
for executable in perception_node team_client_node dataset_recorder_node; do
  if ! grep -Eq "^team_sorting[[:space:]]+${executable}$" <<< "${EXECUTABLES}"; then
    echo "错误：安装副本缺少ROS节点入口：${executable}" >&2
    exit 1
  fi
done
printf "baseline=%s (read-only)\nofficial=%s (read-only)\nruntime=%s\nros_prefix=%s\npip_cache=%s\nobserve_only=true\n" \
  /workspace/baseline "${MATERIAL_SORTING_OFFICIAL_ROOT}" \
  "${TEAM_SORTING_RUNTIME_ROOT}" "${TEAM_SORTING_ROS_PREFIX}" "${PIP_CACHE_DIR}"
ros2 launch team_sorting team.launch.xml')

if [[ "${DRY_RUN}" == "1" ]]; then
    IMAGE_ID="<not-inspected:dry-run>"
else
    command -v docker >/dev/null || { echo "错误：没有找到docker" >&2; exit 1; }
    IMAGE_ID="$(docker image inspect "${IMAGE_NAME}" --format '{{.Id}}')"
fi
printf 'Client image: %s\nClient image ID: %s\n' "${IMAGE_NAME}" "${IMAGE_ID}"
printf 'Project mount: %s -> /workspace/baseline (read-only)\n' "${PROJECT_ROOT}"
printf 'Official mount: %s -> /workspace/official (read-only)\n' "${OFFICIAL_ROOT}"
printf 'ROS_DOMAIN_ID=99\nRMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n'
printf 'GPU behavior: %s\n' "${GPU_BEHAVIOR}"
printf 'Runtime volume: %s -> /opt/team_sorting_runtime\n' "${RUNTIME_VOLUME}"
printf 'Runtime src: /opt/team_sorting_runtime/src\n'
printf 'Prefix root: /opt/team_sorting_runtime/prefix\n'
printf 'ROS prefix: /opt/team_sorting_runtime/prefix/local\n'
printf 'ROS libexec: /opt/team_sorting_runtime/prefix/local/lib/team_sorting\n'
printf 'Pip cache: /opt/team_sorting_runtime/pip-cache\n'
printf 'Clean build: %s\n' "${CLEAN_BUILD}"
printf 'Source fingerprint: stable SHA256 over setup.py/setup.cfg/package.xml/resource/team_sorting/launch/config\n'
printf 'Install mode: offline pip (--no-index --no-deps --no-build-isolation)\n'
printf 'observe_only=true\n'
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi
exec "${command[@]}"
