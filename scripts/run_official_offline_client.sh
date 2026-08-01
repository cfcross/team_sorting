#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${TEAM_SORTING_CLIENT_IMAGE:-material_sorting:offline-client}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DRY_RUN="${DRY_RUN:-0}"
OFFICIAL_ROOT="${MATERIAL_SORTING_OFFICIAL_ROOT:-}"
CLIENT_GPUS="${TEAM_SORTING_CLIENT_GPUS-all}"
YOLO_CHECKPOINT="${MATERIAL_SORTING_YOLO_CHECKPOINT:-}"
MJCF_PATH="${TEAM_SORTING_MJCF:-}"

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
    -v "${PROJECT_ROOT}:/workspace/baseline:rw"
    -v "${OFFICIAL_ROOT}:/workspace/official:ro")
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
    'set -euo pipefail
source /opt/ros/humble/setup.bash
test -d /workspace/baseline/team_sorting
test -d "${MATERIAL_SORTING_OFFICIAL_ROOT}"
if [[ -n "${MATERIAL_SORTING_YOLO_CHECKPOINT:-}" ]]; then test -f "${MATERIAL_SORTING_YOLO_CHECKPOINT}"; fi
if [[ -n "${TEAM_SORTING_MJCF:-}" ]]; then test -f "${TEAM_SORTING_MJCF}"; fi
printf "baseline=%s\nofficial=%s\nyolo=%s\nmjcf=%s\nobserve_only=true\n" \
  /workspace/baseline "${MATERIAL_SORTING_OFFICIAL_ROOT}" \
  "${MATERIAL_SORTING_YOLO_CHECKPOINT:-<adapter-search>}" "${TEAM_SORTING_MJCF:-<adapter-search>}"
cd /workspace/baseline
colcon build --packages-select team_sorting \
  --build-base /tmp/team_sorting_build --install-base /tmp/team_sorting_install \
  --log-base /tmp/team_sorting_log
source /tmp/team_sorting_install/setup.bash
ros2 launch team_sorting team.launch.xml')

printf 'Client image: %s\nProject mount: %s -> /workspace/baseline\n' "${IMAGE_NAME}" "${PROJECT_ROOT}"
printf 'ROS_DOMAIN_ID=99\nRMW_IMPLEMENTATION=rmw_cyclonedds_cpp\n'
printf 'GPU behavior: %s\n' "${GPU_BEHAVIOR}"
printf 'Command:'; printf ' %q' "${command[@]}"; printf '\n'
if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi
command -v docker >/dev/null || { echo "错误：没有找到docker" >&2; exit 1; }
docker image inspect "${IMAGE_NAME}" --format 'Client image ID: {{.Id}}'
exec "${command[@]}"
