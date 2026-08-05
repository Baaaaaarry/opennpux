#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="${IMAGE:-${CORAL_DISK_IMG:-/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img}}"
MODEL="${ROOT_DIR}/build/models/qwen-tiny.npxm"

echo "Qwen guest asset image: ${IMAGE}"
echo "Qwen model output: ${MODEL}"

"${ROOT_DIR}/tools/models/prepare_qwen_tiny.sh" "${MODEL}"
"${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"
"${ROOT_DIR}/tools/guest_tools/build_qwen_inspect.sh"
"${ROOT_DIR}/tools/guest_tools/install_coralctl_to_image.sh" "${IMAGE}"
"${ROOT_DIR}/tools/guest_tools/install_qwen_assets_to_image.sh" "${IMAGE}" "${MODEL}"

echo "Qwen guest assets installed into ${IMAGE}"
echo "Rebuild the boot checkpoint once before running Qwen guest tests"
