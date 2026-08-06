#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="${IMAGE:-${CORAL_DISK_IMG:-/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img}}"
MODEL="${ROOT_DIR}/build/models/heterogeneous-smoke.npxm"

echo "Baseline guest asset image: ${IMAGE}"
"${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"
"${ROOT_DIR}/tools/models/create_sample_model.py" "${MODEL}"
"${ROOT_DIR}/tools/kernel/install_opennpux_init_to_image.sh" "${IMAGE}"
"${ROOT_DIR}/tools/guest_tools/install_coralctl_to_image.sh" "${IMAGE}"
"${ROOT_DIR}/tools/guest_tools/install_model_to_image.sh" "${IMAGE}" "${MODEL}"

echo "Baseline guest assets installed into ${IMAGE}"
echo "Rebuild the boot checkpoint once before running model tests"
