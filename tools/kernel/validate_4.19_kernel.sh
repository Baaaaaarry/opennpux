#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

IMAGE="${IMAGE:-${HOME}/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img}"
KERNEL_BASE_CONFIG="${KERNEL_BASE_CONFIG:-${ROOT_DIR}/tools/kernel/gem5-4.18.config}"
LINUX_BRANCH="${LINUX_BRANCH:-linux-4.19.y}"

if [ "${LINUX_BRANCH}" != "linux-4.19.y" ]; then
    echo "error: baseline kernel validation is pinned to linux-4.19.y" >&2
    echo "hint: run generic tools/kernel/build_arm64_kernel.sh for other branches" >&2
    exit 1
fi
if [ ! -f "${KERNEL_BASE_CONFIG}" ]; then
    echo "error: KERNEL_BASE_CONFIG not found: ${KERNEL_BASE_CONFIG}" >&2
    echo "hint: copy the booting 4.18 config to tools/kernel/gem5-4.18.config" >&2
    exit 1
fi
if [ ! -f "${IMAGE}" ]; then
    echo "error: IMAGE not found: ${IMAGE}" >&2
    exit 1
fi

cd "${ROOT_DIR}"

echo "[kernel-baseline] building ${LINUX_BRANCH} with ${KERNEL_BASE_CONFIG}"
KERNEL_BASE_CONFIG="${KERNEL_BASE_CONFIG}" \
LINUX_BRANCH="${LINUX_BRANCH}" \
"${SCRIPT_DIR}/build_arm64_kernel.sh"

echo "[kernel-baseline] checking gem5 kernel config"
"${SCRIPT_DIR}/check_gem5_kernel_config.sh" "${ROOT_DIR}/build/linux-arm64/.config"

echo "[kernel-baseline] building opennpux_coral.ko"
"${SCRIPT_DIR}/build_opennpux_coral_ko.sh"

echo "[kernel-baseline] building driver-aware coralctl"
"${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"

kernel_release="$(cat "${ROOT_DIR}/build/kernel/kernel.release")"
kernel_image="${ROOT_DIR}/build/kernel/Image-${kernel_release}"
gem5_kernel="${ROOT_DIR}/build/kernel/vmlinux-${kernel_release}"
ko="${ROOT_DIR}/build/kernel/opennpux_coral-${kernel_release}.ko"

echo "[kernel-baseline] installing kernel/modules/driver into image"
sudo "${SCRIPT_DIR}/install_kernel_to_image.sh" \
    "${IMAGE}" \
    "${ROOT_DIR}/build/linux-arm64" \
    "${kernel_image}" \
    "${ko}"

echo "[kernel-baseline] installing OpenNPUX minimal init"
"${SCRIPT_DIR}/install_opennpux_init_to_image.sh" "${IMAGE}"

echo "[kernel-baseline] installing coralctl"
"${ROOT_DIR}/tools/guest_tools/install_coralctl_to_image.sh" "${IMAGE}"

cat <<EOF
[kernel-baseline] build complete
kernel_release=${kernel_release}
gem5_kernel=${gem5_kernel}
guest_kernel_image=${kernel_image}
driver_module=${ko}
image=${IMAGE}

Validate in gem5:
  cd ${ROOT_DIR}/thirdparty/gem5
  CORAL_KERNEL_IMAGE=${gem5_kernel} \\
  CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \\
  CORAL_REBUILD_CKPT=1 \\
  ./run_multicore.sh

Then validate the driver path from the boot checkpoint:
  CORAL_KERNEL_IMAGE=${gem5_kernel} \\
  CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \\
  CORAL_RESUME_BOOTSCRIPT="\$PWD/configs/coralnpu/coral-driver-info-test.rcS" \\
  ./run_multicore.sh

Expected guest output:
  [coral-driver-info-test] kernel=${kernel_release}
  transport=driver
  backend=stage-a
  [coral-driver-info-test] PASS

After building the RTL bridge, validate driver mmap and DMA with:
  CORAL_KERNEL_IMAGE=${gem5_kernel} \
  CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
  ${ROOT_DIR}/tools/coralnpu/run_driver_dma_test.sh
EOF
