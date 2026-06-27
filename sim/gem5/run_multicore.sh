#!/bin/sh

set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
GEM5_ROOT="${SCRIPT_DIR}"
SUPER_ROOT="$(CDPATH= cd -- "${GEM5_ROOT}/../.." && pwd -P)"
cd "${GEM5_ROOT}"

export IMAGE_PATH="/home/barry/wlk/gem5_arm_linux_images"
export M5_PATH="${IMAGE_PATH}"
export CORAL_CKPT_BOOTSCRIPT="${CORAL_CKPT_BOOTSCRIPT:-${GEM5_ROOT}/configs/coralnpu/boot-to-checkpoint.rcS}"
export CORAL_RESUME_BOOTSCRIPT="${CORAL_RESUME_BOOTSCRIPT:-${GEM5_ROOT}/configs/coralnpu/fs-run.rcS}"
export CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/gem5-init.sh}"
export CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE:-${IMAGE_PATH}/vmlinux.arm64}"
export CORAL_KERNEL_CMDLINE="${CORAL_KERNEL_CMDLINE:-}"
export BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"
export CORAL_DISK_IMG_BUILT_DEFAULT="${GEM5_ROOT}/image/out/ubuntu-22.04-arm64-runtime-min-nosystemd.img"
export CORAL_DISK_IMG_BUILT_ALT1="${GEM5_ROOT}/image/out/ubuntu-22.04-arm64-runtime-min.img"
export CORAL_DISK_IMG_BUILT_ALT2="${GEM5_ROOT}/image/out/ubuntu-18.04-arm64-dev-dense.img"
export CORAL_DISK_IMG_BUILT_ALT3="${GEM5_ROOT}/image/out/ubuntu-18.04-arm64-dev.img"
export CORAL_DISK_IMG_FALLBACK="/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img"
export CORAL_DISK_IMG="${CORAL_DISK_IMG:-${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img}"
export CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${SUPER_ROOT}/m5out/coralnpu_ckpt}"
export CORAL_BOOTED_CKPT="${CORAL_CKPT_ROOT}/booted"
export CORAL_REBUILD_CKPT="${CORAL_REBUILD_CKPT:-0}"
export CORAL_CKPT_IMAGE_META="${CORAL_CKPT_ROOT}/disk_image_path.txt"
export CORAL_CKPT_INIT_META="${CORAL_CKPT_ROOT}/kernel_init_path.txt"
export CORAL_CKPT_KERNEL_META="${CORAL_CKPT_ROOT}/kernel_image_path.txt"
export CORAL_CKPT_CMDLINE_META="${CORAL_CKPT_ROOT}/kernel_cmdline.txt"
export CORAL_CKPT_FORMAT_META="${CORAL_CKPT_ROOT}/format_version.txt"
export CORAL_CKPT_FORMAT_VERSION=5
export CORAL_NPU_BACKEND="${CORAL_NPU_BACKEND:-stage-a}"
export CORAL_RTL_BRIDGE="${CORAL_RTL_BRIDGE:-${SUPER_ROOT}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
export CORAL_RTL_FIRMWARE="${CORAL_RTL_FIRMWARE:-${SUPER_ROOT}/build/coralnpu/gem5_smoke_halt.elf}"
export CORAL_RTL_TICK_PERIOD="${CORAL_RTL_TICK_PERIOD:-1ns}"
export CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1}"
export GEM5_OPTIONS="${GEM5_OPTIONS:-}"

NPU_BACKEND_ARGS="--npu-backend=${CORAL_NPU_BACKEND}"
BOOTSTRAP_NPU_BACKEND_ARGS="--npu-backend=stage-a"
if [ "${CORAL_NPU_BACKEND}" = "verilated-coral" ]; then
  if [ ! -f "${CORAL_RTL_BRIDGE}" ]; then
    echo "Coral RTL bridge not found: ${CORAL_RTL_BRIDGE}" >&2
    echo "Build it with tools/coralnpu/phase2_build_bridge.sh" >&2
    exit 1
  fi
  if [ ! -f "${CORAL_RTL_FIRMWARE}" ]; then
    echo "Coral RTL firmware not found: ${CORAL_RTL_FIRMWARE}" >&2
    echo "Build it with tools/coralnpu/phase2_build_bridge.sh" >&2
    exit 1
  fi
  if command -v ldd >/dev/null 2>&1 &&
     ldd "${CORAL_RTL_BRIDGE}" 2>/dev/null | grep -qi systemc; then
    echo "Coral RTL bridge links libsystemc and is incompatible with gem5:" >&2
    echo "  ${CORAL_RTL_BRIDGE}" >&2
    echo "Rebuild it with ./tools/coralnpu/phase2_build_bridge.sh" >&2
    exit 1
  fi
  NPU_BACKEND_ARGS="${NPU_BACKEND_ARGS} --npu-verilated-wrapper=${CORAL_RTL_BRIDGE}"
  NPU_BACKEND_ARGS="${NPU_BACKEND_ARGS} --npu-rtl-firmware=${CORAL_RTL_FIRMWARE}"
  NPU_BACKEND_ARGS="${NPU_BACKEND_ARGS} --npu-rtl-tick-period=${CORAL_RTL_TICK_PERIOD}"
  NPU_BACKEND_ARGS="${NPU_BACKEND_ARGS} --npu-rtl-cycles-per-event=${CORAL_RTL_CYCLES_PER_EVENT}"
fi

if [ "${CORAL_DISK_IMG}" = "${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img" ]; then
  for candidate in \
    "${CORAL_DISK_IMG_BUILT_DEFAULT}" \
    "${CORAL_DISK_IMG_BUILT_ALT1}" \
    "${CORAL_DISK_IMG_BUILT_ALT2}" \
    "${CORAL_DISK_IMG_BUILT_ALT3}" \
    "${CORAL_DISK_IMG_FALLBACK}"
  do
    if [ -f "${candidate}" ]; then
      CORAL_DISK_IMG="${candidate}"
      break
    fi
  done
fi

if [ "${CORAL_REBUILD_CKPT}" = "1" ]; then
  rm -rf "${CORAL_CKPT_ROOT}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  saved_format="$(cat "${CORAL_CKPT_FORMAT_META}" 2>/dev/null || true)"
  if [ "${saved_format}" != "${CORAL_CKPT_FORMAT_VERSION}" ]; then
    echo "Checkpoint lacks the dynamic resume trampoline; rebuilding once"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ "${CORAL_DISK_IMG}" -nt "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Disk image contents changed; rebuilding boot checkpoint"
  rm -rf "${CORAL_CKPT_ROOT}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ ! -f "${CORAL_CKPT_IMAGE_META}" ]; then
  echo "Legacy checkpoint image metadata missing; recording current image"
  mkdir -p "${CORAL_CKPT_ROOT}"
  printf '%s\n' "${CORAL_DISK_IMG}" > "${CORAL_CKPT_IMAGE_META}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ -f "${CORAL_CKPT_IMAGE_META}" ]; then
  if [ "$(cat "${CORAL_CKPT_IMAGE_META}")" != "${CORAL_DISK_IMG}" ]; then
    echo "Disk image changed; rebuilding boot checkpoint"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ ! -f "${CORAL_CKPT_INIT_META}" ]; then
  echo "Legacy checkpoint kernel-init metadata missing; recording current init"
  mkdir -p "${CORAL_CKPT_ROOT}"
  printf '%s\n' "${CORAL_KERNEL_INIT}" > "${CORAL_CKPT_INIT_META}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ -f "${CORAL_CKPT_INIT_META}" ]; then
  if [ "$(cat "${CORAL_CKPT_INIT_META}")" != "${CORAL_KERNEL_INIT}" ]; then
    echo "Kernel init changed; rebuilding boot checkpoint"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ ! -f "${CORAL_CKPT_KERNEL_META}" ]; then
  echo "Legacy checkpoint kernel image metadata missing; recording current kernel"
  mkdir -p "${CORAL_CKPT_ROOT}"
  printf '%s\n' "${CORAL_KERNEL_IMAGE}" > "${CORAL_CKPT_KERNEL_META}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ -f "${CORAL_CKPT_KERNEL_META}" ]; then
  if [ "$(cat "${CORAL_CKPT_KERNEL_META}")" != "${CORAL_KERNEL_IMAGE}" ]; then
    echo "Kernel image changed; rebuilding boot checkpoint"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ "${CORAL_KERNEL_IMAGE}" -nt "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Kernel image contents changed; rebuilding boot checkpoint"
  rm -rf "${CORAL_CKPT_ROOT}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ ! -f "${CORAL_CKPT_CMDLINE_META}" ]; then
  echo "Legacy checkpoint kernel cmdline metadata missing; recording current cmdline"
  mkdir -p "${CORAL_CKPT_ROOT}"
  printf '%s\n' "${CORAL_KERNEL_CMDLINE}" > "${CORAL_CKPT_CMDLINE_META}"
fi

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ -f "${CORAL_CKPT_CMDLINE_META}" ]; then
  if [ "$(cat "${CORAL_CKPT_CMDLINE_META}")" != "${CORAL_KERNEL_CMDLINE}" ]; then
    echo "Kernel cmdline changed; rebuilding boot checkpoint"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

mkdir -p "${CORAL_CKPT_ROOT}"

# 默认启动命令d9300，可对应修改d9300 -> d9200
#./build/ARM/gem5.opt configs/example/arm/arm_multicore_d9200.py --cpu-type="D9200" --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img --kernel=${IMAGE_PATH}/vmlinux.arm64 --enable-gemmini --gemmini-cpu-idx 0
#./build/ARM/gem5.opt configs/example/arm/arm_multicore_d9200.py --cpu-type="D9200" --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img --kernel=${IMAGE_PATH}/vmlinux.arm64 --bootscript configs/gemmini/fs-run.rcS --enable-gemmini --gemmini-mmio-iobus --gemmini-ctrl-addr 0x10030000 --gemmini-ctrl-size 0x1000
echo "Launching ARM multicore FS with Coral NPU enabled"
echo "Guest terminal socket: ./util/term/gem5term localhost 4567"
echo "Disk image: ${CORAL_DISK_IMG}"
echo "Kernel image: ${CORAL_KERNEL_IMAGE}"
echo "Kernel init: ${CORAL_KERNEL_INIT}"
[ -z "${CORAL_KERNEL_CMDLINE}" ] || echo "Kernel cmdline override: ${CORAL_KERNEL_CMDLINE}"
echo "Checkpoint root: ${CORAL_CKPT_ROOT}"
echo "NPU backend: ${CORAL_NPU_BACKEND}"
[ "${CORAL_NPU_BACKEND}" != "verilated-coral" ] || echo "Coral RTL bridge: ${CORAL_RTL_BRIDGE}"
[ "${CORAL_NPU_BACKEND}" != "verilated-coral" ] || echo "Coral RTL firmware: ${CORAL_RTL_FIRMWARE}"
ls -lh "${CORAL_DISK_IMG}" || exit 1
ls -lh "${CORAL_KERNEL_IMAGE}" || exit 1
echo "Building build/ARM/gem5.opt with -j${BUILD_JOBS}"
scons build/ARM/gem5.opt -j"${BUILD_JOBS}"

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Mode: restore from booted checkpoint"
  echo "Checkpoint: ${CORAL_BOOTED_CKPT}"
  echo "Resume script: ${CORAL_RESUME_BOOTSCRIPT}"
  ./build/ARM/gem5.opt ${GEM5_OPTIONS} configs/example/arm/arm_multicore_d9300.py \
    --cpu-type="D9300" \
    --disk="${CORAL_DISK_IMG}" \
    --kernel="${CORAL_KERNEL_IMAGE}" \
    --kernel-init="${CORAL_KERNEL_INIT}" \
    --enable-npu \
    ${NPU_BACKEND_ARGS} \
    --ckpt-dir="${CORAL_CKPT_ROOT}" \
    --restore-from="${CORAL_BOOTED_CKPT}" \
    --bootscript="${CORAL_RESUME_BOOTSCRIPT}" \
    --kernel-cmd="${CORAL_KERNEL_CMDLINE}"
else
  echo "Mode: bootstrap and create checkpoint"
  echo "Bootstrap NPU backend: stage-a"
  echo "Bootstrap script: ${CORAL_CKPT_BOOTSCRIPT}"
  echo "Set CORAL_REBUILD_CKPT=1 to force rebuilding the boot checkpoint"
  ./build/ARM/gem5.opt ${GEM5_OPTIONS} configs/example/arm/arm_multicore_d9300.py \
    --cpu-type="D9300" \
    --disk="${CORAL_DISK_IMG}" \
    --kernel="${CORAL_KERNEL_IMAGE}" \
    --kernel-init="${CORAL_KERNEL_INIT}" \
    --enable-npu \
    ${BOOTSTRAP_NPU_BACKEND_ARGS} \
    --ckpt-dir="${CORAL_CKPT_ROOT}" \
    --exit-after-checkpoint \
    --bootscript="${CORAL_CKPT_BOOTSCRIPT}" \
    --kernel-cmd="${CORAL_KERNEL_CMDLINE}"
  printf '%s\n' "${CORAL_DISK_IMG}" > "${CORAL_CKPT_IMAGE_META}"
  printf '%s\n' "${CORAL_KERNEL_INIT}" > "${CORAL_CKPT_INIT_META}"
  printf '%s\n' "${CORAL_KERNEL_IMAGE}" > "${CORAL_CKPT_KERNEL_META}"
  printf '%s\n' "${CORAL_KERNEL_CMDLINE}" > "${CORAL_CKPT_CMDLINE_META}"
  printf '%s\n' "${CORAL_CKPT_FORMAT_VERSION}" > "${CORAL_CKPT_FORMAT_META}"
  echo "Boot checkpoint saved at ${CORAL_BOOTED_CKPT}"
  echo "Run ./run_multicore.sh again to restore and execute the Coral NPU test script"
fi

# 按照指定slice dump启动命令
#./build/ARM/gem5.opt configs/example/arm/arm_multicore_slc.py --slice-run --slice-ticks 10ms --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img --kernel=${IMAGE_PATH}/../binaries/vmlinux.arm64 --caches --last-cache-level 3 --cpu-type="x4a720"
#
# slice dump + slice save ckpt
#./build/ARM/gem5.opt configs/example/arm/arm_multicore_slc.py --slice-run --slice-ticks 10ms --save-slices --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img --kernel=${IMAGE_PATH}/../binaries/vmlinux.arm64 --caches --last-cache-level 3 --cpu-type="x4a720"

# 从保存的slice_id 断点启动
#./build/ARM/gem5.opt configs/example/arm/arm_multicore_slc.py --slice-run --slice-ticks 10ms --restore-slice 6 --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img --kernel=${IMAGE_PATH}/../binaries/vmlinux.arm64 --caches --last-cache-level 3 --cpu-type="x4a720"

#taskset -c 0 ./geekbench_aarch64 --no-upload --multi-core --iterations 1 --skip-sysinfo --workload 204
