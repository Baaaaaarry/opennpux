#!/bin/sh
#
# run_multicore.sh — Launch gem5 ARM full-system simulation with Coral NPU.
#
# This is the main entry point for all gem5 full-system work.  It compiles
# gem5 (scons), validates the environment, manages boot checkpoints, and
# launches the D9300 ARM configuration with the selected NPU backend.
#
# Two execution modes:
#
#   Bootstrap (no checkpoint exists):
#     gem5 boots ARM Linux with --npu-backend=stage-a, runs the checkpoint
#     bootscript, takes a checkpoint at CORAL_BOOTED_CKPT, exits.
#     All guest modifications (kernel modules, coralctl, model files) must
#     be installed into the disk image before this step.
#
#   Restore (checkpoint exists):
#     gem5 restores from the booted checkpoint with the requested NPU
#     backend (stage-a or verilated-coral), injects the resume bootscript
#     via "m5 readfile", and runs the test payload.
#
# Checkpoint invalidation — the checkpoint is rebuilt when any of these change:
#   1. CORAL_REBUILD_CKPT=1 (explicit force)
#   2. CORAL_CKPT_FORMAT_VERSION mismatch (resume mechanism changed)
#   3. Disk image timestamp is newer than the checkpoint file
#   4. Disk image path differs from the recorded metadata
#   5. Kernel init path differs from the recorded metadata
#   6. Kernel image path differs from the recorded metadata
#   7. Kernel image timestamp is newer than the checkpoint file
#   8. Kernel cmdline differs from the recorded metadata
#
# Environment (all overridable, listed with defaults):
#   CORAL_NPU_BACKEND         stage-a | verilated-coral (default: stage-a)
#   CORAL_RTL_BRIDGE          path to Verilator bridge .so
#   CORAL_RTL_FIRMWARE        path to NPU firmware ELF
#   CORAL_RTL_TICK_PERIOD     gem5 time per RTL cycle (default: 1ns)
#   CORAL_RTL_CYCLES_PER_EVENT  RTL cycles per gem5 event (default: 1)
#   CORAL_KERNEL_IMAGE        vmlinux ELF for gem5 --kernel
#   CORAL_KERNEL_INIT         guest /sbin/init path (default: /sbin/gem5-init.sh)
#   CORAL_KERNEL_CMDLINE      kernel command-line override
#   CORAL_DISK_IMG            ARM64 disk image path
#   CORAL_REBUILD_CKPT        1 = force checkpoint rebuild
#   CORAL_AUTO_RESUME_AFTER_CKPT  1 = auto-resume after bootstrap
#   CORAL_CKPT_ROOT           checkpoint directory
#   CORAL_CKPT_BOOTSCRIPT     bootscript for checkpoint creation
#   CORAL_RESUME_BOOTSCRIPT   bootscript injected on restore
#   IMAGE_PATH                base directory for disk images and kernels
#   GEM5_OPTIONS              additional gem5 flags (e.g. --debug-flags)
#   GEM5_BUILD_TARGET         scons build target (default: build/ARM/gem5.opt)
#   GEM5_CONFIG               gem5 Python config (default: arm_multicore_d9300.py)
#   GEM5_CPU_TYPE             CPU model (default: D9300)
#   GEM5_REBUILD              1 = force scons even if binary is up to date (default: 0)
#   CORAL_CONFIG_OPTIONS      additional gem5 Python config flags
#
# @gem5-sim-spec  v1  2025-07-29

set -eu

# ---------------------------------------------------------------------------
# Resolve paths.
#   GEM5_ROOT  = thirdparty/gem5/ (this script's directory)
#   SUPER_ROOT = superproject root (../../ from here)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
GEM5_ROOT="${SCRIPT_DIR}"
SUPER_ROOT="$(CDPATH= cd -- "${GEM5_ROOT}/../.." && pwd -P)"
cd "${GEM5_ROOT}"

# ---------------------------------------------------------------------------
# Image and kernel paths.
#   IMAGE_PATH is the canonical directory for gem5 ARM assets.
#   Override: IMAGE_PATH=/your/path ./run_multicore.sh
# ---------------------------------------------------------------------------
export IMAGE_PATH="${IMAGE_PATH:-${HOME}/wlk/gem5_arm_linux_images}"
export M5_PATH="${IMAGE_PATH}"

# ---------------------------------------------------------------------------
# Boot and resume scripts.
#   The checkpoint bootscript runs during bootstrap (stage-a backend).
#   The resume bootscript is injected via "m5 readfile" on restore.
# ---------------------------------------------------------------------------
export CORAL_CKPT_BOOTSCRIPT="${CORAL_CKPT_BOOTSCRIPT:-${GEM5_ROOT}/configs/coralnpu/boot-to-checkpoint.rcS}"
export CORAL_RESUME_BOOTSCRIPT="${CORAL_RESUME_BOOTSCRIPT:-${GEM5_ROOT}/configs/coralnpu/fs-run.rcS}"
export CORAL_KERNEL_INIT="${CORAL_KERNEL_INIT:-/sbin/gem5-init.sh}"
export CORAL_KERNEL_IMAGE="${CORAL_KERNEL_IMAGE:-${IMAGE_PATH}/vmlinux.arm64}"
export CORAL_KERNEL_CMDLINE="${CORAL_KERNEL_CMDLINE:-}"
export BUILD_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)"

# ---------------------------------------------------------------------------
# Disk image discovery.
#   Searches built gem5 images first (produced by gem5's image builder),
#   then falls back to the IMAGE_PATH convention.  The first match wins.
# ---------------------------------------------------------------------------
export CORAL_DISK_IMG_BUILT_DEFAULT="${GEM5_ROOT}/image/out/ubuntu-22.04-arm64-runtime-min-nosystemd.img"
export CORAL_DISK_IMG_BUILT_ALT1="${GEM5_ROOT}/image/out/ubuntu-22.04-arm64-runtime-min.img"
export CORAL_DISK_IMG_BUILT_ALT2="${GEM5_ROOT}/image/out/ubuntu-18.04-arm64-dev-dense.img"
export CORAL_DISK_IMG_BUILT_ALT3="${GEM5_ROOT}/image/out/ubuntu-18.04-arm64-dev.img"
export CORAL_DISK_IMG_FALLBACK="${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img"
export CORAL_DISK_IMG="${CORAL_DISK_IMG:-${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img}"

# ---------------------------------------------------------------------------
# Checkpoint paths and metadata.
#   CORAL_CKPT_FORMAT_VERSION is bumped when checkpoint-visible boot state
#   changes, including the resume mechanism or the generated device tree.
#   (e.g. a new dynamic-resume trampoline).  Old-format checkpoints are
#   automatically invalidated.
# ---------------------------------------------------------------------------
export CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${SUPER_ROOT}/checkpoint/coralnpu_ckpt}"
export CORAL_BOOTED_CKPT="${CORAL_CKPT_ROOT}/booted"
export CORAL_REBUILD_CKPT="${CORAL_REBUILD_CKPT:-0}"
export CORAL_AUTO_RESUME_AFTER_CKPT="${CORAL_AUTO_RESUME_AFTER_CKPT:-0}"
export CORAL_CKPT_IMAGE_META="${CORAL_CKPT_ROOT}/disk_image_path.txt"
export CORAL_CKPT_INIT_META="${CORAL_CKPT_ROOT}/kernel_init_path.txt"
export CORAL_CKPT_KERNEL_META="${CORAL_CKPT_ROOT}/kernel_image_path.txt"
export CORAL_CKPT_CMDLINE_META="${CORAL_CKPT_ROOT}/kernel_cmdline.txt"
export CORAL_CKPT_FORMAT_META="${CORAL_CKPT_ROOT}/format_version.txt"
# Version 8 requires the Coral DT node to reference its reserved DMA window
# through memory-region. Linux parses the DT before the checkpoint is taken,
# so a checkpoint created with the previous DT cannot be repaired on restore.
export CORAL_CKPT_FORMAT_VERSION=8

# ---------------------------------------------------------------------------
# NPU backend configuration.
#   stage-a: transaction-level run-to-halt stub (bootstrap + smoke).
#   verilated-coral: official Coral Verilator RTL via C ABI bridge.
# ---------------------------------------------------------------------------
export CORAL_NPU_BACKEND="${CORAL_NPU_BACKEND:-stage-a}"
export CORAL_RTL_BRIDGE="${CORAL_RTL_BRIDGE:-${SUPER_ROOT}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
export CORAL_RTL_FIRMWARE="${CORAL_RTL_FIRMWARE:-${SUPER_ROOT}/build/coralnpu/gem5_smoke_halt.elf}"
export CORAL_RTL_TICK_PERIOD="${CORAL_RTL_TICK_PERIOD:-1ns}"
export CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1}"
export GEM5_OPTIONS="${GEM5_OPTIONS:-}"
export GEM5_BUILD_TARGET="${GEM5_BUILD_TARGET:-build/ARM/gem5.opt}"
export GEM5_REBUILD="${GEM5_REBUILD:-0}"
export GEM5_CONFIG="${GEM5_CONFIG:-configs/example/arm/arm_multicore_d9300.py}"
export GEM5_CPU_TYPE="${GEM5_CPU_TYPE:-D9300}"
export CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-}"

# ---------------------------------------------------------------------------
# Backend validation.
#   When using the verilated-coral backend, the bridge .so and firmware
#   ELF must exist, and the bridge must not link a second libsystemc
#   (gem5 already provides one — a second copy causes symbol conflicts).
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Disk image candidate search.
#   When CORAL_DISK_IMG is still at its default value, scan the built-image
#   candidates in the gem5 tree before falling back to IMAGE_PATH.  If a
#   built image exists it takes priority over the download image.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Checkpoint validity logic.
#
#   Eight independent triggers invalidate the checkpoint.  Each trigger
#   removes CORAL_CKPT_ROOT, which causes the script to enter bootstrap
#   mode on the next run.  Triggers are checked in order:
#
#   (1) Explicit CORAL_REBUILD_CKPT=1
#   (2) Format version changed → resume trampoline is incompatible
#   (3) Disk image mtime newer than checkpoint → guest contents changed
#   (4) Disk image path changed → different image from the one checkpointed
#   (5) Kernel init path changed → different init script
#   (6) Kernel image path changed → different vmlinux file
#   (7) Kernel image mtime newer than checkpoint → kernel was recompiled
#   (8) Kernel cmdline changed → different boot parameters
#
#   Metadata files (disk_image_path.txt, kernel_init_path.txt, etc.) are
#   recorded during bootstrap so subsequent invocations can compare.
# ---------------------------------------------------------------------------

# (1) Explicit force.
if [ "${CORAL_REBUILD_CKPT}" = "1" ]; then
  rm -rf "${CORAL_CKPT_ROOT}"
fi

# (2) Format version.
if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  saved_format="$(cat "${CORAL_CKPT_FORMAT_META}" 2>/dev/null || true)"
  if [ "${saved_format}" != "${CORAL_CKPT_FORMAT_VERSION}" ]; then
    echo "Checkpoint lacks the dynamic resume trampoline; rebuilding once"
    rm -rf "${CORAL_CKPT_ROOT}"
  fi
fi

# (3) Disk image mtime newer than checkpoint.
if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ "${CORAL_DISK_IMG}" -nt "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Disk image contents changed; rebuilding boot checkpoint"
  rm -rf "${CORAL_CKPT_ROOT}"
fi

# (4) Record/compare disk image path.
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

# (5) Record/compare kernel init path.
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

# (6) Record/compare kernel image path.
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

# (7) Kernel image mtime newer than checkpoint.
if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ] && [ "${CORAL_KERNEL_IMAGE}" -nt "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Kernel image contents changed; rebuilding boot checkpoint"
  rm -rf "${CORAL_CKPT_ROOT}"
fi

# (8) Record/compare kernel cmdline.
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

# ---------------------------------------------------------------------------
# Configuration report.
# ---------------------------------------------------------------------------
echo "Launching ARM multicore FS with Coral NPU enabled"
echo "Config: ${GEM5_CONFIG}"
echo "CPU type: ${GEM5_CPU_TYPE}"
echo "Build target: ${GEM5_BUILD_TARGET}"
echo "Guest terminal socket: ./util/term/gem5term localhost 4567"
echo "Disk image: ${CORAL_DISK_IMG}"
echo "Kernel image: ${CORAL_KERNEL_IMAGE}"
echo "Kernel init: ${CORAL_KERNEL_INIT}"
[ -z "${CORAL_KERNEL_CMDLINE}" ] || echo "Kernel cmdline override: ${CORAL_KERNEL_CMDLINE}"
echo "Checkpoint root: ${CORAL_CKPT_ROOT}"
echo "NPU backend: ${CORAL_NPU_BACKEND}"
[ "${CORAL_NPU_BACKEND}" != "verilated-coral" ] || echo "Coral RTL bridge: ${CORAL_RTL_BRIDGE}"
[ "${CORAL_NPU_BACKEND}" != "verilated-coral" ] || echo "Coral RTL firmware: ${CORAL_RTL_FIRMWARE}"
[ -z "${CORAL_CONFIG_OPTIONS}" ] || echo "Coral config options: ${CORAL_CONFIG_OPTIONS}"
ls -lh "${CORAL_DISK_IMG}" || exit 1
ls -lh "${CORAL_KERNEL_IMAGE}" || exit 1

# ---------------------------------------------------------------------------
# Build gem5.
#
#   scons regenerates tags.cc (git hash + timestamp) on every invocation,
#   which triggers a ~30 s relink even when no source changed.  To avoid
#   this, we compare the binary's mtime against the real source directories
#   (src/, ext/, SConstruct, and the sim/gem5 overlay).  If the binary is
#   strictly newer than all of them, we skip the build.
#
#   Set GEM5_REBUILD=1 to force scons regardless (e.g. after pulling code).
# ---------------------------------------------------------------------------
NEED_BUILD=1  # default: build

if [ -x "${GEM5_BUILD_TARGET}" ] && [ "${GEM5_REBUILD}" != "1" ]; then
    # SConstruct (the root build file) is a sentinel for build-system changes.
    # The sim/gem5 overlay copies and touches files, so any overlay update
    # naturally produces source files newer than the binary.
    if [ "${GEM5_ROOT}/SConstruct" -nt "${GEM5_BUILD_TARGET}" ]; then
        :
    elif [ -n "$(find "${GEM5_ROOT}/src" "${GEM5_ROOT}/ext" \
                    \( -name '*.cc' -o -name '*.hh' -o -name '*.py' \
                       -o -name 'SConscript' \) \
                    -newer "${GEM5_BUILD_TARGET}" -print -quit 2>/dev/null)" ]; then
        :
    else
        NEED_BUILD=0
    fi
fi

if [ "${NEED_BUILD}" = "1" ]; then
    echo "Building ${GEM5_BUILD_TARGET} with -j${BUILD_JOBS}"
    # --ignore-style skips SCons install hooks that look for git pre-commit.
    scons "${GEM5_BUILD_TARGET}" --ignore-style -j"${BUILD_JOBS}"
else
    echo "Skipping gem5 build (${GEM5_BUILD_TARGET} is up to date)"
    echo "(set GEM5_REBUILD=1 to force)"
fi

# ---------------------------------------------------------------------------
# Launch gem5.
#
#   Restore path:  checkpoint exists → restore with requested backend.
#   Bootstrap path: no checkpoint → boot with stage-a, create checkpoint,
#                   optionally auto-resume (CORAL_AUTO_RESUME_AFTER_CKPT=1).
#
#   The D9300 SoC configuration includes the Coral NPU (--enable-npu).
#   For the verilated-coral backend, the bridge .so is dlopen'd by gem5
#   and the firmware ELF is loaded into Coral TCM before boot.
#
#   Runtime output (system.terminal, stats.txt) goes to logs/sim/m5out/.
#   Checkpoints remain under checkpoint/coralnpu_ckpt/.
# ---------------------------------------------------------------------------
GEM5_OUTDIR="${SUPER_ROOT}/logs/sim/m5out"
mkdir -p "${GEM5_OUTDIR}"

if [ -f "${CORAL_BOOTED_CKPT}/m5.cpt" ]; then
  echo "Mode: restore from booted checkpoint"
  echo "Checkpoint: ${CORAL_BOOTED_CKPT}"
  echo "Resume script: ${CORAL_RESUME_BOOTSCRIPT}"
  ./"${GEM5_BUILD_TARGET}" ${GEM5_OPTIONS} --outdir="${GEM5_OUTDIR}" "${GEM5_CONFIG}" \
    ${CORAL_CONFIG_OPTIONS} \
    --cpu-type="${GEM5_CPU_TYPE}" \
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
  ./"${GEM5_BUILD_TARGET}" ${GEM5_OPTIONS} --outdir="${GEM5_OUTDIR}" "${GEM5_CONFIG}" \
    ${CORAL_CONFIG_OPTIONS} \
    --cpu-type="${GEM5_CPU_TYPE}" \
    --disk="${CORAL_DISK_IMG}" \
    --kernel="${CORAL_KERNEL_IMAGE}" \
    --kernel-init="${CORAL_KERNEL_INIT}" \
    --enable-npu \
    ${BOOTSTRAP_NPU_BACKEND_ARGS} \
    --ckpt-dir="${CORAL_CKPT_ROOT}" \
    --exit-after-checkpoint \
    --bootscript="${CORAL_CKPT_BOOTSCRIPT}" \
    --kernel-cmd="${CORAL_KERNEL_CMDLINE}"
  # Record metadata so subsequent runs can detect changes.
  printf '%s\n' "${CORAL_DISK_IMG}" > "${CORAL_CKPT_IMAGE_META}"
  printf '%s\n' "${CORAL_KERNEL_INIT}" > "${CORAL_CKPT_INIT_META}"
  printf '%s\n' "${CORAL_KERNEL_IMAGE}" > "${CORAL_CKPT_KERNEL_META}"
  printf '%s\n' "${CORAL_KERNEL_CMDLINE}" > "${CORAL_CKPT_CMDLINE_META}"
  printf '%s\n' "${CORAL_CKPT_FORMAT_VERSION}" > "${CORAL_CKPT_FORMAT_META}"
  echo "Boot checkpoint saved at ${CORAL_BOOTED_CKPT}"
  if [ "${CORAL_AUTO_RESUME_AFTER_CKPT}" = "1" ]; then
    echo "Automatically restoring the new checkpoint with backend ${CORAL_NPU_BACKEND}"
    export CORAL_REBUILD_CKPT=0
    exec "$0"
  fi
  echo "Run ./run_multicore.sh again to restore and execute the Coral NPU test script"
fi

# ---------------------------------------------------------------------------
# Legacy command-line references (uncomment as needed).
#
#   D9200 with Gemmini accelerator:
#     ./build/ARM/gem5.opt configs/example/arm/arm_multicore_d9200.py \
#       --cpu-type="D9200" \
#       --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img \
#       --kernel=${IMAGE_PATH}/vmlinux.arm64 \
#       --enable-gemmini --gemmini-cpu-idx 0
#
#   Slice-based simulation (save/restore individual time slices):
#     ./build/ARM/gem5.opt configs/example/arm/arm_multicore_slc.py \
#       --slice-run --slice-ticks 10ms \
#       --disk=${IMAGE_PATH}/ubuntu-18.04-arm64-docker.img \
#       --kernel=${IMAGE_PATH}/../binaries/vmlinux.arm64 \
#       --caches --last-cache-level 3 --cpu-type="x4a720"
#
#   Geekbench on aarch64:
#     taskset -c 0 ./geekbench_aarch64 --no-upload --multi-core \
#       --iterations 1 --skip-sysinfo --workload 204
# ---------------------------------------------------------------------------
