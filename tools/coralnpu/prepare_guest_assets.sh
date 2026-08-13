#!/bin/sh
#
# phase45_prepare_guest_assets.sh — Install guest-side assets for Phase 4/5 tests.
#
# Convenience wrapper that chains four steps into one:
#   1. Build coralctl (aarch64 NPU control binary)
#   2. Generate the heterogeneous smoke model (.npxm container)
#   3. Install coralctl into the guest disk image
#   4. Install the model file into the guest disk image
#
# The model mixes an official Coral software operator and a custom RTL
# accelerator operator in one .npxm container, exercising the same
# command ABI that MobileNet and other workloads use.
#
# Pipeline:
#   build_coralctl.sh                          → build/guest-tools/coralctl-aarch64
#   create_sample_model.py                     → build/models/heterogeneous-smoke.npxm
#   install_coralctl_to_image.sh <image>       → image:/usr/local/bin/coralctl
#   install_model_to_image.sh <image> <model>  → image:/tmp/model.npxm
#
# Environment:
#   IMAGE    path to the ARM64 disk image
#            (default: $IMAGE_PATH/ubuntu-18.04-arm64-docker.img)
#
# After running this script, rebuild the boot checkpoint once so the new
# coralctl and model are captured in the checkpoint's tmpfs.
#
# @guest-tools-spec  v1  2025-07-29

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="${IMAGE:-${CORAL_DISK_IMG:-${IMAGE_PATH:-${HOME}/wlk/gem5_arm_linux_images}/ubuntu-18.04-arm64-docker.img}}"
MODEL="${ROOT_DIR}/build/models/heterogeneous-smoke.npxm"

echo "Baseline guest asset image: ${IMAGE}"

# ---------------------------------------------------------------------------
# Step 1: Build coralctl (cross-compile for aarch64).
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"

# ---------------------------------------------------------------------------
# Step 2: Generate the heterogeneous smoke model.
#   create_sample_model.py produces a .npxm container with two operators:
#   one official software node and one custom RTL node.
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/models/create_sample_model.py" "${MODEL}"

# ---------------------------------------------------------------------------
# Step 3: Install init and coralctl into the guest image (requires sudo).
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/kernel/install_opennpux_init_to_image.sh" "${IMAGE}"
"${ROOT_DIR}/tools/guest_tools/install_coralctl_to_image.sh" "${IMAGE}"

# ---------------------------------------------------------------------------
# Step 4: Install the model .npxm into the guest image.
# ---------------------------------------------------------------------------
"${ROOT_DIR}/tools/guest_tools/install_model_to_image.sh" "${IMAGE}" "${MODEL}"

echo "Baseline guest assets installed into ${IMAGE}"
echo "Rebuild the boot checkpoint once before running model tests"
