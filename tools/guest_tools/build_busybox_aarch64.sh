#!/bin/sh
#
# build_busybox_aarch64.sh — Build a minimal aarch64 BusyBox for the gem5 guest.
#
# Cross-compiles a static BusyBox binary containing only the insmod applet.
# The binary is installed into the guest disk image so the boot checkpoint
# can preload it into tmpfs.  After checkpoint restore, BusyBox loads the
# opennpux-coral kernel module without touching virtio-blk — the current
# gem5 virtio queue has a known restoration defect that triggers SIGBUS
# on post-restore disk reads.
#
# The build is pinned to BusyBox 1.36.1 with SHA-256 verification.
#
# Pipeline:
#   Download busybox-1.36.1.tar.bz2 (cached under .cache/busybox/)
#     → allnoconfig → enable STATIC + BUSYBOX + INSMOD
#       → make busybox → aarch64-linux-gnu-strip
#         → validate with qemu-aarch64
#           → build/guest-tools/busybox-aarch64
#
# Output:
#   build/guest-tools/busybox-aarch64    static aarch64 binary (~1 MB)
#
# Environment:
#   BUSYBOX_VERSION      source version (default: 1.36.1)
#   BUSYBOX_URL          download URL
#   BUSYBOX_SHA256       expected SHA-256 (pinned for 1.36.1)
#   BUSYBOX_TARBALL      path to pre-downloaded tarball (offline mode)
#   CROSS_COMPILE        cross-compiler prefix (default: aarch64-linux-gnu-)
#   JOBS                 parallel build jobs (default: nproc)
#   BUSYBOX_SKIP_VALIDATE  1 = skip qemu-aarch64 post-build validation
#
# Host dependencies:
#   gcc-aarch64-linux-gnu  libc6-dev-arm64-cross  make  tar  curl/wget
#   qemu-user (for post-build validation of the aarch64 binary)
#
# @guest-tools-spec  v1  2025-07-29
# @synchronized-with  tools/guest_tools/install_module_loader_to_image.sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

BUSYBOX_VERSION="${BUSYBOX_VERSION:-1.36.1}"
BUSYBOX_URL="${BUSYBOX_URL:-https://busybox.net/downloads/busybox-${BUSYBOX_VERSION}.tar.bz2}"
CROSS_COMPILE="${CROSS_COMPILE:-aarch64-linux-gnu-}"
CACHE_DIR="${BUSYBOX_CACHE_DIR:-${ROOT_DIR}/.cache/busybox}"
BUILD_DIR="${BUSYBOX_BUILD_DIR:-${ROOT_DIR}/build/busybox-aarch64}"
OUT="${BUSYBOX_OUT:-${ROOT_DIR}/build/guest-tools/busybox-aarch64}"
TARBALL="${BUSYBOX_TARBALL:-${CACHE_DIR}/busybox-${BUSYBOX_VERSION}.tar.bz2}"

# ---------------------------------------------------------------------------
# Pinned SHA-256 for BusyBox 1.36.1.
# ---------------------------------------------------------------------------
case "${BUSYBOX_VERSION}" in
    1.36.1)
        DEFAULT_SHA256=b8cc24c9574d809e7279c3be349795c5d5ceb6fdf19ca709f80cde50e47de314
        ;;
    *)
        DEFAULT_SHA256=""
        ;;
esac
BUSYBOX_SHA256="${BUSYBOX_SHA256:-${DEFAULT_SHA256}}"

if command -v nproc >/dev/null 2>&1; then
    DEFAULT_JOBS="$(nproc)"
else
    DEFAULT_JOBS=1
fi
JOBS="${JOBS:-${DEFAULT_JOBS}}"

# ---------------------------------------------------------------------------
# Helper for consistent error messages.
# ---------------------------------------------------------------------------
fail() {
    echo "error: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Preflight: host tools must be available.
# ---------------------------------------------------------------------------
for tool in make tar; do
    command -v "${tool}" >/dev/null 2>&1 || fail "${tool} not found"
done
command -v "${CROSS_COMPILE}gcc" >/dev/null 2>&1 || {
    echo "error: ${CROSS_COMPILE}gcc not found" >&2
    echo "hint: sudo apt-get install gcc-aarch64-linux-gnu libc6-dev-arm64-cross" >&2
    exit 1
}

# Build output is saved to a timestamped log for CI archiving.
BUILD_TS="$(date +%Y-%m-%dT%H-%M-%S)"
BUILD_LOG="${ROOT_DIR}/logs/build/busybox-build-${BUILD_TS}.log"
BUILD_LATEST="${ROOT_DIR}/logs/build/busybox-build.log"
mkdir -p "$(dirname "${BUILD_LOG}")"
ln -sfn "busybox-build-${BUILD_TS}.log" "${BUILD_LATEST}"
echo "[busybox] build started $(date -Iseconds)" | tee -a "${BUILD_LOG}"

# ---------------------------------------------------------------------------
# Step 1: Download or reuse the source tarball (with SHA-256 verification).
# ---------------------------------------------------------------------------
mkdir -p "${CACHE_DIR}" "$(dirname -- "${TARBALL}")" "$(dirname -- "${OUT}")"
rm -f "${OUT}"

if [ ! -f "${TARBALL}" ]; then
    echo "Downloading ${BUSYBOX_URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --retry-delay 2 -o "${TARBALL}.tmp" "${BUSYBOX_URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${TARBALL}.tmp" "${BUSYBOX_URL}"
    else
        fail "curl or wget is required to download BusyBox"
    fi
    mv "${TARBALL}.tmp" "${TARBALL}"
else
    echo "Using cached tarball: ${TARBALL}"
    echo "  (set BUSYBOX_TARBALL=/path/to/busybox-${BUSYBOX_VERSION}.tar.bz2 for offline builds)"
fi 2>&1 | tee -a "${BUILD_LOG}"

if [ -n "${BUSYBOX_SHA256}" ]; then
    sha_ok=0
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "${BUSYBOX_SHA256}" "${TARBALL}" | sha256sum -c - && sha_ok=1
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "${TARBALL}" | awk '{print $1}')"
        [ "${actual}" = "${BUSYBOX_SHA256}" ] && sha_ok=1
    else
        fail "sha256sum or shasum is required to verify BusyBox"
    fi
    if [ "${sha_ok}" != "1" ]; then
        cat >&2 <<EOF
error: BusyBox SHA-256 verification failed.
The downloaded tarball may be corrupted or tampered with.
To re-download:  rm "${TARBALL}" && re-run this script.
To skip verification (not recommended):  BUSYBOX_SHA256="" $0
To use a pre-verified tarball:  BUSYBOX_TARBALL=/path/to/verified.tar.bz2 $0
EOF
        exit 1
    fi
else
    echo "warning: no SHA-256 is pinned for BusyBox ${BUSYBOX_VERSION}" >&2
    echo "warning: set BUSYBOX_SHA256 to verify the source archive" >&2
fi 2>&1 | tee -a "${BUILD_LOG}"

# ---------------------------------------------------------------------------
# Step 2: Extract source (cached — only done once per version).
# ---------------------------------------------------------------------------
SRC_DIR="${CACHE_DIR}/busybox-${BUSYBOX_VERSION}"
if [ ! -f "${SRC_DIR}/Makefile" ]; then
    rm -rf "${SRC_DIR}"
    tar -xjf "${TARBALL}" -C "${CACHE_DIR}"
fi

# ---------------------------------------------------------------------------
# Step 3: Configure — start from allnoconfig, then enable only what we need.
#   STATIC: no shared-lib dependency in the guest (libc not required).
#   BUSYBOX: the multicall binary entry point.
#   INSMOD: the only applet we need — load kernel modules.
# ---------------------------------------------------------------------------
# Safety: BUILD_DIR must be an absolute path under the project root.
case "${BUILD_DIR}" in
    "${ROOT_DIR}/build/"*) ;;
    *) fail "BUILD_DIR is outside the project build tree: ${BUILD_DIR}" ;;
esac
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

make -C "${SRC_DIR}" O="${BUILD_DIR}" \
    ARCH=arm64 CROSS_COMPILE="${CROSS_COMPILE}" allnoconfig

enable_config() {
    key="CONFIG_$1"
    awk -v key="${key}" '
        $0 == "# " key " is not set" || index($0, key "=") == 1 {
            if (!written) {
                print key "=y"
                written = 1
            }
            next
        }
        { print }
        END {
            if (!written)
                print key "=y"
        }
    ' "${BUILD_DIR}/.config" > "${BUILD_DIR}/.config.tmp"
    mv "${BUILD_DIR}/.config.tmp" "${BUILD_DIR}/.config"
}

enable_config STATIC
enable_config BUSYBOX
enable_config INSMOD

# ---------------------------------------------------------------------------
# Step 4: Resolve dependencies (oldconfig) and verify the three options stuck.
# ---------------------------------------------------------------------------
yes '' | make -C "${SRC_DIR}" O="${BUILD_DIR}" \
    ARCH=arm64 CROSS_COMPILE="${CROSS_COMPILE}" oldconfig >/dev/null
for option in STATIC BUSYBOX INSMOD; do
    grep -q "^CONFIG_${option}=y$" "${BUILD_DIR}/.config" || \
        fail "BusyBox configuration rejected CONFIG_${option}=y"
done

# ---------------------------------------------------------------------------
# Step 5: Build and strip.
# ---------------------------------------------------------------------------
make -C "${SRC_DIR}" O="${BUILD_DIR}" \
    ARCH=arm64 CROSS_COMPILE="${CROSS_COMPILE}" -j"${JOBS}" busybox 2>&1 \
    | tee -a "${BUILD_LOG}"

# Verify the make step actually produced a binary.
if [ ! -f "${BUILD_DIR}/busybox" ]; then
    fail "make completed but ${BUILD_DIR}/busybox was not produced"
fi

"${CROSS_COMPILE}strip" -s "${BUILD_DIR}/busybox" 2>/dev/null || true
install -m 0755 "${BUILD_DIR}/busybox" "${OUT}"

# ---------------------------------------------------------------------------
# Step 6: Validate the binary runs on aarch64 and contains the insmod applet.
#   qemu-aarch64 runs the binary in user-mode emulation; --list prints all
#   compiled-in applets.  Validation is skipped when qemu is unavailable
#   (common in CI environments without qemu-user) or when explicitly
#   disabled via BUSYBOX_SKIP_VALIDATE=1.
# ---------------------------------------------------------------------------
if [ "${BUSYBOX_SKIP_VALIDATE:-0}" = "1" ]; then
    echo "Skipping qemu-aarch64 validation (BUSYBOX_SKIP_VALIDATE=1)"
elif command -v qemu-aarch64 >/dev/null 2>&1; then
    applets="$(qemu-aarch64 "${OUT}" --list)"
    if printf '%s\n' "${applets}" | grep -qx insmod; then
        echo "verified with qemu-aarch64: insmod"
    else
        rm -f "${OUT}"
        fail "insmod applet is missing from the compiled binary"
    fi
else
    echo "warning: qemu-aarch64 not found; skipping binary validation" >&2
    echo "warning: install qemu-user or set BUSYBOX_SKIP_VALIDATE=1 to suppress" >&2
fi

file "${OUT}" 2>/dev/null || true
size_bytes="$(wc -c < "${OUT}" | tr -d ' ')"
echo "built: ${OUT} (${size_bytes} bytes)" | tee -a "${BUILD_LOG}"
echo "[busybox] build succeeded $(date -Iseconds)" >> "${BUILD_LOG}"
