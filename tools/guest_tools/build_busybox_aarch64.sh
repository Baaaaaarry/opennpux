#!/bin/sh

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

fail() {
    echo "error: $*" >&2
    exit 1
}

for tool in make tar; do
    command -v "${tool}" >/dev/null 2>&1 || fail "${tool} not found"
done
command -v "${CROSS_COMPILE}gcc" >/dev/null 2>&1 || {
    echo "error: ${CROSS_COMPILE}gcc not found" >&2
    echo "hint: sudo apt-get install gcc-aarch64-linux-gnu libc6-dev-arm64-cross" >&2
    exit 1
}

mkdir -p "${CACHE_DIR}" "$(dirname -- "${TARBALL}")" "$(dirname -- "${OUT}")"

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
fi

if [ -n "${BUSYBOX_SHA256}" ]; then
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s  %s\n' "${BUSYBOX_SHA256}" "${TARBALL}" | sha256sum -c -
    elif command -v shasum >/dev/null 2>&1; then
        actual="$(shasum -a 256 "${TARBALL}" | awk '{print $1}')"
        [ "${actual}" = "${BUSYBOX_SHA256}" ] || fail "BusyBox SHA-256 mismatch"
    else
        fail "sha256sum or shasum is required to verify BusyBox"
    fi
else
    echo "warning: no SHA-256 is pinned for BusyBox ${BUSYBOX_VERSION}" >&2
    echo "warning: set BUSYBOX_SHA256 to verify the source archive" >&2
fi

SRC_DIR="${CACHE_DIR}/busybox-${BUSYBOX_VERSION}"
if [ ! -f "${SRC_DIR}/Makefile" ]; then
    rm -rf "${SRC_DIR}"
    tar -xjf "${TARBALL}" -C "${CACHE_DIR}"
fi

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

# Select both applet names, but use their shared small implementation.
enable_config STATIC
enable_config PLATFORM_LINUX
enable_config INSMOD
enable_config MODPROBE
enable_config MODPROBE_SMALL

yes '' | make -C "${SRC_DIR}" O="${BUILD_DIR}" \
    ARCH=arm64 CROSS_COMPILE="${CROSS_COMPILE}" oldconfig >/dev/null
for option in STATIC PLATFORM_LINUX INSMOD MODPROBE MODPROBE_SMALL; do
    grep -q "^CONFIG_${option}=y$" "${BUILD_DIR}/.config" || \
        fail "BusyBox configuration rejected CONFIG_${option}=y"
done
make -C "${SRC_DIR}" O="${BUILD_DIR}" \
    ARCH=arm64 CROSS_COMPILE="${CROSS_COMPILE}" -j"${JOBS}" busybox

"${CROSS_COMPILE}strip" -s "${BUILD_DIR}/busybox" 2>/dev/null || true
install -m 0755 "${BUILD_DIR}/busybox" "${OUT}"

if command -v "${CROSS_COMPILE}strings" >/dev/null 2>&1; then
    applet_strings="$("${CROSS_COMPILE}strings" "${OUT}")"
elif command -v strings >/dev/null 2>&1; then
    applet_strings="$(strings "${OUT}")"
else
    fail "strings not found; install binutils to validate BusyBox applets"
fi
printf '%s\n' "${applet_strings}" | grep -qx insmod || fail "insmod applet is missing"
printf '%s\n' "${applet_strings}" | grep -qx modprobe || fail "modprobe applet is missing"

file "${OUT}" 2>/dev/null || true
size_bytes="$(wc -c < "${OUT}" | tr -d ' ')"
echo "built: ${OUT} (${size_bytes} bytes)"

if command -v qemu-aarch64 >/dev/null 2>&1; then
    applets="$(qemu-aarch64 "${OUT}" --list)"
    printf '%s\n' "${applets}" | grep -qx insmod || fail "insmod applet is missing"
    printf '%s\n' "${applets}" | grep -qx modprobe || fail "modprobe applet is missing"
    echo "verified with qemu-aarch64: insmod, modprobe"
else
    echo "note: install qemu-user to execute the optional host-side applet check"
fi
