#!/bin/sh
#
# phase2_prepare_bazel.sh — Install a repository-local Bazel for Coral NPU builds.
#
# The official Coral build system requires a specific Bazel version (pinned in
# thirdparty/coralnpu/.bazelversion).  This script downloads that exact version
# and installs it under .cache/coralnpu/bin/, isolated from any system Bazel.
#
# The host must be x86_64 Linux — Coral toolchains are registered for that
# exec platform and do not support macOS or arm64.
#
# Pipeline:
#   phase2_prepare_bazel.sh
#     → .cache/coralnpu/bin/bazel          (versioned binary)
#     → .cache/coralnpu/bin/bazel-<ver>    (symlink target)
#
# Environment:
#   BAZEL_BINARY          path to a pre-downloaded Bazel executable (offline mode)
#   BAZEL_DOWNLOAD_URL    override download URL
#   BAZEL_INSTALL_DIR     installation directory (default: .cache/coralnpu/bin)
#
# @coral-build-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
INSTALL_DIR="${BAZEL_INSTALL_DIR:-${ROOT_DIR}/.cache/coralnpu/bin}"
VERSION_FILE="${CORAL_REPO}/.bazelversion"

# ---------------------------------------------------------------------------
# Preflight: the Coral submodule must be initialized.
# ---------------------------------------------------------------------------
if [ ! -f "${VERSION_FILE}" ]; then
    echo "error: missing Coral Bazel version file: ${VERSION_FILE}" >&2
    exit 1
fi

VERSION="$(head -n 1 "${VERSION_FILE}" | tr -d '\r')"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"

# ---------------------------------------------------------------------------
# Coral toolchains are registered for x86_64 Linux only.
# ---------------------------------------------------------------------------
if [ "${HOST_OS}" != "Linux" ] || [ "${HOST_ARCH}" != "x86_64" ]; then
    echo "error: Coral RTL bridge requires x86_64 Linux; found ${HOST_OS} ${HOST_ARCH}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Install Bazel: from BAZEL_BINARY if provided, reuse cached copy, or download.
# ---------------------------------------------------------------------------
mkdir -p "${INSTALL_DIR}"
DEST="${INSTALL_DIR}/bazel-${VERSION}-linux-x86_64"
LINK="${INSTALL_DIR}/bazel"

if [ -n "${BAZEL_BINARY:-}" ]; then
    # Offline mode: user provided a pre-downloaded binary.
    if [ ! -f "${BAZEL_BINARY}" ]; then
        echo "error: BAZEL_BINARY not found: ${BAZEL_BINARY}" >&2
        exit 1
    fi
    cp "${BAZEL_BINARY}" "${DEST}"
elif [ -s "${DEST}" ]; then
    # Already cached; do not redownload Bazel on restricted hosts.
    echo "[coral-bazel] reusing cached ${DEST}"
else
    # Download from the official Bazel release mirror.
    URL="${BAZEL_DOWNLOAD_URL:-https://releases.bazel.build/${VERSION}/release/bazel-${VERSION}-linux-x86_64}"
    TMP="${DEST}.tmp"
    rm -f "${DEST}" "${TMP}"
    echo "[coral-bazel] downloading ${URL}"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --retry 3 --connect-timeout 20 -o "${TMP}" "${URL}"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "${TMP}" "${URL}"
    else
        echo "error: install curl/wget or set BAZEL_BINARY to a local Bazel executable" >&2
        exit 1
    fi
    mv "${TMP}" "${DEST}"
fi

# ---------------------------------------------------------------------------
# Make executable, create convenience symlink, and verify the version.
# ---------------------------------------------------------------------------
chmod 0755 "${DEST}"
ln -sfn "$(basename "${DEST}")" "${LINK}"

ACTUAL="$("${LINK}" --version)"
case "${ACTUAL}" in
    *"${VERSION}"*) ;;
    *)
        echo "error: installed Bazel version mismatch: ${ACTUAL}" >&2
        exit 1
        ;;
esac

echo "[coral-bazel] ready: ${LINK}"
echo "[coral-bazel] ${ACTUAL}"
