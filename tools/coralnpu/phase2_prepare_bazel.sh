#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
CORAL_REPO="${CORAL_REPO:-${ROOT_DIR}/thirdparty/coralnpu}"
INSTALL_DIR="${BAZEL_INSTALL_DIR:-${ROOT_DIR}/.cache/coralnpu/bin}"
VERSION_FILE="${CORAL_REPO}/.bazelversion"

if [ ! -f "${VERSION_FILE}" ]; then
    echo "error: missing Coral Bazel version file: ${VERSION_FILE}" >&2
    exit 1
fi

VERSION="$(head -n 1 "${VERSION_FILE}" | tr -d '\r')"
HOST_OS="$(uname -s)"
HOST_ARCH="$(uname -m)"

if [ "${HOST_OS}" != "Linux" ] || [ "${HOST_ARCH}" != "x86_64" ]; then
    echo "error: Coral Phase 2 requires x86_64 Linux; found ${HOST_OS} ${HOST_ARCH}" >&2
    exit 1
fi

mkdir -p "${INSTALL_DIR}"
DEST="${INSTALL_DIR}/bazel-${VERSION}-linux-x86_64"
LINK="${INSTALL_DIR}/bazel"

if [ -n "${BAZEL_BINARY:-}" ]; then
    if [ ! -f "${BAZEL_BINARY}" ]; then
        echo "error: BAZEL_BINARY not found: ${BAZEL_BINARY}" >&2
        exit 1
    fi
    cp "${BAZEL_BINARY}" "${DEST}"
elif [ ! -x "${DEST}" ]; then
    URL="${BAZEL_DOWNLOAD_URL:-https://releases.bazel.build/${VERSION}/release/bazel-${VERSION}-linux-x86_64}"
    TMP="${DEST}.tmp"
    rm -f "${TMP}"
    echo "[phase2-bazel] downloading ${URL}"
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

echo "[phase2-bazel] ready: ${LINK}"
echo "[phase2-bazel] ${ACTUAL}"
