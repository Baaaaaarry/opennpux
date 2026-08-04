#!/bin/sh
#
# download_bazel_deps.sh — Pre-download all Bazel dependencies into distdir.
#
# Extracts every http_archive / git_repository URL from the Coral Bazel workspace
# and downloads them with wget into thirdparty/coralnpu/distdir/.  Once populated,
# Bazel finds the files locally (via --distdir) and skips all network access.
#
# This is a one-time bootstrap step, only needed when the host cannot reach
# GitHub / Maven / SourceForge from within the Bazel JVM.
#
# Environment:
#   HTTPS_PROXY    proxy for HTTPS downloads (e.g. http://10.126.126.1:7897)
#
# @bootstrap-assets-spec  v1  2025-07-28

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
DISTDIR="${ROOT_DIR}/thirdparty/coralnpu/distdir"
CACHE_DIR="${ROOT_DIR}/.cache/coralnpu"

mkdir -p "${DISTDIR}"

# All external URLs referenced by the Coral WORKSPACE and rules/*.bzl.
# Extracted from:
#   thirdparty/coralnpu/WORKSPACE
#   thirdparty/coralnpu/rules/repos.bzl
#   sim/coralnpu/rules/repos.bzl
URLS="
https://github.com/bazelbuild/rules_cc/releases/download/0.2.9/rules_cc-0.2.9.tar.gz
https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.9.0.tar.gz
https://github.com/bazelbuild/rules_proto/releases/download/7.1.0/rules_proto-7.1.0.tar.gz
https://github.com/bazelbuild/rules_python/releases/download/0.26.0/rules_python-0.26.0.tar.gz
https://github.com/bazelbuild/rules_python/releases/download/0.40.0/rules_python-0.40.0.tar.gz
https://github.com/bazelbuild/rules_scala/releases/download/v6.6.0/rules_scala-v6.6.0.tar.gz
https://github.com/bazel-contrib/bazel_features/releases/download/v1.32.0/bazel_features-v1.32.0.tar.gz
https://github.com/abseil/abseil-cpp/releases/download/20250127.1/abseil-cpp-20250127.1.tar.gz
https://github.com/protocolbuffers/protobuf/releases/download/v29.6/protobuf-29.6.tar.gz
https://github.com/pybind/pybind11/archive/v3.0.1.zip
https://github.com/pybind/pybind11_abseil/archive/54b34dd0e8afb8a4febb9508c69410e708b43515.tar.gz
https://github.com/pybind/pybind11_bazel/releases/download/v2.13.6/pybind11_bazel-2.13.6.tar.gz
https://github.com/hedronvision/bazel-compile-commands-extractor/archive/1266d6a25314d165ca78d0061d3399e909b7920e.tar.gz
https://github.com/tensorflow/tflite-micro/archive/b75c6ff4e2270047f2b48fa01f833c8101c31f43.zip
https://github.com/chipsalliance/rocket-chip/archive/f517abbf41abb65cea37421d3559f9739efd00a9.zip
https://github.com/chipsalliance/diplomacy/archive/6590276fa4dac315ae7c7c01371b954c5687a473.zip
https://github.com/pulp-platform/common_cells/archive/6aeee85d0a34fedc06c14f04fd6363c9f7b4eeea.zip
https://github.com/pulp-platform/fpu_div_sqrt_mvp/archive/86e1f558b3c95e91577c41b2fc452c86b04e85ac.zip
https://github.com/openhwgroup/cvfpu/archive/58ca3c376beb914b2b80b811d4b270c063d4e6f7.zip
https://github.com/google/mpact-riscv/archive/cb68bd4a2cb80dea24d9760dc6397b5854ea41bd.tar.gz
https://github.com/google-coral/coralnpu-mpact/archive/e2a26e6d983f13d4c10875e4e5878a6171c04a06.zip
https://github.com/lowRISC/opentitan/archive/0e3cf62211004443d6d29f8f6120882376da499a.zip
https://github.com/riscv-software-src/riscv-tests/archive/fd4e6cdd033d9075632be9dd207c848181ca474c.zip
https://github.com/riscv-verification/RVVI/archive/5786f0d39b84f3fd15ef75b792bdea4281941afe.zip
https://github.com/FreeRTOS/FreeRTOS-Kernel/archive/refs/tags/V11.1.0.tar.gz
https://github.com/westes/flex/releases/download/v2.6.4/flex-2.6.4.tar.gz
https://github.com/accellera-official/systemc/archive/refs/tags/2.3.4.tar.gz
https://sourceforge.net/projects/srecord/files/srecord/1.65/srecord-1.65.0-Source.tar.gz/download
https://opensecura.googlesource.com/3p/ip/isp/+archive/d53dc0e0ce2605cea2e3b3fc5b97e9dd40f8d55a.tar.gz
https://github.com/coreutils/gnulib/archive/dbc5605c3b37a14d7c7e56fcf6c305d542e73210.zip
https://repo1.maven.org/maven2/org/chipsalliance/llvm-firtool/1.114.0/llvm-firtool-1.114.0.jar
https://cdn.azul.com/zulu/bin/zulu21.40.17-ca-jdk21.0.6-linux_x64.tar.gz
https://raw.githubusercontent.com/verilator/verilator/v5.028/include/vltstd/svdpi.h
"

echo "[distdir] downloading dependencies into ${DISTDIR}"
echo "[distdir] proxy: ${HTTPS_PROXY:-none}"

MAX_RETRIES=3
retry=0
TOTAL_DOWN=0

while [ "${retry}" -lt "${MAX_RETRIES}" ]; do
    retry=$((retry + 1))
    COUNT=0
    FAIL=0
    PASS=0

    for url in ${URLS}; do
        # Extract basename: last path component before any ? or /download suffix.
        filename=$(printf '%s' "${url}" | sed 's|/download$||' | rev | cut -d/ -f1 | rev)
        dest="${DISTDIR}/${filename}"

        if [ -f "${dest}" ]; then
            continue
        fi

        COUNT=$((COUNT + 1))
        echo "[distdir] (pass ${retry}) ${filename}"
        if wget -q --timeout=60 --tries=3 -O "${dest}" "${url}"; then
            PASS=$((PASS + 1))
        else
            echo "        FAILED: ${url}" >&2
            rm -f "${dest}"
            FAIL=$((FAIL + 1))
        fi
    done

    TOTAL_DOWN=$((TOTAL_DOWN + PASS))

    if [ "${FAIL}" -eq 0 ]; then
        echo "[distdir] pass ${retry}: ${PASS} downloaded, 0 failed — complete"
        break
    fi

    echo "[distdir] pass ${retry}: ${PASS} downloaded, ${FAIL} failed"
    if [ "${retry}" -lt "${MAX_RETRIES}" ]; then
        echo "[distdir] retrying failed downloads (pass $((retry + 1))/${MAX_RETRIES})..."
        sleep 2
    fi
done

echo
echo "[distdir] total: ${TOTAL_DOWN} downloaded, ${FAIL} failed after ${retry} pass(es)"

if [ "${FAIL}" -gt 0 ]; then
    echo "[distdir] some downloads could not be completed; re-run this script later" >&2
    echo "[distdir] or set HTTPS_PROXY and retry" >&2
    exit 1
fi
