#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
CORAL_REPO="${SUPER_ROOT}/thirdparty/coralnpu"
IMAGE_TAG="${CORAL_STANDALONE_DOCKER_IMAGE:-gem5-coral-standalone:latest}"
DOCKER_PLATFORM="${DOCKER_PLATFORM:-linux/amd64}"
BASE_IMAGE="${BASE_IMAGE:-ubuntu:24.04}"
BAZEL_CACHE_DIR="${CORAL_STANDALONE_BAZEL_CACHE_DIR:-${SUPER_ROOT}/.cache/coralnpu/bazel}"
REPO_CACHE_DIR="${CORAL_STANDALONE_REPO_CACHE_DIR:-${SUPER_ROOT}/.cache/coralnpu/repository}"
BAZELISK_CACHE_DIR="${CORAL_STANDALONE_BAZELISK_CACHE_DIR:-${SUPER_ROOT}/.cache/coralnpu/bazelisk}"
DISTDIR="${CORAL_STANDALONE_DISTDIR:-${CORAL_REPO}/distdir}"
CORAL_STANDALONE_CLEAN="${CORAL_STANDALONE_CLEAN:-0}"

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker not found in PATH" >&2
    exit 1
fi

if [ ! -d "${CORAL_REPO}" ]; then
    echo "error: Coral submodule missing: ${CORAL_REPO}" >&2
    exit 1
fi

mkdir -p "${BAZEL_CACHE_DIR}" "${REPO_CACHE_DIR}" "${BAZELISK_CACHE_DIR}" "${DISTDIR}"

if [ "${CORAL_STANDALONE_CLEAN}" = "1" ]; then
    echo "[coral-docker] cleaning persisted Bazel caches"
    rm -rf "${BAZEL_CACHE_DIR}" "${REPO_CACHE_DIR}"
    mkdir -p "${BAZEL_CACHE_DIR}" "${REPO_CACHE_DIR}"
fi

OVERRIDE_FLAGS=

prepare_override_from_tarball() {
    repo_name="$1"
    tarball_path="$2"
    vendor_dir="$3"

    if [ ! -f "${tarball_path}" ]; then
        return 0
    fi

    if [ ! -d "${vendor_dir}" ]; then
        echo "[coral-docker] preparing override repository ${repo_name} from $(basename "${tarball_path}")"
        mkdir -p "${vendor_dir}"
        tar -xzf "${tarball_path}" -C "${vendor_dir}"
    fi

    OVERRIDE_FLAGS="${OVERRIDE_FLAGS} --override_repository=${repo_name}=/workspace/super${vendor_dir#${SUPER_ROOT}}"
}

prepare_override_from_tarball \
    "rules_java" \
    "${DISTDIR}/rules_java-8.14.0.tar.gz" \
    "${BAZEL_CACHE_DIR}/vendor_overrides/rules_java-8.14.0"

echo "[coral-docker] checking local base image ${BASE_IMAGE} for ${DOCKER_PLATFORM}"
if ! docker run --rm --platform "${DOCKER_PLATFORM}" "${BASE_IMAGE}" true >/tmp/coral_docker_check.out 2>/tmp/coral_docker_check.err; then
    cat >&2 <<EOF
error: the requested base image is not locally runnable for ${DOCKER_PLATFORM}
  base image: ${BASE_IMAGE}

This Coral standalone flow needs a Linux/amd64 container because the official Coral
toolchains are registered for an x86_64 Linux exec platform.

On this host, the local Ubuntu cache currently only provides an arm64 variant,
so Docker tries to resolve the amd64 manifest from docker.io and falls back to
network/DNS. To proceed offline or with unstable DNS, import an amd64 Ubuntu
rootfs tarball as a local image, then point BASE_IMAGE at that local tag.

Example:
  docker import --platform linux/amd64 /absolute/path/ubuntu-amd64-rootfs.tar.gz local/ubuntu-amd64:noble
  BASE_IMAGE=local/ubuntu-amd64:noble ./tools/coralnpu/build_coral_standalone_docker.sh

Original Docker error:
EOF
    sed -n '1,20p' /tmp/coral_docker_check.err >&2 || true
    rm -f /tmp/coral_docker_check.out /tmp/coral_docker_check.err
    exit 1
fi
rm -f /tmp/coral_docker_check.out /tmp/coral_docker_check.err

echo "[coral-docker] building ${IMAGE_TAG} for ${DOCKER_PLATFORM}"
docker build \
    --pull=false \
    --platform "${DOCKER_PLATFORM}" \
    --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
    -t "${IMAGE_TAG}" \
    -f "${SCRIPT_DIR}/coral_standalone_env.Dockerfile" \
    "${SUPER_ROOT}"

TARGETS="${*:-//examples:coralnpu_v2_hello_world_add_floats //tests/verilator_sim:core_mini_axi_sim}"

echo "[coral-docker] running Coral standalone builds in ${DOCKER_PLATFORM}"
echo "[coral-docker] targets: ${TARGETS}"

exec docker run --rm -t \
    --platform "${DOCKER_PLATFORM}" \
    -v "${SUPER_ROOT}:/workspace/super" \
    -v "${BAZEL_CACHE_DIR}:/workspace/.bazel-cache" \
    -v "${REPO_CACHE_DIR}:/workspace/.bazel-repo-cache" \
    -v "${BAZELISK_CACHE_DIR}:/root/.cache/bazelisk" \
    -v "${DISTDIR}:/workspace/super/thirdparty/coralnpu/distdir" \
    -w /workspace/super/thirdparty/coralnpu \
    "${IMAGE_TAG}" \
    bash -lc "
        set -eu
        bazel shutdown || true
        bazel \
          --output_user_root=/workspace/.bazel-cache \
          build \
          ${OVERRIDE_FLAGS} \
          --java_runtime_version=local_jdk \
          --tool_java_runtime_version=local_jdk \
          --repository_cache=/workspace/.bazel-repo-cache \
          --distdir=/workspace/super/thirdparty/coralnpu/distdir \
          ${TARGETS}
    "
