#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
CORAL_REPO="${CORAL_REPO:-${SUPER_ROOT}/thirdparty/coralnpu}"

required_bazel_version=
if [ -f "${CORAL_REPO}/.bazelversion" ]; then
    required_bazel_version="$(head -n 1 "${CORAL_REPO}/.bazelversion" | tr -d '\r')"
fi

status=0

echo "[coral-host] coral repo: ${CORAL_REPO}"
echo "[coral-host] required bazel: ${required_bazel_version:-unknown}"

host_os="$(uname -s)"
host_arch="$(uname -m)"
echo "[coral-host] host platform: ${host_os} ${host_arch}"

if [ "${host_os}" != "Linux" ] || [ "${host_arch}" != "x86_64" ]; then
    cat <<EOF
[coral-host] note: official Coral example/simulator builds require an x86_64 Linux
[coral-host]       exec platform. Use ./tools/coralnpu/build_coral_standalone_docker.sh
[coral-host]       for Coral standalone validation on this host.
EOF
fi

check_cmd() {
    name="$1"
    cmd="$2"
    if sh -c "${cmd}" >/tmp/coral_host_check.out 2>/tmp/coral_host_check.err; then
        printf '[coral-host] ok:   %s\n' "${name}"
        sed -n '1,2p' /tmp/coral_host_check.out
    else
        printf '[coral-host] miss: %s\n' "${name}"
        sed -n '1,2p' /tmp/coral_host_check.err || true
        status=1
    fi
}

check_cmd "python3.12" "command -v python3.12 >/dev/null 2>&1 && python3.12 --version"
check_cmd "verilator" "command -v verilator >/dev/null 2>&1 && verilator --version"
check_cmd "srec_cat" "command -v srec_cat >/dev/null 2>&1 && srec_cat --version"

if command -v bazel >/dev/null 2>&1; then
    if [ -n "${required_bazel_version}" ] && \
       USE_BAZEL_VERSION="${required_bazel_version}" bazel --version >/tmp/coral_host_check.out 2>/tmp/coral_host_check.err; then
        printf '[coral-host] ok:   bazel %s\n' "${required_bazel_version}"
        sed -n '1,2p' /tmp/coral_host_check.out
    else
        printf '[coral-host] miss: bazel %s\n' "${required_bazel_version:-unknown}"
        sed -n '1,4p' /tmp/coral_host_check.err || true
        status=1
    fi
else
    echo "[coral-host] miss: bazel"
    status=1
fi

rm -f /tmp/coral_host_check.out /tmp/coral_host_check.err

exit "${status}"
