#!/bin/sh

set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
TVM_VERSION="${TVM_VERSION:-0.24.0}"
TVM_ROOT="${TVM_ROOT:-${ROOT_DIR}/.cache/tvm}"
TVM_HOME="${TVM_HOME:-${TVM_ROOT}/apache-tvm-${TVM_VERSION}}"
TVM_BUILD_DIR="${TVM_BUILD_DIR:-${TVM_HOME}/build}"
TVM_VENV="${TVM_VENV:-${ROOT_DIR}/.venv/tvm-byoc}"
TVM_BUILD_JOBS="${TVM_BUILD_JOBS:-4}"

if [ -z "${PYTHON_BOOTSTRAP:-}" ]; then
    for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
        if command -v "${candidate}" >/dev/null 2>&1 &&
           "${candidate}" -c \
               'import sys; raise SystemExit(sys.version_info < (3, 9))'; then
            PYTHON_BOOTSTRAP="${candidate}"
            break
        fi
    done
fi
[ -n "${PYTHON_BOOTSTRAP:-}" ] || {
    echo "error: Python 3.9 or newer is required" >&2
    exit 1
}

command -v git >/dev/null
command -v cmake >/dev/null
command -v "${PYTHON_BOOTSTRAP}" >/dev/null

mkdir -p "${TVM_ROOT}" "${TVM_VENV}"
if [ ! -d "${TVM_HOME}/.git" ]; then
    git clone --branch "v${TVM_VERSION}" --depth 1 \
        https://github.com/apache/tvm.git "${TVM_HOME}"
fi
git -C "${TVM_HOME}" submodule update --init --depth 1 3rdparty/tvm-ffi
git -C "${TVM_HOME}/3rdparty/tvm-ffi" submodule update --init --depth 1

cmake -S "${TVM_HOME}" -B "${TVM_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_LLVM=OFF \
    -DUSE_RPC=OFF \
    -DUSE_SORT=ON \
    -DUSE_RANDOM=ON
cmake --build "${TVM_BUILD_DIR}" --parallel "${TVM_BUILD_JOBS}"

if [ ! -x "${TVM_VENV}/bin/python" ]; then
    "${PYTHON_BOOTSTRAP}" -m venv "${TVM_VENV}"
fi
"${TVM_VENV}/bin/python" -m pip install wheel setuptools
CMAKE_PREFIX_PATH="${TVM_BUILD_DIR}" \
    "${TVM_VENV}/bin/python" -m pip install -e "${TVM_HOME}/3rdparty/tvm-ffi"
"${TVM_VENV}/bin/python" -m pip install \
    numpy cloudpickle psutil scipy tornado typing_extensions pytest

cat >"${TVM_ROOT}/env.sh" <<EOF
export TVM_HOME="${TVM_HOME}"
export TVM_BUILD_DIR="${TVM_BUILD_DIR}"
export TVM_PYTHON="${TVM_VENV}/bin/python"
export PYTHONPATH="${TVM_HOME}/python\${PYTHONPATH:+:\$PYTHONPATH}"
export TVM_LIBRARY_PATH="${TVM_BUILD_DIR}/lib"
export LD_LIBRARY_PATH="${TVM_BUILD_DIR}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export DYLD_LIBRARY_PATH="${TVM_BUILD_DIR}/lib\${DYLD_LIBRARY_PATH:+:\$DYLD_LIBRARY_PATH}"
EOF

echo "TVM environment ready: ${TVM_ROOT}/env.sh"
echo "Run: . ${TVM_ROOT}/env.sh"
