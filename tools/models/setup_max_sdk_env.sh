#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
ENV_DIR="${MAX_SDK_ENV_DIR:-${ROOT_DIR}/.venv/max-sdk}"
BOOTSTRAP_PYTHON="${MAX_SDK_PYTHON:-python3}"
PACKAGE_INDEX="${MAX_PACKAGE_INDEX:-https://whl.modular.com/nightly/simple/}"

"${BOOTSTRAP_PYTHON}" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info[:2] <= (3, 14)):
    raise SystemExit(
        "MAX SDK requires Python 3.10-3.14; set MAX_SDK_PYTHON accordingly"
    )
PY

if [ ! -x "${ENV_DIR}/bin/python" ]; then
    "${BOOTSTRAP_PYTHON}" -m venv "${ENV_DIR}"
fi

"${ENV_DIR}/bin/python" -m pip install --upgrade pip wheel
"${ENV_DIR}/bin/python" -m pip install --pre \
    --index-url "${PACKAGE_INDEX}" modular
"${ENV_DIR}/bin/python" - <<'PY'
import max
from max.graph import Graph, TensorType, ops
print(f"max_sdk_import=PASS module={max.__file__}")
PY

cat >"${ENV_DIR}/env.sh" <<EOF
export MAX_SDK_ENV_DIR='${ENV_DIR}'
export MAX_SDK_PYTHON='${ENV_DIR}/bin/python'
EOF

echo "max_sdk_python=${ENV_DIR}/bin/python"
echo "max_sdk_environment=PASS"
