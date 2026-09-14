#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODULAR_DIR="${MODULAR_SOURCE_DIR:-${ROOT_DIR}/thirdparty/modular}"

"${SCRIPT_DIR}/setup_modular_source.sh"

test -f "${MODULAR_DIR}/max/python/max/graph/graph.py"
test -f "${MODULAR_DIR}/max/python/max/graph/ops/matmul.py"
test -f "${MODULAR_DIR}/max/python/max/driver/driver.py"

# The adapter must stay source-independent: importing it must not import MAX.
PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 - <<'PY'
import sys
import opennpux_backend
import opennpux_mojo

unexpected = sorted(
    name for name in sys.modules if name == "max" or name.startswith("max.")
)
if unexpected:
    raise SystemExit(f"modular_source_adapter=FAIL imports={unexpected}")
print("modular_source_adapter=PASS")
PY

python3 -m unittest \
    tests.unit.models.test_opennpux_mojo_adapter \
    tests.unit.models.test_opennpux_backend_frontend

echo "modular_source_integration=PASS"
