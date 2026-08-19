#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
PYTHON="${OPENNPUX_PYTHON:-python3}"
VENV="${OPENNPUX_HF_VENV:-${ROOT_DIR}/.venv/hf-numerical}"
REQUIREMENTS="${ROOT_DIR}/tools/models/requirements-hf-numerical.txt"

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "error: Python interpreter not found: $PYTHON" >&2
    exit 1
}

if [ ! -x "$VENV/bin/python" ]; then
    echo "[hf-env] creating $VENV" >&2
    "$PYTHON" -m venv --system-site-packages "$VENV" || {
        echo "error: unable to create Python venv" >&2
        echo "Ubuntu: sudo apt-get install python3-venv" >&2
        exit 1
    }
fi

"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel

if ! "$VENV/bin/python" -c 'import torch' >/dev/null 2>&1; then
    echo "[hf-env] PyTorch not found; installing torch and torchvision" >&2
    if [ -n "${OPENNPUX_TORCH_INDEX_URL:-}" ]; then
        "$VENV/bin/python" -m pip install \
            --index-url "$OPENNPUX_TORCH_INDEX_URL" torch torchvision
    else
        "$VENV/bin/python" -m pip install torch torchvision
    fi
elif ! "$VENV/bin/python" -c 'import torchvision' >/dev/null 2>&1; then
    echo "[hf-env] torchvision not found; installing a torch-compatible build" >&2
    if [ -n "${OPENNPUX_TORCH_INDEX_URL:-}" ]; then
        "$VENV/bin/python" -m pip install \
            --index-url "$OPENNPUX_TORCH_INDEX_URL" torchvision
    else
        "$VENV/bin/python" -m pip install torchvision
    fi
fi

"$VENV/bin/python" -m pip install -r "$REQUIREMENTS"

if [ "${OPENNPUX_INSTALL_GPTQMODEL:-1}" != 0 ] &&
   ! "$VENV/bin/python" -c \
       'import importlib.util; assert importlib.util.find_spec("gptqmodel")' \
       >/dev/null 2>&1; then
    echo "[hf-env] installing the GPTQModel backend" >&2
    "$VENV/bin/python" -m pip install --no-build-isolation gptqmodel
fi

"$VENV/bin/python" - <<'PY'
import accelerate
import numpy
import optimum
import safetensors
import torch
import torchvision
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

print(f"hf_env_python={__import__('sys').executable}")
print(f"hf_env_torch={torch.__version__}")
print(f"hf_env_torchvision={torchvision.__version__}")
print(f"hf_env_transformers={transformers.__version__}")
print(f"hf_env_multimodal_loader={AutoModelForMultimodalLM.__name__}")
print(f"hf_env_processor={AutoProcessor.__name__}")
print(f"hf_env_numpy={numpy.__version__}")
print(f"hf_env_accelerate={accelerate.__version__}")
print(f"hf_env_optimum={getattr(optimum, '__version__', 'installed')}")
print(f"hf_env_safetensors={safetensors.__version__}")
print(f"hf_env_cuda={torch.cuda.is_available()}")
print("hf_env=PASS")
PY

echo "[hf-env] use CORAL_HF_PYTHON=$VENV/bin/python" >&2
