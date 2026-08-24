#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)"
PYTHON="${OPENNPUX_PYTHON:-}"
VENV="${OPENNPUX_HF_VENV:-${ROOT_DIR}/.venv/hf-numerical}"
REQUIREMENTS="${ROOT_DIR}/tools/models/requirements-hf-numerical.txt"
PYPI_INDEX_URL="${OPENNPUX_PYPI_INDEX_URL:-}"

check_python_ssl()
{
    interpreter="$1"
    label="$2"
    if "$interpreter" -c \
        'import ssl; print(ssl.OPENSSL_VERSION)' >/dev/null 2>&1; then
        return 0
    fi

    echo "error: $label lacks Python SSL support: $interpreter" >&2
    echo "HTTPS package indexes cannot be used until 'import ssl' succeeds." >&2
    echo "Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y ca-certificates openssl python3-full python3-venv" >&2
    echo "Then select the distro interpreter with OPENNPUX_PYTHON=/usr/bin/python3." >&2
    return 1
}

check_python_version()
{
    interpreter="$1"
    label="$2"
    if "$interpreter" -c \
        'import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)'; then
        return 0
    fi

    version="$($interpreter -c \
        'import sys; print(".".join(map(str, sys.version_info[:3])))' \
        2>/dev/null || echo unknown)"
    echo "error: $label uses Python $version; transformers>=4.57 requires Python 3.9+" >&2
    echo "Install a newer distro Python and select it with OPENNPUX_PYTHON=/path/to/python3." >&2
    return 1
}

pip_install()
{
    if [ -n "$PYPI_INDEX_URL" ]; then
        "$VENV/bin/python" -m pip install \
            --index-url "$PYPI_INDEX_URL" "$@"
    else
        "$VENV/bin/python" -m pip install "$@"
    fi
}

if [ -z "$PYTHON" ]; then
    for candidate in \
        python3 python3.13 python3.12 python3.11 python3.10 python3.9; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c \
               'import ssl, sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)' \
               >/dev/null 2>&1; then
            PYTHON="$candidate"
            break
        fi
    done
fi
PYTHON="${PYTHON:-python3}"

command -v "$PYTHON" >/dev/null 2>&1 || {
    echo "error: Python interpreter not found: $PYTHON" >&2
    exit 1
}

check_python_ssl "$PYTHON" "base interpreter" || exit 1
check_python_version "$PYTHON" "base interpreter" || exit 1
echo "[hf-env] selected Python: $(command -v "$PYTHON") ($("$PYTHON" --version 2>&1))" >&2

if [ ! -x "$VENV/bin/python" ]; then
    echo "[hf-env] creating $VENV" >&2
    "$PYTHON" -m venv --system-site-packages "$VENV" || {
        echo "error: unable to create Python venv" >&2
        echo "Ubuntu: sudo apt-get install python3-venv" >&2
        exit 1
    }
fi

if ! check_python_ssl "$VENV/bin/python" "virtual environment"; then
    echo "If the base interpreter now works, remove the stale environment and rerun:" >&2
    echo "  rm -rf '$VENV'" >&2
    exit 1
fi
check_python_version "$VENV/bin/python" "virtual environment" || exit 1

if [ -n "$PYPI_INDEX_URL" ]; then
    echo "[hf-env] Python package index: $PYPI_INDEX_URL" >&2
fi
pip_install --upgrade pip setuptools wheel

if ! "$VENV/bin/python" -c 'import torch' >/dev/null 2>&1; then
    echo "[hf-env] PyTorch not found; installing torch and torchvision" >&2
    if [ -n "${OPENNPUX_TORCH_INDEX_URL:-}" ]; then
        "$VENV/bin/python" -m pip install \
            --index-url "$OPENNPUX_TORCH_INDEX_URL" torch torchvision
    else
        pip_install torch torchvision
    fi
elif ! "$VENV/bin/python" -c 'import torchvision' >/dev/null 2>&1; then
    echo "[hf-env] torchvision not found; installing a torch-compatible build" >&2
    if [ -n "${OPENNPUX_TORCH_INDEX_URL:-}" ]; then
        "$VENV/bin/python" -m pip install \
            --index-url "$OPENNPUX_TORCH_INDEX_URL" torchvision
    else
        pip_install torchvision
    fi
fi

if ! pip_install -r "$REQUIREMENTS"; then
    echo "error: unable to install the Hugging Face requirements" >&2
    echo "If the configured package mirror is stale, retry with:" >&2
    echo "  OPENNPUX_PYPI_INDEX_URL=https://pypi.org/simple $0" >&2
    exit 1
fi

if [ "${OPENNPUX_INSTALL_GPTQMODEL:-1}" != 0 ] &&
   ! "$VENV/bin/python" -c \
       'import importlib.util; assert importlib.util.find_spec("gptqmodel")' \
       >/dev/null 2>&1; then
    echo "[hf-env] installing the GPTQModel backend" >&2
    pip_install --no-build-isolation gptqmodel
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
print(f"hf_env_torch_cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"hf_env_cuda_device={torch.cuda.get_device_name(0)}")
    print(f"hf_env_cuda_capability={torch.cuda.get_device_capability(0)}")
print("hf_env=PASS")
PY

echo "[hf-env] use CORAL_HF_PYTHON=$VENV/bin/python" >&2
