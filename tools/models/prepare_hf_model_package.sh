#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <model-directory|huggingface-repo-id> [output.npxm]" >&2
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODEL_INPUT="$1"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME:-${HOME}/.cache/huggingface}/hub}"
AUTO_DOWNLOAD="${OPENNPUX_HF_AUTO_DOWNLOAD:-1}"
DEFAULT_REPOSITORY="${OPENNPUX_HF_MODEL_REPO:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}"

resolve_model_dir()
{
    candidate="$1"
    if [ ! -d "${candidate}" ]; then
        case "${candidate}" in
            */*) candidate="${HF_HUB_CACHE}/models--$(printf '%s' "${candidate}" | tr / -)" ;;
        esac
    fi
    if [ -f "${candidate}/config.json" ]; then
        (CDPATH= cd -- "${candidate}" && pwd -P)
        return
    fi
    if [ -d "${candidate}/snapshots" ]; then
        resolved="$(find "${candidate}/snapshots" -mindepth 2 -maxdepth 2 \
            -name config.json -type f -print 2>/dev/null | sort | tail -n 1)"
        if [ -n "${resolved}" ]; then
            (CDPATH= cd -- "$(dirname -- "${resolved}")" && pwd -P)
            return
        fi
    fi
    return 1
}

if ! MODEL_DIR="$(resolve_model_dir "${MODEL_INPUT}")"; then
    if [ "${AUTO_DOWNLOAD}" = "1" ]; then
        case "${MODEL_INPUT}" in
            /*|./*|../*)
                echo "[hf-model] assets missing; downloading ${DEFAULT_REPOSITORY}"
                "${SCRIPT_DIR}/download_hf_model.sh" \
                    "${MODEL_INPUT}" "${DEFAULT_REPOSITORY}"
                MODEL_DIR="$(resolve_model_dir "${MODEL_INPUT}")"
                ;;
        esac
    fi
fi

if [ -z "${MODEL_DIR:-}" ]; then
    cat >&2 <<EOF
error: Hugging Face model assets not found: ${MODEL_INPUT}

Provide a directory containing:
  config.json
  model.safetensors
or:
  config.json
  model.safetensors.index.json
  all referenced model-*.safetensors shards

Search the local Hugging Face cache with:
  find "${HF_HUB_CACHE}" -path '*/snapshots/*/config.json' -print

For an explicit missing directory, automatic download is enabled by default:
  repository: ${DEFAULT_REPOSITORY}
  endpoint:   ${HF_ENDPOINT:-https://hf-mirror.com}

Set OPENNPUX_HF_AUTO_DOWNLOAD=0 to disable it.
EOF
    exit 1
fi

echo "hf_model_directory=${MODEL_DIR}"
OUTPUT="${2:-${MODEL_DIR}/model.npxm}"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CC="${CC:-cc}"

mkdir -p "${OUT_DIR}"
"${SCRIPT_DIR}/import_hf_model.py" "${MODEL_DIR}" "${OUTPUT}"
"${SCRIPT_DIR}/build_qwen_execution_plan.py" "${OUTPUT}"
PLAN="$(dirname -- "${OUTPUT}")/execution-plan.npxp"
if ! "${SCRIPT_DIR}/compile_npu_executable.py" \
    "${OUTPUT}" "${PLAN}" "$(dirname -- "${OUTPUT}")/model.npxe"; then
    echo "error: generic NPU executable compilation failed" >&2
    echo "inspect decoder coverage in: ${PLAN}" >&2
    echo "fields: unknown_decoder_tensor_patterns, unknown_decoder_tensor_samples" >&2
    exit 1
fi
"${SCRIPT_DIR}/compile_npu_weight_plan.py" \
    "${OUTPUT}" "$(dirname -- "${OUTPUT}")/model.npxe" \
    "$(dirname -- "${OUTPUT}")/model.npxw" --require-complete
"${CC}" -O2 -Wall -Wextra -Werror -std=c11 \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/model_package.c" \
    "${ROOT_DIR}/runtime/host/tools/model_inspect.c" \
    -o "${OUT_DIR}/model-inspect"
"${OUT_DIR}/model-inspect" "${OUTPUT}"
echo "hf_model_package_prepare=PASS"
