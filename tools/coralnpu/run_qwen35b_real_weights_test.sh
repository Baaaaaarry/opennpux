#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
KERNEL_RELEASE_FILE="${ROOT_DIR}/build/kernel/kernel.release"

usage() {
    cat <<'EOF'
Usage: run_qwen35b_real_weights_test.sh [options]

Test input:
  --prompt TEXT              Prompt submitted by the guest CPU to the NPU

Inference options:
  --max-new-tokens N         Generate 1..32 tokens (default: 8)
  --prompt-format FORMAT     chat or raw (default: chat)
  --decode-mode MODE         model or greedy (default: greedy)
  --generation-seed N        Non-negative generation seed (default: 42)
  --model-loader LOADER      vllm, transformers, or gptqmodel (default: vllm)
  --gptq-backend BACKEND     GPTQ backend used by the reference runner
  --hf-device DEVICE         auto, cuda, or cpu (default: auto)
  --model-dir PATH           Prepared model directory
  -h, --help                 Show this help

Command-line options override the corresponding CORAL_QWEN_* environment
variables. The external numerical runner is a correctness reference; it is not
the Host C++ functional-kernel result source.
EOF
}

MODEL_DIR="${CORAL_MODEL_DIR:-/data/models/Qwen3.5-35B}"
EXECUTABLE_NAME="${CORAL_NPU_EXECUTABLE_NAME:-model.npxc}"
EXECUTABLE_PLAN_NAME="${CORAL_NPU_EXECUTABLE_PLAN_NAME:-model.npxe}"
MANIFEST_NAME="${CORAL_NPU_MANIFEST_NAME:-model.npxm}"
RANGE_NAME="${CORAL_NPU_RANGE_NAME:-model.npxr}"
TENSOR_PLAN_NAME="${CORAL_NPU_TENSOR_PLAN_NAME:-model.npxtb}"
EXECUTION_PLAN_NAME="${CORAL_NPU_EXECUTION_PLAN_NAME:-execution-plan.npxp}"
POLL_COUNT="${CORAL_PAGED_POLL_COUNT:-100000000}"
BASE="${CORAL_NPU_BASE:-0x1d000000}"
PROMPT="${CORAL_QWEN_PROMPT:-OpenNPUX heterogeneous inference}"
PROMPT_FORMAT="${CORAL_QWEN_PROMPT_FORMAT:-chat}"
MAX_NEW_TOKENS="${CORAL_QWEN_MAX_NEW_TOKENS:-8}"
DECODE_MODE="${CORAL_QWEN_DECODE_MODE:-greedy}"
GENERATION_SEED="${CORAL_QWEN_GENERATION_SEED:-42}"
MODEL_LOADER="${CORAL_QWEN_MODEL_LOADER:-vllm}"
GPTQ_BACKEND="${CORAL_QWEN_GPTQ_BACKEND:-${OPENNPUX_GPTQ_BACKEND:-gptq_torch}}"
HF_DEVICE="${CORAL_QWEN_HF_DEVICE:-auto}"
NUMERICAL_ENV=""
SIM_HOST_PAGING="${CORAL_SIM_HOST_PAGING:-1}"
SIM_HOST_NUMERICAL="${CORAL_SIM_HOST_NUMERICAL:-0}"
SIM_HOST_FUNCTIONAL="${CORAL_SIM_HOST_FUNCTIONAL_CPP:-1}"
TOKEN_REFERENCE="${CORAL_QWEN_TOKEN_REFERENCE:-1}"
REUSE_DECODE_WEIGHTS="${CORAL_REUSE_DECODE_WEIGHTS:-1}"
HF_PYTHON="${CORAL_HF_PYTHON:-${ROOT_DIR}/.venv/hf-numerical/bin/python}"

prompt_from_cli=0
while [ "$#" -gt 0 ]; do
    case "$1" in
        --prompt)
            [ "$#" -ge 2 ] || {
                echo "error: --prompt requires a value" >&2
                exit 2
            }
            PROMPT=$2
            prompt_from_cli=1
            shift 2
            ;;
        --max-new-tokens)
            [ "$#" -ge 2 ] || {
                echo "error: --max-new-tokens requires a value" >&2
                exit 2
            }
            MAX_NEW_TOKENS=$2
            shift 2
            ;;
        --prompt-format)
            [ "$#" -ge 2 ] || {
                echo "error: --prompt-format requires a value" >&2
                exit 2
            }
            PROMPT_FORMAT=$2
            shift 2
            ;;
        --decode-mode)
            [ "$#" -ge 2 ] || {
                echo "error: --decode-mode requires a value" >&2
                exit 2
            }
            DECODE_MODE=$2
            shift 2
            ;;
        --generation-seed)
            [ "$#" -ge 2 ] || {
                echo "error: --generation-seed requires a value" >&2
                exit 2
            }
            GENERATION_SEED=$2
            shift 2
            ;;
        --model-loader)
            [ "$#" -ge 2 ] || {
                echo "error: --model-loader requires a value" >&2
                exit 2
            }
            MODEL_LOADER=$2
            shift 2
            ;;
        --gptq-backend)
            [ "$#" -ge 2 ] || {
                echo "error: --gptq-backend requires a value" >&2
                exit 2
            }
            GPTQ_BACKEND=$2
            shift 2
            ;;
        --hf-device)
            [ "$#" -ge 2 ] || {
                echo "error: --hf-device requires a value" >&2
                exit 2
            }
            HF_DEVICE=$2
            shift 2
            ;;
        --model-dir)
            [ "$#" -ge 2 ] || {
                echo "error: --model-dir requires a value" >&2
                exit 2
            }
            MODEL_DIR=$2
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "error: unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
[ "$#" -eq 0 ] || {
    echo "error: unexpected positional argument: $1" >&2
    exit 2
}

[ "${CORAL_QWEN35B_NUMERICAL:-0}" = 0 ] ||
    NUMERICAL_ENV="OPENNPUX_NPU_NUMERICAL=1"

[ "${#PROMPT}" -lt 128 ] || {
    echo "error: --prompt must be shorter than 128 bytes" >&2
    exit 1
}
case "$MAX_NEW_TOKENS" in
    ''|*[!0-9]*)
        echo "error: CORAL_QWEN_MAX_NEW_TOKENS must be an integer from 1 to 32" >&2
        exit 1
        ;;
esac
[ "$MAX_NEW_TOKENS" -ge 1 ] && [ "$MAX_NEW_TOKENS" -le 32 ] || {
    echo "error: CORAL_QWEN_MAX_NEW_TOKENS must be an integer from 1 to 32" >&2
    exit 1
}
case "$PROMPT" in
    *"'"*)
        echo "error: --prompt cannot contain a single quote" >&2
        exit 1
        ;;
esac

if [ "$prompt_from_cli" = 0 ]; then
    echo "warning: --prompt was not specified; using CORAL_QWEN_PROMPT/default" >&2
fi
echo "[coral-qwen35b-real-weights-test] prompt: $PROMPT" >&2
case "$PROMPT_FORMAT" in
    chat|raw) ;;
    *)
        echo "error: CORAL_QWEN_PROMPT_FORMAT must be chat or raw" >&2
        exit 1
        ;;
esac
case "$DECODE_MODE" in
    model|greedy) ;;
    *)
        echo "error: CORAL_QWEN_DECODE_MODE must be model or greedy" >&2
        exit 1
        ;;
esac
if [ "$SIM_HOST_FUNCTIONAL" != 0 ] && [ "$TOKEN_REFERENCE" != 0 ] &&
   [ "$DECODE_MODE" != greedy ]; then
    echo "error: Host C++ token acceptance requires --decode-mode greedy" >&2
    echo "model mode applies sampling; Host C++ currently implements deterministic argmax only" >&2
    echo "rerun with --decode-mode greedy, or disable exact token-reference acceptance" >&2
    exit 1
fi
case "$MODEL_LOADER" in
    transformers|gptqmodel|vllm) ;;
    *)
        echo "error: CORAL_QWEN_MODEL_LOADER must be transformers, gptqmodel or vllm" >&2
        exit 1
        ;;
esac
[ -n "$GPTQ_BACKEND" ] || {
    echo "error: CORAL_QWEN_GPTQ_BACKEND cannot be empty" >&2
    exit 1
}
case "$HF_DEVICE" in
    auto|cuda|cpu) ;;
    *)
        echo "error: CORAL_QWEN_HF_DEVICE must be auto, cuda or cpu" >&2
        exit 1
        ;;
esac
case "$GENERATION_SEED" in
    ''|*[!0-9]*)
        echo "error: CORAL_QWEN_GENERATION_SEED must be a non-negative integer" >&2
        exit 1
        ;;
esac
case "$MODEL_DIR" in
    *,*)
        echo "error: CORAL_MODEL_DIR cannot contain a comma (9P mount option separator)" >&2
        exit 1
        ;;
esac

for asset in "$EXECUTABLE_NAME" "$MANIFEST_NAME" "$RANGE_NAME"; do
    [ -r "$MODEL_DIR/$asset" ] || {
        echo "error: Qwen model asset missing: $MODEL_DIR/$asset" >&2
        echo "prepare it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    }
done
MODEL_DTYPE="$(python3 - "$MODEL_DIR/$MANIFEST_NAME" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    print(str(json.load(source).get("dtype", "float32")).lower())
PY
)"
HOST_FUNCTIONAL_PRECISION="${CORAL_HOST_FUNCTIONAL_PRECISION:-}"
if [ -z "$HOST_FUNCTIONAL_PRECISION" ]; then
    if [ "$MODEL_DTYPE" = bfloat16 ]; then
        HOST_FUNCTIONAL_PRECISION=bf16
    else
        HOST_FUNCTIONAL_PRECISION=fp32
    fi
fi
case "$HOST_FUNCTIONAL_PRECISION" in
    fp32|bf16) ;;
    *)
        echo "error: CORAL_HOST_FUNCTIONAL_PRECISION must be fp32 or bf16" >&2
        exit 2
        ;;
esac
echo "[coral-qwen35b-real-weights-test] model dtype: $MODEL_DTYPE" >&2
echo "[coral-qwen35b-real-weights-test] Host functional precision: $HOST_FUNCTIONAL_PRECISION" >&2
FUNCTIONAL_GRAPH_REFRESHED=0
if [ "$SIM_HOST_FUNCTIONAL" != 0 ] &&
   ! python3 - "$MODEL_DIR/$MANIFEST_NAME" "$MODEL_DIR/$EXECUTABLE_PLAN_NAME" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as source:
        manifest = json.load(source)
    with open(sys.argv[2], encoding="utf-8") as source:
        executable = json.load(source)
except (OSError, ValueError):
    raise SystemExit(1)
linear_fields = (
    "linear_key_head_dim", "linear_value_head_dim",
    "linear_num_key_heads", "linear_num_value_heads",
    "linear_conv_kernel_dim",
)
valid = (
    manifest.get("functional_graph_revision", 0) >= 9
    and executable.get("functional_graph_revision", 0) >= 9
    and all(int(manifest.get(field, 0)) > 0 for field in linear_fields)
)
raise SystemExit(0 if valid else 1)
PY
then
    if [ "${CORAL_AUTO_REBUILD_FUNCTIONAL_GRAPH:-1}" = 0 ]; then
        echo "error: Host C++ functional graph assets are stale" >&2
        echo "regenerate them with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    fi
    echo "[coral-qwen35b-real-weights-test] rebuilding Host C++ functional graph assets" >&2
    "${ROOT_DIR}/tools/models/import_hf_model.py" \
        "$MODEL_DIR" "$MODEL_DIR/$MANIFEST_NAME"
    "${ROOT_DIR}/tools/models/build_qwen_execution_plan.py" \
        "$MODEL_DIR/$MANIFEST_NAME"
    "${ROOT_DIR}/tools/models/compile_npu_executable.py" \
        "$MODEL_DIR/$MANIFEST_NAME" "$MODEL_DIR/$EXECUTION_PLAN_NAME" \
        "$MODEL_DIR/$EXECUTABLE_PLAN_NAME"
    FUNCTIONAL_GRAPH_REFRESHED=1
fi
WEIGHT_PLAN_VALID=1
if [ "$SIM_HOST_FUNCTIONAL" != 0 ] &&
   ! python3 - "$MODEL_DIR/model.npxw" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as source:
        plan = json.load(source)
except (OSError, ValueError):
    raise SystemExit(1)
valid = (plan.get("tensor_domain") == "text" and
         plan.get("functional_graph_revision", 0) >= 9)
raise SystemExit(0 if valid else 1)
PY
then
    WEIGHT_PLAN_VALID=0
fi
if [ "$SIM_HOST_FUNCTIONAL" != 0 ] &&
   { [ "$FUNCTIONAL_GRAPH_REFRESHED" = 1 ] || [ "$WEIGHT_PLAN_VALID" = 0 ]; }
then
    if [ "${CORAL_AUTO_REBUILD_WEIGHT_PLAN:-1}" = 0 ]; then
        echo "error: Host C++ weight plan predates the functional graph" >&2
        echo "regenerate it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    fi
    echo "[coral-qwen35b-real-weights-test] rebuilding functional text-domain weight plan" >&2
    [ -r "$MODEL_DIR/$EXECUTABLE_PLAN_NAME" ] || {
        echo "error: NPU executable plan missing: $MODEL_DIR/$EXECUTABLE_PLAN_NAME" >&2
        echo "regenerate it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    }
    "${ROOT_DIR}/tools/models/compile_npu_weight_plan.py" \
        "$MODEL_DIR/$MANIFEST_NAME" "$MODEL_DIR/$EXECUTABLE_PLAN_NAME" \
        "$MODEL_DIR/model.npxw" --range-output "$MODEL_DIR/$RANGE_NAME" \
        --require-complete
fi
if [ "$SIM_HOST_FUNCTIONAL" != 0 ] &&
   [ ! -r "$MODEL_DIR/$TENSOR_PLAN_NAME" ]; then
    TENSOR_PLAN_STEM=${TENSOR_PLAN_NAME%.npxtb}
    EXECUTION_PLAN="$MODEL_DIR/$EXECUTION_PLAN_NAME"
    if [ "${CORAL_AUTO_REBUILD_TENSOR_PLAN:-1}" != 0 ] &&
       [ -r "$EXECUTION_PLAN" ]; then
        echo "[coral-qwen35b-real-weights-test] rebuilding missing Host C++ tensor plan" >&2
        "${ROOT_DIR}/tools/models/compile_npu_executable.py" \
            "$MODEL_DIR/$MANIFEST_NAME" "$EXECUTION_PLAN" \
            "$MODEL_DIR/$TENSOR_PLAN_STEM.npxe"
    fi
    [ -r "$MODEL_DIR/$TENSOR_PLAN_NAME" ] || {
        echo "error: Host C++ tensor plan missing: $MODEL_DIR/$TENSOR_PLAN_NAME" >&2
        echo "regenerate it with: ./tools/models/prepare_hf_model_package.sh $MODEL_DIR" >&2
        exit 1
    }
fi
if [ ! -r "$MODEL_DIR/preprocessor_config.json" ] &&
   python3 - "$MODEL_DIR/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    config = json.load(source)
architectures = config.get("architectures", [])
if not ("vision_config" in config or any(
    str(name).endswith("ForConditionalGeneration") for name in architectures
)):
    raise SystemExit(1)
PY
then
    echo "[coral-qwen35b-real-weights-test] completing multimodal processor assets" >&2
    "${ROOT_DIR}/tools/models/download_hf_model.sh" "$MODEL_DIR"
fi
if [ "$SIM_HOST_NUMERICAL" != 0 ] && [ "$SIM_HOST_PAGING" = 0 ]; then
    echo "error: sim-host numerical inference requires sim-host paging" >&2
    exit 1
fi
if [ "$SIM_HOST_NUMERICAL" != 0 ] &&
   [ "${CORAL_QWEN35B_NUMERICAL:-0}" != 0 ]; then
    echo "error: select sim-host numerical or firmware numerical, not both" >&2
    exit 1
fi
if [ "$SIM_HOST_NUMERICAL" != 0 ] && [ "$SIM_HOST_FUNCTIONAL" != 0 ]; then
    echo "error: select Host C++ functional or sim-host numerical, not both" >&2
    exit 1
fi
command -v diod >/dev/null 2>&1 || {
    echo "error: diod is required for the guest model mount" >&2
    echo "Ubuntu: sudo apt-get install diod" >&2
    exit 1
}

SIM_HOST_ENV=""
REUSE_DECODE_WEIGHTS_ENV=""
SIM_HOST_BUNDLE=""
if [ "$SIM_HOST_PAGING" != 0 ]; then
    if [ "${CORAL_QWEN35B_NUMERICAL:-0}" = 0 ]; then
        BUNDLE_MODE="functional"
        BUNDLE_MAX_PAGES=1
    else
        BUNDLE_MODE="numerical"
        BUNDLE_MAX_PAGES=0
    fi
    SIM_HOST_BUNDLE="${CORAL_SIM_HOST_PAGE_BUNDLE:-${ROOT_DIR}/build/model-pages/qwen35b-${BUNDLE_MODE}-64k.npxb}"
    bundle_stale=0
    [ -r "$SIM_HOST_BUNDLE" ] || bundle_stale=1
    if [ "$bundle_stale" -eq 0 ] &&
       { [ "$MODEL_DIR/$MANIFEST_NAME" -nt "$SIM_HOST_BUNDLE" ] ||
         [ "$MODEL_DIR/$RANGE_NAME" -nt "$SIM_HOST_BUNDLE" ]; }; then
        bundle_stale=1
    fi
    if [ "$bundle_stale" -eq 0 ] &&
       find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' \
           -newer "$SIM_HOST_BUNDLE" -print -quit | grep -q .; then
        bundle_stale=1
    fi
    if [ "$bundle_stale" -eq 1 ] ||
       [ "${CORAL_REBUILD_SIM_HOST_BUNDLE:-0}" != 0 ]; then
        echo "[coral-qwen35b-real-weights-test] building sim-host page bundle" >&2
        "${ROOT_DIR}/tools/models/build_sim_host_page_bundle.sh" \
            "$MODEL_DIR/$MANIFEST_NAME" "$MODEL_DIR/$RANGE_NAME" \
            "$SIM_HOST_BUNDLE" 65536 "$BUNDLE_MAX_PAGES"
    fi
    SIM_HOST_ENV="OPENNPUX_SIM_HOST_PAGING=1"
    if [ "$REUSE_DECODE_WEIGHTS" != 0 ]; then
        REUSE_DECODE_WEIGHTS_ENV="OPENNPUX_REUSE_DECODE_WEIGHTS=1"
    fi
    echo "[coral-qwen35b-real-weights-test] sim-host bundle: $SIM_HOST_BUNDLE" >&2
fi

SIM_HOST_NUMERICAL_ENV=""
SIM_HOST_FUNCTIONAL_ENV=""
HOST_FUNCTIONAL_MODEL=""
HOST_FUNCTIONAL_RANGES=""
HOST_FUNCTIONAL_TENSOR_PLAN=""
SIM_HOST_RESULT=""
INPUT_TOKEN_COUNT=1
INPUT_TOKEN_IDS=""
EXPECTED_GENERATED_TOKENS=0
EXPECTED_TOKEN_IDS=""
PREFLIGHT_TOKEN_IDS=""
if [ "$TOKEN_REFERENCE" != 0 ] || [ "$SIM_HOST_NUMERICAL" != 0 ]; then
    PROMPT_TAG="$(python3 - "$PROMPT" <<'PY'
import sys
value = 2166136261
for byte in sys.argv[1].encode():
    value = ((value ^ byte) * 16777619) & 0xffffffff
print(f"{value:08x}")
PY
)"
    SIM_HOST_RESULT="${CORAL_SIM_HOST_INFERENCE_RESULT:-${ROOT_DIR}/build/model-results/qwen35b-${PROMPT_TAG}-${PROMPT_FORMAT}-${MODEL_LOADER}-${GPTQ_BACKEND}-${HF_DEVICE}-${DECODE_MODE}-s${GENERATION_SEED}-t${MAX_NEW_TOKENS}.npxo}"
    result_stale=0
    [ -r "$SIM_HOST_RESULT" ] || result_stale=1
    if [ "$result_stale" -eq 0 ] &&
       { [ "$MODEL_DIR/config.json" -nt "$SIM_HOST_RESULT" ] ||
         [ "$MODEL_DIR/$EXECUTABLE_NAME" -nt "$SIM_HOST_RESULT" ] ||
         [ "${ROOT_DIR}/tools/models/run_hf_next_token.py" -nt "$SIM_HOST_RESULT" ] ||
         [ "${ROOT_DIR}/tools/models/run_vllm_next_token.py" -nt "$SIM_HOST_RESULT" ]; }; then
        result_stale=1
    fi
    if [ "$result_stale" -eq 0 ] &&
       find "$MODEL_DIR" -maxdepth 1 -type f -name '*.safetensors' \
           -newer "$SIM_HOST_RESULT" -print -quit | grep -q .; then
        result_stale=1
    fi
    if [ "$result_stale" -eq 1 ] ||
       [ "${CORAL_REBUILD_SIM_HOST_RESULT:-0}" != 0 ]; then
        if [ ! -x "$HF_PYTHON" ]; then
            echo "error: HF numerical Python environment is missing: $HF_PYTHON" >&2
            echo "run: ./tools/models/setup_hf_numerical_env.sh" >&2
            exit 1
        fi
        if ! "$HF_PYTHON" -c \
            'import accelerate, numpy, optimum, safetensors, torch, torchvision, transformers; from transformers import AutoModelForMultimodalLM, AutoProcessor' \
            >/dev/null 2>&1; then
            echo "error: HF numerical Python dependencies are incomplete" >&2
            echo "run: ./tools/models/setup_hf_numerical_env.sh" >&2
            exit 1
        fi
        if [ "$MODEL_LOADER" != vllm ] &&
           ! "$HF_PYTHON" -c 'import gptqmodel' >/dev/null 2>&1; then
            echo "error: GPTQModel is required for $MODEL_LOADER loading" >&2
            echo "run: ./tools/models/setup_hf_numerical_env.sh" >&2
            exit 1
        fi
        if [ "$MODEL_LOADER" = vllm ] &&
           ! "$HF_PYTHON" -c 'import vllm' >/dev/null 2>&1; then
            echo "error: vLLM is required for CORAL_QWEN_MODEL_LOADER=vllm" >&2
            echo "install the latest Qwen3.5-compatible vLLM build" >&2
            exit 1
        fi
        echo "[coral-qwen35b-real-weights-test] running real HF numerical forward" >&2
        (
            if [ "$HF_DEVICE" = cpu ]; then
                CUDA_VISIBLE_DEVICES=
                export CUDA_VISIBLE_DEVICES
            fi
            if [ "$MODEL_LOADER" = vllm ]; then
                GENERATOR="${ROOT_DIR}/tools/models/run_vllm_next_token.py"
            else
                GENERATOR="${ROOT_DIR}/tools/models/run_hf_next_token.py"
            fi
            set -- "$MODEL_DIR" "$MODEL_DIR/$EXECUTABLE_NAME" \
                "$SIM_HOST_RESULT" --prompt "$PROMPT" \
                --prompt-format "$PROMPT_FORMAT" \
                --decode-mode "$DECODE_MODE" \
                --seed "$GENERATION_SEED" \
                --max-new-tokens "$MAX_NEW_TOKENS"
            if [ "$MODEL_LOADER" != vllm ]; then
                set -- "$@" --model-loader "$MODEL_LOADER" \
                    --gptq-backend "$GPTQ_BACKEND" \
                    --device-map "$HF_DEVICE"
            fi
            exec "$HF_PYTHON" "$GENERATOR" "$@"
        )
    fi
    if [ "$SIM_HOST_NUMERICAL" != 0 ]; then
        SIM_HOST_NUMERICAL_ENV="OPENNPUX_SIM_HOST_NUMERICAL=1"
    fi
    INPUT_TOKEN_COUNT="$(python3 - "$SIM_HOST_RESULT" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
if len(data) != 256 or struct.unpack_from("<III", data) != (0x5258504e, 2, 256):
    raise SystemExit("invalid OpenNPUX numerical result v2")
value = struct.unpack_from("<I", data, 40)[0]
if value == 0 or value > 65504:
    raise SystemExit("invalid numerical input token count")
print(value)
PY
)"
    EXPECTED_GENERATED_TOKENS="$(python3 - "$SIM_HOST_RESULT" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
print(struct.unpack_from("<I", data, 120)[0])
PY
)"
    EXPECTED_TOKEN_IDS="$(python3 - "$SIM_HOST_RESULT" "$DECODE_MODE" <<'PY'
import struct
import sys

data = open(sys.argv[1], "rb").read()
count = struct.unpack_from("<I", data, 120)[0]
if count == 0 or count > 32:
    raise SystemExit("invalid numerical generated token count")
values = struct.unpack_from(f"<{count}I", data, 128)
if sys.argv[2] == "model" and count > 1 and len(set(values)) == 1:
    raise SystemExit("degenerate numerical golden contains one repeated token")
print(",".join(str(value) for value in values))
PY
)"
    echo "[coral-qwen35b-real-weights-test] numerical result: $SIM_HOST_RESULT" >&2
    echo "[coral-qwen35b-real-weights-test] prompt format: $PROMPT_FORMAT" >&2
    echo "[coral-qwen35b-real-weights-test] decode mode: $DECODE_MODE seed=$GENERATION_SEED" >&2
    echo "[coral-qwen35b-real-weights-test] model loader: $MODEL_LOADER" >&2
    echo "[coral-qwen35b-real-weights-test] GPTQ backend: $GPTQ_BACKEND" >&2
    echo "[coral-qwen35b-real-weights-test] HF device: $HF_DEVICE" >&2
    echo "[coral-qwen35b-real-weights-test] input tokens: $INPUT_TOKEN_COUNT" >&2
    echo "[coral-qwen35b-real-weights-test] expected token ids: $EXPECTED_TOKEN_IDS" >&2
fi
if [ "$SIM_HOST_FUNCTIONAL" != 0 ]; then
    if [ ! -x "$HF_PYTHON" ]; then
        echo "error: tokenizer Python environment is missing: $HF_PYTHON" >&2
        echo "run: ./tools/models/setup_hf_numerical_env.sh" >&2
        exit 1
    fi
    TOKENIZED="$($HF_PYTHON "${ROOT_DIR}/tools/models/tokenize_hf_prompt.py" \
        "$MODEL_DIR" --prompt "$PROMPT" --prompt-format "$PROMPT_FORMAT")"
    INPUT_TOKEN_COUNT="$(printf '%s\n' "$TOKENIZED" |
        sed -n 's/^input_token_count=//p')"
    INPUT_TOKEN_IDS="$(printf '%s\n' "$TOKENIZED" |
        sed -n 's/^input_token_ids=//p')"
    [ -n "$INPUT_TOKEN_COUNT" ] && [ -n "$INPUT_TOKEN_IDS" ] || {
        echo "error: CPU tokenizer did not produce token IDs" >&2
        exit 1
    }
    SIM_HOST_FUNCTIONAL_ENV="OPENNPUX_SIM_HOST_FUNCTIONAL_CPP=1"
    HOST_FUNCTIONAL_MODEL="$MODEL_DIR/$MANIFEST_NAME"
    HOST_FUNCTIONAL_RANGES="$MODEL_DIR/$RANGE_NAME"
    HOST_FUNCTIONAL_TENSOR_PLAN="$MODEL_DIR/$TENSOR_PLAN_NAME"
    echo "[coral-qwen35b-real-weights-test] CPU token count: $INPUT_TOKEN_COUNT" >&2
    if [ "${CORAL_SKIP_HOST_FUNCTIONAL_PREFLIGHT:-0}" = 0 ]; then
        PREFLIGHT_LOG="${ROOT_DIR}/simout/qwen35b-host-functional-preflight.log"
        mkdir -p "${ROOT_DIR}/simout"
        TERMINAL_LOG="${ROOT_DIR}/logs/sim/m5out/system.terminal"
        if [ -f "$TERMINAL_LOG" ]; then
            mv -f "$TERMINAL_LOG" "$TERMINAL_LOG.previous"
        fi
        echo "[coral-qwen35b-real-weights-test] running Host C++ preflight" >&2
        : >"$PREFLIGHT_LOG"
        OPENNPUX_HOST_FUNCTIONAL_PRECISION="$HOST_FUNCTIONAL_PRECISION" \
        OPENNPUX_HOST_FUNCTIONAL_PROGRESS="${CORAL_HOST_FUNCTIONAL_PROGRESS:-0}" \
        OPENNPUX_HOST_FUNCTIONAL_LOGITS_TRACE="${CORAL_HOST_FUNCTIONAL_LOGITS_TRACE:-0}" \
        OPENNPUX_HOST_FUNCTIONAL_REFERENCE_TOKENS="$EXPECTED_TOKEN_IDS" \
        "${ROOT_DIR}/tools/models/run_host_functional_preflight.sh" \
            "$MODEL_DIR" "$INPUT_TOKEN_IDS" "$MAX_NEW_TOKENS" \
            >"$PREFLIGHT_LOG" 2>&1 &
        PREFLIGHT_PID=$!
        PREFLIGHT_LINES=0
        while kill -0 "$PREFLIGHT_PID" 2>/dev/null; do
            sleep "${CORAL_PREFLIGHT_PROGRESS_INTERVAL:-5}"
            CURRENT_LINES=$(wc -l <"$PREFLIGHT_LOG" | tr -d ' ')
            if [ "$CURRENT_LINES" -gt "$PREFLIGHT_LINES" ]; then
                sed -n "$((PREFLIGHT_LINES + 1)),${CURRENT_LINES}p" \
                    "$PREFLIGHT_LOG" >&2
                PREFLIGHT_LINES=$CURRENT_LINES
            fi
        done
        if wait "$PREFLIGHT_PID"; then
            PREFLIGHT_RC=0
        else
            PREFLIGHT_RC=$?
        fi
        CURRENT_LINES=$(wc -l <"$PREFLIGHT_LOG" | tr -d ' ')
        if [ "$CURRENT_LINES" -gt "$PREFLIGHT_LINES" ]; then
            sed -n "$((PREFLIGHT_LINES + 1)),${CURRENT_LINES}p" \
                "$PREFLIGHT_LOG" >&2
        fi
        if [ "$PREFLIGHT_RC" -ne 0 ]; then
            echo "error: Host C++ functional preflight failed; gem5 was not started" >&2
            exit 1
        fi
        PREFLIGHT_TOKEN_IDS="$(sed -n \
            's/^host_functional_token_ids=//p' "$PREFLIGHT_LOG" | tail -n 1)"
        [ -n "$PREFLIGHT_TOKEN_IDS" ] || {
            echo "error: Host C++ preflight did not produce token IDs" >&2
            exit 1
        }
        if [ "$TOKEN_REFERENCE" != 0 ] &&
           [ "$PREFLIGHT_TOKEN_IDS" != "$EXPECTED_TOKEN_IDS" ]; then
            echo "error: Host C++ preflight token IDs differ from $MODEL_LOADER" >&2
            echo "host_cpp=$PREFLIGHT_TOKEN_IDS" >&2
            echo "reference=$EXPECTED_TOKEN_IDS" >&2
            exit 1
        fi
        echo "[coral-qwen35b-real-weights-test] host_preflight=PASS" >&2
    fi
fi

CPU_DECODE_TOKEN_IDS="$EXPECTED_TOKEN_IDS"
if [ -z "$CPU_DECODE_TOKEN_IDS" ]; then
    CPU_DECODE_TOKEN_IDS="$PREFLIGHT_TOKEN_IDS"
fi
DECODED_TOKEN_TEXT_B64=""
if [ -n "$CPU_DECODE_TOKEN_IDS" ]; then
    DECODED_TOKEN_TEXT="$($HF_PYTHON \
        "${ROOT_DIR}/tools/models/decode_token_ids.py" "$MODEL_DIR" \
        --token-ids "$CPU_DECODE_TOKEN_IDS" --text-only)"
    DECODED_TOKEN_TEXT_B64="$(printf '%s' "$DECODED_TOKEN_TEXT" | \
        base64 | tr -d '\n')"
fi

if [ -z "${CORAL_KERNEL_IMAGE:-}" ]; then
    if [ ! -r "$KERNEL_RELEASE_FILE" ]; then
        echo "error: kernel release metadata missing: $KERNEL_RELEASE_FILE" >&2
        echo "build the 9P-enabled kernel with: ./tools/kernel/build_arm64_kernel.sh" >&2
        exit 1
    fi
    KERNEL_RELEASE="$(cat "$KERNEL_RELEASE_FILE")"
    CORAL_KERNEL_IMAGE="${ROOT_DIR}/build/kernel/vmlinux-${KERNEL_RELEASE}"
fi
[ -r "$CORAL_KERNEL_IMAGE" ] || {
    echo "error: kernel image missing: $CORAL_KERNEL_IMAGE" >&2
    echo "build it with: ./tools/kernel/build_arm64_kernel.sh" >&2
    exit 1
}

CORALCTL="${ROOT_DIR}/build/guest-tools/coralctl-aarch64"
FIRMWARE="${CORAL_RTL_FIRMWARE:-${ROOT_DIR}/build/coralnpu/gem5_npu_command_processor_smoke.elf}"
BRIDGE="${CORAL_RTL_BRIDGE:-${ROOT_DIR}/build/coralnpu/libcoralnpu_gem5_bridge.so}"
coralctl_stale=0
if [ ! -x "$CORALCTL" ]; then
    coralctl_stale=1
elif find "${ROOT_DIR}/runtime/host" -type f -newer "$CORALCTL" \
     -print -quit | grep -q .; then
    coralctl_stale=1
fi
if [ "$coralctl_stale" -eq 1 ]; then
    echo "[coral-qwen35b-real-weights-test] coralctl missing or stale; rebuilding" >&2
    "${ROOT_DIR}/tools/guest_tools/build_coralctl.sh"
fi
rtl_stale=0
if [ ! -r "$FIRMWARE" ] || [ ! -r "$BRIDGE" ]; then
    rtl_stale=1
elif [ -z "${CORAL_RTL_FIRMWARE:-}" ] &&
     find "${ROOT_DIR}/sim/coralnpu" -type f \
         \( -newer "$FIRMWARE" -o -newer "$BRIDGE" \) \
         -print -quit | grep -q .; then
    rtl_stale=1
fi
if [ "$rtl_stale" -eq 1 ] &&
   [ -z "${CORAL_RTL_FIRMWARE:-}" ] &&
   [ -z "${CORAL_RTL_BRIDGE:-}" ] &&
   [ "${CORAL_AUTO_BUILD_FIRMWARE:-1}" = 1 ]; then
    echo "[coral-qwen35b-real-weights-test] RTL artifacts missing or stale; rebuilding" >&2
    "${ROOT_DIR}/tools/coralnpu/build_rtl_bridge.sh"
fi
[ -r "$FIRMWARE" ] || {
    echo "error: firmware missing or stale: $FIRMWARE" >&2
    echo "run: ./tools/coralnpu/build_rtl_bridge.sh" >&2
    exit 1
}
[ -r "$BRIDGE" ] || { echo "error: RTL bridge missing: $BRIDGE" >&2; exit 1; }

TMP_SCRIPT="$(mktemp)"
trap 'rm -f "$TMP_SCRIPT"' EXIT

cat >"$TMP_SCRIPT" <<EOF
#!/bin/sh
set -u
mkdir -p /proc /sys /dev /tmp /mnt/opennpux-model
mount -t proc proc /proc 2>/dev/null || true
mount -t sysfs sysfs /sys 2>/dev/null || true
mount -t devtmpfs devtmpfs /dev 2>/dev/null || true
decode_base64()
{
    if command -v base64 >/dev/null 2>&1; then
        base64 -d
    elif [ -x /bin/busybox ]; then
        /bin/busybox base64 -d
    elif [ -x /tmp/busybox ]; then
        /tmp/busybox base64 -d
    else
        return 1
    fi
}
decode_base64 >/tmp/coralctl <<'OPENNPUX_CORALCTL_EOF'
EOF
base64 "$CORALCTL" >>"$TMP_SCRIPT"
cat >>"$TMP_SCRIPT" <<EOF
OPENNPUX_CORALCTL_EOF
chmod 0755 /tmp/coralctl
echo '[coral-qwen35b-real-weights-test] started'
if ! grep -qw 9p /proc/filesystems; then
    echo '[coral-qwen35b-real-weights-test] FAIL: kernel lacks CONFIG_9P_FS'
    m5 --inst exit
    exit 1
fi
if ! mount -t 9p \
    -o 'trans=virtio,version=9p2000.L,ro,aname=$MODEL_DIR' \
    gem5 /mnt/opennpux-model; then
    echo '[coral-qwen35b-real-weights-test] FAIL: VirtIO 9P model mount failed'
    dmesg 2>/dev/null | tail -n 20 || true
    m5 --inst exit
    exit 1
fi
for asset in '$EXECUTABLE_NAME' '$MANIFEST_NAME' '$RANGE_NAME'; do
    if [ ! -r "/mnt/opennpux-model/\$asset" ]; then
        echo "[coral-qwen35b-real-weights-test] FAIL: mounted asset missing: \$asset"
        m5 --inst exit
        exit 1
    fi
done
if env $NUMERICAL_ENV $SIM_HOST_ENV $SIM_HOST_NUMERICAL_ENV \
    $SIM_HOST_FUNCTIONAL_ENV \
    $REUSE_DECODE_WEIGHTS_ENV \
    OPENNPUX_PROMPT='$PROMPT' \
    OPENNPUX_MAX_NEW_TOKENS='$MAX_NEW_TOKENS' \
    OPENNPUX_INPUT_TOKEN_COUNT='$INPUT_TOKEN_COUNT' \
    OPENNPUX_INPUT_TOKEN_IDS='$INPUT_TOKEN_IDS' \
    OPENNPUX_MODEL_ROOT=/mnt/opennpux-model \
    /tmp/coralctl executable-run-paged \
    /mnt/opennpux-model/$EXECUTABLE_NAME decode \
    /mnt/opennpux-model/$MANIFEST_NAME \
    /mnt/opennpux-model/$RANGE_NAME \
    $BASE $POLL_COUNT >/tmp/opennpux-inference.log 2>&1; then
    inference_rc=0
else
    inference_rc=\$?
fi
cat /tmp/opennpux-inference.log
if [ "\$inference_rc" -ne 0 ]; then
    echo '[coral-qwen35b-real-weights-test] FAIL: real weight execution failed'
    m5 --inst exit
    exit 1
fi
if [ '$SIM_HOST_FUNCTIONAL' != 0 ]; then
    if ! grep -Fqx 'inference_result_source=host-functional-cpp' \
        /tmp/opennpux-inference.log; then
        echo '[coral-qwen35b-real-weights-test] FAIL: Host C++ result source missing'
        m5 --inst exit
        exit 1
    fi
    echo '[coral-qwen35b-real-weights-test] functional_backend=host-cpp'
fi
if [ '$TOKEN_REFERENCE' != 0 ]; then
    if ! grep -Fqx 'inference_generated_tokens=$EXPECTED_GENERATED_TOKENS' \
        /tmp/opennpux-inference.log; then
        echo '[coral-qwen35b-real-weights-test] FAIL: generated token count differs from HF golden'
        m5 --inst exit
        exit 1
    fi
    if ! grep -Fqx 'inference_token_ids=$EXPECTED_TOKEN_IDS' \
        /tmp/opennpux-inference.log; then
        echo '[coral-qwen35b-real-weights-test] FAIL: token IDs differ from HF golden'
        m5 --inst exit
        exit 1
    fi
    echo '[coral-qwen35b-real-weights-test] token_reference=$MODEL_LOADER'
    echo '[coral-qwen35b-real-weights-test] token_golden=PASS'
fi
if [ -n '$DECODED_TOKEN_TEXT_B64' ]; then
    echo 'inference_text_source=cpu-tokenizer'
    printf 'inference_token_text='
    printf '%s' '$DECODED_TOKEN_TEXT_B64' | decode_base64
    printf '\n'
fi
echo '[coral-qwen35b-real-weights-test] PASS'
m5 --inst exit
exit 0
EOF

"${ROOT_DIR}/sim/gem5/apply_patchset.sh"
cd "${ROOT_DIR}/thirdparty/gem5"
OPENNPUX_HOST_FUNCTIONAL_PRECISION="$HOST_FUNCTIONAL_PRECISION" \
CORAL_NPU_BACKEND=verilated-coral \
CORAL_SIM_HOST_PAGE_BUNDLE="$SIM_HOST_BUNDLE" \
CORAL_SIM_HOST_INFERENCE_RESULT="$SIM_HOST_RESULT" \
CORAL_HOST_FUNCTIONAL_MODEL="$HOST_FUNCTIONAL_MODEL" \
CORAL_HOST_FUNCTIONAL_RANGES="$HOST_FUNCTIONAL_RANGES" \
CORAL_HOST_FUNCTIONAL_TENSOR_PLAN="$HOST_FUNCTIONAL_TENSOR_PLAN" \
CORAL_RTL_BRIDGE="$BRIDGE" \
CORAL_RTL_FIRMWARE="$FIRMWARE" \
CORAL_RTL_CYCLES_PER_EVENT="${CORAL_RTL_CYCLES_PER_EVENT:-1000}" \
CORAL_KERNEL_IMAGE="$CORAL_KERNEL_IMAGE" \
CORAL_AUTO_RESUME_AFTER_CKPT="${CORAL_AUTO_RESUME_AFTER_CKPT:-1}" \
CORAL_CKPT_ROOT="${CORAL_CKPT_ROOT:-${ROOT_DIR}/checkpoint/coralnpu_qwen35b_real_9p_ckpt}" \
CORAL_CONFIG_OPTIONS="${CORAL_CONFIG_OPTIONS:-} --vio-9p --vio-9p-root=$MODEL_DIR --npu-operator-mode=hybrid --npu-dma-shared-base=0x8f000000 --npu-dma-shared-size=8MiB --npu-fast-dma --npu-fast-dma-event-batch=${CORAL_FAST_DMA_EVENT_BATCH:-1} --npu-fast-dma-sync-offset=0 --npu-fast-dma-sync-size=64KiB" \
CORAL_RESUME_BOOTSCRIPT="$TMP_SCRIPT" \
./run_multicore.sh
