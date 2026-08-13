#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <destination-directory> [repository-id]" >&2
    exit 2
fi

DESTINATION="$1"
REPOSITORY="${2:-${OPENNPUX_HF_MODEL_REPO:-Qwen/Qwen3.5-35B-A3B-GPTQ-Int4}}"
ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
REVISION="${OPENNPUX_HF_REVISION:-main}"
CURL="${CURL:-curl}"

if ! command -v "${CURL}" >/dev/null 2>&1; then
    echo "error: curl is required to download model assets" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to parse the safetensors index" >&2
    exit 1
fi

mkdir -p "${DESTINATION}"

download_file()
{
    relative_path="$1"
    output="${DESTINATION}/${relative_path}"
    temporary="${output}.part"
    url="${ENDPOINT%/}/${REPOSITORY}/resolve/${REVISION}/${relative_path}?download=true"
    if [ -s "${output}" ]; then
        echo "[hf-model] reuse ${relative_path}"
        return
    fi
    mkdir -p "$(dirname -- "${output}")"
    echo "[hf-model] download ${relative_path}"
    if ! "${CURL}" --fail --location --retry 8 --retry-delay 5 \
        --retry-all-errors --continue-at - --output "${temporary}" "${url}"; then
        echo "error: download failed: ${relative_path}" >&2
        return 1
    fi
    if [ ! -s "${temporary}" ]; then
        echo "error: downloaded file is empty: ${relative_path}" >&2
        return 1
    fi
    mv -f "${temporary}" "${output}"
}

download_optional()
{
    relative_path="$1"
    if ! download_file "${relative_path}"; then
        rm -f "${DESTINATION}/${relative_path}.part"
        echo "[hf-model] optional asset unavailable: ${relative_path}" >&2
    fi
}

download_file config.json

if ! download_file model.safetensors.index.json; then
    rm -f "${DESTINATION}/model.safetensors.index.json.part"
    echo "[hf-model] sharded index unavailable; trying single safetensors file"
    download_file model.safetensors
else
    INDEX="${DESTINATION}/model.safetensors.index.json"
    TOTAL_SIZE="$(python3 - "${INDEX}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    index = json.load(source)
print(int(index.get("metadata", {}).get("total_size", 0)))
PY
)"
    if [ "${TOTAL_SIZE}" -gt 0 ]; then
        AVAILABLE_KIB="$(df -Pk "${DESTINATION}" | awk 'NR == 2 {print $4}')"
        REQUIRED_KIB="$(( (TOTAL_SIZE + 1023) / 1024 ))"
        # Keep 5% or 1 GiB headroom, whichever is larger.
        HEADROOM_KIB="$(( REQUIRED_KIB / 20 ))"
        [ "${HEADROOM_KIB}" -ge 1048576 ] || HEADROOM_KIB=1048576
        if [ "${AVAILABLE_KIB}" -lt "$(( REQUIRED_KIB + HEADROOM_KIB ))" ]; then
            echo "error: insufficient disk space for ${REPOSITORY}" >&2
            echo "required_kib=${REQUIRED_KIB} available_kib=${AVAILABLE_KIB}" >&2
            exit 1
        fi
    fi
    python3 - "${INDEX}" <<'PY' | while IFS= read -r shard; do
import json
import sys

with open(sys.argv[1], encoding="utf-8") as source:
    index = json.load(source)
for name in sorted(set(index["weight_map"].values())):
    print(name)
PY
        [ -n "${shard}" ] && download_file "${shard}"
    done
fi

for asset in \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    generation_config.json \
    merges.txt \
    vocab.json
do
    download_optional "${asset}"
done

echo "hf_model_repository=${REPOSITORY}"
echo "hf_model_directory=$(CDPATH= cd -- "${DESTINATION}" && pwd -P)"
echo "hf_model_download=PASS"
