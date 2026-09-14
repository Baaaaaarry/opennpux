#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODULAR_DIR="${MODULAR_SOURCE_DIR:-${ROOT_DIR}/thirdparty/modular}"

usage() {
    echo "usage: $0 [--query TARGET | --test TARGET]" >&2
    exit 2
}

mode=validate
target=
case "$#" in
    0) ;;
    2)
        case "$1" in
            --query) mode=query ;;
            --test) mode=test ;;
            *) usage ;;
        esac
        target=$2
        ;;
    *) usage ;;
esac

if [ ! -e "${MODULAR_DIR}/.git" ]; then
    echo "[modular-source] initializing pinned submodule"
    git -C "${ROOT_DIR}" submodule update --init --depth 1 thirdparty/modular
fi

for path in bazelw MODULE.bazel LICENSE max/python/max/graph/graph.py; do
    if [ ! -e "${MODULAR_DIR}/${path}" ]; then
        echo "modular_source=FAIL missing=${path}" >&2
        exit 1
    fi
done

expected="$(git -C "${ROOT_DIR}" ls-tree HEAD thirdparty/modular 2>/dev/null | awk '{print $3}')"
actual="$(git -C "${MODULAR_DIR}" rev-parse HEAD)"
if [ -n "${expected}" ] && [ "${expected}" != "${actual}" ]; then
    echo "modular_source=FAIL expected=${expected} actual=${actual}" >&2
    exit 1
fi

echo "modular_source_path=${MODULAR_DIR}"
echo "modular_source_revision=${actual}"
echo "modular_source=PASS"

case "${mode}" in
    validate) ;;
    query) (cd "${MODULAR_DIR}" && ./bazelw query "${target}") ;;
    test) (cd "${MODULAR_DIR}" && ./bazelw test "${target}") ;;
esac
