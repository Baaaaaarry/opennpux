#!/bin/sh

set -eu

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <new-vmlinux> [known-good-vmlinux]" >&2
    exit 1
fi

NEW_KERNEL="$1"
GOOD_KERNEL="${2:-}"

if [ ! -f "${NEW_KERNEL}" ]; then
    echo "error: kernel not found: ${NEW_KERNEL}" >&2
    exit 1
fi
if [ -n "${GOOD_KERNEL}" ] && [ ! -f "${GOOD_KERNEL}" ]; then
    echo "error: known-good kernel not found: ${GOOD_KERNEL}" >&2
    exit 1
fi

dump_kernel() {
    label="$1"
    kernel="$2"
    echo "== ${label}: ${kernel} =="
    file "${kernel}" || true
    if command -v readelf >/dev/null 2>&1; then
        echo "-- ELF header --"
        readelf -h "${kernel}" | grep -E 'Class:|Data:|Machine:|Entry point address:' || true
        echo "-- LOAD segments --"
        readelf -l "${kernel}" | awk '
            $1 == "LOAD" {
                printf "LOAD off=%s vaddr=%s paddr=%s filesz=%s memsz=%s flags=%s\n",
                    $2, $3, $4, $5, $6, $7
            }
        ' || true
    else
        echo "warning: readelf not found; install binutils" >&2
    fi
    if command -v nm >/dev/null 2>&1; then
        echo "-- key symbols --"
        nm -n "${kernel}" 2>/dev/null | grep -E ' (_text|stext|start_kernel|__primary_switched)$' || true
    fi
    echo
}

dump_kernel "new" "${NEW_KERNEL}"
if [ -n "${GOOD_KERNEL}" ]; then
    dump_kernel "known-good" "${GOOD_KERNEL}"
fi

cat <<'EOF'
Interpretation:
- gem5's ARM boot path expects an ELF vmlinux and relocates LOAD segments into
  simulated RAM before the bootloader branches to the masked entry address.
- If the known-good kernel prints serial but the new kernel does not, with the
  same disk and command line, the failure is before Linux reaches PL011 console.
- Compare the Entry point and LOAD segment shape first. Large differences from
  the known-good vmlinux usually mean the new kernel config/toolchain is not
  compatible with this gem5 boot path.
EOF
