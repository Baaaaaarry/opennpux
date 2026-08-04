# CoralNPU Overlay Rules

`sim/coralnpu` is the source of truth for local CoralNPU changes. It mirrors
`thirdparty/coralnpu` paths and is synced before build.

## Scope

- RTL wrappers, Verilator bridge, firmware, custom operators, and local fixes
  belong here.
- Sync into the submodule with:

```bash
./sim/coralnpu/apply_patchset.sh
```

- Build from `thirdparty/coralnpu` through the project scripts under
  `tools/coralnpu`.

## Required Care

- Do not make lasting edits directly inside `thirdparty/coralnpu`.
- Keep official CoralNPU behavior separable from OpenNPUX additions.
- If a Verilog/SystemVerilog compatibility fix is needed for Verilator, mirror
  the whole modified source file here and document why the change is required.
- C ABI changes must update both bridge and gem5-side loader.
- Firmware-visible mailbox/TCB/operator descriptor changes must update
  `docs/design/coral_operator_abi.md` and runtime headers.
- Operator changes must state whether they are full RTL, sampled RTL, hybrid
  modeling, or TFLM fallback.

## Validation

Expected local/GB10 checks, depending on scope:

```bash
./tools/coralnpu/phase2_build_bridge.sh
./tools/coralnpu/build_rvv_mobilenet_partial.sh
CORAL_MOBILENET_PARTIAL_DEBUG=1 ./tools/coralnpu/run_rvv_mobilenet_partial.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

If full RTL is too slow, provide sampled or partial validation plus the reason
full RTL is deferred.

