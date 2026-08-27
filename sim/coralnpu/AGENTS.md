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
- Firmware-visible mailbox/invocation/command/tensor descriptor changes must update
  `docs/design/coral_operator_abi.md` and runtime headers.
- Operator changes must state whether they are full RTL, sampled RTL, hybrid
  modeling, or TFLM fallback.
- Changes to Coral custom-instruction classification, NPU coprocessor dispatch,
  custom CSRs, second-level decode, issue, synchronization, or retirement must
  follow `docs/design/xopennpux_coprocessor_isa_contract.md`.
- New XOpenNPUX instructions must use the real coprocessor path. Do not lower
  them to the legacy LSU/MMIO doorbell path.
- Coral first-level decode classifies `custom3`; operator-specific `funct3` and
  `funct7` decode belongs in the NPU second-level decoder.
- A normal asynchronous tensor instruction retires in Coral only after the NPU
  durably accepts it. Scalar-result and wait/fence instructions must wait for
  their architectural completion condition.

## Validation

Expected local/GB10 checks, depending on scope:

```bash
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/build_rvv_mobilenet_partial.sh
CORAL_MOBILENET_PARTIAL_DEBUG=1 ./tools/coralnpu/run_rvv_mobilenet_partial.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

If full RTL is too slow, provide sampled or partial validation plus the reason
full RTL is deferred.
