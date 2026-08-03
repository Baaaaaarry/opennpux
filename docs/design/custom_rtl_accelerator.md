# Custom RTL Accelerator

## Architecture

The custom unit is a synthesizable three-cycle 32-bit multiply-accumulate RTL
block under `sim/coralnpu/hdl/verilog/opennpux`. It is built by Coral's Bazel
and Verilator rules and linked into the same bridge as the official
`CoreMiniAxi` model.

Coral firmware accesses the unit through external AXI address `0x30000000`.
The bridge routes only this bounded register aperture to the custom Verilated
model; EXTMEM tensor traffic continues through gem5 coherent DMA.

This keeps the boundaries explicit:

- official Coral core executes firmware and issues AXI transactions;
- custom RTL performs the MAC computation and reports hardware cycles;
- gem5 models the ARM SoC, coherent memory, MMIO shell, and scheduling;
- Linux uses the unchanged driver runtime and command runtime command ABI.

## Register Interface

| Offset | Register |
| --- | --- |
| `0x00` | operand A |
| `0x04` | operand B |
| `0x08` | accumulator |
| `0x0c` | start command |
| `0x10` | result |
| `0x14` | done/busy status |
| `0x18` | accumulated active cycles |
| `0x1c` | accelerator ID `0x4e5058a1` |

The `VECTOR_ADD_CUSTOM_U32` firmware path computes each output as
`input0 * 1 + input1`. The software and custom paths therefore consume the
same model tensors and must produce identical output checksums.

## Verification Layers

1. Standalone Verilator test checks ID, register protocol, result 42, and
   three-cycle completion.
2. Coral firmware checks the RTL ID before accepting a custom command.
3. Linux A/B test executes software and RTL opcodes through the same driver.
4. The heterogeneous sample model contains one software and one RTL node.
