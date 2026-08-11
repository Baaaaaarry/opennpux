# Coral Control Core to NPU Coprocessor Command Path

## Scope

This increment validates the control path before adding detailed TDMA and
compute timing models:

1. Coral scalar Decode recognizes an OpenNPUX `NPU_LAUNCH` instruction.
2. The instruction reaches the existing operator doorbell through the normal
   scalar LSU and AXI master path.
3. `Gem5CoprocessorCommandAdapter` arbitrates the command source and allocates
   a submission tag.
4. The task front-end validates the operator descriptor.
5. The second-level command decoder expands the operator into engine-specific
   micro-commands.
6. `Gem5DependencyScoreboard` controls issue order.

## Instruction Encoding

`NPU_LAUNCH` uses the RISC-V `custom-0` major opcode:

```text
funct7=0000000 rs2=descriptor rs1=operator_base funct3=000 rd=00000 opcode=0001011
```

Firmware emits it with:

```asm
.insn r 0x0b, 0, 0, x0, operator_base, descriptor_address
```

The first-level Coral decoder lowers the instruction to an SW of `rs2` at
`rs1 + 4`. This is the existing operator doorbell address. Consequently the
initial implementation exercises Coral Fetch, Decode/Dispatch, scalar
scoreboarding, LSU, AXI master, and the Verilated bridge without adding a new
top-level RTL interface.

## Second-Level Decode

The bridge converts a validated operator descriptor into this dependency
chain:

```text
FETCH_DESCRIPTOR
    -> TDMA_READ_OPERANDS
    -> EXECUTE_OPERATOR (Tensor, Vector, or SFU)
    -> TDMA_WRITEBACK
    -> COMPLETE
```

The execution engine is selected from the operator opcode:

- Tensor: Conv2D, DepthwiseConv2D, MatMul, FullyConnected.
- Vector: Add, LayerNorm.
- SFU: Softmax.

Each command contains a submission tag, command ID, descriptor address,
engine, opcode, dependency mask, source, and state. A command issues only when
all bits in its dependency mask have completed.

## Compatibility

The adapter accepts both command sources:

- Legacy firmware MMIO doorbell.
- `NPU_LAUNCH` custom instruction.

Both sources share the same descriptor ABI, second-level decoder, queues,
scoreboard, operator backend, and completion semantics. The current hybrid
kernel remains attached only to `EXECUTE_OPERATOR`; later TDMA and compute
models can replace individual micro-command handlers without changing the
firmware ABI.

## Validation

Host-side control-model test:

```bash
./tools/coralnpu/test_coprocessor_command.sh
```

GB10 full build:

```bash
./tools/coralnpu/phase2_build_bridge.sh
./tools/coralnpu/build_rvv_mobilenet.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The host log must contain five ordered command issues and a completion:

```text
Coral command issue ... opcode=0 ... source=1
Coral command issue ... opcode=1 ... source=1
Coral command issue ... opcode=2 ... source=1
Coral command issue ... opcode=3 ... source=1
Coral command issue ... opcode=4 ... source=1
Coral command submission ... state=complete pending=0
```

`source=1` proves the descriptor was submitted through the custom-instruction
path. The guest must still report `mobilenet_test=PASS`.
