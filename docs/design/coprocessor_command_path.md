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
7. A cycle-driven modeling front-end advances issued micro-commands using the
   configured latency model and publishes descriptor completion asynchronously.

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

Generic executables use the same outer command path with
`CORAL_OPERATOR_OP_GENERIC_COMMAND`. The outer descriptor points to a
versioned `opennpux_npu_functional_request` in local EXTMEM and mirrors the
inner generic opcode in `reserved[0]`. The second-level decoder uses that
inner opcode to select Tensor, Vector, or SFU before execution. At the execute
boundary, `DispatchGem5GenericCommand()` validates every EXTMEM range,
translates address-based operands to host pointers, calls
`Gem5HostFunctionalBackend`, and writes operation, traffic, and modeled-cycle
statistics back to both the request and the outer descriptor. This keeps the
firmware/custom-instruction path intact while allowing Host C++ kernels to be
replaced incrementally by timing models or RTL functional units.

The functional request ABI intentionally contains no host pointer. It carries
an operator parameter address plus role-tagged operand address/size pairs.
The compiler-generated tensor plan and CPU runtime resolve logical tensors to
device addresses; NPU firmware materializes the request only after those
addresses are known.

Each command contains a submission tag, command ID, descriptor address,
engine, micro-op, dependency mask, source, state, latency, and remaining
cycles. A command issues only when all bits in its dependency mask have
completed and the selected engine has a free credit.

## Cycle-Driven Modeling

The first modeling increment does not execute the operator immediately at the
doorbell write. Instead:

1. `CoprocessorCommandAdapter::Submit()` validates the descriptor and allocates
   a submission tag.
2. `DecodeOperator()` expands the descriptor into five ordered micro-commands.
3. `AdvanceCycle()` issues dependency-ready commands, decrements their latency,
   and moves completed work to `ReadyToComplete`.
4. The bridge handles the `EXECUTE_OPERATOR` micro-op by invoking the current
   host functional kernel, then reschedules the command for modeled compute
   latency.
5. `WRITEBACK` latency is updated from the descriptor's output byte count.
6. `COMPLETE` publishes descriptor state, error, modeled cycles, and operator
   statistics.

This creates a stable seam for later replacing the host functional kernel with
TDMA, scratchpad, tensor, vector, and SFU timing models or real RTL units while
keeping the firmware-visible descriptor ABI unchanged.

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
./tools/coralnpu/test_hybrid_kernels.sh
```

Focused RTL end-to-end test on x86 Linux/GB10:

```bash
./tools/coralnpu/build_rvv_mobilenet.sh -c opt
./tools/coralnpu/build_npu_launch_smoke.sh -c opt
./tools/coralnpu/run_npu_launch_test.sh
```

The focused firmware constructs one four-element INT8 ADD descriptor in local
EXTMEM and submits it with `NPU_LAUNCH`. The run script requires independent
evidence from both sides of the boundary: the bridge log must show
`source=custom-instruction` and all five second-level micro-commands, while the
guest must report `npu_launch_output=11,22,33,44` and
`npu_launch_test=PASS`. Unlike the full MobileNet test, this path does not load
TFLM or execute a model graph, so it is suitable for per-commit acceptance.

GB10 full build:

```bash
./tools/coralnpu/phase2_build_bridge.sh
./tools/coralnpu/build_rvv_mobilenet.sh
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The host log must contain five ordered command issues and a completion:

```text
Coral command submission ... source=custom-instruction ...
Coral command complete ... micro_op=fetch-descriptor ...
Coral command complete ... micro_op=read-operands ...
Coral command execute ... micro_op=execute-operator ...
Coral command complete ... micro_op=writeback ...
Coral command complete ... micro_op=complete ...
Coral hybrid operator complete ...
```

`source=custom-instruction` proves the descriptor was submitted through the
custom-instruction path. The guest must still report `mobilenet_test=PASS`.
