# XOpenNPUX Coral Controller and NPU Coprocessor ISA Contract

## Status

Normative draft. This document is mandatory reading for changes to the Coral
scalar pipeline, XOpenNPUX CSRs or instructions, NPU coprocessor pipeline,
firmware code generation, assembler/disassembler support, and instruction-level
simulation.

The key words **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are normative.
Encoding conflicts listed in [Open specification items](#open-specification-items)
MUST be resolved and reviewed before RTL depending on those fields is merged.

## Purpose

OpenNPUX implements a Coral RISC-V controller plus an NPU coprocessor. The
architectural target is not the existing descriptor doorbell prototype. The
target is a fixed-width, 32-bit `XOpenNPUX` custom instruction extension whose
extended operator state is carried by custom CSRs.

The implementation proceeds in this order:

1. Execute one operator end to end with standard scalar RISC-V instructions,
   custom CSRs, and one real NPU custom instruction.
2. Build the primitive operator instruction library.
3. Add fused instructions only after primitive semantics are stable.
4. Add compiler graph optimization, parallel scheduling, dynamic tiling, and
   model lowering.
5. Run general models without introducing model-family-specific hardware
   semantics.

## Architectural Boundary

### Coral controller responsibilities

The original Coral scalar pipeline MUST remain responsible for:

- instruction fetch from ITCM or the existing IBus fallback;
- standard RISC-V scalar decode and execution;
- branch, scalar LSU, scalar CSR, trap, and debug behavior;
- reading the GPR fields required by a custom instruction;
- first-level classification of the `custom3` major opcode;
- dispatch handshake with the NPU coprocessor;
- architectural retirement and precise rejection of an invalid instruction.

The Coral first-level decoder MUST NOT implement the operator meaning of
`funct3`, `funct7`, `.ttt`, `.ttr`, `.rt`, `.tr`, MMA, SIMD, reduction, SFU,
MOV, cache, or synchronization operations.

### NPU coprocessor responsibilities

The NPU coprocessor MUST own:

- second-level decode of every accepted `custom3` instruction;
- custom CSR storage and instruction-time CSR snapshots;
- tensor-address, scalar-operand, and instruction-form interpretation;
- tensor dependency and resource scoreboarding;
- engine-specific issue queues;
- MMA, SIMD, reduction, SFU, MOV/TDMA, cache, and synchronization execution;
- tensor completion, asynchronous errors, counters, and interrupt reporting.

### Required two-level flow

```text
ITCM / IBus
    |
Coral Fetch
    |
Coral Decode L1
    |-- standard instruction --> existing Coral scalar pipeline
    |
    `-- opcode == custom3
            |
        NPU dispatch ready/valid
            |
        NPU Decode L2
            |
        NPU dependency scoreboard
            |
        engine issue queue
            |
        NPU functional unit
```

## Fixed-Width ISA Rule

All XOpenNPUX instructions MUST remain 32 bits. Tensor shapes, strides, data
types, cache policy, prefetch policy, and other operands that do not fit in the
instruction MUST be configured through XOpenNPUX custom CSRs. A single-operator
instruction MUST NOT require an in-memory operator descriptor.

The major opcode is RISC-V `custom3`:

```text
custom3 = 0b1111011 = 0x7b
```

The first implementation MUST NOT lower a new XOpenNPUX instruction to an LSU
store or MMIO doorbell. It MUST enter the NPU coprocessor dispatch interface.

## Custom CSR Contract

### Privilege split

Machine-level policy CSRs use the custom machine read/write range beginning at
`0x7c0`. User-visible operator context uses the custom user read/write range
beginning at `0x800`. Identification uses the custom user read-only range
beginning at `0xcc0`.

Machine policy state and user operator state MUST remain separate:

- Machine state controls global hardware policy such as address mode, MMU,
  cache/scratchpad partitioning, compression, and prefetch enablement.
- User state describes the next tensor operation, including shapes, strides,
  data types, padding, and per-context cache or prefetch policy.

### Machine-level logical CSR schema

| CSR | Name | Required logical fields |
| --- | --- | --- |
| `0x7c0` | `tctrl` | `prefetch_en`, `addr_mode`, `mmu_on`, `cache_on`, `compress_on` |
| `0x7c1` | `kv_tile_dim` | `tile_dim0..4` |
| `0x7c2` | `weight_tile_dim` | `tile_dim0..2` |
| `0x7c3..0x7c6` | `dim_grp_entry0..3` | four dimensions per group |
| `0x7c7` | `dim_grp_ctrl` | group dimension ranges and `dim4` values |
| `0x7c8` | `cache_ctrl` | scratchpad/cache partition selection |

The unambiguous `tctrl` bit allocation is:

| Bit | Field | Encoding |
| --- | --- | --- |
| 0 | `prefetch_en` | 0 disabled, 1 enabled |
| 1 | `addr_mode` | 0 normal address, 1 LLM address mode |
| 2 | `mmu_on` | 0 disabled, 1 enabled |
| 3 | `cache_on` | 0 all scratchpad, 1 cache enabled |
| 4 | `compress_on` | 0 disabled, 1 compression/decompression enabled |

`cache_ctrl[7:0]` encodes the scratchpad allocation as 0, 1/2, 1/4, 1/8,
or 100 percent for values 0 through 4 respectively. Other values are reserved.

### User-level logical CSR schema

| CSR | Name | Required logical fields |
| --- | --- | --- |
| `0xcc0` | `tnpuid` | version and vendor identification |
| `0x800` | `mma_shape` | `M`, `N`, `K` |
| `0x801` | `mma_data_type` | source 1, source 2, destination types |
| `0x802` | `tensor_dim` | `dim0..3` |
| `0x803` | `tensor_dim_size` | source dimension counts and `dim4` |
| `0x804..` | stride registers | independent source 1, source 2, and destination strides |
| assigned CSR | `tensor_data_type` | source and destination data types |
| `0x807` | `padding_val` | explicit padding value |
| `0x808` | `tensor_cache_cfg` | eviction and prefetch modes |
| `0x809` | `prefetch_cfg` | stride, threshold, and prefetch count |
| `0x80a` | `prefetch_start_time` | prefetch insertion time |

The supported logical data types are FP16, BF16, FP32, INT16, INT8,
FP8-E4M3, FP8-E5M2, INT4, INT2, MXFP6, and MXFP4. Each instruction definition
MUST state which subset is legal and how accumulation and destination rounding
are performed.

The common source data-type encoding is:

| Value | Type |
| --- | --- |
| 0 | FP16 |
| 1 | BF16 |
| 2 | FP32 |
| 3 | INT16 |
| 4 | INT8 |
| 5 | FP8-E4M3 |
| 6 | FP8-E5M2 |
| 7 | INT4 |
| 8 | INT2 |
| 9 | MXFP6 |
| 10 | MXFP4 |

`mma_data_type` contains distinct source 1, source 2, and destination fields.
The final bit positions are blocked by the open specification items below.

### CSR access and snapshot rules

1. Standard `csrw/csrr` instructions are fetched, decoded, and retired by the
   Coral controller.
2. An XOpenNPUX CSR access MUST be routed to the NPU CSR register file.
3. A CSR write MUST NOT retire until the NPU CSR interface accepts the write.
4. A custom instruction MUST observe all older retired CSR writes.
5. NPU Decode L2 MUST atomically snapshot every CSR needed by the instruction
   when accepting the instruction.
6. A later CSR write MUST NOT alter an already accepted instruction.
7. The implementation SHOULD carry a monotonically changing CSR epoch in the
   dispatch and trace state.
8. The operating system MUST save and restore user-level XOpenNPUX CSR state
   before multiple processes or harts share the coprocessor.

## Instruction Operand Forms

Instruction suffixes are architectural operand classes, not cosmetic assembly
syntax.

| Form | `rd` field | `rs1` field | `rs2` field | Scalar GPR writeback |
| --- | --- | --- | --- | --- |
| `.ttt` | destination tensor address/coordinate | source tensor address/coordinate | source tensor address/coordinate | no |
| `.ttr` | destination tensor address/coordinate | source tensor address/coordinate | scalar value | no |
| `.tt` | destination tensor address/coordinate | source tensor address/coordinate | unused | no |
| `.rt` | scalar destination register | source tensor address/coordinate | unused | yes |
| `.tr` | destination tensor address/coordinate | scalar value | unused | no |
| `.tttt` | first destination tensor | source tensor | scalar or encoded operand | no; second destination is encoded separately |

For tensor-destination forms, the `rd` field is a GPR **source** containing an
address or coordinate. Coral Dispatch and the NPU scoreboard MUST NOT treat it
as an integer destination. For `.rt`, `rd` is a true GPR destination and MUST
remain reserved until the NPU scalar result is returned.

## Instruction Encoding Classes

| `funct3` | Class |
| --- | --- |
| `000` | MMA and tensor-scalar SIMD |
| `001` | tensor-tensor SIMD |
| `010` | unary, reduction, scalar result, tensor fill, and SFU |
| `011` | MOV/copy |
| `100` | multi-output operations such as TopK |
| `101` | cache and prefetch |
| `110` | fence and synchronization |

### MMA

`tmma.ttt (rd),(rs1),(rs2)` uses `custom3`, `funct3=000`, and
`funct7=0000000`. `rs1`, `rs2`, and `rd` contain the source 1, source 2, and
destination first-element addresses or coordinates. The instruction uses
`mma_shape` and `mma_data_type` plus the applicable stride CSRs.

The architectural operation is a full tensor MMA. The initial implementation
MAY use a slow sequential MAC state machine. Tile size, array dimensions,
double buffering, and other microarchitecture choices MUST NOT change the
architectural result.

#### Experimental v0.1 physical CSR map

The first `tmma.ttt` implementation freezes the following experimental RV32
layout so that NPU L2 decode and functional execution can be tested before the
complete tensor CSR map is reviewed:

| CSR | Field | Physical layout |
| --- | --- | --- |
| `0x800` `mma_shape` | `M` | `[9:0]`, legal range 1 through 1023 |
| | `N` | `[19:10]`, legal range 1 through 1023 |
| | `K` | `[29:20]`, legal range 1 through 1023 |
| | reserved | `[31:30]`, write zero |
| `0x801` `mma_data_type` | source 1 | `[3:0]` |
| | source 2 | `[7:4]` |
| | destination | `[11:8]` |
| | reserved | `[31:12]`, write zero |

The v0.1 execution subset accepts only FP32 source 1, FP32 source 2, and FP32
destination with contiguous row-major tensors. It computes `C[M,N] =
A[M,K] * B[K,N]` with a sequential FP32 MAC loop. The GPR values carried in
`rs1`, `rs2`, and `rd` are byte addresses of `A`, `B`, and `C` respectively.

This layout is an experimental implementation profile, not the final general
tensor ABI. Any incompatible replacement MUST change the profile version and
update the shared encoder, NPU decoder, firmware, and tests together.

### SIMD and reduction

- `funct3=000`, `funct7=0000001/0000010`: `tadd.ttr`, `tmul.ttr`.
- `funct3=001`, `funct7=0000001..0000101`: tensor `tadd`, `tmul`, `tmax`,
  `tmin`, and `tcmp`.
- `funct3=010`, `funct7=0010000..0010101`: row reduction/cumulative/argmax.
- `funct3=010`, `funct7=0100000..0100110`: unary arithmetic, transpose, and
  conversion.
- `funct3=010`, `funct7=0110000`: tensor-to-scalar sum form.
- `funct3=010`, `funct7=1000000`: scalar-to-tensor range/fill form.

#### Experimental v0.1 TADD/TMUL functional profile

The first multi-operator functional coprocessor implements FP32 tensor add as
`tadd.ttt` (`funct3=001`, `funct7=0000001`) and multiply as `tmul.ttt`
(`funct3=001`, `funct7=0000010`). `rs1`, `rs2`, and `rd` carry the source 1,
source 2, and destination first-element addresses. The architectural results
are `dst[i] = src1[i] + src2[i]` and `dst[i] = src1[i] * src2[i]`.

Until the `tensor_dim` RTL CSR ports are available, the v0.1 compatibility
profile derives the contiguous element count from `mma_shape.M * N * K` and
uses `mma_data_type` for the three operand types. This temporary CSR mapping
MUST remain confined to the firmware operator library and NPU L2 functional
decoder. Model compilers MUST call `xopennpux_add_fp32()` or
`xopennpux_mul_fp32()` rather than program the compatibility CSRs directly.
Replacing the CSR mapping MUST NOT change the 32-bit instruction encodings.

### SFU

`funct3=010`, `funct7=1000000..1000101` selects `texp`, `ttanh`, `tsqrt`,
`tcos`, `tsin`, and `tlog`. The final allocation MUST resolve the overlap with
the scalar-to-tensor encoding before implementation.

### MOV

`funct3=011` selects linear, tensor, and permute copy, with optional padding.
The implementation MUST use the specified GPR/CSR fields rather than fetching
an operator descriptor. Linear copy interprets `rs2` as its packed control
value; tensor and permute forms interpret it as dimension-group and padding
control.

| `funct7` | Operation |
| --- | --- |
| `0000000` | linear copy |
| `0000001` | tensor copy |
| `0000010` | permute copy |
| `0000100` | linear copy plus padding |
| `0000101` | tensor copy plus padding |
| `0000110` | permute copy plus padding |

For linear copy, the logical `rs2` control value contains a 32-bit byte length,
source data type, padding selection, and padding element count. For tensor and
permute copy, it contains source and destination dimension-group selectors,
source type, padding dimension, first padding location, and padding selection.
RV32 materialization of control values wider than XLEN is an open item and MUST
NOT be implemented through an undocumented side channel.

### Cache and prefetch

`funct3=101` selects tile or non-tile invalidate, clean, flush, and prefetch.
Cache instructions operate on the address or tile coordinate in `rs1`.
Ordering relative to tensor operations MUST be represented in the NPU
scoreboard, not inferred only from Coral retirement order.

The common cache encoding is:

```text
31:29=000, bit28=tile, 27:25=subfn, 24:20=00000,
19:15=rs1, 14:12=101, 11:7=00000, 6:0=custom3
```

| `subfn` | Operation |
| --- | --- |
| `000` | invalidate |
| `001` | clean |
| `010` | flush |
| `011` | prefetch |

`tile=1` selects tile addressing. Non-tile prefetch uses its defined aligned
immediate offset form. Reserved encodings MUST be rejected by NPU Decode L2.

### Fence and synchronization encoding

`tfence` uses `funct3=110` with all other instruction fields zero. The
synchronization family shares `funct3=110`:

| Instruction | Selector | Architectural behavior |
| --- | --- | --- |
| `tsync.wait.r` | register selector `001` | younger covered work waits for the ID |
| `tsync.wait.i` | immediate prefix `101` | immediate-ID wait |
| `tsync.arrive.r` | register selector `010` | older covered work completes before arrival publication |
| `tsync.arrive.i` | immediate prefix `110` | immediate-ID arrival |
| `tsync.r` | register selector `011` | full register-ID barrier |
| `tsync.i` | immediate prefix `111` | full immediate-ID barrier |

The assembler, decoder, and tests MUST share one generated encoding table for
these forms; manually duplicated masks are not acceptable once compiler support
is introduced.

## Dispatch and Retirement Contract

### Dispatch packet

Coral MUST provide at least this information to NPU Decode L2:

```c
struct xnpu_dispatch_packet {
    uint32_t instruction;
    uint32_t pc;
    uint32_t rs1_value;
    uint32_t rs2_value;
    uint32_t rd_value;
    uint32_t hart_id;
    uint32_t privilege;
    uint32_t sequence_id;
    uint32_t csr_epoch;
};
```

The implementation MAY optimize away unused fields after L1 classification,
but the architectural behavior MUST remain equivalent.

### Ready/valid and rejection

The Coral-to-NPU interface MUST provide `valid`, `ready`, and an accept/reject
response.

- If `ready` is false, Coral MUST stall the custom instruction.
- If NPU Decode L2 accepts and durably queues the instruction, Coral MAY mark
  an asynchronous tensor instruction complete and retire it normally.
- If NPU Decode L2 rejects an unsupported or malformed encoding, Coral MUST
  raise a precise illegal-instruction exception and MUST NOT retire it.
- Coral retirement means **accepted by the NPU**, not **tensor execution
  completed**.

### Retirement classes

| Instruction class | Coral completion condition |
| --- | --- |
| asynchronous compute/MOV/cache/prefetch | accepted into a durable NPU queue |
| `tfence`, `tsync.wait`, full `tsync` | synchronization condition completed |
| `.rt` scalar-result instruction | scalar result returned and written to GPR |
| malformed/unsupported custom instruction | never retires; precise trap |

An implementation MUST NOT retire `.rt`, `tfence`, or wait-type synchronization
instructions merely because they entered an NPU queue.

## NPU Decode, Dependency, and Issue Contract

NPU Decode L2 MUST produce an internal instruction context containing the raw
instruction, PC, sequence ID, decoded operand classes, operand values, and CSR
snapshot. It MUST route instructions to distinct logical issue classes:

- MMA;
- SIMD/vector;
- reduction;
- SFU;
- MOV/TDMA;
- cache/prefetch;
- synchronization.

The NPU dependency scoreboard MUST cover:

- tensor RAW, WAR, and WAW hazards;
- scalar-result dependencies for `.rt`;
- CSR snapshot/version dependencies;
- overlapping memory ranges, conservatively at first;
- functional-unit and queue capacity;
- MOV/cache operations relative to compute;
- `tfence` and `tsync` dependency scopes.

The first implementation MAY serialize all NPU instructions. Serialization is
an implementation choice, not an architectural requirement, and MUST preserve
the same visible behavior when parallel issue is introduced later.

## Synchronization and Memory Visibility

`tfence` is a full thread-local NPU barrier. It MUST wait for all older NPU
instructions, implicit memory accesses, tensor writebacks, and relevant cache
operations before completing. No younger instruction covered by the barrier
may begin before it completes.

`tsync.arrive` publishes completion of the older dependency scope.
`tsync.wait` blocks younger dependent work until the specified synchronization
ID arrives. Full `tsync` combines both directions.

Because ordinary asynchronous custom instructions may retire before tensor
completion, scalar code MUST execute `tfence` or the corresponding
`tsync.wait` before consuming an NPU-produced tensor:

```asm
tmma.ttt (a2),(a0),(a1)
tfence
lw t0, 0(a2)
```

Standard memory fences are still required where CPU/NPU coherency does not
provide the required ordering for producer data.

## Error Contract

Errors detected before acceptance are precise Coral exceptions. These include
an illegal encoding, unavailable extension, invalid privilege, or an encoding
that NPU Decode L2 does not support.

Errors detected after an asynchronous instruction retires are NPU completion
errors. They MUST be reported through NPU status/cause state, a completion
record or queue, and optionally an interrupt. An asynchronous error record MUST
identify at least the hart, sequence ID, original PC, instruction, cause,
faulting address when applicable, and CSR epoch.

The first operator implementation SHOULD validate all source and destination
ranges before exposing partial destination writes. If partial completion is
architecturally allowed later, the ISA specification MUST define restart and
visibility behavior first.

## Compiler Contract

The compiler lowers one operator to:

1. standard scalar RISC-V address and control calculations;
2. `csrw` updates for changed XOpenNPUX operator state;
3. one or more 32-bit XOpenNPUX instructions;
4. only the `tfence` or `tsync` operations required by data dependencies.

It SHOULD eliminate redundant CSR writes when consecutive operations use the
same state. It MUST model `rd` as a source for tensor-destination forms and as a
destination only for scalar-result forms. It MUST model custom instructions as
having the specified memory effects so that host compiler scheduling cannot
move dependent loads or stores across them illegally.

The first compiler milestone is a single supported operator. Unsupported
operators MUST fail compilation or use an explicitly selected reference path;
they MUST NOT silently pass acceptance by invoking the legacy host bridge.

## Initial Implementation and Acceptance Gate

### Current implementation status

The first checked-in implementation now spans Coral RTL and
`hw_sim/gem5_bridge`:

- canonical `custom3`/`tmma.ttt` and v0.1 CSR encoding helpers;
- NPU second-level instruction decode;
- CSR storage, monotonically increasing epoch, and accept-time snapshot;
- a four-entry durable submission queue with explicit backpressure;
- sequential FP32 MATMUL over the NPU EXTMEM address model;
- completion metadata and invalid-encoding, invalid-CSR, data-type, and address
  errors;
- unit tests for encoding, snapshot isolation, execution, rejection, and
  queue-full behavior.
- Coral L1 classification of `custom3` without interpreting NPU sub-opcodes;
- a physical Coral-to-NPU ready/valid request and precise reject response;
- `mma_shape` and `mma_data_type` routing with an incrementing CSR epoch;
- three GPR source reads for address-bearing `rs1`, `rs2`, and `rd`, including
  RAW hazard tracking and no false scalar destination mark;
- retirement coupling: a command enters the Coral ROB only on durable NPU
  acceptance, while malformed or invalid commands trap;
- `tfence`, accepted only after the older NPU queue drains;
- firmware intrinsics and `gem5_tmma_smoke.elf` source for CSR configuration,
  custom dispatch, tensor execution, fence, and result checking.

The current physical CSR registers reside in the Coral CSR block as the RV32
architectural access point and are exported as one atomic NPU CSR-state bundle.
The NPU snapshots the bundle at request acceptance; operator execution never
re-reads mutable Coral CSR state.

The source integration is complete, but the generated Verilog, bridge, and
firmware acceptance run remains mandatory on the Linux/GB10 toolchain before
this experimental profile is treated as a validated RTL path.

The first mandatory implementation subset is:

- the CSR state needed by `tmma.ttt`;
- `tmma.ttt` L1 classification and L2 decode;
- Coral-to-NPU dispatch backpressure and rejection;
- NPU CSR snapshot and serialized issue;
- a baseline functional MMA engine;
- tensor memory read/write;
- `tfence`;
- generic operation classification shared by TMMA, TADD, and TMUL;
- FP32 `tadd.ttt`/`tmul.ttt` functional execution and per-operation statistics;
- `xopennpux_ops.h` as the firmware/compiler-facing operator-library boundary;
- trace and error reporting.

Acceptance requires independent evidence that:

1. the firmware ELF contains the intended `custom3` instruction;
2. Coral L1 classifies it without lowering it to LSU `SW`;
3. NPU L2 reports the expected decoded instruction and CSR epoch;
4. the instruction enters the MMA issue path;
5. the MMA engine, not the host functional operator bridge, reads and writes
   tensors;
6. Coral retires the asynchronous instruction only after NPU acceptance;
7. `tfence` retires only after tensor completion;
8. the destination matches an independent scalar CPU reference;
9. queue-full backpressure and invalid-encoding rejection are tested;
10. no legacy descriptor or MMIO doorbell is required by the test.

GB10 validation commands:

```bash
./tools/coralnpu/prepare_coral_bazel.sh
./tools/coralnpu/test_tmma_instruction.sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/coralnpu/run_tmma_instruction_test.sh
./tools/coralnpu/test_tmma_operator_e2e.sh
./tools/coralnpu/run_tmma_operator_e2e_test.sh
```

`test_tmma_instruction.sh` parses executable ELF sections directly and checks
the CSR writes, `custom3` TMMA, and `tfence` encodings. It deliberately does
not require a system `riscv32-unknown-elf-objdump` installation. All Coral
build scripts prefer the repository-local Bazel version pinned by
`thirdparty/coralnpu/.bazelversion` over a system Bazel/Bazelisk wrapper.

The full-system run MUST additionally observe `Coral XOpenNPU accepted`,
`Coral XOpenNPU complete ... error=0`, a successful `tfence` dispatch, firmware
halt without a Coral fault, and the independently calculated EXTMEM writeback
checksum. The baseline 2x2 FP32 smoke expects destination `0x20000200`, 16
bytes, checksum `0xe6d7ed59`, and firmware result word `0x544d4d41` at
`0x20000300`.

The operator end-to-end test preserves that minimal smoke and adds three
independently checked packed FP32 workloads: `2x3 * 3x2`, `3x2 * 2x4`, and
`1x4 * 4x3`. It requires CSR reconfiguration, three TMMA/TFENCE pairs,
rectangular and negative-value arithmetic, complete destination checksums, and
firmware-side word-for-word validation. This test covers the complete v0.1
execution subset; it does not claim stride, transpose, accumulation, or mixed
precision support.

## Legacy Compatibility

The existing `CUSTOM_0 NPU_LAUNCH -> LSU SW -> MMIO doorbell -> descriptor`
path is a bring-up and regression path. It MAY remain available while the new
ISA is developed, but it is not the target implementation for any new
XOpenNPUX instruction.

`docs/design/coprocessor_command_path.md` documents that legacy path. Where it
conflicts with this document for new instruction development, this document is
normative.

The generic CPU runtime and compiled-model formats may continue to submit
coarse jobs. NPU firmware or compiler-generated code lowers those jobs into
XOpenNPUX CSR writes and instructions. Model names and model-family-specific
behavior MUST NOT enter the hardware ISA.

## Open Specification Items

The supplied logical format contains conflicts that MUST be resolved in a
reviewed ISA-map change before affected RTL or compiler code is merged:

1. `mma_shape` assigns `15:0` independently to M, N, and K without defining
   their physical packing or separate CSR addresses.
2. Several logical CSRs contain more than 32 bits. The current RV32 Coral core
   requires explicit low/high CSR pairs or another reviewed access mechanism.
3. CSR `0x805` is assigned to more than one stride register.
4. CSR `0x806` is assigned to both stride and tensor data-type state.
5. Source 2 and destination stride fields are not independently addressable in
   the current table.
6. `tnpuid.vendor_id[15:8]` cannot hold the documented reset value `0x666`.
7. The second destination encoding for `.tttt/.ttti` needs an exact, unique
   bitfield definition.
8. `funct3=010`, `funct7=1000000` overlaps scalar-to-tensor `trng` and SFU
   `texp`.
9. Every multi-field CSR needs exact non-overlapping bit positions and RV32
   access semantics.
10. All stride fields must normatively use bytes or elements; the unit may not
    vary implicitly between instructions.

Until resolved, implementations MAY prototype the unambiguous `tmma.ttt`
subset using a temporary experimental map, but that map MUST be isolated and
MUST NOT be presented as a stable ABI.

## Change Control

Changes to this contract require:

1. a design note or ADR explaining the compatibility impact;
2. updates to RTL decode, firmware headers, compiler encoding, assembler or
   disassembler support, and tests in the same PR when applicable;
3. review by owners of both the Coral controller pipeline and NPU coprocessor;
4. instruction encoding collision tests and CSR-map validation;
5. end-to-end evidence on the required x86/GB10 validation platform.
