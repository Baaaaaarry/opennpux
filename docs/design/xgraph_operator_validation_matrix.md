# XGraph Operator Mapping and Validation Matrix

This document is the canonical record for generic-model operation lowering to
XOpenNPUX functional instructions. It records actual full-system acceptance
data, not projected RTL throughput. The current backend is the C++ functional
coprocessor behind the Coral scalar controller, L1 custom-instruction split and
NPU L2 decode.

## Generic operation mapping

| Generic model operation | XOpenNPUX lowering | Current test shape / semantics | Physical commands | Actual modeled cycles | Status |
| --- | --- | --- | ---: | ---: | --- |
| EMBED | `TGATHER` | 2 rows x 4 features | 1 | 8 | PASS |
| MATMUL, FP32 | `TMMA` | M=2, K=4, N=4 | 1 | 32 | PASS |
| ADD | `TADD` | 8 FP32 elements | 1 | 8 | PASS |
| MUL | `TMUL` | 8 FP32 elements | 1 | 8 | PASS |
| NORMALIZE | `TRMSNORM` | 8 FP32 elements | 1 | 32 | PASS |
| ROPE | `TROPE` | 8 FP32 elements | 1 | 24 | PASS |
| ACTIVATION, SiLU | `TSILU` | 8 FP32 elements | 1 | 24 | PASS |
| SOFTMAX | `TSOFTMAX` | 8 FP32 elements | 1 | 32 | PASS |
| TOPK | `TTOPK`, split values/indices outputs | 8 inputs, K=1 | 1 | 8 | PASS |
| MATMUL, GPTQ INT4 | `TDEQUANT + 2xTMMA` | M=2, K=8, N=2 | 3 | 48 | PASS |
| COMBINE | canonical `TADD` | 4 FP32 elements | 1 | 4 | PASS |
| DMA / KV update | `TDMA(K) + TDMA(V)` | 4-byte Key and Value planes | 2 | 8 | PASS |
| ROUTER | `TMMA + TTOPK + TSOFTMAX + 2xTDMA` | rows=1, features=2, experts=4, K=2 | 5 | 28 | PASS |
| ATTENTION | `TATTENTION` | rows=2, heads=2, KV heads=1, head dim=2, KV length=3; causal GQA | 1 | 80 | PASS |
| CAUSAL_CONVOLUTION | `TCAUSALCONV` | rows=2, features=2, kernel=3, stateful | 1 | 24 | PASS |
| RECURRENT_UPDATE | `TDMA(full output) + TDMA(final row state)` | rows=2, features=2, basic persistent state | 2 | 6 | PASS |
| EXPERT | `gate(TDEQUANT+TMMA) + up(TDEQUANT+TMMA) + TSILU + TMUL + down(TDEQUANT+TMMA)` | rows=1, input=2, intermediate=2, output=2, GPTQ INT4 | 8 | 32 | PASS |
| EXPERT, dynamic routed | `TROUTED_EXPERT` device-control command | Runtime expert IDs/weights, weight-plan indexed GPTQ gate/up/down | 1 | kernel-derived | C++ NPU functional engine PASS; GB10 audit pending |
| ATTENTION, sigmoid gated | `TATTENTION` + gate CSR | rows=1, heads=1, head dim=1, KV length=1 | 1 | 8 | PASS |
| RECURRENT_UPDATE, gated delta | `TRECURRENT` | rows=1, key/value heads=1, key/value dim=1 | 1 | 21 | PASS |
| CONVOLUTION | `TCONV` | FP32 NHWC input/output, OHWI weights, groups=1, input=1x3x3x1, kernel=2x2, output=1x2x2x1, bias | 1 | 16 | PASS (`0x40a9cead`) |

The per-operation values above sum to the current full-system baseline:

```text
xgraph_batches=5
xgraph_completed_requests=20
xgraph_completed_commands=35
xgraph_output=35,5,1053851104,0
xgraph_npu_cycles=451
xgraph_operation_count=451
xgraph_validated_operators=20
xgraph_correctness=PASS
```

Acceptance was run on the GB10 validation host from source commit `ea3dd84`
with `./tools/coralnpu/run_qwen_command_flow_test.sh` after rebuilding the
bridge and firmware. The cycle values are functional-model accounting, not RTL
throughput measurements.

The first batch contains the nine direct primitives. The second contains GPTQ
MatMul, COMBINE and KV DMA. The third contains the atomic Router sequence and
the stateful causal convolution and basic recurrent update requests. The fourth
contains the eight-command GPTQ Expert decomposition and one atomic causal GQA
`TATTENTION`. The fifth contains gated `TATTENTION`, `TRECURRENT` and `TCONV`.
GB10 full-system acceptance validated all 20 logical operators and the aggregate
correctness verdict. `RECURRENT_UPDATE` atomically emits two `TDMA` records: one
copies the complete input tensor to the visible output and one publishes the
final row to persistent state. Its checksum is `0x14a86f48`. Gated-delta now
lowers separately to `TRECURRENT`; it is not reduced to this copy sequence.

## Acceptance evidence rule

After every successful full-system operator or graph test, update this matrix
and the dated section in `docs/design/current_progress.md` in the same change.
Record all of the following fields:

- generic opcode and semantic variant;
- lowered XOpenNPUX instruction sequence;
- tested shape, dtype, layout, stride and quantization assumptions;
- logical request count and physical command count;
- batch count and atomic composite boundaries;
- actual `xgraph_npu_cycles` and `xgraph_operation_count`;
- per-operator checksum or maximum absolute error and PASS verdict;
- test command, execution host and source commit when the acceptance result is
  used as a release baseline.

Do not present these functional cycles as cycle-accurate RTL performance. When
an operator is replaced by an RTL execution unit, add a separate RTL row or
measurement column rather than overwriting the functional-model baseline.

## Host graph replacement status

The generic Host functional graph now has an opt-in execution policy,
`OPENNPUX_HOST_FUNCTIONAL_EXECUTION=xopennpux-primitives`. Eligible primitive
requests and supported GPTQ MatMul sequences are lowered and executed by
`Gem5XOpenNpuFunctionalCoprocessor`, which
uses the same XOpenNPUX L2 instruction decoder and CSR snapshot contract as the
Coral custom-instruction path. External read-only weight operands are mirrored
into one temporary EXTMEM image; output operands must remain in the graph
Tensor Arena. Unsupported composites remain on the strict Host C++ reference
backend.

This is a functional NPU-coprocessor replacement milestone, not an RTL claim.
Acceptance must show both unchanged strict token output and the independent
`host_functional_xgraph_*` counters. A request counted in
`host_functional_xgraph_fallback_requests` has not been replaced, even if its
lowering audit previously passed.
Per-opcode fallback counters identify the next replacement target without
changing the logical request-completion totals.

The current GB10 replacement baseline is 352 logical XGraph requests, 1580
physical commands, 904494592 modeled operations/cycles and 132 fallback
requests with strict token equivalence. All 81 weighted NORMALIZE requests and
all 70 stateful CAUSAL_CONVOLUTION, RECURRENT_UPDATE and ATTENTION requests now
execute through XOpenNPUX. Opcodes 10, 11 and 15 are absent from the accepted
fallback histogram. The remaining fallback distribution is EMBED 1, gated QKV
MATMUL 40, TOPK 1, ROUTER 40, EXPERT 40 and DMA 10.

Host-executor integration for generic KV-cache DMA is locally validated. Each
logical request atomically emits `TDMA(K) + TDMA(V)` and writes the visible tail
of the two persistent state planes. External-weight EMBED now stages its
read-only vocabulary table and executes `TGATHER`. Indices-only TOPK uses an
internal values scratch plane while publishing only token indices through
`TTOPK`. Local tests independently verify selected embedding rows, per-row
Top-1 indices, command counts and modeled cycles. The next GB10 acceptance
target is 364 logical XGraph requests, 1602 physical commands and 120 fallback
requests with opcodes 1, 8 and 14 absent and strict token equivalence unchanged.
GB10 full-model acceptance matched these values exactly and reported 904558592
modeled operations/cycles. Strict token equivalence remained PASS. The accepted
fallback histogram now contains only QKV MATMUL opcode 2, ROUTER opcode 12 and
EXPERT opcode 13, with 40 requests each.

GPTQ ROUTER Host-executor integration is locally validated. Its atomic sequence
uses disjoint logits, packed TopK and dequant-tile scratch ranges and executes a
tiled GPTQ projection followed by `TTOPK + TSOFTMAX + 2xTDMA`. The fixture
checks normalized route weights, valid expert IDs and zero XGraph fallback. The
next GB10 target is 404 XGraph requests and 80 fallback requests with opcode 12
absent; command, operation and cycle totals must be recorded from the real tile
plan rather than extrapolated from the fixture.

The accepted fourth batch appends one generic `TATTENTION` command after the
eight-command GPTQ `EXPERT` decomposition. The fifth batch adds one gated
`TATTENTION`, one `TRECURRENT` and one `TCONV`. The measured totals are 5
batches / 20 requests / 35 commands / 451 modeled cycles. The fixture covers causal
visibility, GQA head mapping, Expert gate/up/down execution, sigmoid attention
gating, persistent gated-delta state update and generic grouped Conv2D. The
Conv2D fixture produced checksum `0x40a9cead` with zero maximum absolute error.
