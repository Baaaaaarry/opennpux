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
| TOPK | `TTOPK` | 8 inputs, K=1 | 1 | 8 | PASS |
| MATMUL, GPTQ INT4 | `TDEQUANT + 2xTMMA` | M=2, K=8, N=2 | 3 | 48 | PASS |
| COMBINE | canonical `TADD` | 4 FP32 elements | 1 | 4 | PASS |
| DMA / KV update | `TDMA(K) + TDMA(V)` | 4-byte Key and Value planes | 2 | 8 | PASS |
| ROUTER | `TMMA + TTOPK + TSOFTMAX + 2xTDMA` | rows=1, features=2, experts=4, K=2 | 5 | 28 | PASS |
| ATTENTION | decomposition pending | Generic attention geometry is preserved in `.npxtb` | - | - | TODO |
| CAUSAL_CONVOLUTION | `TCAUSALCONV` | rows=2, features=2, kernel=3, stateful | 1 | 24 | PASS |
| RECURRENT_UPDATE | decomposition pending | Persistent recurrent state | - | - | TODO |
| EXPERT | GPTQ projection and activation composition pending | Selected-expert execution only | - | - | TODO |

The per-operation values above sum to the current full-system baseline:

```text
xgraph_batches=3
xgraph_completed_requests=14
xgraph_completed_commands=21
xgraph_npu_cycles=288
xgraph_operation_count=288
xgraph_op_CAUSAL_CONVOLUTION=PASS checksum=0xaa4fb265 max_abs_error=0
xgraph_validated_operators=14
xgraph_correctness=PASS
```

The first batch contains the nine direct primitives. The second contains GPTQ
MatMul, COMBINE and KV DMA. The third contains the atomic Router sequence and
the stateful causal convolution request. GB10 full-system acceptance validated
all 14 logical operators with zero maximum absolute error. The causal
convolution checksum is `0xaa4fb265`; the complete per-operator checksums remain
part of the corresponding test log.

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
