# Simulation Tests

Add smoke and end-to-end simulation tests here.

Phase-2 smoke sequence:

```bash
./tools/coralnpu/phase2_check_abi.sh
./tools/coralnpu/phase2_build_bridge.sh
./sim/gem5/apply_patchset.sh
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```
