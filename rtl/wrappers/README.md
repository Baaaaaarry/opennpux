# RTL Wrappers

`coralnpu_gem5_abi.h` is the canonical versioned ABI shared by the gem5
backend and the Coral Bazel adapter. Run
`tools/coralnpu/check_rtl_bridge_abi.sh` after changing it.

Generated Verilator sources and shared objects are build outputs and must not
be committed here. The staged runtime library belongs under
`build/coralnpu/`.
