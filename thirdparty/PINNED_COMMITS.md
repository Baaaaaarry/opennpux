# Pinned Submodule Commits

- `thirdparty/gem5`
  - remote: `git@github.com:Baaaaaarry/gem5-npu.git`
  - pinned commit: `245dfe23e0387ed852e5a553859ea80a1cc6107b`
- `thirdparty/coralnpu`
  - remote: `https://github.com/google-coral/coralnpu.git`
  - pinned commit: `8baac41897c40c6eafdbaabe491117dc60c6175e`

## Update policy

- Keep both submodules close to their upstream histories.
- Land local integration changes in this superproject as patches, overlays,
  wrappers, scripts, and runbooks.
- Avoid carrying long-lived product code directly inside either submodule.
