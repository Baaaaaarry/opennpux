# Pinned Submodule Commits

- `thirdparty/gem5`
  - remote: `https://github.com/gem5/gem5.git`
  - target upstream branch: `stable`
  - intended pinned commit: `c8222cc67a399bfc01e8658dd14b30d5bfd634f9`
- `thirdparty/coralnpu`
  - remote: `https://github.com/google-coral/coralnpu.git`
  - pinned commit: `8baac41897c40c6eafdbaabe491117dc60c6175e`

## Update policy

- Keep both submodules close to their official upstream histories.
- Land local integration changes in this superproject as mirrored source trees
  under `sim/`, plus wrappers, scripts, and runbooks.
- Avoid carrying long-lived product code directly inside either submodule.
