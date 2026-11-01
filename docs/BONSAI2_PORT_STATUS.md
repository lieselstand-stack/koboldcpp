# Bonsai 2 support port — status

Branch: `koboldcpp-bonsai2`  
Target: KoboldCpp fork, primary target Intel Arc B580 (12 GB).

## Correctness contract

Bonsai 2 is not supported by recognizing GGUF tensor types alone. The model uses signed blockwise Hadamard transforms folded into weights. Runtime must apply matching activation transforms and preserve embedding/output and Qwen35 GatedDeltaNet ordering. A model that loads but emits incoherent text is a failed implementation.

## Formats and current facts

| Format | GGML type | Current status |
|---|---:|---|
| Official `Q2_0` | 42 | Supported by recent upstream llama.cpp only when the Bonsai 2 Hadamard/model integration is present. This fork still requires audit and port. |
| Prism `PQ2_0` | 142 | Prism-private format; not ported in this branch. |
| Prism `PTQ1_0` | 143 | Prism-private format; not ported in this branch. |

The Prism formats need GGML type traits, GGUF metadata/serialization, quant/dequant, CPU/backend kernels, and compatible transform handling. Do not add only type IDs.

## Source audit — 2026-11-01

### Target Prism implementation
Reference: `PrismML-Eng/llama.cpp`, branch `prism`.

Verified Prism components:
- `llama_model_base::load_hparams_hadamard`: validates `prism.hadamard.*` metadata, version, transform, axis, sign mode, folded weight names, inverse-after-lookup names, and tied-output constraints.
- `llama_model_base::load_tensors_hadamard`: registers folded tensors and rotation/sign data; validates tensor dimensions, buffers, and Qwen35 GDN geometry.
- Graph helpers apply activation rotations for folded matmul, restore latent embeddings, and handle inverse-after-lookup paths.
- Qwen35 contains model-specific handling for latent token embeddings, attention projections, and output transforms.
- GGML defines Prism-private types `PQ2_0=142`, `PTQ1_0=143`, and a Hadamard source hint.

### This KoboldCpp fork
- Contains Qwen35 and Qwen35-MoE implementations.
- Contains partial Prism/Hadamard infrastructure in model and graph files.
- Presence of helper names is not proof of parity with Prism's current metadata validation or end-to-end graph behavior.
- No build or model inference has been performed by the GitHub connector.

## Implementation plan

1. [x] Confirm upstream format facts and identify the Prism runtime contract.
2. [ ] Diff exact metadata structures and validation against the fork; map missing fields and architecture coverage.
3. [ ] Port safe metadata validation and folded tensor registration without enabling unsupported execution paths.
4. [ ] Port graph-level activation transforms and embedding/output inverse handling for Qwen35.
5. [ ] Add model-load guards: reject unsupported metadata/architectures rather than silently generating corrupted output.
6. [ ] Build and test ordinary Qwen35 regression.
7. [ ] Test a small Bonsai 2 `Q2_0` GGUF with coherent generation/logit checks.
8. [ ] Port `PQ2_0` and `PTQ1_0` GGML formats and CPU/backend kernels.
9. [ ] Validate Vulkan/Intel Arc B580; investigate SYCL/XMX separately if supported by this environment.

## Validation checklist

- [ ] Source audit complete
- [ ] Patch implemented
- [ ] Build succeeds
- [ ] Qwen35 regression passes
- [ ] Bonsai 2 Q2_0 loads and generates coherently
- [ ] PQ2_0 loads and generates coherently
- [ ] PTQ1_0 loads and generates coherently
- [ ] Intel Arc B580 backend validated

Status labels are intentionally conservative: source-audited, implemented, built, and runtime-validated are distinct states.
