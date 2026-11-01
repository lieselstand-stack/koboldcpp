# Bonsai 2 support port — status

Branch: `koboldcpp-bonsai2`  
Target: KoboldCpp fork, with Intel Arc B580 (12 GB) as the primary GPU.

## Important correctness requirement

Bonsai 2 is not supported by merely recognizing its GGUF tensor types. The checkpoint uses ternary weights with folded signed blockwise Hadamard rotations. The runtime must apply the corresponding activation transforms and embedding/output handling, or inference can load while producing incorrect results.

## Formats

| Format | GGML type | Status in this branch |
|---|---:|---|
| Official `Q2_0` | 42 | Baseline target; must verify the exact Qwen35 Hadamard path and model metadata |
| Prism `PQ2_0` | 142 | Not yet ported |
| Prism `PTQ1_0` | 143 | Not yet ported |

The Prism-specific formats require type traits, GGUF parsing/serialization, quantization/dequantization, CPU/backend kernels, and compatible transform handling. They are intentionally not being enabled by a type-ID-only patch.

## Existing code observed

- This fork contains a Qwen35 model implementation in `src/models/qwen35.cpp`.
- The fork contains partial Prism/Hadamard infrastructure in the model and graph code.
- A source-level presence check is not proof of end-to-end Bonsai 2 correctness.

## Port sequence

1. Compare the fork's Hadamard metadata validation and graph transform application against Prism's `prism` branch.
2. Confirm every folded Qwen35 tensor listed by Bonsai 2 metadata reaches a transform-aware matmul path, including embedding/output handling and GatedDeltaNet output ordering.
3. Add only missing correctness behavior for official `Q2_0`.
4. Build and test: ordinary Qwen35 regression, then a small Bonsai 2 GGUF with coherent generation/logit checks.
5. Port `PQ2_0` and `PTQ1_0` only after baseline correctness; then investigate Intel SYCL/XMX acceleration separately.

## Validation status

- [ ] Source audit complete
- [ ] Patch implemented
- [ ] Build succeeds
- [ ] Qwen35 regression test passes
- [ ] Bonsai 2 Q2_0 loads and generates coherently
- [ ] Intel Arc B580 backend tested

No runtime build or model inference has been performed by the GitHub connector. Do not describe this branch as Bonsai 2-compatible until the relevant checks pass.
