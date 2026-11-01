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
- Contains substantial Prism/Hadamard infrastructure: metadata validation, folded tensor registration, graph helper calls, and Qwen35 latent token embedding restoration.
- The model/graph files are close in Hadamard symbol coverage, but not line-equivalent to Prism; fork graph/model files have diverged.
- Qwen35 fork implementation is substantially shorter than Prism's current version and does not show the full set of model-specific Hadamard paths (including Prism's GDN-related transforms and inverse/output handling). This is a concrete port gap and must be resolved before claiming Bonsai 2 Qwen35 correctness.
- Targeted comparison of Qwen35 MTP: fork has a single-step `graph_mtp` path calling `build_attn(...)`; Prism additionally implements `graph_mtp::build_chain()` for chained speculative drafts. Prism's chain explicitly applies K/V Hadamard rotations before KV-cache writes and the V inverse after attention. The fork has no matching `build_chain()` symbol in `src/models/qwen35.cpp` at this branch snapshot. Treat chained MTP as an unported feature; do not add isolated rotations to the single-step path without confirming `build_attn()` semantics.
- Exact metadata and loader validation audit is complete at source level; execution correctness remains unverified.
- No build or model inference has been performed by the GitHub connector.

## Implementation plan

1. [x] Confirm upstream format facts and identify the Prism runtime contract.
2. [x] Diff exact metadata structures and validation against the fork; map missing fields and architecture coverage.
3. [ ] Port safe metadata validation and folded tensor registration without enabling unsupported execution paths.
4. [ ] Port graph-level activation transforms and embedding/output inverse handling for Qwen35, including model-specific MTP chain support or an explicit guard.
5. [ ] Add model-load guards: reject unsupported metadata/architectures rather than silently generating corrupted output.
6. [ ] Build and test ordinary Qwen35 regression.
7. [ ] Test a small Bonsai 2 `Q2_0` GGUF with coherent generation/logit checks.
8. [ ] Port `PQ2_0` and `PTQ1_0` GGML formats and CPU/backend kernels.
9. [ ] Validate Vulkan/Intel Arc B580; investigate SYCL/XMX separately if supported by this environment.

## Validation checklist

- [x] Source audit complete (source-level only; chained MTP gap recorded)
- [ ] Patch implemented
- [ ] Build succeeds
- [ ] Qwen35 regression passes
- [ ] Bonsai 2 Q2_0 loads and generates coherently
- [ ] PQ2_0 loads and generates coherently
- [ ] PTQ1_0 loads and generates coherently
- [ ] Intel Arc B580 backend validated

Status labels are intentionally conservative: source-audited, implemented, built, and runtime-validated are distinct states.

## Detailed execution plan — planning phase

**Current phase: planning. No source-code changes authorized yet.** Work proceeds in small reviewed steps. Each implementation step must have a defined scope, evidence, and exit criterion before code is changed.

### Phase 0 — Freeze baseline and scope
- [ ] Record the exact target branch/base commit and reference Prism revision used for comparisons.
- [ ] Preserve the current source-audit findings as the baseline; do not treat the local `lel` branch as evidence.
- [ ] Define the initial acceptance target: ordinary Qwen35 remains unchanged, and a supported Bonsai 2 Qwen35 `Q2_0` model loads and generates coherently.
- [ ] Decide whether chained MTP is required for the first usable milestone or can be explicitly disabled/rejected with a clear diagnostic.
- **Exit criterion:** scope and exclusions are written down and agreed before implementation.

### Phase 1 — Close remaining source/API questions
- [ ] Inspect the fork's exact `build_attn()` implementation and establish whether its normal attention path already applies the relevant K/V rotations, including ordering relative to RoPE, KV-cache writes, and inverse transforms.
- [ ] Map the fork's Qwen35 GDN, attention, embedding, output, and MTP data structures against the Prism implementation.
- [ ] Identify Prism-only APIs/types and determine the smallest compatible adaptation; do not wholesale replace the fork's Qwen35 implementation.
- [ ] Trace GGUF metadata → model loader → tensor registration → graph execution for the target model.
- **Exit criterion:** every required transform has a known insertion point, and every unresolved compatibility issue is listed explicitly.

### Phase 2 — Metadata and tensor loading, safely gated
- [ ] Specify accepted `prism.hadamard.*` metadata, versions, architecture scope, dimensions, tensor names, and tied-output constraints.
- [ ] Specify validation errors for malformed, incomplete, or unsupported metadata.
- [ ] Plan folded-tensor and sign/rotation tensor registration, including buffer/device constraints.
- [ ] Ensure unsupported combinations fail during model load rather than silently proceeding.
- **Exit criterion:** loader contract and negative cases are reviewed before coding.

### Phase 3 — Runtime graph integration for Qwen35
- [ ] Map each folded projection to its required activation transform and inverse/restore path.
- [ ] Preserve latent token embedding restoration and verify output/head behavior.
- [ ] Handle GDN and attention ordering explicitly; avoid assumptions based only on matching symbol names.
- [ ] Decide and document MTP-chain behavior for the first milestone: implement it only if the required API/data flow is understood; otherwise reject/disable that path explicitly.
- **Exit criterion:** a graph-level data-flow map exists and can be checked against source changes.

### Phase 4 — GGML format support
- [ ] Treat official Bonsai 2 `Q2_0` separately from Prism-private `PQ2_0` and `PTQ1_0`.
- [ ] For each private format, audit type traits, quantization/dequantization, GGUF serialization, CPU kernels, backend dispatch, and Hadamard compatibility.
- [ ] Do not expose a format as supported until all required layers are implemented and tested.
- **Exit criterion:** per-format support matrix lists implementation and validation status independently.

### Phase 5 — Incremental implementation and review
- [ ] Implement only the smallest reviewed slice from Phases 2–4.
- [ ] Review the exact diff for scope, ordering, API compatibility, and unsupported-path guards.
- [ ] Update this document in the same checkpoint with what changed, what did not, and what remains unverified.
- [ ] Commit only after the diff is reviewed; no unrelated cleanup or broad file replacement.
- **Exit criterion:** each commit corresponds to one bounded technical change with a clear review note.

### Phase 6 — Build and validation gates
1. [ ] Build the target branch; record exact command, configuration, and output.
2. [ ] Run ordinary Qwen35 regression first.
3. [ ] Load a small, known Bonsai 2 Qwen35 `Q2_0` GGUF; capture loader logs and verify metadata/tensor acceptance.
4. [ ] Run deterministic or repeatable generation/logit checks; reject “loads successfully” as sufficient evidence.
5. [ ] Test unsupported/malformed metadata and confirm clean rejection.
6. [ ] Validate Intel Arc B580/Vulkan separately; do not infer backend correctness from CPU success.
7. [ ] Test `PQ2_0` and `PTQ1_0` only after their full format paths are implemented.
- **Exit criterion:** each result is recorded as pass/fail/not-run with evidence; no claim of runtime support without actual execution.

### Working rules
- No source-code edits during the planning phase.
- Do not mark a task complete based on inspection alone when it requires a build or runtime test.
- Keep “audited”, “implemented”, “built”, and “runtime-validated” as separate status labels.
- Stop at each checkpoint for review before expanding scope.

