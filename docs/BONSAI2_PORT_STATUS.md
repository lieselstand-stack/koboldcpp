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

**Current phase: planning. No source-code changes authorized yet.** Priority is correctness first, robustness second, backend validation third, and optimization only after functional correctness is demonstrated. Work proceeds in small reviewed steps. Each implementation step must have a defined scope, evidence, and exit criterion before code is changed.

### Phase 0 — Freeze baseline and scope
- [ ] Record the exact target branch/base commit and reference Prism revision used for comparisons.
- [ ] Preserve the current source-audit findings as the baseline; do not treat the local `lel` branch as evidence.
- [ ] Define the initial acceptance target: ordinary Qwen35 remains unchanged, and a supported Bonsai 2 Qwen35 `Q2_0` model loads and generates coherently.
- [x] Scope approved: first milestone targets ordinary Qwen35 regression plus one Bonsai 2 Qwen35 `Q2_0` model; MTP is excluded initially.
- [x] MTP-chain support is deferred; do not implement or imply it is supported in this milestone. If the selected model/runtime path requires MTP, stop and report the dependency rather than silently expanding scope.
- [x] `PQ2_0`, `PTQ1_0`, other architectures, SYCL/XMX, performance optimization, and broad refactoring are out of scope for this milestone.
- **Exit criterion:** scope and exclusions recorded and approved before implementation.

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
- [x] MTP-chain behavior for the first milestone: excluded; no MTP-chain implementation. Required-model dependencies must be identified before coding.
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

### Optimization policy
- [x] User priority confirmed: **make it run correctly first; optimize afterward**.
- [ ] Establish a correct baseline before performance work: successful build, ordinary Qwen35 regression, and coherent Bonsai 2 Qwen35 `Q2_0` generation.
- [ ] Record baseline performance only after correctness checks pass.
- [ ] Optimize one measured bottleneck at a time; rerun correctness tests after every optimization.
- [ ] Do not accept speedups that alter correctness or bypass unsupported-path validation.
- **Exit criterion:** optimization begins only after functional acceptance criteria are met and a baseline is recorded.

### Working rules
- No source-code edits during the planning phase.
- Do not mark a task complete based on inspection alone when it requires a build or runtime test.
- Keep “audited”, “implemented”, “built”, and “runtime-validated” as separate status labels.
- Stop at each checkpoint for review before expanding scope.



## Milestone 1 — approved scope

**Status: scope approved; implementation not started.**

### Included
- Bonsai 2 Qwen35 using official `Q2_0` format.
- Required Hadamard metadata validation and folded tensor registration.
- Qwen35 graph transforms, including embedding/output and GDN/attention ordering as required by the selected model.
- Clear load-time rejection for unsupported or malformed metadata/configurations.
- Regression for ordinary Qwen35.
- Build and coherent generation validation for one known Bonsai 2 Qwen35 `Q2_0` GGUF.

### Explicitly excluded
- MTP, including chained speculative MTP. If the chosen model requires MTP for its intended execution, pause and reassess rather than expanding scope silently.
- Prism-private `PQ2_0` and `PTQ1_0` formats.
- Architectures other than Qwen35.
- SYCL/XMX work, performance optimization, and unrelated refactoring.

### Acceptance criteria
- [ ] Target branch builds; exact command/configuration and output recorded.
- [ ] Ordinary Qwen35 regression passes.
- [ ] Known Bonsai 2 Qwen35 `Q2_0` model loads with expected metadata/tensors.
- [ ] Repeatable generation/logit checks show coherent behavior; load success alone is insufficient.
- [ ] Unsupported/malformed Bonsai metadata is rejected clearly.
- [ ] Intel Arc B580/Vulkan result recorded separately; not inferred from CPU validation.

### Next action
Complete Phase 1 source/API mapping for the non-MTP Qwen35 path. No source-code changes until that mapping and its insertion points are reviewed.


## Priority clarification

The user confirmed: **“Quero rodar corretamente. Depois otimiza.”** Therefore, this port is correctness-first. Performance tuning is explicitly deferred until the supported Bonsai 2 Qwen35 `Q2_0` path builds, passes ordinary Qwen35 regression, and produces coherent, repeatable output. No speed optimization is part of the initial correctness milestone.


## Phase 1 source/API mapping — non-MTP path (2026-11-01)

**Status: partial source mapping completed; no source-code changes.**

### Confirmed in the fork
- `src/llama-graph.cpp`: the KV-cache `build_attn(...)` overload applies `self_k_rot[il]` to **Q and K**, and `self_v_rot[il]` to **V**, before the graph expands these nodes and before K/V are copied into the KV cache.
- The same path applies `self_v_rot[il]` to the attention result after MHA. Therefore the earlier uncertainty is resolved: the generic KV attention helper already contains these rotation hooks; Qwen35 callers do not need isolated duplicate rotations in that helper.
- `src/models/qwen35.cpp`: the regular Qwen35 attention path applies MRoPE to Q/K, then calls the generic `build_attn(inp, ...)`. The Hadamard hooks therefore execute inside that helper after the caller's RoPE operations and before KV-cache writes.
- `src/llama-graph.cpp` also contains Hadamard-aware paths for other attention variants and graph input setup; these must not be assumed equivalent to the Qwen35 path without tracing the specific caller.
- `src/models/qwen35.cpp` has a single-block MTP graph, but MTP is excluded from Milestone 1. No MTP changes are planned.

### Still to map before implementation
- [ ] Trace how the selected Qwen35 Bonsai2 GGUF folded-weight names map to the fork's actual Qwen35 projection tensors.
- [ ] Trace Qwen35 linear-attention/GDN path end-to-end and identify each required rotation/inverse relative to convolution, recurrent state update, and output projection.
- [ ] Verify embedding inverse-after-lookup and output/head handling for the selected metadata/version, including tied-output behavior if applicable.
- [ ] Confirm all intended folded projections reach a Hadamard-aware matmul path; loader validation must reject any that do not.
- [ ] Record the exact target model's metadata and tensor inventory as the implementation/test fixture, without assuming every Prism metadata option is required.

### Interim conclusion
The generic KV attention helper already has K/Q/V rotation hooks, so the likely work is **Qwen35-specific data-flow coverage and loader-to-graph binding**, not blindly adding duplicate rotations to `build_attn()`. This is a source-level conclusion only; numerical correctness remains untested.


## Phase 1 source/API mapping — Qwen35 GDN path (2026-11-01)

**Status: source-level trace advanced; no source-code changes.**

### Fork path traced
In `src/models/qwen35.cpp`, `build_layer_attn_linear()` follows this order:

1. `build_qkvz()` computes the combined Q/K/V projection (`wqkv`) and gate projection (`wqkv_gate`).
2. Q/K/V pass through the recurrent convolution input/state path: `build_conv_state()`, `ggml_ssm_conv()`, SiLU, then Q/K/V views.
3. Q and K receive `build_gdn_l2_norm()`; optional repeat aligns K/Q heads with V heads.
4. `build_recurrent_attn()` performs the recurrent GDN update using the resulting Q/K/V, gate, beta, and state.
5. Gated normalization combines recurrent output and `z`; `ssm_out` projects the result back to model width.

The fork's shown GDN function contains no explicit Bonsai/Hadamard transform calls around these stages. This does **not** establish the exact missing transform placement by itself; it establishes that the existing dense-attention hooks do not automatically cover this separate recurrent path.

### Tensor-name mapping observed in the fork
Qwen35 loader names relevant to this path are represented through architecture tensor identifiers:
- `LLM_TENSOR_ATTN_QKV` → `layer.wqkv`
- `LLM_TENSOR_ATTN_GATE` → `layer.wqkv_gate`
- `LLM_TENSOR_SSM_CONV1D` → `layer.ssm_conv1d`
- `LLM_TENSOR_SSM_BETA` / `LLM_TENSOR_SSM_ALPHA` → `layer.ssm_beta` / `layer.ssm_alpha`
- `LLM_TENSOR_SSM_OUT` → `layer.ssm_out`
- Dense attention uses `create_tensor_qkv()` plus `LLM_TENSOR_ATTN_OUT` (`layer.wo`).
- FFN uses `LLM_TENSOR_FFN_GATE`, `LLM_TENSOR_FFN_UP`, and `LLM_TENSOR_FFN_DOWN`.

These are fork-side architecture mappings, **not yet a verified mapping from a concrete Bonsai 2 GGUF's folded tensor names**. The target fixture's metadata and tensor inventory are still required to close that gap.

### Phase 1 status update
- [x] Trace fork Qwen35 GDN execution order at source level.
- [x] Identify that the GDN path is separate from generic KV attention and currently has no explicit Hadamard operations in the inspected function.
- [ ] Compare Prism's exact Qwen35 GDN transform placement and identify the minimum compatible fork insertion points.
- [ ] Resolve folded GGUF tensor names to the fork architecture tensor identifiers using a concrete target fixture.
- [ ] Finish embedding/output/tied-head mapping and the complete folded-projection coverage matrix.

### Interim conclusion
The port cannot be treated as “just wire the existing KV rotations.” Dense attention and GDN are separate execution paths. The GDN transform contract must be copied/adapted from Prism with ordering preserved around convolution, normalization, recurrent state update, gated normalization, and `ssm_out`. No implementation or numerical correctness is claimed yet.


## Phase 1 source/API mapping — Prism GDN comparison and loader contract (2026-11-01)

**Status: comparison advanced; no source-code changes.**

### Prism Qwen35 GDN execution order
The inspected Prism `src/models/qwen35.cpp` linear-attention path has this order:
1. `wqkv` and `wqkv_gate` projections; beta/alpha projections and gate preparation.
2. Recurrent convolution input/state preparation, then `ggml_ssm_conv` and SiLU.
3. Q/K/V views from the convolved output; joint QK L2 normalization; optional K/Q head repeat.
4. Recurrent GDN update.
5. Gated normalization with `z`, reshape, then `ssm_out` projection.

The fork follows the same broad stages, but its Q/K normalization uses `build_gdn_l2_norm()` separately, while the inspected Prism code uses a joint `ggml_l2_norm()` view. Treat this as a numerical-compatibility question, not an automatic code-copy instruction: Prism itself comments that a normalization migration changes packed-model logits.

### Prism Hadamard call sites found in Qwen35
The inspected Prism file contains explicit Hadamard calls for:
- latent token embedding restoration (`tok_embd` via a rotation table);
- dense attention Q/K and V transforms, including post-attention V-basis restoration;
- an inverse-after-lookup embedding path;
- additional attention Q/K/V and attention-output transforms.

No explicit Hadamard call was found in the inspected Prism `build_layer_attn_linear()` GDN body. This is an important unresolved point: do not infer that GDN projections are exempt, nor invent transforms there. The concrete target GGUF's folded-weight inventory and the full Prism loader/graph contract must determine whether GDN uses pre-folded projections, another shared helper, or additional architecture handling.

### Loader contract observed in Prism
`load_tensors_hadamard()` iterates metadata-listed folded and inverse weight names, resolves each through `get_tensor(weight_name)`, and validates presence, block-size divisibility, and allocated buffer. It creates/caches block-size-specific normalized Walsh-Hadamard rotation tensors in the target weight buffer type, and creates sign tensors keyed by width and buffer type when sign metadata is present. Tied-output handling can bind the token embedding name to output or embedding tensors, with an explicit consistency check.

This is the source-level contract to compare against the fork implementation. It does **not** yet prove that the fork's metadata parser, tensor registration, or Qwen35 architecture mapping matches the target model's actual tensor names.

### Updated unresolved items
- [ ] Inspect Prism's complete Qwen35 tensor-registration/loading path and compare it to the fork's `load_tensors_hadamard()` implementation, not just helper names.
- [ ] Obtain or inspect a concrete Bonsai 2 Qwen35 `Q2_0` GGUF metadata/tensor inventory; map every folded/inverse name to the fork's actual tensor object.
- [ ] Resolve whether GDN projections are folded and how the model format expects runtime activation transforms to be applied; do not extrapolate from dense-attention hooks.
- [ ] Finish token embedding, inverse-after-lookup, output head, and tied-output mapping for the selected fixture.
- [ ] Compare exact metadata validation and unsupported-path errors between Prism and fork.

### Interim conclusion
The Prism loader is metadata-driven and validates actual named tensors before registering rotation/sign buffers. Qwen35 dense attention has visible graph transform call sites; GDN remains unresolved at the concrete folded-weight contract level. Implementation should wait until the target GGUF inventory closes this gap.


## Phase 1 source/API mapping — fork Hadamard-aware matmul and loader comparison (2026-11-01)

**Status: core generic mechanism confirmed; target model mapping still open. No source-code changes.**

### Fork graph mechanism confirmed
In `src/llama-graph.cpp`, `build_lora_mm(w, cur, ...)` calls `build_hadamard_input(w, cur)` before `ggml_mul_mat(w, transformed_input)`. The helper:
- looks up the actual weight tensor pointer in the graph's registered Hadamard rotation map;
- optionally performs the grouped-head feature permutation;
- applies sign multiplication when sign metadata exists;
- applies the normalized Hadamard rotation;
- caches/reuses the transformed activation;
- leaves LoRA's delta path operating on the original, untransformed activation.

This means the generic folded-weight execution path already exists. The remaining issue is proving that each target GGUF weight is registered to the correct fork tensor and that its graph projection uses this helper.

### Fork loader validation observed
The fork's `load_hparams_hadamard()` already validates metadata version 1/2, tied-output/version consistency, block size, transform, axis, sign mode, sign tables, nonempty weight list, supported architectures, duplicate names, and an allowlist of weight-name kinds. The allowlist includes `attn_qkv` and `ssm_out`, but does not include every Qwen35 projection kind (for example, the gate projection is not listed as a generic folded kind).

The loader's allowlist is an explicit compatibility boundary: do not widen it based only on architecture naming. First confirm the concrete model's folded tensor inventory and whether each projection is actually routed through a verified Hadamard-aware matmul.

### Updated interpretation
- Generic activation transformation and folded matmul support are present in the fork.
- Qwen35 GDN's combined `wqkv` projection appears structurally eligible through the `attn_qkv` kind, and `ssm_out` is explicitly allowlisted. This is a name-kind observation, not confirmation that a real target GGUF contains those folded tensors.
- GDN `alpha`, `beta`, `wqkv_gate`, convolution kernel, and normalization parameters are not automatically covered by the current allowlist. Their required treatment must come from target metadata and Prism's format contract; do not assume they should be folded.
- The largest remaining unknown is now concrete: obtain the target GGUF metadata/tensor inventory and construct the mapping matrix, then trace each mapped tensor through loader registration and graph call sites.

### Next source audit step
Inspect Prism's full tensor-name/architecture registration path and locate a trustworthy concrete Bonsai 2 Qwen35 `Q2_0` GGUF fixture or metadata listing. If no fixture is available from the repository or user, report that limitation explicitly rather than fabricating tensor names.


## Phase 1 source/API mapping — public Bonsai 2 fixture evidence (2026-11-01)

**Status: public model facts found; exact target GGUF tensor-name mapping remains incomplete. No source-code changes.**

### Public Bonsai 2 Qwen35 evidence

The public PrismML model card for Bonsai 2 9B identifies:
- Base architecture: Qwen3.5-9B; 32 decoder blocks, hidden width 4096, separate embedding and LM head.
- Hybrid layout: 24 linear-attention/GDN layers and 8 full-attention layers.
- Available language GGUFs: official `Q2_0`, Prism `PQ2_0`, and `PTQ1_0`; a separate MTP-bearing artifact is also listed.
- Weight basis: ternary weights with blockwise Hadamard folded into weights; runtime activation transforms are required.
- Higher-precision GDN state-path tensors include `ssm_alpha` and `ssm_beta`; the card describes normalization and state-path tensors as higher precision.

Sources: [PrismML Bonsai 2 9B model card](https://huggingface.co/prism-ml/Ternary-Bonsai-2-9B-gguf), [PrismML Bonsai demo format guide](https://github.com/PrismML-Eng/Bonsai-demo/blob/main/MODEL-FORMATS.md).

### Concrete inventory limitation

A public detailed tensor inventory was found for Bonsai 2 27B, including folded GDN projections (`attn_qkv.weight`, `attn_gate.weight`, `ssm_out.weight`), full-attention projections, FFN projections, embedding and output tensors. This is useful as architecture-family evidence, but **it is not the 9B fixture** and must not be copied as the 9B tensor manifest.

No authoritative complete tensor-name inventory for the selected 9B official `Q2_0` file has yet been captured in this audit. The model card confirms file availability and architecture, but does not enumerate all `prism.hadamard.weight_names` and inverse names. Exact mapping therefore remains blocked on one of:
1. inspecting the actual 9B GGUF metadata/tensor directory, or
2. obtaining an authoritative complete manifest for that exact artifact.

Do not widen the loader allowlist or assert GDN transform coverage based only on the 27B inventory.

### Mapping matrix — current evidence

| GGUF tensor family | Fork Qwen35 object/path | Hadamard-aware execution | Status |
|---|---|---|---|
| `token_embd.weight` | `model.tok_embd` → `build_inp_embd()` | Generic inverse-after-lookup infrastructure exists | Exact target metadata/version and tied-output behavior pending |
| `output.weight` | `model.output` → final `build_lora_mm()` | Generic Hadamard-aware matmul exists | Exact target folded-name registration pending |
| Dense attention Q/K/V | `create_tensor_qkv()` and `layer.wo` | Generic KV attention helper has Q/K/V rotations; projection matmul helper exists | Exact target folded names and complete coverage pending |
| GDN combined QKV | `layer.wqkv` / `LLM_TENSOR_ATTN_QKV` | `build_lora_mm()` uses Hadamard-aware matmul | Target manifest and transform contract pending |
| GDN gate | `layer.wqkv_gate` / `LLM_TENSOR_ATTN_GATE` | `build_lora_mm()` call exists, but loader allowlist currently does not establish support | Must resolve from exact metadata; reject if unsupported |
| GDN output | `layer.ssm_out` / `LLM_TENSOR_SSM_OUT` | `build_lora_mm()`; loader kind is allowlisted | Target manifest and transform placement pending |
| GDN alpha/beta/conv/norm | `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_norm` | Not assumed folded | Verify against exact metadata; no unsupported transform inferred |
| FFN gate/up/down | `layer.ffn_gate/up/down` | Generic Hadamard-aware matmul path | Exact target names/registration pending |

This table is a **source-path map**, not a claim that each listed tensor is folded in the 9B GGUF.

### Phase 1 decision

The public evidence supports selecting Bonsai 2 9B as a plausible small Qwen35 fixture, but the exact artifact must be fixed before implementation. The next audit checkpoint is to capture its GGUF metadata and tensor inventory, then verify each listed folded tensor against loader registration and graph callsites. If the user has already downloaded the exact file, a metadata dump from that file is preferable to relying on secondary manifests.
