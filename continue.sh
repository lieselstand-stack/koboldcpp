#!/usr/bin/env bash
# Safe progress helper for the Bonsai 2 port.
# It never modifies source files, switches branches, resets, or commits automatically.
set -Eeuo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
  echo "ERROR: run this script from inside a KoboldCpp git checkout." >&2
  exit 2
}
cd "$ROOT"

EXPECTED_BRANCH="koboldcpp-bonsai2"
STATE_DIR=".continue"
STATE_FILE="$STATE_DIR/STATE.md"
TASKS_FILE="$STATE_DIR/TASKS.md"

branch="$(git branch --show-current)"
if [[ "$branch" != "$EXPECTED_BRANCH" ]]; then
  echo "ERROR: expected branch '$EXPECTED_BRANCH'; current branch is '$branch'." >&2
  echo "No changes were made." >&2
  exit 2
fi

mkdir -p "$STATE_DIR"
if [[ ! -f "$STATE_FILE" ]]; then
  cat > "$STATE_FILE" <<'STATE'
# Bonsai 2 port — current state

## Goal
Bring Bonsai 2 model support to this KoboldCpp fork, then validate correctness and investigate Intel Arc B580 acceleration.

## Safety
- Keep changes on `koboldcpp-bonsai2`; do not rewrite `concedo`.
- Prefer the smallest compatible change.
- Do not claim runtime support until a real model load and generation test pass.
- Keep PQ2/PTQ and specialized SYCL/XMX work separate from baseline correctness.

## Current checkpoint
- [ ] Audit fork vs Prism implementation and record exact missing pieces.
- [ ] Verify Bonsai 2 Q2_0 metadata/transform path.
- [ ] Implement minimal correctness path.
- [ ] Build and run a small-model smoke test.
- [ ] Validate with a Bonsai 2 GGUF.
- [ ] Investigate Intel Arc B580 backend optimization.
STATE
fi

if [[ ! -f "$TASKS_FILE" ]]; then
  cat > "$TASKS_FILE" <<'TASKS'
# Task queue

1. Compare the fork's Qwen35 and Hadamard implementation against Prism's `prism` branch.
2. Identify the minimum Bonsai 2 Q2_0 compatibility delta.
3. Implement the minimum delta without importing Prism-only quantization formats.
4. Build and test on the target Intel Arc B580 environment.
5. Only after correctness, assess PQ2_0/PTQ1_0 and SYCL/XMX porting.
TASKS
fi

echo "=== KoboldCpp Bonsai 2 continuation checkpoint ==="
echo "Repository: $ROOT"
echo "Branch:    $branch"
echo
echo "--- Working tree ---"
git status --short
echo
echo "--- Latest commit ---"
git log -1 --oneline
echo
echo "--- State ---"
cat "$STATE_FILE"
echo
echo "--- Tasks ---"
cat "$TASKS_FILE"
echo
echo "Checkpoint ready. This helper is read-only except for creating missing .continue/*.md files."
