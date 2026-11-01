#!/usr/bin/env bash
# Bonsai 2 Auto-Continue: creates a portable checkpoint for leaving and resuming work.
# Safe by design: never switches branches, resets files, commits, or edits project source.
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
NOTES_FILE="$STATE_DIR/NOTES.md"
RESUME_FILE="$STATE_DIR/RESUME.md"
STATUS_DOC="docs/BONSAI2_PORT_STATUS.md"

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
- Distinguish audited, implemented, built, and runtime-validated.

## Current checkpoint
- [x] Initial scope and source audit documented.
- [ ] Complete concrete GGUF tensor-to-loader-to-graph mapping.
- [ ] Implement minimal correctness path for official Bonsai 2 Q2_0.
- [ ] Build and run regression/smoke tests.
- [ ] Validate coherent generation with a Bonsai 2 GGUF.
- [ ] Investigate Intel Arc B580 backend optimization after correctness.
STATE
fi

if [[ ! -f "$TASKS_FILE" ]]; then
  cat > "$TASKS_FILE" <<'TASKS'
# Task queue

1. Compare complete Prism and fork tensor-registration/loading paths.
2. Map concrete target GGUF folded tensor names to fork tensors and graph callsites.
3. Complete embedding/output/tied-output mapping and projection coverage matrix.
4. Implement the minimum official Bonsai 2 Q2_0 correctness delta.
5. Build, run ordinary Qwen35 regression, then validate Bonsai 2 generation.
6. Only after correctness, assess PQ2_0/PTQ1_0 and SYCL/XMX porting.
TASKS
fi

if [[ ! -f "$NOTES_FILE" ]]; then
  cat > "$NOTES_FILE" <<'NOTES'
# Session notes

Append short decisions, findings, and the exact next action with:
  ./continue.sh checkpoint "note"
NOTES
fi

write_resume() {
  local now head status
  now="$(date --iso-8601=seconds)"
  head="$(git log -1 --format='%h %s')"
  status="$(git status --short)"

  {
    echo "# Bonsai 2 — Auto-Continue checkpoint"
    echo
    echo "- Generated: $now"
    echo "- Repository: $ROOT"
    echo "- Branch: $branch"
    echo "- HEAD: $head"
    echo
    echo "## Working tree"
    if [[ -n "$status" ]]; then
      printf '%s\n' "$status"
    else
      echo "Clean"
    fi
    echo
    echo "## Current state"
    cat "$STATE_FILE"
    echo
    echo "## Task queue"
    cat "$TASKS_FILE"
    echo
    echo "## Session notes"
    tail -n 40 "$NOTES_FILE"
    echo
    echo "## Source-of-truth status document"
    echo "$STATUS_DOC"
    echo
    echo "## Resume instruction"
    echo "Continue the Bonsai 2 port from this checkpoint. First inspect the current GitHub branch and this status document. Do not use the local 'lel' branch. Do not claim build/runtime validation unless actually performed. Continue with the first unchecked task, make bounded changes, and update the checkpoint."
  } > "$RESUME_FILE"
}

case "${1:-resume}" in
  resume|status)
    write_resume
    echo "=== Auto-Continue: resume packet ==="
    cat "$RESUME_FILE"
    echo
    echo "Saved to: $RESUME_FILE"
    echo "Copy this file's contents into the next conversation to resume."
    ;;
  checkpoint)
    shift || true
    note="${*:-Checkpoint saved without an additional note.}"
    printf '\n- [%s] %s\n' "$(date --iso-8601=seconds)" "$note" >> "$NOTES_FILE"
    write_resume
    echo "Checkpoint saved: $RESUME_FILE"
    echo "Note: $note"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage:
  ./continue.sh                 Generate and display the resume packet (default)
  ./continue.sh resume          Same as default
  ./continue.sh status          Show current checkpoint
  ./continue.sh checkpoint NOTE Append a timestamped note and refresh checkpoint
  ./continue.sh help            Show this help

This script only writes .continue/*.md files. It never edits source, changes branches,
resets, stages, or commits automatically.
HELP
    ;;
  *)
    echo "ERROR: unknown command '$1'. Run ./continue.sh help." >&2
    exit 2
    ;;
esac
