#!/bin/bash
# Best-response child run (2026-08-27): delegates to start.sh, whose
# preamble already stops any live learner gracefully (Ctrl-C -> the
# KeyboardInterrupt handler's synchronous checkpoint) before launching.
# The BR trains in its own ckpts/gen{N}/br/<tag>/ subtree against the
# frozen target, and publishes its latest params into the parent's
# players/ dir on every stop. Resuming an interrupted BR is re-running
# the SAME command; resuming main afterwards is plain `bash start.sh`.
#
# Usage: bash start_br.sh <target_ckpt_dir> <num_steps> [run_tag]
set -euo pipefail

TARGET=${1:?usage: start_br.sh <target_ckpt_dir> <num_steps> [run_tag]}
STEPS=${2:?usage: start_br.sh <target_ckpt_dir> <num_steps> [run_tag]}
TAG=${3:-}

if [ ! -d "$TARGET" ]; then
  echo "target checkpoint dir '$TARGET' does not exist" >&2
  exit 1
fi

ARGS=(--br-target "$TARGET" --num-steps "$STEPS")
if [ -n "$TAG" ]; then
  ARGS+=(--run-tag "$TAG")
fi

exec bash "$(dirname "$0")/start.sh" "${ARGS[@]}"
