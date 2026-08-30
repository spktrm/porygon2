#!/bin/bash
# Best-response child run (2026-08-27): delegates to start.sh, whose
# preamble already stops any live learner gracefully (Ctrl-C -> the
# KeyboardInterrupt handler's synchronous checkpoint) before launching.
# The BR trains in its own ckpts/gen{N}/br/<tag>/ subtree against the
# frozen target, and publishes its latest params into the parent's
# players/ dir on every stop. Resuming an interrupted BR is re-running
# the SAME command; resuming main afterwards is plain `bash start.sh`.
#
# Usage: bash start_br.sh <target_ckpt_dir> [num_steps] [run_tag] [extra flags...]
#   num_steps omitted -> train until winrate vs the target clears 0.7
#   (the --br-winrate default), unbounded step budget.
#   Anything after the positionals (or any leading --flag) is forwarded
#   to rl.online.main verbatim — e.g. --br-init shrink-perturb
#   --br-perturb-frac 0.75. A non-default init wants a fresh run_tag,
#   or an existing subtree resumes and the init flag does nothing.
set -euo pipefail

TARGET=${1:?usage: start_br.sh <target_ckpt_dir> [num_steps] [run_tag] [extra flags...]}
shift
STEPS=""
TAG=""
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
  STEPS=$1
  shift
fi
if [ $# -gt 0 ] && [[ $1 != --* ]]; then
  TAG=$1
  shift
fi

if [ ! -d "$TARGET" ]; then
  echo "target checkpoint dir '$TARGET' does not exist" >&2
  exit 1
fi

ARGS=(--br-target "$TARGET")
if [ -n "$STEPS" ]; then
  ARGS+=(--num-steps "$STEPS")
fi
if [ -n "$TAG" ]; then
  ARGS+=(--run-tag "$TAG")
fi
ARGS+=("$@")

exec bash "$(dirname "$0")/start.sh" "${ARGS[@]}"
