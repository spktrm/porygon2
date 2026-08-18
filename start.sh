#!/bin/bash
SESSION=train

# Safely capture all script arguments, preserving spaces and quotes.
# Guard the no-arg case: printf %q with an empty arg list still runs the
# format once and emits a literal '' that main.py rejects.
ARGS=""
[ $# -gt 0 ] && ARGS=$(printf "%q " "$@")

cd service
cd ../

# Ask any wandb runs from a previous "train" session to stop gracefully.
# Run.stop() just sets a flag the OLD process's own heartbeat has to pick
# up, so this only works while that process is still alive -- it MUST run
# before the tmux kill-session below, or there's nothing left to see the
# flag and the runs just sit "Running" until W&B's own timeout eventually
# marks them Crashed instead of Killed.
echo "Stopping stale wandb runs from any previous session..."
env/bin/python -c "
import wandb
api = wandb.Api()
for run in api.runs(f'{api.default_entity}/pokemon-rl', filters={'state': 'running'}):
    try:
        print(f'  stopping {run.name}')
        run.stop()
    except Exception as e:
        # Best-effort: e.g. Run.stop() doesn't exist before some SDK
        # version (AttributeError on 0.27.2, present by 0.28.1). Never
        # worth blocking the restart over — falls back to W&B's own
        # heartbeat timeout marking it Crashed instead of Killed.
        print(f'  could not stop {run.name}: {e}')
" 2>&1 || echo "  (skipping wandb cleanup — could not reach the API, e.g. network issue)"
sleep 5

# Graceful stop first: Ctrl-C the learner so its KeyboardInterrupt
# handler writes a synchronous full checkpoint (main + live exploiter
# populations + scheduler) before the session is killed. A hard kill
# mid-background-write leaves stray "<component>.tmp.<pid>.<tid>" files
# (loader skips them since 2026-08-15, but the state since the last
# periodic save is still lost) — with this, a deliberate restart loses
# nothing. Bounded wait, then the kill below is the backstop either way.
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux send-keys -t "$SESSION":service.1 C-c 2>/dev/null || true
  for _ in $(seq 1 45); do
    pgrep -f 'rl\.online\.main' >/dev/null || break
    sleep 2
  done
fi

# Start clean
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the first window
tmux new-session -d -s "$SESSION" -n service

# Keep panes open on exit & show pane titles
tmux set-option -t "$SESSION" remain-on-exit on
tmux setw -t "$SESSION" -g pane-border-status top
tmux setw -t "$SESSION" -g pane-border-format '#{pane_index} #{pane_title}'

# ----- Pane 0: npm service -----
tmux select-pane -t "$SESSION":service.0 -T "service"
tmux send-keys  -t "$SESSION":service.0 "cd service" C-m
tmux send-keys  -t "$SESSION":service.0 "npm run start" C-m

# ----- Pane 1: python rl -----
tmux split-window -h -t "$SESSION":service.0
tmux select-pane -t "$SESSION":service.1 -T "rl"
tmux send-keys  -t "$SESSION":service.1 "source env/bin/activate" C-m
# .env is the single definition point for ALL env vars (see .env.example
# for the documented set; .env itself is gitignored, per-box). It's
# sourced into the pane shell BEFORE python starts because main.py's own
# load_dotenv() runs inside the already-started interpreter — fine for
# variables python code reads later (LOAD_STATE_MODE, WANDB_*, XLA
# flags), but silently useless for anything the C runtime consumes at
# process startup (MALLOC_ARENA_MAX, PYTHONMALLOC, LD_*). Sourcing here
# covers both classes from one file. Single-quoted so expansion happens
# in the pane shell.
# unset LOAD_STATE_MODE first: the pane inherits the tmux server's
# environment, which can carry an export from a long-gone .env revision
# (sourced with set -a in whatever shell first started the server). A
# commented-out line in today's .env can't clear it, and a stale "params"
# silently resets step counts + league on every launch.
tmux send-keys  -t "$SESSION":service.1 'unset LOAD_STATE_MODE; set -a; [ -f .env ] && source .env || echo "WARNING: no .env found — see .env.example (MALLOC_ARENA_MAX etc. unset)"; set +a' C-m
# Inject the captured arguments at the end of the python command. Piped
# through tee into runtime/learner.log (timestamped, prior logs kept) so a
# postmortem survives even if tmux itself dies (e.g. a host reboot) —
# logging.basicConfig only writes to the pane otherwise. `set -o pipefail`
# so a learner crash still fails the pane command (not masked by tee's own
# exit code); mkdir is idempotent, runtime/ is already gitignored.
mkdir -p runtime
LOG_FILE="runtime/learner_$(date +%Y%m%d_%H%M%S).log"
tmux send-keys  -t "$SESSION":service.1 "set -o pipefail; python -m rl.online.main $ARGS 2>&1 | tee $(printf '%q' "$LOG_FILE")" C-m

tmux attach -t "$SESSION"