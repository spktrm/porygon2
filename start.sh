#!/bin/bash
SESSION=train

# Safely capture all script arguments, preserving spaces and quotes
ARGS=$(printf "%q " "$@")

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
    print(f'  stopping {run.name}')
    run.stop()
" 2>/dev/null || true
sleep 5

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
# Inject the captured arguments at the end of the python command
tmux send-keys  -t "$SESSION":service.1 "python -m rl.online.main $ARGS" C-m

tmux attach -t "$SESSION"