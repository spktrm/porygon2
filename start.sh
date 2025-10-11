#!/bin/bash
SESSION=train

cd service
cd ../

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
tmux send-keys  -t "$SESSION":service.1 "python rl/main.py" C-m

tmux attach -t "$SESSION"
