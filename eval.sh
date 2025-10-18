#!/bin/bash
SESSION=train

cd service
cd ../

# Start clean
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the first window
tmux new-session -d -s "$SESSION" -n evalclient

# Keep panes open on exit & show pane titles
tmux set-option -t "$SESSION" remain-on-exit on
tmux setw -t "$SESSION" -g pane-border-status top
tmux setw -t "$SESSION" -g pane-border-format '#{pane_index} #{pane_title}'

# ----- Pane 0: npm service -----
tmux select-pane -t "$SESSION":evalclient.0 -T "evalclient"
tmux send-keys  -t "$SESSION":evalclient.0 "cd service" C-m
tmux send-keys  -t "$SESSION":evalclient.0 "npm run start-evaluation-client" C-m

# ----- Pane 1: python rl -----
tmux split-window -h -t "$SESSION":evalclient.0
tmux select-pane -t "$SESSION":evalclient.1 -T "evalserver"
tmux send-keys  -t "$SESSION":evalclient.1 "source env/bin/activate" C-m
tmux send-keys  -t "$SESSION":evalclient.1 "python inference/server.py" C-m

tmux attach -t "$SESSION"
