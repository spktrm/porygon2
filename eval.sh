#!/bin/bash
SESSION=eval
# Every path below is relative to the repo root, and the panes inherit it.
cd "$(dirname "$0")" || exit 1
# shlex quotes each argument for the command sent to the tmux pane.
ARGS=$(env/bin/python -c 'import shlex, sys; print(shlex.join(sys.argv[1:]))' "$@")

# Start clean
tmux kill-session -t "$SESSION" 2>/dev/null || true

# Create session with the first window
tmux new-session -d -s "$SESSION" -n evalclient

# Keep panes open on exit & show pane titles
tmux set-option -t "$SESSION" remain-on-exit on
tmux setw -t "$SESSION":evalclient pane-border-status top
tmux setw -t "$SESSION":evalclient pane-border-format '#{pane_index} #{pane_title}'

# ----- Pane 0: npm service -----
tmux select-pane -t "$SESSION":evalclient.0 -T "evalclient"
tmux send-keys  -t "$SESSION":evalclient.0 "cd service" C-m
tmux send-keys  -t "$SESSION":evalclient.0 "npm run start-evaluation-client" C-m

# ----- Pane 1: python rl -----
tmux split-window -h -t "$SESSION":evalclient.0
tmux select-pane -t "$SESSION":evalclient.1 -T "evalserver"
tmux send-keys  -t "$SESSION":evalclient.1 "source env/bin/activate" C-m
# .env carries the python-side flags; sourced in the pane shell for the same
# reason as start.sh. The eval client reads the same file itself (dotenv) for
# its Showdown login and RL_SERVER_URL. -m so the repo root is on sys.path
# regardless of cwd tricks.
tmux send-keys  -t "$SESSION":evalclient.1 'set -a; [ -f .env ] && source .env; set +a' C-m
tmux send-keys  -t "$SESSION":evalclient.1 "python -m inference.server $ARGS" C-m

tmux attach -t "$SESSION"
