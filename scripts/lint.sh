#!/bin/sh

# Get the directory where the script is located, then navigate one level up
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Define the function in a POSIX-compliant way
run_in_dir() {
    dir=$1
    echo "Running function in $dir"

    autoflake -r --in-place --remove-unused-variables $dir
    isort $dir
    black $dir
}

# List of directories in the parent directory (modify as needed)
DIRS="$PARENT_DIR/heuristics $PARENT_DIR/inference $PARENT_DIR/ml $PARENT_DIR/replays $PARENT_DIR/rlenv $PARENT_DIR/viz"

# Iterate over each directory
for DIR in $DIRS; do
    if [ -d "$DIR" ]; then
        run_in_dir "$DIR"
    else
        echo "$DIR is not a valid directory!"
    fi
done
