#!/bin/sh

# Get the directory where the script is located, then navigate one level up
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Define the function in a POSIX-compliant way
run_in_dir_or_file() {
    path=$1
    echo "Processing: $path"

    if [ -d "$path" ]; then
        # Process directory
        autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports "$path"
        isort "$path"
        black "$path"
    elif [ -f "$path" ]; then
        # Process single file
        autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$path"
        isort "$path"
        black "$path"
    else
        echo "Warning: $path is not a valid directory or file!"
    fi
}

# List of directories and file patterns in the parent directory (space-separated)
DIRS="$PARENT_DIR/heuristics/*.py $PARENT_DIR/inference $PARENT_DIR/ml $PARENT_DIR/replays $PARENT_DIR/rlenv/*.py $PARENT_DIR/viz $PARENT_DIR/embeddings"

# Iterate over each item and process it
for path in $DIRS; do
    run_in_dir_or_file "$path"
done
