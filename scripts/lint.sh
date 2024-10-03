#!/bin/sh

# Get the directory where the script is located, then navigate one level up
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Define the function in a POSIX-compliant way
run_in_dir_or_file() {
    path=$1
    echo "Running function on $path"

    if [ -d "$path" ]; then
        autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports $path
        isort $path
        black $path
    elif [ -f "$path" ]; then
        autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$path"
        isort "$path"
        black "$path"
    else
        echo "$path is not a valid directory or file!"
    fi
}

# List of directories and file patterns in the parent directory (modify as needed)
DIRS="$PARENT_DIR/heuristics $PARENT_DIR/inference $PARENT_DIR/ml $PARENT_DIR/replays $PARENT_DIR/rlenv/*.py $PARENT_DIR/viz"

# Iterate over each item
for ITEM in $DIRS; do
    for FILE_OR_DIR in $ITEM; do
        run_in_dir_or_file "$FILE_OR_DIR"
    done
done
