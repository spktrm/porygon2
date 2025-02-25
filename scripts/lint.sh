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

# Find all Python files in the parent directory, excluding the env/, proto/, and protos/ directories
find "$PARENT_DIR" -name "*.py" -not -path "$PARENT_DIR/env/*" -not -path "*/proto/*" -not -path "*/protos/*" | while read -r file; do
    run_in_dir_or_file "$file"
done
