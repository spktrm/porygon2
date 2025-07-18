run_in_dir_or_file() {
    echo "Processing: $@"

    autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$@"
    isort "$@"
    black "$@"
}

npx prettier service/src/server/*.ts -w

run_in_dir_or_file rlenv/*.py
run_in_dir_or_file ml/**/*.py
run_in_dir_or_file embeddings/*.py