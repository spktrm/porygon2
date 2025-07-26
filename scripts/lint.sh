run_in_dir_or_file() {
    echo "Processing: $@"

    autoflake --in-place --remove-unused-variables --remove-all-unused-imports "$@"
    isort "$@"
    black "$@"
}


npx prettier service/src/**/*.ts -w
npx prettier data/src/**/*.ts -w

run_in_dir_or_file embeddings/*.py
run_in_dir_or_file inference/*.py
run_in_dir_or_file rl/**/*.py
run_in_dir_or_file rl/*.py
run_in_dir_or_file embeddings/*.py