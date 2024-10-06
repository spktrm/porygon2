#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating Python virtual environment..."
python3 -m virtualenv venv
. venv/bin/activate

echo "Installing Python requirements..."
# Install all requirements.txt files found in the repository
find . -maxdepth 2 -name 'requirements.txt' -exec pip install -r {} \;
pip install -U "jax[cuda12]"

echo "Installing npm packages..."
# Find all directories containing package.json and run npm install
find . -maxdepth 2 -type f -name "package.json" -not -path "*/node_modules/*" -execdir npm install \;

echo "Setup complete!"
