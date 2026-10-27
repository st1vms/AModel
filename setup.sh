#!/usr/bin/env bash
# One-time setup: create the virtualenv and install dependencies.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    echo "Creating .venv ..."
    python3 -m venv .venv
fi

echo "Installing dependencies ..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "Done. Launch the app with ./run.sh"
