#!/usr/bin/env bash
# Launch the AModel Streamlit UI.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m streamlit run src/ui/app.py
