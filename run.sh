#!/usr/bin/env bash
# One-command launcher for the Nagpur cross-modal debris dashboard (macOS / Linux).
#
#   ./run.sh            # set up a virtualenv, install deps, launch the dashboard
#   ./run.sh demo       # run the batch benchmark instead of the dashboard
#   ./run.sh test       # run the test suite
#
set -euo pipefail
cd "$(dirname "$0")"

# 1. Find Python 3.9+
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.9+ from https://python.org and retry." >&2
  exit 1
fi
echo "Using $($PY --version)"

# 2. Create / reuse a virtual environment
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment (.venv) ..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install dependencies (quietly; skip if already satisfied)
echo "Installing dependencies (first run only, ~1-2 min) ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# 4. Run the requested mode
MODE="${1:-app}"
case "$MODE" in
  demo)  echo "Running benchmark ..."; python run_demo.py ;;
  test)  echo "Running tests ...";     python tests/test_pipeline.py ;;
  app|*)
    echo
    echo "======================================================================"
    echo " Launching the dashboard.  Open the 'Local URL' it prints below"
    echo " (usually http://localhost:8501) in your browser."
    echo " Press Ctrl+C here to stop."
    echo "======================================================================"
    echo
    exec streamlit run app.py
    ;;
esac
