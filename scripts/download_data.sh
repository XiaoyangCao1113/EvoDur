#!/usr/bin/env bash
# Download the public datasets (and the NOAA weather file of the Beijing data) into data/ (or $EVODUR_DATA), then check them.
#   scripts/download_data.sh [sydney] [ny511] [seattle] [sanfrancisco] [weather]      (default: the four public datasets)
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
PYTHON=${PYTHON:-python}
$PYTHON -m evodur.download_data "$@"
$PYTHON -m evodur.check_data sydney ny511early seattle sanfrancisco || true
