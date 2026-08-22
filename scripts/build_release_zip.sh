#!/usr/bin/env bash
# Package the complete pipeline for manual upload to GitHub.
#
#   bash scripts/build_release_zip.sh
#
# Includes: all source, scripts, tests, the executed EDA notebook, every
# report, figure and result table, and a git bundle carrying the full commit
# history (so the branch can be restored with `git clone` from the bundle).
#
# Excludes: data/raw/*.csv (1.3 GB, redistributable from Kaggle - see
# data/raw/README.md) and data/interim/*.parquet (regenerable by scripts/03
# and scripts/04).
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-/tmp/ieee-fraud-detection-complete.zip}"
BUNDLE="ieee-fraud-detection-history.bundle"

rm -f "$OUT" "$BUNDLE"
git bundle create "$BUNDLE" --all >/dev/null 2>&1

zip -r -q "$OUT" \
    src scripts notebooks reports \
    README.md claude.md IMPLEMENTATION_PLAN.md requirements.txt .gitignore \
    data/raw/README.md \
    "$BUNDLE" \
    -x '*__pycache__*' '*.pyc' '*.ipynb_checkpoints*'

rm -f "$BUNDLE"
echo "wrote $OUT"
unzip -l "$OUT" | tail -1
