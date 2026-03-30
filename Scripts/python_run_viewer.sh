#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
VENV_DIR="${VENV_DIR:-$ROOT/.venv-python}"
INPUT_DIR="${1:-$ROOT/input}"
OUTPUT_DIR="${2:-$ROOT/output}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Virtualenv not found: $VENV_DIR"
  echo "Run: ./Scripts/python_setup.sh"
  exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
  echo "Input directory not found: $INPUT_DIR"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

if [[ "$OVERWRITE" == "1" ]]; then
  rsync -a --delete --prune-empty-dirs \
    --include='*/' \
    --include='*.png' \
    --include='*.jpg' \
    --include='*.jpeg' \
    --include='*.webp' \
    --include='*.bmp' \
    --include='*.tif' \
    --include='*.tiff' \
    --exclude='*' \
    "$INPUT_DIR/" "$OUTPUT_DIR/"
else
  rsync -a --ignore-existing --prune-empty-dirs \
    --include='*/' \
    --include='*.png' \
    --include='*.jpg' \
    --include='*.jpeg' \
    --include='*.webp' \
    --include='*.bmp' \
    --include='*.tif' \
    --include='*.tiff' \
    --exclude='*' \
    "$INPUT_DIR/" "$OUTPUT_DIR/"
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT"
python ui_viewer.py --dir "$OUTPUT_DIR" --lama-device cpu --lama-roi-pad 96 --lama-max-side 1280
