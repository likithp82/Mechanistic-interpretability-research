#!/usr/bin/env bash
set -euo pipefail

# One-command Day 6 real dataset pipeline:
# 1) Prepare COCO inputs (download/extract only if missing)
# 2) Build real spatial dataset
# 3) Print quick class-balance summary
#
# Usage:
#   bash src/day6/run_real_dataset_pipeline.sh /path/to/coco_root [output_dir]
#
# Example:
#   bash src/day6/run_real_dataset_pipeline.sh /data/coco /data/coco/spatial_real_dataset

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/day6/run_real_dataset_pipeline.sh /path/to/coco_root [output_dir]"
  exit 1
fi

COCO_ROOT="$1"
OUT_DIR="${2:-$COCO_ROOT/spatial_real_dataset}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PREP_SCRIPT="$SCRIPT_DIR/prepare_coco_inputs.sh"
BUILD_SCRIPT="$SCRIPT_DIR/build_real_spatial_dataset.py"

if [[ ! -f "$PREP_SCRIPT" ]]; then
  echo "[error] Missing script: $PREP_SCRIPT"
  exit 1
fi

if [[ ! -f "$BUILD_SCRIPT" ]]; then
  echo "[error] Missing script: $BUILD_SCRIPT"
  exit 1
fi

echo "[1/3] Preparing COCO inputs (idempotent)"
bash "$PREP_SCRIPT" "$COCO_ROOT"

IMAGES_DIR="$COCO_ROOT/train2017"
ANN_JSON="$COCO_ROOT/annotations/instances_train2017.json"

echo "[2/3] Building real spatial dataset"
python "$BUILD_SCRIPT" \
  --images-dir "$IMAGES_DIR" \
  --annotations "$ANN_JSON" \
  --output-dir "$OUT_DIR" \
  --max-images 3000 \
  --max-pairs-per-image 8 \
  --min-box-area 900 \
  --axis-gap 18 \
  --dominance-ratio 1.25 \
  --transforms original,hflip,vflip

SUMMARY_JSON="$OUT_DIR/summary.json"
MANIFEST_JSONL="$OUT_DIR/manifest.jsonl"

echo "[3/3] Quick balance summary"
if [[ -f "$SUMMARY_JSON" ]]; then
  echo "Summary file: $SUMMARY_JSON"
  python - <<'PY' "$SUMMARY_JSON"
import json, sys
p = sys.argv[1]
with open(p, 'r', encoding='utf-8') as f:
    s = json.load(f)
print("rows_written:", s.get("rows_written"))
print("relation_counts:", s.get("relation_counts"))
print("transform_counts:", s.get("transform_counts"))
PY
else
  echo "[warn] summary.json not found at $SUMMARY_JSON"
fi

if [[ -f "$MANIFEST_JSONL" ]]; then
  echo "manifest rows: $(wc -l < "$MANIFEST_JSONL")"
else
  echo "[warn] manifest.jsonl not found at $MANIFEST_JSONL"
fi

echo "Done. Output dir: $OUT_DIR"
