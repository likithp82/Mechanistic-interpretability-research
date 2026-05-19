#!/usr/bin/env bash
set -euo pipefail

# Prepare COCO 2017 inputs for real spatial dataset building.
# Downloads and extraction are idempotent: existing files/folders are reused.
#
# Usage:
#   bash src/day6/prepare_coco_inputs.sh /path/to/coco_root
#
# Output paths:
#   Images dir:      /path/to/coco_root/train2017
#   Annotations json:/path/to/coco_root/annotations/instances_train2017.json

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/day6/prepare_coco_inputs.sh /path/to/coco_root"
  exit 1
fi

COCO_ROOT="$1"
mkdir -p "$COCO_ROOT"

TRAIN_ZIP="$COCO_ROOT/train2017.zip"
ANN_ZIP="$COCO_ROOT/annotations_trainval2017.zip"
TRAIN_DIR="$COCO_ROOT/train2017"
ANN_JSON="$COCO_ROOT/annotations/instances_train2017.json"

TRAIN_URL_HTTPS="https://images.cocodataset.org/zips/train2017.zip"
ANN_URL_HTTPS="https://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_URL_HTTP="http://images.cocodataset.org/zips/train2017.zip"
ANN_URL_HTTP="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

is_valid_zip() {
  local zip_path="$1"
  unzip -tq "$zip_path" >/dev/null 2>&1
}

download_if_missing() {
  local url_https="$1"
  local url_http="$2"
  local out="$3"
  local need_download=1
  local attempt=1
  local max_attempts=200

  if [[ -f "$out" ]]; then
    if is_valid_zip "$out"; then
      echo "[skip] Already downloaded: $out"
      need_download=0
    else
      echo "[resume] Partial or invalid archive detected: $out"
    fi
  fi

  if [[ $need_download -eq 1 ]]; then
    while true; do
      echo "[download] $url_https (attempt $attempt/$max_attempts)"
      # Resume partial downloads and retry all network/server errors.
      if curl -fL --continue-at - --retry 20 --retry-delay 5 --retry-all-errors "$url_https" -o "$out"; then
        if is_valid_zip "$out"; then
          echo "[ok] Downloaded archive: $out"
          break
        fi
        echo "[warn] Archive integrity check failed after download, retrying"
      else
        curl_rc=$?
        if [[ $curl_rc -eq 60 ]]; then
          echo "[warn] HTTPS certificate mismatch. Falling back to HTTP for COCO host."
          if curl -fL --continue-at - --retry 20 --retry-delay 5 --retry-all-errors "$url_http" -o "$out"; then
            if is_valid_zip "$out"; then
              echo "[ok] Downloaded archive via HTTP fallback: $out"
              break
            fi
            echo "[warn] Archive integrity check failed after HTTP download, retrying"
          else
            echo "[warn] HTTP fallback download failed, retrying"
          fi
        else
          echo "[warn] Download attempt failed (curl rc=$curl_rc), retrying"
        fi
      fi

      if (( attempt >= max_attempts )); then
        echo "[error] Failed downloading after $max_attempts attempts: $url_https"
        return 1
      fi

      attempt=$((attempt + 1))
      sleep 5
    done
  fi
}

extract_if_missing_dir() {
  local zip_path="$1"
  local target_dir="$2"
  if [[ -d "$target_dir" ]]; then
    echo "[skip] Already extracted dir: $target_dir"
  else
    echo "[extract] $zip_path -> $(dirname "$target_dir")"
    unzip -q "$zip_path" -d "$(dirname "$target_dir")"
  fi
}

extract_if_missing_file() {
  local zip_path="$1"
  local target_file="$2"
  if [[ -f "$target_file" ]]; then
    echo "[skip] Already extracted file: $target_file"
  else
    echo "[extract] $zip_path"
    unzip -q "$zip_path" -d "$(dirname "$(dirname "$target_file")")"
  fi
}

download_if_missing "$TRAIN_URL_HTTPS" "$TRAIN_URL_HTTP" "$TRAIN_ZIP"
download_if_missing "$ANN_URL_HTTPS" "$ANN_URL_HTTP" "$ANN_ZIP"

extract_if_missing_dir "$TRAIN_ZIP" "$TRAIN_DIR"
extract_if_missing_file "$ANN_ZIP" "$ANN_JSON"

if [[ ! -d "$TRAIN_DIR" ]]; then
  echo "[error] Missing images directory: $TRAIN_DIR"
  exit 1
fi

if [[ ! -f "$ANN_JSON" ]]; then
  echo "[error] Missing annotations JSON: $ANN_JSON"
  exit 1
fi

echo
echo "COCO inputs ready:"
echo "  images-dir   : $TRAIN_DIR"
echo "  annotations  : $ANN_JSON"
echo
echo "Next:"
echo "python src/day6/build_real_spatial_dataset.py \\\n  --images-dir $TRAIN_DIR \\\n  --annotations $ANN_JSON \\\n  --output-dir $COCO_ROOT/spatial_real_dataset \\\n  --max-images 3000 \\\n  --max-pairs-per-image 8 \\\n  --transforms original,hflip,vflip"
