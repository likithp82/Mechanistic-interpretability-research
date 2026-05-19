#!/usr/bin/env bash
set -euo pipefail

# Download minimal pilot datasets for spatial research:
# 1) COCO val2017 images
# 2) COCO 2017 annotations
# 3) VSR benchmark (attempt from Hugging Face dataset repos)
#
# Usage:
#   bash src/day6/download_pilot_datasets.sh /path/to/dataset_root [--vsr-only]
#
# Output layout under dataset_root:
#   coco/
#     val2017.zip
#     annotations_trainval2017.zip
#     val2017/
#     annotations/instances_val2017.json
#   vsr/
#     <downloaded dataset snapshot files>

if [[ $# -lt 1 ]]; then
  echo "Usage: bash src/day6/download_pilot_datasets.sh /path/to/dataset_root [--vsr-only]"
  exit 1
fi

DATA_ROOT="$1"
MODE="${2:-all}"
VSR_ONLY=0
if [[ "$MODE" == "--vsr-only" ]]; then
  VSR_ONLY=1
fi

if [[ "$MODE" != "all" && "$MODE" != "--vsr-only" ]]; then
  echo "Usage: bash src/day6/download_pilot_datasets.sh /path/to/dataset_root [--vsr-only]"
  exit 1
fi

COCO_ROOT="$DATA_ROOT/coco"
VSR_ROOT="$DATA_ROOT/vsr"
mkdir -p "$COCO_ROOT" "$VSR_ROOT"

VAL_ZIP="$COCO_ROOT/val2017.zip"
ANN_ZIP="$COCO_ROOT/annotations_trainval2017.zip"
VAL_DIR="$COCO_ROOT/val2017"
ANN_JSON="$COCO_ROOT/annotations/instances_val2017.json"

# Use plain HTTP for COCO: macOS curl SSL cert mismatch on images.cocodataset.org
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"
ANN_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

is_valid_zip() {
  local zip_path="$1"
  unzip -tq "$zip_path" >/dev/null 2>&1
}

download_resumable() {
  local url="$1"
  local out="$2"
  local attempt=1
  local max_attempts="${COCO_MAX_ATTEMPTS:-200}"

  if [[ -f "$out" ]] && is_valid_zip "$out"; then
    echo "[skip] Already downloaded: $out"
    return 0
  fi

  if [[ -f "$out" ]]; then
    echo "[resume] Partial archive detected: $out"
  fi

  while true; do
    echo "[download] $url (attempt $attempt/$max_attempts)"
    if curl -fL --continue-at - --retry 5 --retry-delay 5 "$url" -o "$out"; then
      if is_valid_zip "$out"; then
        echo "[ok] Downloaded: $out"
        return 0
      fi
      echo "[warn] Archive integrity check failed, retrying"
    else
      echo "[warn] Download attempt failed (curl rc=$?), retrying in 5s"
    fi

    if (( attempt >= max_attempts )); then
      echo "[error] Failed after $max_attempts attempts: $url"
      return 1
    fi

    attempt=$((attempt + 1))
    sleep 5
  done
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

download_vsr_from_hf() {
  local out_dir="$1"

  if [[ -f "$out_dir/.download_ok" ]]; then
    echo "[skip] VSR already downloaded at: $out_dir"
    return 0
  fi

  echo "[download] Attempting VSR from Hugging Face dataset repositories"
  python - "$out_dir" <<'PY'
import os
import sys

out_dir = sys.argv[1]
os.makedirs(out_dir, exist_ok=True)

candidates = [
    "cambridgeltl/vsr_random",
    "cambridgeltl/vsr_zeroshot",
    "cambridgeltl/vsr",
]

ok = False
errors = []

try:
    from huggingface_hub import snapshot_download
except Exception as e:
    print(f"[warn] huggingface_hub import failed: {e}")
    print("[hint] Install once: pip install huggingface_hub")
    sys.exit(2)

for repo_id in candidates:
    try:
        target = os.path.join(out_dir, repo_id.replace('/', '__'))
        os.makedirs(target, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=target,
        )
        with open(os.path.join(out_dir, ".download_ok"), "w", encoding="utf-8") as f:
            f.write(repo_id)
        print(f"[ok] Downloaded VSR dataset snapshot from: {repo_id}")
        ok = True
        break
    except Exception as e:
        errors.append((repo_id, str(e)))

if not ok:
    print("[warn] Could not auto-download VSR from candidate repositories.")
    for repo_id, err in errors:
        print(f"  - {repo_id}: {err}")
    print("[hint] You can still continue with COCO val now and add VSR manually later.")
    sys.exit(3)
PY
}

if [[ $VSR_ONLY -eq 0 ]]; then
  echo "[1/3] Downloading COCO val2017 + annotations"
  download_resumable "$VAL_URL" "$VAL_ZIP"
  download_resumable "$ANN_URL" "$ANN_ZIP"

  echo "[2/3] Extracting COCO inputs"
  extract_if_missing_dir "$VAL_ZIP" "$VAL_DIR"
  extract_if_missing_file "$ANN_ZIP" "$ANN_JSON"
else
  echo "[mode] --vsr-only enabled: skipping COCO download/extract"
fi

echo "[3/3] Downloading VSR benchmark"
if ! download_vsr_from_hf "$VSR_ROOT"; then
  echo "[warn] VSR auto-download failed. COCO val is still ready."
fi

echo
if [[ $VSR_ONLY -eq 1 ]]; then
  echo "Pilot datasets status:"
  echo "  VSR root        : $VSR_ROOT"
  echo
  echo "COCO was skipped by --vsr-only."
  exit 0
fi

if [[ -d "$VAL_DIR" && -f "$ANN_JSON" ]]; then
  echo "Pilot datasets status:"
  echo "  COCO images-dir : $VAL_DIR"
  echo "  COCO annotation : $ANN_JSON"
  echo "  VSR root        : $VSR_ROOT"
  echo
  echo "Use for pilot build:"
  echo "python src/day6/build_real_spatial_dataset.py \\\n  --images-dir $VAL_DIR \\\n  --annotations $ANN_JSON \\\n  --output-dir $DATA_ROOT/spatial_real_dataset_val2017 \\\n  --max-images 1000 \\\n  --max-pairs-per-image 8 \\\n  --transforms original,hflip,vflip"
else
  echo "[error] COCO val setup incomplete."
  exit 1
fi
