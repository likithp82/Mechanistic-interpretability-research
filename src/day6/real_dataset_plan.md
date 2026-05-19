# Real Image Dataset Plan (Automatic Builder)

## Why This

Synthetic images were ideal for circuit discovery, but next-stage claims need real-image stress tests and distribution shift.

## What This Builder Does

Script: src/day6/build_real_spatial_dataset.py

Inputs:
- COCO-style image folder
- COCO-style annotations JSON

Pipeline:
1. Mine object pairs from real images using annotation boxes.
2. Assign robust 4-way labels with center geometry and ambiguity filters.
3. Generate transformed variants with exact label remapping:
- original
- horizontal flip (left/right swap)
- vertical flip (above/below swap)
4. Write manifest.jsonl with image path, pair labels, transformed boxes, and prompt text.
5. Write summary.json with class counts and build stats.

## Full Pipeline (Including Data Acquisition)

Step 1. Download real images and annotations (one-time).

- Download COCO images and COCO annotations to a local folder.
- If files already exist, skip download.

Step 2. Extract and verify input paths.

- Verify `train2017/` image folder exists.
- Verify `annotations/instances_train2017.json` exists.
- If already extracted, skip extraction.

Step 3. Build pairwise spatial dataset using annotation geometry.

- Run `src/day6/build_real_spatial_dataset.py` with selected parameters.
- Script mines object pairs, assigns robust 4-way labels, and writes manifest.

Step 4. Audit quality and class balance.

- Check `summary.json` for relation counts.
- Sample-check a subset for label quality and ambiguity handling.

Step 5. Use outputs in Day 6 experiments.

- Train relation probe and steering vectors on subset.
- Evaluate on held-out real splits and compare with synthetic results.

## Usage Example

python src/day6/build_real_spatial_dataset.py \
  --images-dir /path/to/coco/train2017 \
  --annotations /path/to/coco/annotations/instances_train2017.json \
  --output-dir /path/to/spatial_real_dataset \
  --max-images 3000 \
  --max-pairs-per-image 8 \
  --transforms original,hflip,vflip

## Optional Helper For Step 1 and Step 2 (Skip If Exists)

Use helper script:

`src/day6/prepare_coco_inputs.sh`

Example:

`bash src/day6/prepare_coco_inputs.sh /path/to/coco`

This script will:

1. Download `train2017.zip` and `annotations_trainval2017.zip` only if missing.
2. Extract them only if the target folders/files are missing.
3. Print final verified paths for `--images-dir` and `--annotations`.

## Recommended Settings For First Run

- max_images: 2000 to 5000
- max_pairs_per_image: 6 to 10
- min_box_area: 900
- axis_gap: 18
- dominance_ratio: 1.25

## Integration With Day 6 Experiments

1. Train relation probe / steering directions on a subset.
2. Evaluate causal steering on held-out real images.
3. Re-run bottleneck and head-localization protocols on real data.
4. Report clean-vs-hard-vs-real comparisons.

## Notes

- This is annotation-driven and does not hallucinate labels.
- Ambiguous geometric cases are filtered rather than forced.
- Transform label remapping is deterministic and auditable.
