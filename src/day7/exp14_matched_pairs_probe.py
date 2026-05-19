"""
Experiment 14 — Matched-Pairs Layer Sweep (Day 7)
==================================================
Goal
----
Determine whether layer 21's spatial-encoding peak (found on synthetic images in
Days 4/5) also holds for real COCO images.

Methodology — matched pairs
---------------------------
The manifest contains every image in three transforms: original, hflip, vflip.
A vertical flip swaps above↔below while keeping *all* other visual content
identical (same photo, same objects, same colours, same lighting).

We build pairs:
  • image A = original crop  →  true_relation = "below",  question label = 1
  • image B = vflip of same crop  →  true_relation = "above",  label = 1

Both images are shown with the *same fixed question*:
  "Is object A above object B?"

Image A (below) should make the model say "no" → probe label 0.
Image B (vflip/above) should make the model say "yes" → probe label 1.

Because both images come from the *same photo* (just flipped), visual content is
matched. The *only* difference is spatial arrangement. Any probe accuracy above
chance (0.50) must come from spatial encoding, not content.

Probe
-----
At each of the 28 layers we extract the mean of image-pad token activations
(token id for <|image_pad|>) from a single forward pass. A logistic regression
with 5-fold stratified CV is fit on label (above=1, below=0). The CV accuracy
profile across layers reveals *where* spatial information is encoded.

Expected: early layers ≈ 0.50 (no cross-modal attention yet), peak emerging
around layers 15–25, consistent with Day 4/5 synthetic finding of layer 21.

Usage
-----
python src/day7/exp14_matched_pairs_probe.py \\
    --manifest src/dataset/spatial_real_dataset_val2017/manifest.jsonl \\
    --images-root src/dataset/spatial_real_dataset_val2017/images \\
    --max-pairs 100 \\
    --out-dir src/day7
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
for d in [SRC / "day5", SRC / "day1"]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from day5_utils import load_model
from utils import ActivationCollector

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen2-VL-2B-Instruct"
PROBE_LAYER = 21          # Day 4/5 causal peak on synthetic images
FIXED_Q     = "Is object A above object B?"   # identical for both pair members


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",    type=Path, required=True)
    p.add_argument("--images-root", type=Path, required=True)
    p.add_argument("--max-pairs",   type=int, default=100,
                   help="Max matched pairs to use (default 100).")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out-dir",     type=Path, default=Path("src/day7"))
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_matched_pairs(rows: list[dict]) -> list[tuple[dict, dict]]:
    """Return list of (above_row, below_row) matched pairs.

    above_row: vflip transform, true_relation='above', label=1
    below_row: original transform, true_relation='below', label=1
    They share the same (image_id, subject ann_id, object ann_id).
    """
    Key = tuple  # (image_id, subj_ann_id, obj_ann_id)

    vflip_above = {}
    orig_below  = {}

    for r in rows:
        k = (r["image_id"], r["subject"]["ann_id"], r["object"]["ann_id"])
        if r.get("transform") == "vflip"     and r.get("true_relation") == "above" and r.get("label") == 1:
            vflip_above[k] = r
        elif r.get("transform") == "original" and r.get("true_relation") == "below" and r.get("label") == 1:
            orig_below[k] = r

    shared = set(vflip_above.keys()) & set(orig_below.keys())
    pairs = [(vflip_above[k], orig_below[k]) for k in sorted(shared)]
    return pairs


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question} Answer yes or no."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_image_path(row: dict, images_root: Path) -> Path | None:
    p = images_root / Path(row["output_file"]).name
    if p.exists():
        return p
    p2 = images_root.parent / row["output_file"]
    if p2.exists():
        return p2
    return None


def probe_accuracy(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n_splits = min(cv, min(np.bincount(np.unique(y, return_inverse=True)[1])))
    n_splits = max(n_splits, 2)
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0),
    )
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
    return float(scores.mean())


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 14: Matched-Pairs Layer Sweep (Day 7)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Manifest: {args.manifest}")
    print(f"Fixed question: '{FIXED_Q}'")
    print(f"Max pairs: {args.max_pairs}")
    print()

    # ── load and pair data ────────────────────────────────────────────────────
    all_rows = load_manifest(args.manifest)
    print(f"Loaded {len(all_rows)} manifest rows.")

    all_pairs = build_matched_pairs(all_rows)
    print(f"Available matched pairs (vflip-above / orig-below): {len(all_pairs)}")

    random.shuffle(all_pairs)
    pairs = all_pairs[: args.max_pairs]
    print(f"Using {len(pairs)} pairs for sweep.")

    # ── load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    collector = ActivationCollector(model)
    print(f"Model loaded in {time.time()-t0:.1f}s  |  layers={num_layers}  hidden={hidden_dim}")

    image_token_id: int = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    print(f"image_pad token id: {image_token_id}")

    # ── single-pass activation collection ────────────────────────────────────
    # Each pair contributes two rows:
    #   above_row (vflip) → label 1
    #   below_row (original) → label 0
    # Both use the same FIXED_Q so text is identical.
    print(f"\n[Phase 1] Collecting activations — {len(pairs)*2} inferences total "
          f"(one pass per image, all {num_layers} layers simultaneously)")
    print(f"  Token position: mean of image-pad tokens  (id={image_token_id})")

    all_layer_acts: dict[int, list[np.ndarray]] = {i: [] for i in range(num_layers)}
    labels: list[int] = []

    prompt = build_prompt(FIXED_Q)
    done = 0
    total = len(pairs) * 2

    for above_row, below_row in pairs:
        for label_val, row in ((1, above_row), (0, below_row)):
            img_path = get_image_path(row, args.images_root)
            if img_path is None:
                continue

            image = Image.open(img_path).convert("RGB")
            collector.attach()
            collector.activations = {}
            inputs = processor(
                text=prompt, images=[image], return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)

            input_ids = inputs["input_ids"]
            for layer_idx in range(num_layers):
                try:
                    act = collector.get_image_token_mean_activation(
                        layer_idx, input_ids, image_token_id
                    )
                    all_layer_acts[layer_idx].append(act)
                except (KeyError, ValueError):
                    pass
            collector.detach()

            labels.append(label_val)
            done += 1
            if done % 20 == 0 or done == total:
                print(f"  collected {done}/{total}")

    # ── per-layer probe ───────────────────────────────────────────────────────
    print(f"\n[Phase 2] Fitting per-layer binary probe (5-fold CV)")
    print(f"  Chance baseline = 0.500")
    print()

    layer_accs: dict[int, float | None] = {}
    y_np = np.array(labels)

    for layer_idx in range(num_layers):
        acts = all_layer_acts[layer_idx]
        yl = y_np[: len(acts)]
        if len(acts) >= 4 and len(set(yl.tolist())) == 2:
            layer_accs[layer_idx] = probe_accuracy(
                np.array(acts, dtype=np.float32), yl
            )
        else:
            layer_accs[layer_idx] = None

    # Print table
    print(f"  {'Layer':>5}  {'Probe CV acc':>12}  {'▲ above chance':>14}")
    for i in range(num_layers):
        acc = layer_accs[i]
        if acc is not None:
            delta = acc - 0.5
            bar = "█" * int(abs(delta) * 40)
            sign = "+" if delta >= 0 else "-"
            print(f"  {i:>5}  {acc:>12.4f}  {sign}{abs(delta):>6.4f}  {bar}")
        else:
            print(f"  {i:>5}  {'N/A':>12}")

    valid = {k: v for k, v in layer_accs.items() if v is not None}
    if valid:
        best = max(valid, key=lambda k: valid[k])
        print(f"\n  Best layer: {best}  (acc={valid[best]:.4f})")
        print(f"  Day4/5 synthetic peak: layer {PROBE_LAYER} — "
              + ("MATCHES ✓" if best == PROBE_LAYER else f"DIFFERS → {best}"))
    else:
        print("\n  No valid probe results.")

    # ── save ──────────────────────────────────────────────────────────────────
    results = {
        "experiment": "exp14_matched_pairs_probe",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "n_pairs": len(pairs),
        "n_inferences": done,
        "fixed_question": FIXED_Q,
        "probe_token": "image-mean",
        "image_pad_token_id": image_token_id,
        "layer_probe_accs": {str(k): v for k, v in layer_accs.items()},
        "best_layer": best if valid else None,
        "synthetic_peak_layer": PROBE_LAYER,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results_exp14.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print("=" * 60)
    print("EXPERIMENT 14 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
