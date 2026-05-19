"""
Experiment 15 — Attention Head Sweep at Layer 21 (Day 7)
=========================================================
Goal
----
Identify *which* of the 12 attention heads in layer 21 are responsible for the
spatial-processing circuit found in Days 4/5.

Methodology — head ablation
---------------------------
For each head h in {0…11} at layer 21 (and optionally a ±3 layer window around
it), we run activation patching by zeroing out that head's contribution to the
output:

  1. Run a normal forward pass → record baseline yes/no answer.
  2. Run a second forward pass with a hook that zeros the h-th head's slice of
     the attention output (pre-projection) → record ablated answer.
  3. If the model's answer *changes*, that head is causally necessary for the
     spatial judgement.

We measure **flip rate**: fraction of images where ablating head h changes the
answer. A high flip rate = causally important head.

We test on matched-pair images where the model is *correct* (answers "yes" to
"Is object A above object B?" for a true-above image). Those are the only cases
where the model has actually encoded spatial info — ablating a spatial head
should cause a flip to "no".

Architecture note (Qwen2-VL-2B)
--------------------------------
  hidden_size = 1536, num_heads = 12, head_dim = 128
  GQA: num_kv_heads = 2  (heads 0–5 share KV group 0, heads 6–11 share KV group 1)

  In the attention output tensor (before o_proj), each head h occupies
  positions [:, :, h*128 : (h+1)*128].  Zeroing those positions ablates head h.

Usage
-----
python src/day7/exp15_attention_head_sweep.py \\
    --manifest src/dataset/spatial_real_dataset_val2017/manifest.jsonl \\
    --images-root src/dataset/spatial_real_dataset_val2017/images \\
    --max-images 60 \\
    --layer-window 3 \\
    --out-dir src/day7
"""

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# ── path setup ────────────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
for d in [SRC / "day5", SRC / "day1"]:
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from day5_utils import load_model

# ── constants ─────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2-VL-2B-Instruct"
PEAK_LAYER   = 21          # Day 4/5 causal peak
HEAD_DIM     = 128
NUM_HEADS    = 12
FIXED_Q      = "Is object A above object B?"


# ── helpers ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",      type=Path, required=True)
    p.add_argument("--images-root",   type=Path, required=True)
    p.add_argument("--max-images",    type=int, default=60,
                   help="Max images to test per head (default 60).")
    p.add_argument("--layer-window",  type=int, default=0,
                   help="Sweep heads in layers [PEAK_LAYER-window, PEAK_LAYER+window]. "
                        "0 = only layer 21 (default).")
    p.add_argument("--ablate-mlp",    action="store_true", default=False,
                   help="Also run MLP ablation: zero the MLP output at each layer in "
                        "the sweep window on the same correct images. Distinguishes "
                        "MLP-driven vs attention-driven spatial circuit.")
    p.add_argument("--relation",      type=str, default="above",
                   choices=["above", "below", "left", "right"],
                   help="Which spatial relation to use for candidate images and the "
                        "fixed yes/no question. 'above' (default) uses vflip transform; "
                        "'left'/'right'/'below' use original transform images.")
    p.add_argument("--skip-heads",    action="store_true", default=False,
                   help="Skip the attention head ablation sweep entirely. Use with "
                        "--ablate-mlp when only MLP results are needed, to avoid "
                        "the ~3.5hr head sweep on a 7-layer window.")
    p.add_argument("--seed",          type=int, default=42)
    p.add_argument("--out-dir",       type=Path, default=Path("src/day7"))
    return p.parse_args()


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_prompt(question: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question} Answer yes or no."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def get_image_path(row: dict, images_root: Path) -> Optional[Path]:
    p = images_root / Path(row["output_file"]).name
    if p.exists():
        return p
    p2 = images_root.parent / row["output_file"]
    if p2.exists():
        return p2
    return None


def get_yes_no(model, processor, image: Image.Image, question: str) -> str:
    """Return 'yes' or 'no' (first token of model output, lowercase)."""
    prompt = build_prompt(question)
    inputs = processor(
        text=prompt, images=[image], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=3)
    text = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return "yes" if text.startswith("yes") else "no"


def ablate_head(
    model, processor, layers_ref,
    image: Image.Image, question: str,
    layer_idx: int, head_idx: int,
) -> str:
    """Run forward pass with head `head_idx` at `layer_idx` zeroed out.

    We hook the self_attn module and zero the head's slice in the
    output (the tensor passed to o_proj).  This removes that head's
    contribution without touching any other computation.
    """
    start = head_idx * HEAD_DIM
    end   = start + HEAD_DIM
    handle = None

    def _hook(module, args, kwargs, output):
        # output of self_attn is (attn_output, ...) where attn_output is
        # (batch, seq_len, hidden_dim).  Zero the head slice.
        if isinstance(output, tuple):
            h = output[0].clone()
            h[:, :, start:end] = 0.0
            return (h,) + output[1:]
        else:
            out = output.clone()
            out[:, :, start:end] = 0.0
            return out

    attn_module = layers_ref[layer_idx].self_attn
    handle = attn_module.register_forward_hook(_hook, with_kwargs=True)

    try:
        prompt = build_prompt(question)
        inputs = processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3)
        text = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
    finally:
        if handle is not None:
            handle.remove()

    return "yes" if text.startswith("yes") else "no"


def ablate_mlp(
    model, processor, layers_ref,
    image: Image.Image, question: str,
    layer_idx: int,
) -> str:
    """Run forward pass with the MLP output at `layer_idx` zeroed out entirely.

    Zeros the output of the MLP block (after its activation function, before
    it is added back to the residual stream).  A high flip rate here means
    the MLP — not individual attention heads — is driving the spatial circuit.
    """
    handle = None

    def _hook(module, args, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    mlp_module = layers_ref[layer_idx].mlp
    handle = mlp_module.register_forward_hook(_hook)

    try:
        prompt = build_prompt(question)
        inputs = processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3)
        text = processor.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().lower()
    finally:
        if handle is not None:
            handle.remove()

    return "yes" if text.startswith("yes") else "no"


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Build relation-specific question and candidate filter
    rel = args.relation
    _prep = "" if rel in ("above", "below") else " of"
    fixed_q = f"Is object A {rel}{_prep} object B?"
    # above/below use vflip (flipping an orig-below gives above); others use original
    candidate_transform = "vflip" if rel == "above" else "original"

    print("=" * 60)
    print("EXPERIMENT 15: Attention Head Sweep at Layer 21 (Day 7)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Peak layer: {PEAK_LAYER}  |  layer window: ±{args.layer_window}")
    print(f"Heads per layer: {NUM_HEADS}  |  head_dim: {HEAD_DIM}")
    print(f"Relation: '{rel}'  |  Fixed question: '{fixed_q}'")
    print(f"Candidate transform: '{candidate_transform}'")
    print()

    # ── select images: true-<relation>, model correct ─────────────────────────
    all_rows = load_manifest(args.manifest)
    print(f"Loaded {len(all_rows)} manifest rows.")

    candidates = [
        r for r in all_rows
        if r.get("transform") == candidate_transform
        and r.get("true_relation") == rel
        and r.get("label") == 1
    ]
    random.shuffle(candidates)
    print(f"Candidate {candidate_transform}-{rel} rows: {len(candidates)}")
    print(f"Will test up to {args.max_images} (model-correct subset identified at runtime.)")

    # ── load model ────────────────────────────────────────────────────────────
    print("\nLoading model...")
    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    print(f"Model loaded in {time.time()-t0:.1f}s  |  layers={num_layers}  hidden={hidden_dim}")

    # ── verify NUM_HEADS against actual model config ──────────────────────────
    cfg = model.config
    actual_heads = (
        cfg.num_attention_heads if hasattr(cfg, "num_attention_heads")
        else cfg.text_config.num_attention_heads
    )
    if actual_heads != NUM_HEADS:
        print(f"  WARNING: expected {NUM_HEADS} heads, got {actual_heads} — updating.")
    n_heads = actual_heads

    # ── collect model-correct images (baseline pass) ──────────────────────────
    print(f"\n[Phase 1] Baseline pass — finding images where model answers 'yes' correctly")
    correct_images: list[tuple[Image.Image, dict]] = []

    for row in candidates:
        if len(correct_images) >= args.max_images:
            break
        img_path = get_image_path(row, args.images_root)
        if img_path is None:
            continue
        image = Image.open(img_path).convert("RGB")
        ans = get_yes_no(model, processor, image, fixed_q)
        if ans == "yes":
            correct_images.append((image, row))

    print(f"  Found {len(correct_images)} model-correct images (answered 'yes' to above question).")
    if len(correct_images) < 5:
        print("  WARNING: very few correct images — results may be noisy.")

    # Compute layers_to_sweep here so both MLP and head ablation can use it
    layers_to_sweep = list(range(
        max(0, PEAK_LAYER - args.layer_window),
        min(num_layers, PEAK_LAYER + args.layer_window + 1),
    ))

    # ── MLP ablation (optional) ───────────────────────────────────────────────
    mlp_flip_rates: dict[int, float] = {}
    if args.ablate_mlp:
        print(f"\n[Phase 2a] MLP ablation — zeroing MLP output at each swept layer")
        print(f"  Layers: {layers_to_sweep}  |  Images: {len(correct_images)}")
        for layer_idx in layers_to_sweep:
            flips = sum(
                1 for image, row in correct_images
                if ablate_mlp(model, processor, layers_ref, image, fixed_q, layer_idx) != "yes"
            )
            mlp_flip_rates[layer_idx] = flips / max(len(correct_images), 1)
            print(f"  layer={layer_idx}  MLP flip_rate={mlp_flip_rates[layer_idx]:.4f}  "
                  f"({'█' * int(mlp_flip_rates[layer_idx] * 30)})")
        print()
        # Interpretation hint
        max_mlp_fr = max(mlp_flip_rates.values()) if mlp_flip_rates else 0
        if max_mlp_fr > 0.30:
            print(f"  → MLP dominates (max flip_rate={max_mlp_fr:.3f} > 0.30): spatial circuit is MLP-driven")
        elif max_mlp_fr > 0.10:
            print(f"  → MLP contributes (max flip_rate={max_mlp_fr:.3f}): mixed attn+MLP circuit")
        else:
            print(f"  → MLP not dominant (max flip_rate={max_mlp_fr:.3f} ≤ 0.10): circuit likely distributed across attention heads")
    # ── head ablation sweep ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────
    total_combos = len(layers_to_sweep) * n_heads
    results: dict[int, dict[int, float]] = {}

    if args.skip_heads:
        print(f"\n[Phase 2] Head sweep skipped (--skip-heads). Saving MLP-only results.")
    else:
        print(f"\n[Phase 2] Ablation sweep — {len(layers_to_sweep)} layers × {n_heads} heads = {total_combos} combos")
        print(f"  Layers: {layers_to_sweep}")
        print(f"  Images per combo: {len(correct_images)}")
        print()

        # results[layer_idx][head_idx] = flip_rate (fraction of images that flipped)
        combo_done = 0
        for layer_idx in layers_to_sweep:
            results[layer_idx] = {}
            for head_idx in range(n_heads):
                flips = 0
                for image, row in correct_images:
                    ablated = ablate_head(
                        model, processor, layers_ref,
                        image, fixed_q,
                        layer_idx=layer_idx, head_idx=head_idx,
                    )
                    if ablated != "yes":   # baseline was "yes"; any change = flip
                        flips += 1
                flip_rate = flips / max(len(correct_images), 1)
                results[layer_idx][head_idx] = flip_rate
                combo_done += 1
                if combo_done % 12 == 0 or combo_done == total_combos:
                    print(f"  [{combo_done}/{total_combos}] layer={layer_idx} head={head_idx} "
                          f"flip_rate={flip_rate:.3f}")

        # ── print summary table ───────────────────────────────────────────────
        print(f"\n{'Layer':>6}  {'Head':>5}  {'Flip rate':>10}  Bar")
        print("-" * 50)
        for layer_idx in layers_to_sweep:
            for head_idx in range(n_heads):
                fr = results[layer_idx][head_idx]
                bar = "█" * int(fr * 30)
                print(f"{layer_idx:>6}  {head_idx:>5}  {fr:>10.4f}  {bar}")
            print()

        # Find top causal heads across the sweep window
        all_scores = [
            (layer_idx, head_idx, results[layer_idx][head_idx])
            for layer_idx in layers_to_sweep
            for head_idx in range(n_heads)
        ]
        all_scores.sort(key=lambda x: x[2], reverse=True)
        print(f"\nTop 5 causal heads (by flip rate):")
        for layer_idx, head_idx, fr in all_scores[:5]:
            print(f"  Layer {layer_idx:2d}  Head {head_idx:2d}  flip_rate={fr:.4f}")

    # ── save ──────────────────────────────────────────────────────────────────
    top5 = sorted(
        [(li, hi, results[li][hi]) for li in results for hi in results[li]],
        key=lambda x: x[2], reverse=True
    )[:5]
    out = {
        "experiment": "exp15_attention_head_sweep",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "peak_layer": PEAK_LAYER,
        "layer_window": args.layer_window,
        "layers_swept": layers_to_sweep,
        "n_heads": n_heads,
        "head_dim": HEAD_DIM,
        "n_correct_images": len(correct_images),
        "relation": rel,
        "fixed_question": fixed_q,
        "flip_rates": {
            str(layer_idx): {str(h): results[layer_idx][h] for h in range(n_heads)}
            for layer_idx in results
        },
        "top5_causal_heads": [
            {"layer": l, "head": h, "flip_rate": fr} for l, h, fr in top5
        ],
        "mlp_flip_rates": {str(k): v for k, v in mlp_flip_rates.items()},
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "results_exp15.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print("=" * 60)
    print("EXPERIMENT 15 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
