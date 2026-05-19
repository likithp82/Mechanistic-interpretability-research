"""Experiment 13: Real-Data Baseline on COCO val2017 spatial dataset.

Goals
-----
1. Measure Qwen2-VL-2B-Instruct accuracy on real-image yes/no spatial questions
   (manifest produced by build_real_spatial_dataset.py).
2. Collect last-token residual-stream activations at every layer for a random
   sample of positive examples to check whether the layer-21 causal localisation
   from Day 4/5 synthetic experiments holds on real data.
3. Train a logistic-regression direction probe per relation class on those
   activations and compare train accuracy to the Day 4/5 synthetic probes.
4. Save full results to results_exp13.json for later Day 6 comparison.

Usage
-----
python src/day6/exp13_real_data_baseline.py \
    --manifest src/dataset/spatial_real_dataset_val2017/manifest.jsonl \
    --images-root src/dataset/spatial_real_dataset_val2017/images \
    --max-eval 500 \
    --max-probe-per-class 50
"""

from __future__ import annotations

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

# ── repo path setup ──────────────────────────────────────────────────────────
SRC = Path(__file__).resolve().parent.parent
DAY1_DIR = SRC / "day1"
DAY5_DIR = SRC / "day5"
for p in (str(DAY1_DIR), str(DAY5_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from day5_utils import load_model, run_generation
from utils import ActivationCollector

# ── constants ────────────────────────────────────────────────────────────────
RELATIONS = ["above", "below", "left", "right"]
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
YES_TOKENS = {"yes", "Yes", "YES"}
NO_TOKENS  = {"no",  "No",  "NO"}
PROBE_LAYER = 21   # layer found causal in Day 4/5 synthetic experiments


# ── helpers ──────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",          type=Path,  required=True)
    p.add_argument("--images-root",       type=Path,  required=True)
    p.add_argument("--max-eval",          type=int,   default=500,
                   help="Max rows for accuracy evaluation (positives only).")
    p.add_argument("--max-probe-per-class", type=int, default=50,
                   help="Max positive rows per relation class for probing.")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--out-dir",           type=Path,  default=Path("src/day6"))
    p.add_argument("--probe-only",        action="store_true", default=False,
                   help="Skip Phase 1 accuracy eval; run probing/layer sweep only.")
    p.add_argument("--sweep-rows-per-class", type=int, default=50,
                   help="Examples per class for Phase 3 layer sweep (default 50).")
    p.add_argument("--blind-probe",       action="store_true", default=False,
                   help="Replace category names with 'object A/B' in probe questions "
                        "to remove text-label leakage and force visual-only probing.")
    p.add_argument("--binary-probe",      action="store_true", default=False,
                   help="Binary yes/no probe: pos-relation images (label=1) vs "
                        "neg-relation images (label=0), both shown with the SAME fixed "
                        "question asking about the pos relation.  Text is identical for "
                        "both classes so only visual information can drive the probe.")
    p.add_argument("--binary-rel",        type=str, default="left,right",
                   help="Comma-separated pair 'pos,neg' for --binary-probe. "
                        "pos images get label=1, neg images label=0; fixed question "
                        "asks 'Is object A <pos> of object B?'. Default: left,right. "
                        "Use 'above,below' for the higher-accuracy pair.")
    p.add_argument("--probe-token",       type=str, default="last",
                   choices=["last", "image-mean"],
                   help="Which token position(s) to probe. "
                        "'last' = final text token (default, current behaviour). "
                        "'image-mean' = mean over image-pad token positions, "
                        "which is where spatial visual info actually lives.")
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
    """Wrap question in Qwen2-VL chat format expecting yes/no answer."""
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|image_pad|><|vision_end|>"
        f"{question} Answer yes or no."
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def blind_question(row: dict) -> str:
    """Return a question where subject/object category names are replaced with
    'object A' and 'object B', forcing the model to rely on visual content.
    E.g. 'Is the dog left of the cat?' -> 'Is object A left of object B?'
    """
    relation = row.get("relation") or row.get("true_relation", "")
    return f"Is object A {relation} object B?"


def infer_yn(model, processor, image: Image.Image, question: str) -> str:
    """Return model's first token as lowercase string."""
    prompt = build_prompt(question)
    inputs = processor(
        text=prompt, images=[image], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5)
    answer = processor.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip().lower()
    return answer


def collect_activation(
    model, processor, collector: ActivationCollector,
    image: Image.Image, question: str, layer_idx: int,
    probe_token: str = "last", image_token_id: int | None = None,
) -> np.ndarray:
    """Return hidden state at layer_idx as float32 numpy array.

    probe_token='last'       -> last token position (old behaviour).
    probe_token='image-mean' -> mean over image-pad token positions.
    """
    collector.attach()
    collector.activations = {}
    prompt = build_prompt(question)
    inputs = processor(
        text=prompt, images=[image], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    if probe_token == "image-mean":
        act = collector.get_image_token_mean_activation(
            layer_idx, inputs["input_ids"], image_token_id
        )
    else:
        act = collector.get_last_token_activation(layer_idx)
    collector.detach()
    return act


def probe_accuracy(X: np.ndarray, y: np.ndarray, cv: int = 5) -> float:
    """Cross-validated probe accuracy to prevent overfitting in high-dim space."""
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


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("EXPERIMENT 13: Real-Data Baseline")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Manifest: {args.manifest}")
    print(f"Max eval: {args.max_eval}  |  Max probe/class: {args.max_probe_per_class}")
    print()

    # ── load data ────────────────────────────────────────────────────────────
    all_rows = load_manifest(args.manifest)
    print(f"Loaded {len(all_rows)} manifest rows.")

    # For accuracy eval: sample positive rows (label=1) evenly across relations
    pos_by_rel: dict[str, list[dict]] = {r: [] for r in RELATIONS}
    for row in all_rows:
        if row.get("label") == 1 and row.get("transform") == "original":
            rel = row.get("true_relation") or row.get("relation")
            if rel in pos_by_rel:
                pos_by_rel[rel].append(row)

    per_rel = args.max_eval // len(RELATIONS)
    eval_rows: list[dict] = []
    for rel in RELATIONS:
        sample = pos_by_rel[rel][: per_rel]
        random.shuffle(sample)
        eval_rows.extend(sample)
    random.shuffle(eval_rows)

    print(f"Eval set: {len(eval_rows)} rows "
          f"({per_rel} per relation, original transform only).")

    # Probe set: up to max_probe_per_class per relation (may overlap eval)
    probe_by_rel: dict[str, list[dict]] = {r: [] for r in RELATIONS}
    for rel in RELATIONS:
        probe_by_rel[rel] = pos_by_rel[rel][: args.max_probe_per_class]

    # ── load model ───────────────────────────────────────────────────────────
    print("\nLoading model...")
    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    collector = ActivationCollector(model)  # hooks all layers; layer_indices=None
    print(f"Model loaded in {time.time()-t0:.1f}s  |  "
          f"layers={num_layers}  hidden={hidden_dim}")

    # Resolve image-pad token id for --probe-token=image-mean
    image_token_id: int | None = None
    if args.probe_token == "image-mean":
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        print(f"  probe_token=image-mean  |  image_pad id={image_token_id}")
    else:
        print(f"  probe_token=last (final text token)")

    # ── accuracy evaluation ──────────────────────────────────────────────────
    overall_acc = None
    per_rel_acc = {r: None for r in RELATIONS}

    if args.probe_only:
        print("\n[Phase 1] SKIPPED (--probe-only)")
    else:
        print(f"\n[Phase 1] Accuracy evaluation on {len(eval_rows)} real-image rows")

        per_rel_correct: dict[str, int] = {r: 0 for r in RELATIONS}
        per_rel_total:   dict[str, int] = {r: 0 for r in RELATIONS}
        total_correct = 0

        for idx, row in enumerate(eval_rows, 1):
            img_path = args.images_root / Path(row["output_file"]).name
            if not img_path.exists():
                img_path = args.images_root.parent / row["output_file"]
            if not img_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            question = row["question"]
            rel = row.get("true_relation") or row.get("relation")

            answer = infer_yn(model, processor, image, question)
            correct = answer.startswith("yes")

            per_rel_total[rel] += 1
            if correct:
                per_rel_correct[rel] += 1
                total_correct += 1

            if idx % 50 == 0 or idx == len(eval_rows):
                run_acc = total_correct / sum(per_rel_total.values())
                print(f"  [{idx}/{len(eval_rows)}] running accuracy = {run_acc:.3f}")

        overall_acc = total_correct / max(sum(per_rel_total.values()), 1)
        per_rel_acc = {r: per_rel_correct[r] / max(per_rel_total[r], 1) for r in RELATIONS}

        print(f"\n  Overall accuracy (real data, label=1): {overall_acc:.4f}")
        for rel in RELATIONS:
            print(f"    {rel:6s}: {per_rel_acc[rel]:.4f}  ({per_rel_correct[rel]}/{per_rel_total[rel]})")

    # ── activation probing ───────────────────────────────────────────────────
    # --binary-probe: scientifically valid visual-only probe.
    # Use left (label=1) vs right (label=0) images with IDENTICAL fixed question
    # "Is object A left of object B?" — text is the same for both classes so
    # only visual information can distinguish them.
    if args.binary_probe:
        rel_parts = [s.strip() for s in args.binary_rel.split(",")]
        if len(rel_parts) != 2:
            raise ValueError(f"--binary-rel must be 'pos,neg' (got '{args.binary_rel}')")
        pos_rel, neg_rel = rel_parts
        # "above"/"below" take no preposition; "left"/"right" take "of"
        _prep = "" if pos_rel in ("above", "below") else " of"
        BINARY_Q = f"Is object A {pos_rel}{_prep} object B?"
        n_per = args.max_probe_per_class

        pos_rows = [r for r in all_rows
                    if r.get("true_relation") == pos_rel and r.get("label") == 1][:n_per]
        neg_rows = [r for r in all_rows
                    if r.get("true_relation") == neg_rel and r.get("label") == 1][:n_per]

        print(f"\n[Phase 2] BINARY probe at layer {PROBE_LAYER} — "
              f"{pos_rel}(1) n={len(pos_rows)}  {neg_rel}(0) n={len(neg_rows)}")
        print(f"  Fixed question: '{BINARY_Q}' (identical for both classes)")

        X_probe: list[np.ndarray] = []
        y_probe: list[int] = []

        for label_val, rows in ((1, pos_rows), (0, neg_rows)):
            for row in rows:
                img_path = args.images_root / Path(row["output_file"]).name
                if not img_path.exists():
                    img_path = args.images_root.parent / row["output_file"]
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                act = collect_activation(
                    model, processor, collector, image, BINARY_Q, PROBE_LAYER,
                    probe_token=args.probe_token, image_token_id=image_token_id,
                )
                X_probe.append(act)
                y_probe.append(label_val)

        probe_acc = None
        if len(set(y_probe)) == 2 and len(X_probe) >= 4:
            X_np = np.array(X_probe, dtype=np.float32)
            y_np = np.array(y_probe)
            probe_acc = probe_accuracy(X_np, y_np)
            print(f"\n  Layer-{PROBE_LAYER} BINARY probe CV accuracy: {probe_acc:.4f}")
            print(f"  Chance baseline = 0.500")
        else:
            print("  Not enough samples for binary probe.")

        # ── binary layer sweep ──────────────────────────────────────────────
        sweep_n = args.sweep_rows_per_class
        pos_sweep = [r for r in all_rows
                     if r.get("true_relation") == pos_rel and r.get("label") == 1][:sweep_n]
        neg_sweep = [r for r in all_rows
                     if r.get("true_relation") == neg_rel and r.get("label") == 1][:sweep_n]

        print(f"\n[Phase 3] BINARY layer sweep ({sweep_n} per class, "
              f"all {num_layers} layers — single-pass, VISUAL-ONLY)")
        print(f"  Fixed question: '{BINARY_Q}'  |  pair: {pos_rel}(1) vs {neg_rel}(0)")

        all_layer_acts: dict[int, list] = {i: [] for i in range(num_layers)}
        sweep_y: list[int] = []

        all_sweep_rows = [(1, r) for r in pos_sweep] + [(0, r) for r in neg_sweep]
        total_sweep = len(all_sweep_rows)
        done = 0
        for label_val, row in all_sweep_rows:
            img_path = args.images_root / Path(row["output_file"]).name
            if not img_path.exists():
                img_path = args.images_root.parent / row["output_file"]
            if not img_path.exists():
                continue

            image = Image.open(img_path).convert("RGB")
            collector.attach()
            collector.activations = {}
            prompt = build_prompt(BINARY_Q)
            inputs = processor(
                text=prompt, images=[image], return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)

            for layer_idx in range(num_layers):
                try:
                    if args.probe_token == "image-mean":
                        act = collector.get_image_token_mean_activation(
                            layer_idx, inputs["input_ids"], image_token_id
                        )
                    else:
                        act = collector.get_last_token_activation(layer_idx)
                    all_layer_acts[layer_idx].append(act)
                except (KeyError, ValueError):
                    pass
            collector.detach()
            sweep_y.append(label_val)
            done += 1
            if done % 20 == 0 or done == total_sweep:
                print(f"  collected {done}/{total_sweep} sweep examples")

        # Fit per-layer binary probe
        layer_probe_accs: dict[int, float | None] = {}
        for layer_idx in range(num_layers):
            acts = all_layer_acts[layer_idx]
            yl = sweep_y[:len(acts)]
            if len(acts) >= 4 and len(set(yl)) == 2:
                layer_probe_accs[layer_idx] = probe_accuracy(
                    np.array(acts, dtype=np.float32), np.array(yl)
                )
            else:
                layer_probe_accs[layer_idx] = None

            if (layer_idx + 1) % 7 == 0 or layer_idx == num_layers - 1:
                acc_str = (f"{layer_probe_accs[layer_idx]:.3f}"
                           if layer_probe_accs[layer_idx] is not None else "N/A")
                print(f"  Layer {layer_idx:2d}: probe acc = {acc_str}")

    else:
        # ── original 4-class relation probe ──────────────────────────────────
        print(f"\n[Phase 2] Activation probing at layer {PROBE_LAYER} "
              f"(up to {args.max_probe_per_class} per class"
              f"{', BLIND mode' if args.blind_probe else ''})")

        X_probe: list[np.ndarray] = []
        y_probe: list[str] = []

        for rel in RELATIONS:
            rows_for_rel = probe_by_rel[rel]
            print(f"  Collecting {len(rows_for_rel)} activations for '{rel}' ...")
            for row in rows_for_rel:
                img_path = args.images_root / Path(row["output_file"]).name
                if not img_path.exists():
                    img_path = args.images_root.parent / row["output_file"]
                if not img_path.exists():
                    continue
                image = Image.open(img_path).convert("RGB")
                question = blind_question(row) if args.blind_probe else row["question"]
                act = collect_activation(
                    model, processor, collector, image, question, PROBE_LAYER,
                    probe_token=args.probe_token, image_token_id=image_token_id,
                )
                X_probe.append(act)
                y_probe.append(rel)

        probe_acc = None
        if len(set(y_probe)) == len(RELATIONS) and len(X_probe) >= len(RELATIONS) * 2:
            X_np = np.array(X_probe, dtype=np.float32)
            y_np = np.array(y_probe)
            probe_acc = probe_accuracy(X_np, y_np)
            print(f"\n  Layer-{PROBE_LAYER} probe train accuracy (real data): {probe_acc:.4f}")
            print(f"  (Day4/5 synthetic benchmark: ~0.95-1.00 expected)")
        else:
            print("  Not enough samples to fit probe.")

        # ── layer sweep probe ─────────────────────────────────────────────────
        sweep_rows_per_class = args.sweep_rows_per_class
        print(f"\n[Phase 3] Layer sweep probe ({sweep_rows_per_class} rows/class, "
              f"all {num_layers} layers — single-pass collection"
              f"{', BLIND mode' if args.blind_probe else ''})")
        sweep_by_rel = {r: pos_by_rel[r][:sweep_rows_per_class] for r in RELATIONS}

        all_layer_acts: dict[int, list] = {i: [] for i in range(num_layers)}

        total_sweep = sum(len(v) for v in sweep_by_rel.values())
        done = 0
        for rel in RELATIONS:
            for row in sweep_by_rel[rel]:
                img_path = args.images_root / Path(row["output_file"]).name
                if not img_path.exists():
                    img_path = args.images_root.parent / row["output_file"]
                if not img_path.exists():
                    continue

                image = Image.open(img_path).convert("RGB")
                question = blind_question(row) if args.blind_probe else row["question"]
                collector.attach()
                collector.activations = {}
                prompt = build_prompt(question)
                inputs = processor(
                    text=prompt, images=[image], return_tensors="pt"
                ).to(model.device)
                with torch.no_grad():
                    model.generate(**inputs, max_new_tokens=1)

                for layer_idx in range(num_layers):
                    try:
                        if args.probe_token == "image-mean":
                            act = collector.get_image_token_mean_activation(
                                layer_idx, inputs["input_ids"], image_token_id
                            )
                        else:
                            act = collector.get_last_token_activation(layer_idx)
                        all_layer_acts[layer_idx].append(act)
                    except (KeyError, ValueError):
                        pass
                collector.detach()
                done += 1
                if done % 20 == 0 or done == total_sweep:
                    print(f"  collected {done}/{total_sweep} sweep examples")

        layer_probe_accs: dict[int, float | None] = {}
        for layer_idx in range(num_layers):
            acts = all_layer_acts[layer_idx]
            yl = []
            for rel in RELATIONS:
                yl.extend([rel] * len(sweep_by_rel[rel]))
            yl = yl[:len(acts)]

            if len(acts) >= len(RELATIONS) * 2 and len(set(yl)) == len(RELATIONS):
                layer_probe_accs[layer_idx] = probe_accuracy(
                    np.array(acts, dtype=np.float32), np.array(yl)
                )
            else:
                layer_probe_accs[layer_idx] = None

            if (layer_idx + 1) % 7 == 0 or layer_idx == num_layers - 1:
                acc_str = (f"{layer_probe_accs[layer_idx]:.3f}"
                           if layer_probe_accs[layer_idx] is not None else "N/A")
                print(f"  Layer {layer_idx:2d}: probe acc = {acc_str}")

    valid_layers = [k for k, v in layer_probe_accs.items() if v is not None]
    if valid_layers:
        best_layer = max(valid_layers, key=lambda k: layer_probe_accs[k])
        print(f"\n  Best probe layer on real data: {best_layer} "
              f"(acc={layer_probe_accs[best_layer]:.4f})")
        print(f"  Day4/5 causal peak was layer {PROBE_LAYER} — "
              + ("MATCHES" if best_layer == PROBE_LAYER else f"DIFFERS (now layer {best_layer})"))
    else:
        print("\n  No valid probe layers found (all layers returned N/A — not enough samples?)")

    # ── save results ─────────────────────────────────────────────────────────
    results = {
        "experiment": "exp13_real_data_baseline",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "eval_rows": len(eval_rows),
        "overall_accuracy": overall_acc,
        "per_relation_accuracy": per_rel_acc,
        "probe_layer": PROBE_LAYER,
        "probe_train_accuracy_real": probe_acc,
        "layer_probe_sweep": {str(k): v for k, v in layer_probe_accs.items()},
        "best_probe_layer_real": best_layer if valid_layers else None,
        "probe_mode": (f"binary_{args.binary_rel.replace(',', '_vs_')}" if args.binary_probe
                       else ("blind_4class" if args.blind_probe else "standard_4class")),
        "params": {
            "max_eval": args.max_eval,
            "max_probe_per_class": args.max_probe_per_class,
            "seed": args.seed,
        },
    }

    out_path = args.out_dir / "results_exp13.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved: {out_path}")
    print("=" * 60)
    print("EXPERIMENT 13 COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
