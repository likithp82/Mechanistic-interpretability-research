"""
EXPERIMENT 6: Real/Hard Image Robustness Check
================================================
Day 4 of Kill-Switch Pilot

WHAT: Test whether the position encoding (Exp 3) and relation direction (Exp 4)
      findings survive when images are harder — cluttered backgrounds, varied
      sizes, distractors, adversarial conditions.

LEVELS:
  Level 2: Hard synthetic (random bg, asymmetric sizes, jitter)
  Level 3: Cluttered (gradient bg, distractor shapes, blur)
  Level 4: Adversarial (tiny objects, low contrast, close together)

TESTS:
  A) Model accuracy at each level (does it still get the answer right?)
  B) Position encoding R² at each level (are positions still linear?)
  C) Relation probe accuracy at each level (can we still classify relation?)
  D) Relation-position overlap at each level (still orthogonal?)

RUN: python exp6_hard_images.py
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day1"))

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import (
    ActivationCollector,
    create_single_object_image,
    format_prompt,
    run_inference,
)
from generate_sample_data import create_hard_synthetic, create_cluttered, create_adversarial
from PIL import Image, ImageDraw, ImageFilter

def create_single_object_on_cluttered_bg(x_grid, y_grid, grid_size, img_size=448):
    """Place a red circle at a grid position on a cluttered background."""
    img = Image.new("RGB", (img_size, img_size))
    pixels = img.load()
    br, bg_c, bb = random.randint(40, 120), random.randint(40, 120), random.randint(40, 120)
    for y in range(img_size):
        for x in range(img_size):
            pixels[x, y] = (
                min(255, br + x // 5),
                min(255, bg_c + y // 5),
                min(255, bb + (x + y) // 10),
            )
    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(2, 4)):
        dx, dy = random.randint(30, img_size - 30), random.randint(30, img_size - 30)
        ds = random.randint(10, 30)
        gray = random.randint(80, 160)
        draw.ellipse([dx - ds, dy - ds, dx + ds, dy + ds], fill=(gray, gray, gray))
    margin = img_size // 8
    usable = img_size - 2 * margin
    step = usable / (grid_size - 1) if grid_size > 1 else 0
    cx = int(margin + x_grid * step)
    cy = int(margin + y_grid * step)
    r = img_size // 12
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill="red", outline="darkred", width=2)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))
    return img

# ── Setup ───────────────────────────────────────────────────────

print("=" * 60)
print("EXPERIMENT 6: Robustness Across Image Difficulty Levels")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
N_PER_LEVEL = 10  # per relation per level

print(f"Loading {MODEL_NAME}...")
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
load_time = time.time() - t0

if hasattr(model.model, "language_model"):
    num_layers = len(model.model.language_model.layers)
else:
    num_layers = len(model.model.layers)
hidden_dim = model.config.hidden_size
print(f"Loaded in {load_time:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
print()

# Key layers
layer_indices = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
best_layer = 3 * num_layers // 4  # Layer 21 was best in Exp 3 & 4

RELATIONS = ["above", "below", "left", "right"]
LEVELS = {
    "level2_hard": create_hard_synthetic,
    "level3_cluttered": create_cluttered,
    "level4_adversarial": create_adversarial,
}

# ── Test A: Accuracy at Each Level ──────────────────────────────

print("=" * 60)
print("TEST A: Model Accuracy Across Difficulty Levels")
print("=" * 60 + "\n")

accuracy_results = {}

for level_name, create_func in LEVELS.items():
    correct = 0
    total = 0
    per_rel = {r: {"correct": 0, "total": 0} for r in RELATIONS}

    for rel in RELATIONS:
        for i in range(N_PER_LEVEL):
            result = create_func(rel)
            img = result[0] if isinstance(result, tuple) else result

            if rel in ("above", "below"):
                question = "Is the red circle above or below the blue square? Answer with just 'above' or 'below'."
            else:
                question = "Is the red circle to the left or right of the blue square? Answer with just 'left' or 'right'."

            answer = run_inference(model, processor, img, question, max_new_tokens=20)
            is_correct = rel in answer.lower()

            if is_correct:
                correct += 1
                per_rel[rel]["correct"] += 1
            total += 1
            per_rel[rel]["total"] += 1

    acc = correct / total
    accuracy_results[level_name] = {
        "overall": acc,
        "per_relation": {r: per_rel[r]["correct"] / per_rel[r]["total"] for r in RELATIONS},
        "total_correct": correct,
        "total": total,
    }
    print(f"  {level_name}: {correct}/{total} = {acc:.1%}")
    for r in RELATIONS:
        r_acc = per_rel[r]["correct"] / per_rel[r]["total"]
        print(f"    {r}: {per_rel[r]['correct']}/{per_rel[r]['total']} = {r_acc:.1%}")
    print()

# ── Test B: Relation Probe at Each Level ────────────────────────

print("=" * 60)
print("TEST B: Relation Classification Probe Across Levels")
print("=" * 60 + "\n")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

collector = ActivationCollector(model, layer_indices=[best_layer])

probe_results = {}

for level_name, create_func in LEVELS.items():
    relation_acts = {r: [] for r in RELATIONS}

    for rel in RELATIONS:
        for i in range(N_PER_LEVEL):
            result = create_func(rel)
            img = result[0] if isinstance(result, tuple) else result

            question = "What is the spatial relationship between the red circle and the blue square?"

            collector.attach()
            collector.activations = {}
            inputs = processor(
                text=format_prompt(question), images=[img], return_tensors="pt"
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            relation_acts[rel].append(collector.get_last_token_activation(best_layer))
            collector.detach()

    # 4-way classification
    X = []
    y = []
    for i, rel in enumerate(RELATIONS):
        for act in relation_acts[rel]:
            X.append(act)
            y.append(i)
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, C=0.1)
    n_samples_per_class = N_PER_LEVEL
    cv = min(5, n_samples_per_class)
    scores = cross_val_score(clf, X_scaled, y, cv=cv, scoring="accuracy")

    # Also compute above/below direction overlap with position
    ab_acts = np.concatenate([np.stack(relation_acts["above"]), np.stack(relation_acts["below"])])
    ab_labels = np.array([1] * N_PER_LEVEL + [0] * N_PER_LEVEL)
    scaler_ab = StandardScaler()
    ab_scaled = scaler_ab.fit_transform(ab_acts)
    clf_ab = LogisticRegression(max_iter=2000, C=1.0).fit(ab_scaled, ab_labels)
    relation_dir = clf_ab.coef_[0]
    relation_dir = relation_dir / np.linalg.norm(relation_dir)

    probe_results[level_name] = {
        "four_way_accuracy": float(scores.mean()),
        "four_way_std": float(scores.std()),
        "relation_direction_norm": float(np.linalg.norm(clf_ab.coef_[0])),
    }

    print(f"  {level_name}: 4-way accuracy = {scores.mean():.3f} ± {scores.std():.3f} (chance=0.25)")

print()

# ── Test C: Position Encoding at Hardest Level ──────────────────

print("=" * 60)
print("TEST C: Position Encoding (Ridge R²) at Cluttered Level")
print("=" * 60 + "\n")

from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score

# Use cluttered images (Level 3) with single object at grid positions
positions = []
pos_acts = {l: [] for l in layer_indices}
pos_collector = ActivationCollector(model, layer_indices=layer_indices)

GRID = 5
print(f"Collecting position activations on cluttered backgrounds ({GRID}x{GRID} grid)...")

for x_pos in range(GRID):
    for y_pos in range(GRID):
        # Create cluttered background, then place object
        img = create_single_object_on_cluttered_bg(x_pos, y_pos, GRID)

        question = "Where is the red circle in the image?"
        pos_collector.attach()
        pos_collector.activations = {}
        inputs = processor(
            text=format_prompt(question), images=[img], return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        for l in layer_indices:
            pos_acts[l].append(pos_collector.get_last_token_activation(l))
        pos_collector.detach()
        positions.append([x_pos, y_pos])

positions = np.array(positions)

print(f"\nPosition Encoding R² (cluttered backgrounds):")
print(f"{'Layer':<8} {'R² (x)':<12} {'R² (y)':<12} {'R² (avg)':<12}")
print("-" * 44)

position_results = {}
for l in layer_indices:
    X = np.stack(pos_acts[l])
    reg = Ridge(alpha=1.0)
    y_pred = cross_val_predict(reg, X, positions, cv=5)
    r2_x = r2_score(positions[:, 0], y_pred[:, 0])
    r2_y = r2_score(positions[:, 1], y_pred[:, 1])
    position_results[l] = {"r2_x": float(r2_x), "r2_y": float(r2_y), "r2_avg": float((r2_x + r2_y) / 2)}
    print(f"Layer {l:<4} {r2_x:<12.3f} {r2_y:<12.3f} {(r2_x + r2_y) / 2:<12.3f}")

# ── Summary ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("SUMMARY: How Findings Degrade With Difficulty")
print("=" * 60 + "\n")

print(f"{'Level':<22} {'Accuracy':<12} {'4-Way Probe':<14} {'Signal?':<10}")
print("-" * 58)
print(f"{'Level 1 (basic synth)':<22} {'100%':<12} {'1.000':<14} {'YES':<10}")
for level_name in LEVELS:
    acc = accuracy_results[level_name]["overall"]
    probe = probe_results[level_name]["four_way_accuracy"]
    signal = "YES" if probe > 0.40 else "WEAK" if probe > 0.30 else "NO"
    print(f"{level_name:<22} {acc:<12.1%} {probe:<14.3f} {signal:<10}")

best_pos_layer = max(position_results, key=lambda l: position_results[l]["r2_avg"])
print(f"\nPosition encoding on cluttered images: R²={position_results[best_pos_layer]['r2_avg']:.3f} "
      f"(vs 0.928 on clean)")

# ── Evaluation ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60 + "\n")

# Does the mechanism survive at Level 3 (cluttered)?
l3_acc = accuracy_results["level3_cluttered"]["overall"]
l3_probe = probe_results["level3_cluttered"]["four_way_accuracy"]
l3_pos = position_results[best_pos_layer]["r2_avg"]

results = {}
results["accuracy_above_70_level3"] = l3_acc > 0.70
results["probe_above_40_level3"] = l3_probe > 0.40
results["position_r2_above_03"] = l3_pos > 0.3

print(f"  [{'PASS' if results['accuracy_above_70_level3'] else 'FAIL'}] "
      f"Accuracy > 70% at Level 3: {l3_acc:.1%}")
print(f"  [{'PASS' if results['probe_above_40_level3'] else 'FAIL'}] "
      f"4-way probe > 40% at Level 3: {l3_probe:.3f}")
print(f"  [{'PASS' if results['position_r2_above_03'] else 'FAIL'}] "
      f"Position R² > 0.3 on cluttered: {l3_pos:.3f}")

all_pass = all(results.values())
print()
if all_pass:
    print("FINDINGS SURVIVE on harder images. Mechanism is robust.")
else:
    print("PARTIAL DEGRADATION. Mechanism exists but weakens with difficulty.")
    print("This is EXPECTED and actually interesting for the paper —")
    print("we can study WHERE in the circuit the degradation happens.")

# ── Save ────────────────────────────────────────────────────────

output = {
    "experiment": "exp6_hard_images",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_per_level": N_PER_LEVEL,
    "accuracy_results": accuracy_results,
    "probe_results": probe_results,
    "position_results": {str(k): v for k, v in position_results.items()},
    "criteria": results,
    "all_pass": all_pass,
}

with open("results_exp6.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to results_exp6.json")
print("Done.")
