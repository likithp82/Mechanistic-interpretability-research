"""
EXPERIMENT 8: Cross-Image Generalization
==========================================
Day 4 of Kill-Switch Pilot

WHAT: Train the relation direction on ONE set of stimuli (red circle + blue square)
      and test on COMPLETELY DIFFERENT stimuli (green triangle + orange diamond).
      If the direction generalizes, it's about SPATIAL RELATIONS, not about
      specific object appearances.

METHOD:
  1. Train: Collect above/below activations using red circle + blue square
  2. Fit logistic regression to find the "above/below" direction
  3. Test: Collect activations using green triangle + orange diamond
  4. Apply the SAME direction — does it still classify correctly?
  5. Also test with different sizes, backgrounds, and object counts

SUCCESS CRITERIA:
  - Cross-stimulus accuracy > 70% (direction is about relations, not objects)
  - Cross-stimulus accuracy should be close to within-stimulus accuracy

RUN: python exp8_cross_generalization.py
"""

import sys
import time
import json
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day1"))

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import (
    ActivationCollector,
    create_two_object_image,
    format_prompt,
)

# ── Setup ───────────────────────────────────────────────────────

print("=" * 60)
print("EXPERIMENT 8: Cross-Image Generalization")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

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

target_layer = 3 * num_layers // 4  # Layer 21
collector = ActivationCollector(model, layer_indices=[target_layer])


# ── Helper: Create Different Stimulus Sets ──────────────────────

def create_novel_objects_image(relation, obj1_type="triangle", obj2_type="diamond",
                                obj1_color="green", obj2_color="orange", size=448):
    """Completely different shapes and colors from training set."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    obj_size = size // 6
    margin = size // 6
    center = size // 2

    if relation == "above":
        x1, y1 = center, margin + obj_size
        x2, y2 = center, size - margin - obj_size
    elif relation == "below":
        x1, y1 = center, size - margin - obj_size
        x2, y2 = center, margin + obj_size
    elif relation == "left":
        x1, y1 = margin + obj_size, center
        x2, y2 = size - margin - obj_size, center
    elif relation == "right":
        x1, y1 = size - margin - obj_size, center
        x2, y2 = margin + obj_size, center

    # Obj1: triangle
    if obj1_type == "triangle":
        points = [(x1, y1 - obj_size), (x1 - obj_size, y1 + obj_size), (x1 + obj_size, y1 + obj_size)]
        draw.polygon(points, fill=obj1_color, outline="black", width=2)
    else:
        draw.ellipse([x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
                     fill=obj1_color, outline="black", width=2)

    # Obj2: diamond
    if obj2_type == "diamond":
        points = [(x2, y2 - obj_size), (x2 + obj_size, y2), (x2, y2 + obj_size), (x2 - obj_size, y2)]
        draw.polygon(points, fill=obj2_color, outline="black", width=2)
    else:
        draw.rectangle([x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
                       fill=obj2_color, outline="black", width=2)

    return img


def create_small_objects_image(relation, size=448):
    """Same concept but with very small objects."""
    img = Image.new("RGB", (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    obj_size = size // 12  # Much smaller
    margin = size // 5
    center = size // 2

    if relation == "above":
        x1, y1 = center, margin + obj_size
        x2, y2 = center, size - margin - obj_size
    elif relation == "below":
        x1, y1 = center, size - margin - obj_size
        x2, y2 = center, margin + obj_size
    elif relation == "left":
        x1, y1 = margin + obj_size, center
        x2, y2 = size - margin - obj_size, center
    elif relation == "right":
        x1, y1 = size - margin - obj_size, center
        x2, y2 = margin + obj_size, center

    draw.ellipse([x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
                 fill="purple", outline="darkviolet", width=2)
    draw.rectangle([x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
                   fill="brown", outline="saddlebrown", width=2)

    return img


def create_three_objects_image(relation, size=448):
    """Three objects — the question asks about two specific ones, third is distractor."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    obj_size = size // 7
    margin = size // 6
    center = size // 2

    if relation == "above":
        x1, y1 = center, margin + obj_size
        x2, y2 = center, size - margin - obj_size
    elif relation == "below":
        x1, y1 = center, size - margin - obj_size
        x2, y2 = center, margin + obj_size
    elif relation == "left":
        x1, y1 = margin + obj_size, center
        x2, y2 = size - margin - obj_size, center
    elif relation == "right":
        x1, y1 = size - margin - obj_size, center
        x2, y2 = margin + obj_size, center

    # Target objects
    draw.ellipse([x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
                 fill="red", outline="darkred", width=2)
    draw.rectangle([x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
                   fill="blue", outline="darkblue", width=2)

    # Distractor (random position, gray) — offset from main objects
    if relation in ("above", "below"):
        dx = center + random.choice([-1, 1]) * random.randint(size // 4, size // 3)
        dy = random.randint(size // 4, 3 * size // 4)
    else:
        dx = random.randint(size // 4, 3 * size // 4)
        dy = center + random.choice([-1, 1]) * random.randint(size // 4, size // 3)
    dx = max(obj_size, min(size - obj_size, dx))
    dy = max(obj_size, min(size - obj_size, dy))
    draw.polygon([(dx, dy - obj_size // 2), (dx + obj_size // 2, dy + obj_size // 2),
                  (dx - obj_size // 2, dy + obj_size // 2)],
                 fill="gray", outline="darkgray", width=2)

    return img


# ── Step 1: Train Direction on Standard Stimuli ─────────────────

print("=" * 60)
print("STEP 1: Train Direction on Standard Stimuli (Red Circle + Blue Square)")
print("=" * 60 + "\n")

N_TRAIN = 20
train_above_acts = []
train_below_acts = []

COLOR_PAIRS = [("red", "blue"), ("red", "blue"), ("red", "blue"),
               ("red", "blue"), ("red", "blue")]  # All same for training

print(f"Collecting {N_TRAIN} above + {N_TRAIN} below (red circle / blue square)...")

for i in range(N_TRAIN):
    # Above
    img = create_two_object_image(relation="above", obj1_color="red", obj2_color="blue")
    question = "Is the red circle above or below the blue square?"
    collector.attach()
    collector.activations = {}
    inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    train_above_acts.append(collector.get_last_token_activation(target_layer))
    collector.detach()

    # Below
    img = create_two_object_image(relation="below", obj1_color="red", obj2_color="blue")
    collector.attach()
    collector.activations = {}
    inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    train_below_acts.append(collector.get_last_token_activation(target_layer))
    collector.detach()

print("  Done.")

# Fit classifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

X_train = np.concatenate([np.stack(train_above_acts), np.stack(train_below_acts)])
y_train = np.array([1] * N_TRAIN + [0] * N_TRAIN)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

clf = LogisticRegression(max_iter=2000, C=1.0)
clf.fit(X_train_scaled, y_train)
train_acc = clf.score(X_train_scaled, y_train)
print(f"  Training accuracy: {train_acc:.3f}")

# ── Step 2: Test on Different Stimulus Sets ─────────────────────

print("\n" + "=" * 60)
print("STEP 2: Test Generalization Across Different Stimuli")
print("=" * 60 + "\n")

N_TEST = 15

test_sets = {
    "novel_shapes": {
        "description": "Green triangle + orange diamond",
        "create_fn": lambda rel: create_novel_objects_image(rel),
        "question": "Is the green triangle above or below the orange diamond?",
    },
    "small_objects": {
        "description": "Small purple circle + brown square",
        "create_fn": lambda rel: create_small_objects_image(rel),
        "question": "Is the purple circle above or below the brown square?",
    },
    "three_objects": {
        "description": "Red circle + blue square + gray distractor",
        "create_fn": lambda rel: create_three_objects_image(rel),
        "question": "Is the red circle above or below the blue square?",
    },
    "swapped_colors": {
        "description": "Blue circle + red square (swapped from training)",
        "create_fn": lambda rel: create_two_object_image(relation=rel, obj1_color="blue", obj2_color="red"),
        "question": "Is the blue circle above or below the red square?",
    },
}

generalization_results = {}

for test_name, test_config in test_sets.items():
    print(f"--- {test_name}: {test_config['description']} ---")

    test_above_acts = []
    test_below_acts = []

    for i in range(N_TEST):
        # Above
        img = test_config["create_fn"]("above")
        question = test_config["question"]
        collector.attach()
        collector.activations = {}
        inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        test_above_acts.append(collector.get_last_token_activation(target_layer))
        collector.detach()

        # Below
        img = test_config["create_fn"]("below")
        collector.attach()
        collector.activations = {}
        inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        test_below_acts.append(collector.get_last_token_activation(target_layer))
        collector.detach()

    # Test with the classifier trained on red/blue
    X_test = np.concatenate([np.stack(test_above_acts), np.stack(test_below_acts)])
    y_test = np.array([1] * N_TEST + [0] * N_TEST)
    X_test_scaled = scaler.transform(X_test)

    test_acc = clf.score(X_test_scaled, y_test)
    predictions = clf.predict(X_test_scaled)

    # Per-class accuracy
    above_correct = sum(predictions[:N_TEST] == 1)
    below_correct = sum(predictions[N_TEST:] == 0)

    generalization_results[test_name] = {
        "description": test_config["description"],
        "accuracy": float(test_acc),
        "above_correct": int(above_correct),
        "below_correct": int(below_correct),
        "n_test": N_TEST,
    }

    print(f"  Accuracy: {test_acc:.3f} (above: {above_correct}/{N_TEST}, below: {below_correct}/{N_TEST})")
    print()

# ── Step 3: Also Test 4-Way (Above/Below/Left/Right) ───────────

print("=" * 60)
print("STEP 3: 4-Way Generalization (train on red/blue, test on novel)")
print("=" * 60 + "\n")

# Train 4-way on standard stimuli
RELATIONS = ["above", "below", "left", "right"]
train_4way_acts = {r: [] for r in RELATIONS}

for rel in RELATIONS:
    for i in range(N_TRAIN):
        img = create_two_object_image(relation=rel, obj1_color="red", obj2_color="blue")
        question = "What is the spatial relationship between the red circle and the blue square?"
        collector.attach()
        collector.activations = {}
        inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        train_4way_acts[rel].append(collector.get_last_token_activation(target_layer))
        collector.detach()

X_train_4 = []
y_train_4 = []
for i, rel in enumerate(RELATIONS):
    for act in train_4way_acts[rel]:
        X_train_4.append(act)
        y_train_4.append(i)
X_train_4 = np.array(X_train_4)
y_train_4 = np.array(y_train_4)

scaler_4 = StandardScaler()
X_train_4_scaled = scaler_4.fit_transform(X_train_4)
clf_4 = LogisticRegression(max_iter=2000, C=0.1)
clf_4.fit(X_train_4_scaled, y_train_4)
train_4_acc = clf_4.score(X_train_4_scaled, y_train_4)
print(f"4-way training accuracy: {train_4_acc:.3f}")

# Test on novel shapes
test_4way_acts = {r: [] for r in RELATIONS}
for rel in RELATIONS:
    for i in range(N_TEST):
        img = create_novel_objects_image(rel)
        question = "What is the spatial relationship between the green triangle and the orange diamond?"
        collector.attach()
        collector.activations = {}
        inputs = processor(text=format_prompt(question), images=[img], return_tensors="pt").to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=1)
        test_4way_acts[rel].append(collector.get_last_token_activation(target_layer))
        collector.detach()

X_test_4 = []
y_test_4 = []
for i, rel in enumerate(RELATIONS):
    for act in test_4way_acts[rel]:
        X_test_4.append(act)
        y_test_4.append(i)
X_test_4 = np.array(X_test_4)
y_test_4 = np.array(y_test_4)

X_test_4_scaled = scaler_4.transform(X_test_4)
test_4_acc = clf_4.score(X_test_4_scaled, y_test_4)
predictions_4 = clf_4.predict(X_test_4_scaled)

print(f"4-way cross-stimulus accuracy: {test_4_acc:.3f} (chance=0.25)")
print(f"\nPer-relation accuracy (test on novel shapes):")
for i, rel in enumerate(RELATIONS):
    mask = y_test_4 == i
    rel_acc = (predictions_4[mask] == i).mean()
    print(f"  {rel}: {rel_acc:.3f}")

generalization_results["four_way_novel"] = {
    "description": "4-way on green triangle + orange diamond",
    "train_accuracy": float(train_4_acc),
    "test_accuracy": float(test_4_acc),
}

# ── Evaluation ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60 + "\n")

# Key metric: does the direction generalize?
novel_acc = generalization_results["novel_shapes"]["accuracy"]
small_acc = generalization_results["small_objects"]["accuracy"]
three_acc = generalization_results["three_objects"]["accuracy"]
swapped_acc = generalization_results["swapped_colors"]["accuracy"]

avg_generalization = np.mean([novel_acc, small_acc, three_acc, swapped_acc])

results = {}
results["novel_shapes_above_70"] = novel_acc > 0.70
results["avg_generalization_above_70"] = avg_generalization > 0.70
results["four_way_above_40"] = test_4_acc > 0.40

print(f"  Within-stimulus training accuracy: {train_acc:.3f}")
print(f"  Cross-stimulus accuracies:")
print(f"    Novel shapes:    {novel_acc:.3f}")
print(f"    Small objects:   {small_acc:.3f}")
print(f"    Three objects:   {three_acc:.3f}")
print(f"    Swapped colors:  {swapped_acc:.3f}")
print(f"  Average cross-generalization: {avg_generalization:.3f}")
print(f"  4-way cross-generalization:  {test_4_acc:.3f}")
print()
print(f"  [{'PASS' if results['novel_shapes_above_70'] else 'FAIL'}] "
      f"Novel shapes > 70%: {novel_acc:.3f}")
print(f"  [{'PASS' if results['avg_generalization_above_70'] else 'FAIL'}] "
      f"Average generalization > 70%: {avg_generalization:.3f}")
print(f"  [{'PASS' if results['four_way_above_40'] else 'FAIL'}] "
      f"4-way cross > 40%: {test_4_acc:.3f}")

all_pass = all(results.values())
print()
print("=" * 60)
if all_pass:
    print("RESULT: DIRECTION GENERALIZES")
    print("The spatial relation direction is about RELATIONS, not specific objects.")
    print("It transfers across shapes, colors, sizes, and with distractors.")
    print("This confirms it's a genuine spatial reasoning feature.")
elif avg_generalization > 0.60:
    print("RESULT: PARTIAL GENERALIZATION")
    print("The direction partially transfers but is somewhat stimulus-specific.")
    print("May need more diverse training data for robust directions.")
else:
    print("RESULT: DIRECTION IS STIMULUS-SPECIFIC")
    print("The direction learned on red/blue doesn't transfer to other stimuli.")
    print("The model may use different features for different object pairs.")
print("=" * 60)

# ── Save ────────────────────────────────────────────────────────

output = {
    "experiment": "exp8_cross_generalization",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "target_layer": target_layer,
    "n_train": N_TRAIN,
    "n_test": N_TEST,
    "train_accuracy": float(train_acc),
    "generalization_results": generalization_results,
    "avg_generalization": float(avg_generalization),
    "criteria": results,
    "all_pass": all_pass,
}

with open("results_exp8.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to results_exp8.json")
print("Done.")
