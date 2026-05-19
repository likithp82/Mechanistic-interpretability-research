"""
EXPERIMENT 1B: Activation Statistics & Spatial Signal Check
===========================================================
Day 1 (after exp1 passes)

WHAT: Compare activations across multiple spatial questions to see
      if there's any signal before running the full probe (Day 2).
      This is a quick directional check, not a rigorous experiment.

RUN AFTER: exp1_activation_hooks.py passes
RUN: python exp1b_activation_stats.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("EXPERIMENT 1B: Activation Statistics Across Spatial Tasks")
print("=" * 60)
print()

# ── Load Model ──────────────────────────────────────────────────

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import (
    ActivationCollector,
    create_two_object_image,
    create_random_two_object_image,
    format_prompt,
    run_inference_with_activations,
)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
num_layers = len(model.model.language_model.layers)
hidden_dim = model.config.hidden_size
print(f"Loaded. {num_layers} layers, hidden_dim={hidden_dim}")
print()

# ── Collect Activations: Spatial vs Non-Spatial ─────────────────

# We pick 3 representative layers: early, middle, late
layer_indices = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
print(f"Hooking layers: {layer_indices}")

collector = ActivationCollector(model, layer_indices=layer_indices)

# Create 10 quick test cases
N = 10
spatial_acts = {l: [] for l in layer_indices}
nonspatial_acts = {l: [] for l in layer_indices}

print(f"\nRunning {N} paired spatial/non-spatial queries...")

for i in range(N):
    img, obj1, obj2, vert_rel, horiz_rel = create_random_two_object_image()

    # Spatial question
    spatial_q = f"Is the {obj1['color']} circle above or below the {obj2['color']} square?"
    collector.attach()
    collector.activations = {}
    inputs = processor(
        text=format_prompt(spatial_q), images=[img], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    for l in layer_indices:
        spatial_acts[l].append(collector.get_last_token_activation(l))
    collector.detach()

    # Non-spatial question (same image)
    nonspatial_q = f"What color is the circle in this image?"
    collector.attach()
    collector.activations = {}
    inputs = processor(
        text=format_prompt(nonspatial_q), images=[img], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    for l in layer_indices:
        nonspatial_acts[l].append(collector.get_last_token_activation(l))
    collector.detach()

    print(f"  Pair {i+1}/{N} done")

# ── Analyze ─────────────────────────────────────────────────────

print("\n--- Activation Comparison: Spatial vs Non-Spatial ---\n")
print(f"{'Layer':<10} {'Cos Sim':<12} {'L2 Dist':<12} {'Spatial Norm':<14} {'NonSpatial Norm':<16}")
print("-" * 64)

analysis = {}
for l in layer_indices:
    s_acts = np.stack(spatial_acts[l])      # [N, hidden_dim]
    ns_acts = np.stack(nonspatial_acts[l])   # [N, hidden_dim]

    # Mean activation vectors
    s_mean = s_acts.mean(axis=0)
    ns_mean = ns_acts.mean(axis=0)

    # Cosine similarity between mean spatial and mean non-spatial
    cos_sim = np.dot(s_mean, ns_mean) / (np.linalg.norm(s_mean) * np.linalg.norm(ns_mean) + 1e-8)

    # L2 distance
    l2_dist = np.linalg.norm(s_mean - ns_mean)

    # Norms
    s_norm = np.linalg.norm(s_mean)
    ns_norm = np.linalg.norm(ns_mean)

    analysis[l] = {
        "cosine_similarity": float(cos_sim),
        "l2_distance": float(l2_dist),
        "spatial_norm": float(s_norm),
        "nonspatial_norm": float(ns_norm),
    }

    print(f"Layer {l:<5} {cos_sim:<12.4f} {l2_dist:<12.2f} {s_norm:<14.2f} {ns_norm:<16.2f}")

# ── Quick Classification Check ──────────────────────────────────

print("\n--- Quick Linear Separability Check ---\n")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    for l in layer_indices:
        X = np.concatenate([np.stack(spatial_acts[l]), np.stack(nonspatial_acts[l])])
        y = np.array([1] * N + [0] * N)

        clf = LogisticRegression(max_iter=2000, C=1.0)
        if N >= 5:
            cv = min(5, N)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            print(f"Layer {l}: accuracy = {scores.mean():.3f} +/- {scores.std():.3f} (chance=0.5)")
            analysis[l]["probe_accuracy"] = float(scores.mean())
        else:
            clf.fit(X, y)
            acc = clf.score(X, y)
            print(f"Layer {l}: train accuracy = {acc:.3f} (N too small for CV)")
            analysis[l]["probe_accuracy"] = float(acc)

except ImportError:
    print("sklearn not available, skipping classification check")

# ── Interpretation ──────────────────────────────────────────────

print("\n--- Interpretation ---\n")
best_layer = max(analysis.keys(), key=lambda l: analysis[l].get("probe_accuracy", 0))
best_acc = analysis[best_layer].get("probe_accuracy", 0)

print(f"Best discriminating layer: {best_layer} (accuracy: {best_acc:.3f})")

if best_acc > 0.7:
    print("SIGNAL: Spatial vs non-spatial activations ARE distinguishable.")
    print("This is encouraging for the full probe experiment (Day 2).")
elif best_acc > 0.55:
    print("WEAK SIGNAL: Some difference exists but it's subtle.")
    print("The full experiment with N=50 may show clearer results.")
else:
    print("NO SIGNAL: Spatial and non-spatial activations look the same.")
    print("WARNING: This suggests the SAE approach may struggle.")

# Note about cosine similarity
max_cos = max(analysis[l]["cosine_similarity"] for l in layer_indices)
min_cos = min(analysis[l]["cosine_similarity"] for l in layer_indices)
print(f"\nCosine similarity range: {min_cos:.4f} to {max_cos:.4f}")
if max_cos > 0.99:
    print("High cosine similarity is expected (most activation encodes non-spatial info).")
    print("The probe accuracy is the better signal.")

# ── Save ────────────────────────────────────────────────────────

results = {
    "experiment": "exp1b_activation_stats",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "n_samples": N,
    "layer_indices": layer_indices,
    "analysis": {str(k): v for k, v in analysis.items()},
    "best_layer": int(best_layer),
    "best_accuracy": float(best_acc),
}

with open("results_exp1b.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to results_exp1b.json")
print("Done.")
