"""
EXPERIMENT 7: Activation Patching (Causal Intervention)
========================================================
Day 4 of Kill-Switch Pilot

WHAT: The CRITICAL causal experiment. We take the "above/below" relation
      direction discovered in Exp 4 and INJECT it into activations to
      see if we can flip the model's spatial answer.

METHOD:
  1. Get the "above" → "below" direction vector from Exp 4's probe
  2. Run a clean "above" example, record activations
  3. Add the direction vector (scaled) to the activation at the best layer
  4. Re-run generation with the patched activation
  5. Check if the answer flips from "above" to "below"

  Also test:
  - Patching in OPPOSITE direction (below → above)
  - Patching at different layers (early vs middle vs late)
  - Patching with different magnitudes (how much signal is needed?)

SUCCESS CRITERIA:
  - Answer flips in > 50% of cases when patching correct direction
  - Answer does NOT flip when patching random directions (control)
  - Effect is stronger at middle/late layers than early layers

RUN: python exp7_activation_patching.py
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
    create_two_object_image,
    format_prompt,
)

# ── Setup ───────────────────────────────────────────────────────

print("=" * 60)
print("EXPERIMENT 7: Activation Patching (Causal Intervention)")
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
    layers_ref = model.model.language_model.layers
else:
    num_layers = len(model.model.layers)
    layers_ref = model.model.layers
hidden_dim = model.config.hidden_size
print(f"Loaded in {load_time:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
print()

# ── Step 1: Learn the Relation Direction ────────────────────────

print("=" * 60)
print("STEP 1: Learning Above/Below Direction Vector")
print("=" * 60 + "\n")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Target layer for patching (layer 21 was best in Exp 3 & 4)
target_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
primary_layer = 3 * num_layers // 4  # Layer 21

N_TRAIN = 20  # samples per relation for learning direction

collector = ActivationCollector(model, layer_indices=target_layers)

above_acts = {l: [] for l in target_layers}
below_acts = {l: [] for l in target_layers}

COLOR_PAIRS = [("red", "blue"), ("green", "orange"), ("purple", "cyan"),
               ("red", "green"), ("blue", "orange")]

print(f"Collecting {N_TRAIN} above + {N_TRAIN} below activations...")

for i in range(N_TRAIN):
    c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]

    # Above
    img_above = create_two_object_image(relation="above", obj1_color=c1, obj2_color=c2)
    question = f"Is the {c1} circle above or below the {c2} square?"
    collector.attach()
    collector.activations = {}
    inputs = processor(text=format_prompt(question), images=[img_above], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    for l in target_layers:
        above_acts[l].append(collector.get_last_token_activation(l))
    collector.detach()

    # Below
    img_below = create_two_object_image(relation="below", obj1_color=c1, obj2_color=c2)
    collector.attach()
    collector.activations = {}
    inputs = processor(text=format_prompt(question), images=[img_below], return_tensors="pt").to(model.device)
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=1)
    for l in target_layers:
        below_acts[l].append(collector.get_last_token_activation(l))
    collector.detach()

    if (i + 1) % 10 == 0:
        print(f"  {i+1}/{N_TRAIN} pairs done")

# Compute direction vectors per layer
directions = {}
for l in target_layers:
    above_mean = np.stack(above_acts[l]).mean(axis=0)
    below_mean = np.stack(below_acts[l]).mean(axis=0)
    # Direction: above - below (pointing toward "above")
    direction = above_mean - below_mean
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    directions[l] = {
        "raw": direction,
        "normalized": direction_norm,
        "magnitude": float(np.linalg.norm(direction)),
    }
    print(f"  Layer {l}: direction magnitude = {np.linalg.norm(direction):.2f}")

print(f"\nPrimary patching layer: {primary_layer}")
print(f"Direction magnitude: {directions[primary_layer]['magnitude']:.2f}")

# ── Step 2: Patching Function ───────────────────────────────────

print("\n" + "=" * 60)
print("STEP 2: Activation Patching Tests")
print("=" * 60 + "\n")


def patch_and_generate(model, processor, image, question, layer_idx, direction_vector, scale=1.0):
    """
    Run model with a hook that ADDS direction_vector to the residual stream
    at the specified layer. Returns the generated answer.
    """
    direction_tensor = torch.tensor(direction_vector * scale, dtype=torch.float16).to(model.device)

    def patch_hook(module, input, output):
        if isinstance(output, tuple):
            patched = output[0].clone()
            # Add direction to the last token position
            patched[0, -1, :] += direction_tensor
            return (patched,) + output[1:]
        else:
            patched = output.clone()
            patched[0, -1, :] += direction_tensor
            return patched

    # Register hook
    handle = layers_ref[layer_idx].register_forward_hook(patch_hook)

    # Run generation
    inputs = processor(
        text=format_prompt(question), images=[image], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=30)
    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    handle.remove()
    return answer


# ── Test 2A: Flip "above" to "below" ───────────────────────────

print("--- Test 2A: Patch 'above' examples with NEGATIVE direction (push toward 'below') ---\n")

N_TEST = 10
flip_results_a2b = []

for i in range(N_TEST):
    c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
    img = create_two_object_image(relation="above", obj1_color=c1, obj2_color=c2)
    question = f"Is the {c1} circle above or below the {c2} square?"

    # Clean answer (no patching)
    clean_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                       np.zeros(hidden_dim), scale=0)

    # Patched answer (subtract the "above" direction → push toward "below")
    patched_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                         -directions[primary_layer]["raw"], scale=1.0)

    clean_says_above = "above" in clean_answer.lower()
    patched_says_below = "below" in patched_answer.lower()
    flipped = clean_says_above and patched_says_below

    flip_results_a2b.append({
        "clean_answer": clean_answer,
        "patched_answer": patched_answer,
        "clean_says_above": clean_says_above,
        "patched_says_below": patched_says_below,
        "flipped": flipped,
    })
    status = "FLIPPED" if flipped else "no flip"
    print(f"  [{status}] clean=\"{clean_answer}\" → patched=\"{patched_answer}\"")

flip_rate_a2b = sum(r["flipped"] for r in flip_results_a2b) / N_TEST
print(f"\n  Flip rate (above→below): {flip_rate_a2b:.1%} ({sum(r['flipped'] for r in flip_results_a2b)}/{N_TEST})")

# ── Test 2B: Flip "below" to "above" ───────────────────────────

print("\n--- Test 2B: Patch 'below' examples with POSITIVE direction (push toward 'above') ---\n")

flip_results_b2a = []

for i in range(N_TEST):
    c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
    img = create_two_object_image(relation="below", obj1_color=c1, obj2_color=c2)
    question = f"Is the {c1} circle above or below the {c2} square?"

    clean_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                       np.zeros(hidden_dim), scale=0)

    patched_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                         directions[primary_layer]["raw"], scale=1.0)

    clean_says_below = "below" in clean_answer.lower()
    patched_says_above = "above" in patched_answer.lower()
    flipped = clean_says_below and patched_says_above

    flip_results_b2a.append({
        "clean_answer": clean_answer,
        "patched_answer": patched_answer,
        "clean_says_below": clean_says_below,
        "patched_says_above": patched_says_above,
        "flipped": flipped,
    })
    status = "FLIPPED" if flipped else "no flip"
    print(f"  [{status}] clean=\"{clean_answer}\" → patched=\"{patched_answer}\"")

flip_rate_b2a = sum(r["flipped"] for r in flip_results_b2a) / N_TEST
print(f"\n  Flip rate (below→above): {flip_rate_b2a:.1%} ({sum(r['flipped'] for r in flip_results_b2a)}/{N_TEST})")

# ── Test 2C: Control — Random Direction (should NOT flip) ───────

print("\n--- Test 2C: Control — Patch with RANDOM direction (should NOT flip) ---\n")

np.random.seed(42)
random_direction = np.random.randn(hidden_dim).astype(np.float32)
random_direction = random_direction / np.linalg.norm(random_direction)
random_direction = random_direction * directions[primary_layer]["magnitude"]  # same magnitude

control_flips = 0
for i in range(N_TEST):
    c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
    img = create_two_object_image(relation="above", obj1_color=c1, obj2_color=c2)
    question = f"Is the {c1} circle above or below the {c2} square?"

    clean_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                       np.zeros(hidden_dim), scale=0)
    patched_answer = patch_and_generate(model, processor, img, question, primary_layer,
                                         random_direction, scale=1.0)

    clean_correct = "above" in clean_answer.lower()
    patched_wrong = "below" in patched_answer.lower() and "above" not in patched_answer.lower()
    if clean_correct and patched_wrong:
        control_flips += 1

    print(f"  clean=\"{clean_answer}\" → random_patch=\"{patched_answer}\"")

control_flip_rate = control_flips / N_TEST
print(f"\n  Control flip rate (random direction): {control_flip_rate:.1%} (should be low)")

# ── Test 2D: Layer Comparison ───────────────────────────────────

print("\n--- Test 2D: Flip Rate Across Layers ---\n")

layer_flip_rates = {}
for l in target_layers:
    flips = 0
    for i in range(5):  # Quick test with 5 samples
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        img = create_two_object_image(relation="above", obj1_color=c1, obj2_color=c2)
        question = f"Is the {c1} circle above or below the {c2} square?"

        clean_answer = patch_and_generate(model, processor, img, question, l,
                                           np.zeros(hidden_dim), scale=0)
        patched_answer = patch_and_generate(model, processor, img, question, l,
                                             -directions[l]["raw"], scale=1.0)

        if "above" in clean_answer.lower() and "below" in patched_answer.lower():
            flips += 1

    layer_flip_rates[l] = flips / 5
    print(f"  Layer {l}: flip rate = {flips}/5 = {flips/5:.1%}")

# ── Test 2E: Magnitude Scaling ──────────────────────────────────

print("\n--- Test 2E: Effect of Patching Magnitude ---\n")

scales = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
scale_results = {}

img = create_two_object_image(relation="above", obj1_color="red", obj2_color="blue")
question = "Is the red circle above or below the blue square?"

for scale in scales:
    flips = 0
    for i in range(5):
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        img = create_two_object_image(relation="above", obj1_color=c1, obj2_color=c2)
        q = f"Is the {c1} circle above or below the {c2} square?"

        patched = patch_and_generate(model, processor, img, q, primary_layer,
                                      -directions[primary_layer]["raw"], scale=scale)
        if "below" in patched.lower():
            flips += 1

    scale_results[scale] = flips / 5
    print(f"  Scale {scale:.2f}: flip rate = {flips}/5 = {flips/5:.1%}")

# ── Evaluation ──────────────────────────────────────────────────

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60 + "\n")

avg_flip_rate = (flip_rate_a2b + flip_rate_b2a) / 2

results = {}
results["flip_rate_above_50"] = avg_flip_rate > 0.50
results["control_below_20"] = control_flip_rate < 0.20
results["causal_gap"] = avg_flip_rate - control_flip_rate

print(f"  Average flip rate (targeted): {avg_flip_rate:.1%}")
print(f"  Control flip rate (random): {control_flip_rate:.1%}")
print(f"  Causal gap: {results['causal_gap']:.1%}")
print()
print(f"  [{'PASS' if results['flip_rate_above_50'] else 'FAIL'}] "
      f"Targeted flip rate > 50%: {avg_flip_rate:.1%}")
print(f"  [{'PASS' if results['control_below_20'] else 'FAIL'}] "
      f"Control flip rate < 20%: {control_flip_rate:.1%}")

print()
print("=" * 60)
if results["flip_rate_above_50"] and results["control_below_20"]:
    print("RESULT: CAUSAL EVIDENCE CONFIRMED")
    print("The relation direction vector CAUSALLY controls the model's spatial answer.")
    print("This is publishable-grade evidence of a mechanistic finding.")
elif avg_flip_rate > 0.30:
    print("RESULT: PARTIAL CAUSAL EVIDENCE")
    print("The direction has some causal effect but isn't fully controlling.")
    print("May need better direction estimation or different patching strategy.")
else:
    print("RESULT: WEAK CAUSAL EFFECT")
    print("The correlation from Exp 4 doesn't translate to causal control.")
    print("The mechanism may be more distributed or non-linear.")
print("=" * 60)

# ── Save ────────────────────────────────────────────────────────

output = {
    "experiment": "exp7_activation_patching",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "primary_layer": primary_layer,
    "direction_magnitude": directions[primary_layer]["magnitude"],
    "flip_rate_above_to_below": float(flip_rate_a2b),
    "flip_rate_below_to_above": float(flip_rate_b2a),
    "average_flip_rate": float(avg_flip_rate),
    "control_flip_rate": float(control_flip_rate),
    "causal_gap": float(results["causal_gap"]),
    "layer_flip_rates": {str(k): v for k, v in layer_flip_rates.items()},
    "scale_results": {str(k): v for k, v in scale_results.items()},
    "criteria": {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in results.items()},
    "flip_details_a2b": flip_results_a2b,
    "flip_details_b2a": flip_results_b2a,
}

with open("results_exp7.json", "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to results_exp7.json")
print("Done.")
