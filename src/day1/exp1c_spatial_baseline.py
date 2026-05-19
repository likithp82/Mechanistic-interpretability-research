"""
EXPERIMENT 1C: Model Spatial Accuracy Baseline
===============================================
Day 1 (run alongside or after exp1b)

WHAT: Test Qwen2-VL-2B on simple spatial questions with synthetic images
      to establish a baseline accuracy. Do we see the near-chance performance
      that the literature reports?

RUN: python exp1c_spatial_baseline.py
"""

import json
import random
from datetime import datetime
from collections import defaultdict

import torch
from PIL import Image

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from utils import create_two_object_image, run_inference

# ── Setup ───────────────────────────────────────────────────────

print("=" * 60)
print("EXPERIMENT 1C: Spatial Accuracy Baseline")
print("=" * 60)

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
)
print("Loaded.\n")

# ── Test Cases ──────────────────────────────────────────────────

# Test 4 relations, 10 images each = 40 total
RELATIONS = ["above", "below", "left", "right"]
N_PER_RELATION = 10
COLOR_PAIRS = [
    ("red", "blue"),
    ("green", "orange"),
    ("purple", "cyan"),
    ("red", "green"),
    ("blue", "orange"),
]

results_per_relation = defaultdict(list)
all_results = []

print(f"Testing {len(RELATIONS)} relations x {N_PER_RELATION} images = "
      f"{len(RELATIONS) * N_PER_RELATION} total\n")

for relation in RELATIONS:
    for i in range(N_PER_RELATION):
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        img = create_two_object_image(relation=relation, obj1_color=c1, obj2_color=c2)

        if relation in ("above", "below"):
            question = f"Is the {c1} circle above or below the {c2} square? Answer with just 'above' or 'below'."
            correct_answer = relation
        else:
            question = f"Is the {c1} circle to the left or right of the {c2} square? Answer with just 'left' or 'right'."
            correct_answer = relation

        answer = run_inference(model, processor, img, question, max_new_tokens=20)
        answer_lower = answer.lower().strip()

        # Check correctness
        is_correct = correct_answer in answer_lower

        # Also check if the WRONG answer is present (explicit error)
        if relation in ("above", "below"):
            wrong = "below" if relation == "above" else "above"
        else:
            wrong = "right" if relation == "left" else "left"
        is_wrong = wrong in answer_lower and correct_answer not in answer_lower

        result = {
            "relation": relation,
            "colors": (c1, c2),
            "question": question,
            "answer": answer,
            "correct": correct_answer,
            "is_correct": is_correct,
            "is_explicit_wrong": is_wrong,
        }
        all_results.append(result)
        results_per_relation[relation].append(is_correct)

        status = "OK" if is_correct else "WRONG" if is_wrong else "UNCLEAR"
        print(f"  [{status}] {relation}: \"{answer}\" (expected: {correct_answer})")

# ── Summary ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

total_correct = sum(r["is_correct"] for r in all_results)
total = len(all_results)
overall_acc = total_correct / total

print(f"\nOverall accuracy: {total_correct}/{total} = {overall_acc:.1%}")
print()

print(f"{'Relation':<12} {'Correct':<10} {'Accuracy':<10}")
print("-" * 32)
for rel in RELATIONS:
    correct = sum(results_per_relation[rel])
    n = len(results_per_relation[rel])
    acc = correct / n if n > 0 else 0
    print(f"{rel:<12} {correct}/{n:<8} {acc:.1%}")

print()
if overall_acc > 0.8:
    print("MODEL IS GOOD at spatial reasoning on synthetic images.")
    print("This is actually somewhat expected for simple colored-shape scenes.")
    print("The real test is on natural images (WhatsUp, VSR benchmarks).")
elif overall_acc > 0.6:
    print("MODEL IS MODERATE. Some spatial capability exists.")
elif overall_acc > 0.4:
    print("MODEL IS NEAR CHANCE. Consistent with literature findings.")
    print("This is the expected result and validates our research motivation.")
else:
    print("MODEL IS BELOW CHANCE. May have systematic biases.")
    print("Check if answers are always the same (e.g., always 'above').")

# Check for systematic bias
answer_counts = defaultdict(int)
for r in all_results:
    for word in ["above", "below", "left", "right"]:
        if word in r["answer"].lower():
            answer_counts[word] += 1
print(f"\nAnswer distribution: {dict(answer_counts)}")

# ── Save ────────────────────────────────────────────────────────

output = {
    "experiment": "exp1c_spatial_baseline",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "overall_accuracy": overall_acc,
    "per_relation": {
        rel: sum(results_per_relation[rel]) / len(results_per_relation[rel])
        for rel in RELATIONS
    },
    "answer_distribution": dict(answer_counts),
    "total_correct": total_correct,
    "total": total,
    "all_results": all_results,
}

with open("results_exp1c.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to results_exp1c.json")
print("Done.")
