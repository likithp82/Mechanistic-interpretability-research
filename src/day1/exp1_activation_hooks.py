"""
EXPERIMENT 1: Can We Hook Into VLM Activations?
================================================
Day 1 of Kill-Switch Pilot

WHAT: Load Qwen2-VL-2B-Instruct, hook into all decoder layers,
      run a spatial reasoning question, verify activations are collected.

PASS CRITERIA:
  [x] Model loads on GPU (T4 16GB / fp16)
  [x] Activations collected for ALL layers
  [x] Each activation has shape [1, seq_len, hidden_dim]
  [x] No NaN/Inf in activations
  [x] Model produces a coherent spatial answer

FAIL => STOP THE PROJECT

Run: python exp1_activation_hooks.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image

# ── Setup ───────────────────────────────────────────────────────

print("=" * 60)
print("EXPERIMENT 1: Activation Hook Verification")
print("=" * 60)
print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    print("Device: Apple MPS (Metal)")
    DEVICE = "mps"
else:
    print("WARNING: No GPU detected. This will be very slow.")
    DEVICE = "cpu"
print()

# ── Step 1: Load Model ─────────────────────────────────────────

print("[Step 1/6] Loading Qwen2-VL-2B-Instruct...")
t0 = time.time()

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

try:
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    load_time = time.time() - t0
    print(f"  Model loaded in {load_time:.1f}s")
    print(f"  Model device: {model.device}")
    # Qwen2-VL uses model.model.language_model.layers
    _layers = model.model.language_model.layers
    print(f"  Number of layers: {len(_layers)}")
    print(f"  Hidden dim: {model.config.hidden_size}")
    num_layers = len(_layers)
    hidden_dim = model.config.hidden_size
    STEP1_PASS = True
except Exception as e:
    print(f"  FAILED to load model: {e}")
    STEP1_PASS = False
    sys.exit(1)

print()

# ── Step 2: Create Synthetic Test Image ─────────────────────────

print("[Step 2/6] Creating synthetic spatial image...")

from utils import create_two_object_image

img = create_two_object_image(relation="above", obj1_color="red", obj2_color="blue")
img.save("test_image_above.png")
print("  Created: red circle ABOVE blue square (saved to test_image_above.png)")
print()

# ── Step 3: Prepare Input ───────────────────────────────────────

print("[Step 3/6] Preparing model input...")

from utils import format_prompt

question = "Is the red circle above or below the blue square?"
prompt = format_prompt(question)

inputs = processor(
    text=prompt,
    images=[img],
    return_tensors="pt",
).to(model.device)

input_ids = inputs["input_ids"]
print(f"  Question: {question}")
print(f"  Input token count: {input_ids.shape[1]}")
print(f"  Input keys: {list(inputs.keys())}")
print()

# ── Step 4: Attach Hooks & Run Forward Pass ─────────────────────

print("[Step 4/6] Attaching hooks and running forward pass...")

from utils import ActivationCollector

collector = ActivationCollector(model)
collector.attach()
print(f"  Hooks attached to {len(collector.layer_indices)} layers")

t0 = time.time()
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=50)
gen_time = time.time() - t0

# Decode answer
input_len = input_ids.shape[1]
answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
print(f"  Forward pass completed in {gen_time:.1f}s")
print(f"  Model answer: \"{answer}\"")
print()

# ── Step 5: Validate Activations ────────────────────────────────

print("[Step 5/6] Validating activations...")

passed, report = collector.validate()
print(f"  Validation: {'PASS' if passed else 'FAIL'}")
print(f"  Report: {report}")

# Detailed per-layer report
print()
print("  Per-layer activation summary:")
print(f"  {'Layer':<10} {'Shape':<25} {'Dtype':<12} {'Norm':<12} {'HasNaN':<8} {'HasInf':<8}")
print("  " + "-" * 75)

layer_info = []
for name in sorted(collector.activations.keys(), key=lambda x: int(x.split("_")[1])):
    act = collector.activations[name]
    info = {
        "name": name,
        "shape": list(act.shape),
        "dtype": str(act.dtype),
        "norm": act.float().norm().item(),
        "has_nan": bool(torch.isnan(act).any()),
        "has_inf": bool(torch.isinf(act).any()),
        "mean": act.float().mean().item(),
        "std": act.float().std().item(),
    }
    layer_info.append(info)
    shape_str = str(list(act.shape))
    print(
        f"  {name:<10} {shape_str:<25} {str(act.dtype):<12} "
        f"{info['norm']:<12.2f} {str(info['has_nan']):<8} {str(info['has_inf']):<8}"
    )

# Detach hooks
collector.detach()
print()

# ── Step 6: Check Pass/Fail Criteria ────────────────────────────

print("[Step 6/6] Evaluating PASS/FAIL criteria...")
print()

results = {}

# Criterion 1: Model loaded
results["model_loaded"] = STEP1_PASS
print(f"  [{'PASS' if results['model_loaded'] else 'FAIL'}] Model loads on GPU")

# Criterion 2: All layers captured
results["all_layers_captured"] = len(collector.activations) == num_layers
print(
    f"  [{'PASS' if results['all_layers_captured'] else 'FAIL'}] "
    f"Activations for all layers ({len(collector.activations)}/{num_layers})"
)

# Criterion 3: Correct shape [1, seq_len, hidden_dim]
shapes_ok = all(
    len(info["shape"]) == 3 and info["shape"][0] == 1 and info["shape"][2] == hidden_dim
    for info in layer_info
)
results["correct_shapes"] = shapes_ok
print(f"  [{'PASS' if shapes_ok else 'FAIL'}] Activation shapes are [1, seq_len, {hidden_dim}]")

# Criterion 4: No NaN/Inf
no_nan_inf = not any(info["has_nan"] or info["has_inf"] for info in layer_info)
results["no_nan_inf"] = no_nan_inf
print(f"  [{'PASS' if no_nan_inf else 'FAIL'}] No NaN/Inf in activations")

# Criterion 5: Coherent answer
answer_lower = answer.lower()
coherent = "above" in answer_lower or "below" in answer_lower
results["coherent_answer"] = coherent
print(f"  [{'PASS' if coherent else 'FAIL'}] Model answer is coherent (\"{answer}\")")

# Overall
all_pass = all(results.values())
print()
print("=" * 60)
if all_pass:
    print("EXPERIMENT 1 RESULT: *** PASS ***")
    print("Proceed to Experiment 2.")
else:
    print("EXPERIMENT 1 RESULT: *** FAIL ***")
    failed = [k for k, v in results.items() if not v]
    print(f"Failed criteria: {failed}")
    print("RECOMMENDATION: Debug failures before continuing.")
    if not results["model_loaded"]:
        print("  -> Try int8 quantization: load_in_8bit=True")
    if not results["all_layers_captured"]:
        print("  -> Check model architecture: model.model.layers may differ")
    if not results["no_nan_inf"]:
        print("  -> Try float32 instead of float16")
    if not results["coherent_answer"]:
        print("  -> Model may need different prompt format. Check Qwen2-VL docs.")
print("=" * 60)

# ── Save Results ────────────────────────────────────────────────

results_data = {
    "experiment": "exp1_activation_hooks",
    "timestamp": datetime.now().isoformat(),
    "model": MODEL_NAME,
    "num_layers": num_layers,
    "hidden_dim": hidden_dim,
    "question": question,
    "answer": answer,
    "load_time_s": load_time,
    "gen_time_s": gen_time,
    "input_tokens": input_ids.shape[1],
    "criteria": results,
    "overall_pass": all_pass,
    "layer_info": layer_info,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else DEVICE,
}

output_path = Path("results_exp1.json")
with open(output_path, "w") as f:
    json.dump(results_data, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")


# ── Bonus: Quick Sanity Checks ──────────────────────────────────

print("\n--- Bonus Sanity Checks ---")

# Check activation norms across layers (should vary, not constant)
norms = [info["norm"] for info in layer_info]
print(f"Activation norms range: {min(norms):.1f} to {max(norms):.1f}")
print(f"Norm std across layers: {np.std(norms):.1f}")
if np.std(norms) < 0.1:
    print("  WARNING: Norms are nearly constant across layers. Unusual.")

# Check seq_len consistency
seq_lens = [info["shape"][1] for info in layer_info]
if len(set(seq_lens)) == 1:
    print(f"Sequence length: {seq_lens[0]} (consistent across all layers)")
else:
    print(f"WARNING: Sequence lengths vary across layers: {set(seq_lens)}")

# Test a second image to make sure hooks are reusable
print("\n--- Testing hook reusability ---")
img2 = create_two_object_image(relation="left", obj1_color="green", obj2_color="orange")
collector2 = ActivationCollector(model, layer_indices=[0, num_layers // 2, num_layers - 1])
collector2.attach()

q2 = "Is the green circle to the left or right of the orange square?"
inputs2 = processor(text=format_prompt(q2), images=[img2], return_tensors="pt").to(model.device)
with torch.no_grad():
    out2 = model.generate(**inputs2, max_new_tokens=30)
answer2 = processor.decode(out2[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True).strip()
collector2.detach()

print(f"  Second question: {q2}")
print(f"  Second answer: \"{answer2}\"")
print(f"  Activations collected: {len(collector2.activations)} layers")
valid2, report2 = collector2.validate()
print(f"  Validation: {'PASS' if valid2 else 'FAIL'} - {report2}")

print("\nDone.")
