"""Experiment 12: Head-Level Localization via Steering Degradation."""

import json
import random
import time
from datetime import datetime

import numpy as np
import torch

from day5_utils import (
    MODEL_NAME,
    contains_relation,
    create_adversarial,
    create_cluttered,
    create_novel_objects_image,
    create_three_objects_image,
    create_two_object_image,
    load_model,
    patch_layer_with_vector,
    relation_question,
)
from utils import ActivationCollector


TARGET_LAYER = 21
N_TRAIN = 12
N_TEST = 4
PAIRINGS = [("above", "below"), ("below", "above"), ("left", "right"), ("right", "left")]
COLOR_PAIRS = [("red", "blue"), ("green", "orange"), ("purple", "cyan"), ("red", "green")]
STEER_SCALE = 1.5

# Harder evaluation sets (shared spirit with Exp11)
EVAL_SETS = {
    "cluttered": create_cluttered,
    "adversarial": create_adversarial,
    "novel_shapes": create_novel_objects_image,
    "three_objects": create_three_objects_image,
}


def ablation_hook_factory(head_indices, head_dim):
    head_indices = sorted(set(head_indices))

    def hook(module, inputs):
        hidden = inputs[0].clone()
        for head_idx in head_indices:
            start = head_idx * head_dim
            end = start + head_dim
            hidden[:, :, start:end] = 0
        return (hidden,)

    return hook


def collect_layer_means(model, processor, collector):
    class_acts = {rel: [] for rel in ["above", "below", "left", "right"]}
    for i in range(N_TRAIN):
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        for rel in ["above", "below", "left", "right"]:
            img = create_two_object_image(relation=rel, obj1_color=c1, obj2_color=c2)
            question = relation_question(rel, c1=c1, c2=c2)
            collector.attach()
            collector.activations = {}
            inputs = processor(
                text=f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
                images=[img],
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            class_acts[rel].append(collector.get_last_token_activation(TARGET_LAYER))
            collector.detach()

    class_means = {rel: np.stack(class_acts[rel]).mean(axis=0) for rel in class_acts}
    return class_means


def build_eval_cases():
    cases = []
    set_names = list(EVAL_SETS.keys())
    for source_rel, target_rel in PAIRINGS:
        for i in range(N_TEST):
            set_name = set_names[i % len(set_names)]
            create_fn = EVAL_SETS[set_name]
            img = create_fn(source_rel)
            question = relation_question(source_rel)
            cases.append(
                {
                    "set": set_name,
                    "source_rel": source_rel,
                    "target_rel": target_rel,
                    "image": img,
                    "question": question,
                }
            )
    return cases


def evaluate_steering_rate(model, processor, layers_ref, class_means, cases):
    targeted = 0
    total = 0
    for case in cases:
        source_rel = case["source_rel"]
        target_rel = case["target_rel"]
        direction = class_means[target_rel] - class_means[source_rel]
        patched = patch_layer_with_vector(
            model,
            processor,
            layers_ref,
            TARGET_LAYER,
            case["image"],
            case["question"],
            direction * STEER_SCALE,
            max_new_tokens=20,
        )
        if contains_relation(patched, target_rel):
            targeted += 1
        total += 1
    return targeted / total


def evaluate_steering_with_ablation(model, processor, layer_ref, layers_ref, class_means, cases, head_indices):
    handle = None
    if head_indices:
        handle = layer_ref.self_attn.o_proj.register_forward_pre_hook(
            ablation_hook_factory(head_indices, layer_ref.self_attn.head_dim)
        )
    try:
        return evaluate_steering_rate(model, processor, layers_ref, class_means, cases)
    finally:
        if handle is not None:
            handle.remove()


def main():
    print("=" * 60)
    print("EXPERIMENT 12: Head-Level Localization via Steering Degradation")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
    print()

    layer = layers_ref[TARGET_LAYER]
    num_heads = layer.self_attn.num_heads
    head_dim = layer.self_attn.head_dim
    print(f"Target layer: {TARGET_LAYER}")
    print(f"Attention heads: {num_heads} (head_dim={head_dim})")
    print()

    collector = ActivationCollector(model, layer_indices=[TARGET_LAYER])

    print("=" * 60)
    print("STEP 1: Build Relation Means and Hard Eval Cases")
    print("=" * 60)
    class_means = collect_layer_means(model, processor, collector)
    cases = build_eval_cases()
    print(f"Collected means from N_TRAIN={N_TRAIN} per relation")
    print(f"Evaluation cases: {len(cases)} (hard/mixed only)")
    print()

    print("=" * 60)
    print("STEP 2: Baseline Steering Rate")
    print("=" * 60)
    baseline_steering = evaluate_steering_rate(model, processor, layers_ref, class_means, cases)
    print(f"Baseline steering rate: {baseline_steering:.3f}")

    print("\n" + "=" * 60)
    print("STEP 3: Single-Head Steering Degradation")
    print("=" * 60)
    head_results = {}
    for head_idx in range(num_heads):
        ablated_rate = evaluate_steering_with_ablation(
            model, processor, layer, layers_ref, class_means, cases, [head_idx]
        )
        drop = baseline_steering - ablated_rate
        head_results[str(head_idx)] = {
            "ablated_steering_rate": ablated_rate,
            "steering_drop": drop,
        }
        print(f"  Head {head_idx}: rate={ablated_rate:.3f}, drop={drop:.3f}")

    ranked = sorted(head_results.items(), key=lambda item: item[1]["steering_drop"], reverse=True)
    top_head = ranked[0][0]
    top3_heads = [int(h) for h, _ in ranked[:3]]

    print("\n" + "=" * 60)
    print("STEP 4: Grouped Ablation vs Random Controls")
    print("=" * 60)
    top3_rate = evaluate_steering_with_ablation(
        model, processor, layer, layers_ref, class_means, cases, top3_heads
    )

    rng = random.Random(42)
    random_group_rates = []
    head_ids = list(range(num_heads))
    for _ in range(5):
        group = rng.sample(head_ids, 3)
        rate = evaluate_steering_with_ablation(
            model, processor, layer, layers_ref, class_means, cases, group
        )
        random_group_rates.append({"group": group, "rate": rate})

    random_mean_rate = float(np.mean([x["rate"] for x in random_group_rates]))

    print(f"Top3 heads: {top3_heads}, steering rate={top3_rate:.3f}")
    print(f"Random group mean steering rate: {random_mean_rate:.3f}")

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    top_head_drop = head_results[top_head]["steering_drop"]
    results = {
        "baseline_steering_above_0p50": baseline_steering > 0.50,
        "top_head_drop_above_0p05": top_head_drop > 0.05,
        "top3_worse_than_random_by_0p05": top3_rate < (random_mean_rate - 0.05),
    }

    print(f"  Baseline steering rate: {baseline_steering:.3f}")
    print(f"  Top head: {top_head}, drop={top_head_drop:.3f}")
    print(f"  Top3 rate: {top3_rate:.3f}")
    print(f"  Random-group mean rate: {random_mean_rate:.3f}")
    for key, value in results.items():
        print(f"  [{'PASS' if value else 'FAIL'}] {key}")

    all_pass = all(results.values())
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: HEAD-LEVEL LOCALIZATION SUPPORTED BY STEERING DEGRADATION")
    else:
        print("RESULT: HEAD EFFECTS APPEAR WEAK OR DISTRIBUTED UNDER THIS PROTOCOL")
    print("=" * 60)

    output = {
        "experiment": "exp12_head_level_localization",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "pairings": PAIRINGS,
        "eval_sets": list(EVAL_SETS.keys()),
        "baseline_steering": baseline_steering,
        "head_results": head_results,
        "top_head": top_head,
        "top3_heads": top3_heads,
        "top3_rate": top3_rate,
        "random_group_rates": random_group_rates,
        "random_group_mean_rate": random_mean_rate,
        "criteria": results,
        "all_pass": all_pass,
    }

    with open("results_exp12.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to results_exp12.json")
    print("Done.")


if __name__ == "__main__":
    main()
