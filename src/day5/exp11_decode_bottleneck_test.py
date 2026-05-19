"""Experiment 11: Decode Bottleneck Test (Principled Mixed Evaluation)."""

import json
import math
import time
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from day5_utils import (
    MODEL_NAME,
    RELATIONS,
    contains_relation,
    create_adversarial,
    create_cluttered,
    create_hard_synthetic,
    create_novel_objects_image,
    create_roof_floor_conflict_image,
    create_sky_ground_conflict_image,
    create_small_objects_image,
    create_three_objects_image,
    create_two_object_image,
    load_model,
    relation_question,
    run_generation,
)
from utils import ActivationCollector


TARGET_LAYER = 21
N_TRAIN = 20
N_TEST = 10

TEST_SETS = {
    "clean_standard": lambda rel: create_two_object_image(relation=rel, obj1_color="red", obj2_color="blue"),
    "hard_synthetic": create_hard_synthetic,
    "cluttered": create_cluttered,
    "adversarial": create_adversarial,
    "novel_shapes": create_novel_objects_image,
    "small_objects": create_small_objects_image,
    "three_objects": create_three_objects_image,
    "swapped_colors": lambda rel: create_two_object_image(relation=rel, obj1_color="blue", obj2_color="red"),
}

PRIOR_CONFLICTS = {
    "sky_ground": create_sky_ground_conflict_image,
    "roof_floor": create_roof_floor_conflict_image,
}


def ci95_from_counts(successes, total):
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    se = math.sqrt(max(0.0, p * (1.0 - p) / total))
    low = max(0.0, p - 1.96 * se)
    high = min(1.0, p + 1.96 * se)
    return [float(low), float(high)]


def train_probe(model, processor, collector):
    acts = []
    labels = []
    for _ in range(N_TRAIN):
        for rel in RELATIONS:
            img = create_two_object_image(relation=rel, obj1_color="red", obj2_color="blue")
            question = relation_question(rel)
            collector.attach()
            collector.activations = {}
            inputs = processor(
                text=f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
                images=[img],
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            acts.append(collector.get_last_token_activation(TARGET_LAYER))
            labels.append(rel)
            collector.detach()

    X = np.array(acts)
    y = np.array(labels)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, multi_class="multinomial")
    clf.fit(X_scaled, y)
    return scaler, clf, float(clf.score(X_scaled, y))


def evaluate_set(model, processor, collector, scaler, clf, create_fn, set_name):
    model_correct = 0
    probe_correct = 0
    bottleneck_hits = 0
    total = 0

    details = []
    for rel in RELATIONS:
        for _ in range(N_TEST):
            img = create_fn(rel)
            question = relation_question(rel)

            collector.attach()
            collector.activations = {}
            inputs = processor(
                text=f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>\n<|im_start|>assistant\n",
                images=[img],
                return_tensors="pt",
            ).to(model.device)
            with torch.no_grad():
                model.generate(**inputs, max_new_tokens=1)
            act = collector.get_last_token_activation(TARGET_LAYER)
            collector.detach()

            answer = run_generation(model, processor, img, question, max_new_tokens=20)
            pred = clf.predict(scaler.transform([act]))[0]

            model_is_correct = contains_relation(answer, rel)
            probe_is_correct = pred == rel

            if model_is_correct:
                model_correct += 1
            if probe_is_correct:
                probe_correct += 1
            if probe_is_correct and not model_is_correct:
                bottleneck_hits += 1
            total += 1

            if len(details) < 20:
                details.append(
                    {
                        "set": set_name,
                        "relation": rel,
                        "answer": answer,
                        "probe_pred": pred,
                        "model_correct": model_is_correct,
                        "probe_correct": probe_is_correct,
                    }
                )

    return {
        "model_accuracy": model_correct / total,
        "probe_accuracy": probe_correct / total,
        "bottleneck_rate": bottleneck_hits / total,
        "model_correct": model_correct,
        "probe_correct": probe_correct,
        "bottleneck_hits": bottleneck_hits,
        "total": total,
        "ci95": {
            "model_accuracy": ci95_from_counts(model_correct, total),
            "probe_accuracy": ci95_from_counts(probe_correct, total),
            "bottleneck_rate": ci95_from_counts(bottleneck_hits, total),
        },
        "sample_details": details,
    }


def evaluate_prior_conflict(model, processor, create_fn):
    results = {
        "congruent": {"model_correct": 0, "total": 0},
        "incongruent": {"model_correct": 0, "total": 0},
    }
    for rel in ("above", "below"):
        for _ in range(N_TEST):
            img = create_fn(rel)
            question = "Is the sky above or below the ground?"
            answer = run_generation(model, processor, img, question, max_new_tokens=20)
            bucket = "congruent" if rel == "above" else "incongruent"
            results[bucket]["total"] += 1
            if contains_relation(answer, rel):
                results[bucket]["model_correct"] += 1

    for key in results:
        total = results[key]["total"]
        correct = results[key]["model_correct"]
        results[key]["accuracy"] = correct / total
        results[key]["ci95"] = ci95_from_counts(correct, total)

    return results


def summarize_group(level_results, keys):
    model_correct = sum(level_results[k]["model_correct"] for k in keys)
    probe_correct = sum(level_results[k]["probe_correct"] for k in keys)
    bottleneck_hits = sum(level_results[k]["bottleneck_hits"] for k in keys)
    total = sum(level_results[k]["total"] for k in keys)

    return {
        "model_accuracy": model_correct / total,
        "probe_accuracy": probe_correct / total,
        "bottleneck_rate": bottleneck_hits / total,
        "model_correct": model_correct,
        "probe_correct": probe_correct,
        "bottleneck_hits": bottleneck_hits,
        "total": total,
        "ci95": {
            "model_accuracy": ci95_from_counts(model_correct, total),
            "probe_accuracy": ci95_from_counts(probe_correct, total),
            "bottleneck_rate": ci95_from_counts(bottleneck_hits, total),
        },
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 11: Decode Bottleneck Test (Principled Mixed Evaluation)")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
    print()

    collector = ActivationCollector(model, layer_indices=[TARGET_LAYER])

    print("=" * 60)
    print("STEP 1: Train Probe on Clean (Red/Blue) Only")
    print("=" * 60)
    scaler, clf, train_acc = train_probe(model, processor, collector)
    print(f"Clean probe training accuracy: {train_acc:.3f}")

    print("\n" + "=" * 60)
    print("STEP 2: Evaluate on Mixed-Difficulty Held-Out Sets")
    print("=" * 60)

    level_results = {}
    for set_name, create_fn in TEST_SETS.items():
        print(f"\n--- {set_name} ---")
        result = evaluate_set(model, processor, collector, scaler, clf, create_fn, set_name)
        level_results[set_name] = result
        print(f"  model accuracy:  {result['model_accuracy']:.3f}")
        print(f"  probe accuracy:  {result['probe_accuracy']:.3f}")
        print(f"  bottleneck rate: {result['bottleneck_rate']:.3f}")
        print(f"  bottleneck 95% CI: [{result['ci95']['bottleneck_rate'][0]:.3f}, {result['ci95']['bottleneck_rate'][1]:.3f}]")

    clean_keys = ["clean_standard"]
    hard_keys = [k for k in TEST_SETS if k not in clean_keys]
    clean_summary = summarize_group(level_results, clean_keys)
    hard_summary = summarize_group(level_results, hard_keys)

    print("\n" + "=" * 60)
    print("STEP 3: Prior-Conflict Check")
    print("=" * 60)
    prior_results = {}
    for prior_name, create_fn in PRIOR_CONFLICTS.items():
        prior_results[prior_name] = evaluate_prior_conflict(model, processor, create_fn)
        print(
            f"  {prior_name}: congruent={prior_results[prior_name]['congruent']['accuracy']:.3f}, incongruent={prior_results[prior_name]['incongruent']['accuracy']:.3f}"
        )

    bottleneck_gap = hard_summary["bottleneck_rate"] - clean_summary["bottleneck_rate"]

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"  Clean group model/probe: {clean_summary['model_accuracy']:.3f} / {clean_summary['probe_accuracy']:.3f}")
    print(f"  Hard group model/probe:  {hard_summary['model_accuracy']:.3f} / {hard_summary['probe_accuracy']:.3f}")
    print(f"  Clean bottleneck rate:   {clean_summary['bottleneck_rate']:.3f}")
    print(f"  Hard bottleneck rate:    {hard_summary['bottleneck_rate']:.3f}")
    print(f"  Bottleneck gap (hard-clean): {bottleneck_gap:.3f}")

    results = {
        "probe_not_worse_than_model_hard": hard_summary["probe_accuracy"] >= hard_summary["model_accuracy"] - 0.02,
        "clean_probe_high": clean_summary["probe_accuracy"] > 0.90,
        "bottleneck_gap_hard_vs_clean_above_0p05": bottleneck_gap > 0.05,
    }

    for key, value in results.items():
        print(f"  [{'PASS' if value else 'FAIL'}] {key}")

    all_pass = all(results.values())
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: DECODE BOTTLENECK EVIDENCE ON HARDER HELD-OUT SETS")
    else:
        print("RESULT: NO STRONG BOTTLENECK SIGNAL UNDER CURRENT EVALUATION MIX")
    print("=" * 60)

    output = {
        "experiment": "exp11_decode_bottleneck_test",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "train_accuracy": train_acc,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "level_results": level_results,
        "clean_summary": clean_summary,
        "hard_summary": hard_summary,
        "bottleneck_gap_hard_minus_clean": bottleneck_gap,
        "prior_results": prior_results,
        "criteria": results,
        "all_pass": all_pass,
    }

    with open("results_exp11.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to results_exp11.json")
    print("Done.")


if __name__ == "__main__":
    main()
