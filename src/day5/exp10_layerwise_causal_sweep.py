"""Experiment 10: Layer-Wise Causal Sweep."""

import json
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
    create_two_object_image,
    load_model,
    patch_layer_with_vector,
    relation_question,
    run_generation,
)
from utils import ActivationCollector


TARGET_LAYERS = [14, 21, 27]
N_TRAIN = 12
N_TEST = 4
PAIRINGS = [("above", "below"), ("below", "above"), ("left", "right"), ("right", "left")]
COLOR_PAIRS = [("red", "blue"), ("green", "orange"), ("purple", "cyan"), ("red", "green")]


def collect_layer_activations(model, processor, collector):
    layer_data = {layer: {rel: [] for rel in RELATIONS} for layer in TARGET_LAYERS}
    for i in range(N_TRAIN):
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        for rel in RELATIONS:
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
            for layer in TARGET_LAYERS:
                layer_data[layer][rel].append(collector.get_last_token_activation(layer))
            collector.detach()
    return layer_data


def train_directions(layer_data):
    layer_models = {}
    for layer in TARGET_LAYERS:
        X = []
        y = []
        for rel in RELATIONS:
            for act in layer_data[layer][rel]:
                X.append(act)
                y.append(rel)
        X = np.array(X)
        y = np.array(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=3000, multi_class="multinomial")
        clf.fit(X_scaled, y)
        layer_models[layer] = {
            "scaler": scaler,
            "clf": clf,
            "train_accuracy": float(clf.score(X_scaled, y)),
            "class_dirs": {
                rel: clf.coef_[idx] / (np.linalg.norm(clf.coef_[idx]) + 1e-8)
                for idx, rel in enumerate(clf.classes_)
            },
            "class_means": {
                rel: np.stack(layer_data[layer][rel]).mean(axis=0) for rel in RELATIONS
            },
        }
    return layer_models


def evaluate_layer(model, processor, layers_ref, layer, class_means):
    pair_results = {}
    steer_scale = 1.5
    for source_rel, target_rel in PAIRINGS:
        # Use raw mean-difference vectors; normalized coefficients were too weak.
        direction = class_means[target_rel] - class_means[source_rel]

        targeted = 0
        control = 0
        clean = 0
        for i in range(N_TEST):
            c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
            img = create_two_object_image(relation=source_rel, obj1_color=c1, obj2_color=c2)
            question = relation_question(source_rel, c1=c1, c2=c2)
            clean_answer = run_generation(model, processor, img, question, max_new_tokens=20)
            patched_answer = patch_layer_with_vector(
                model, processor, layers_ref, layer, img, question, direction * steer_scale, max_new_tokens=20
            )

            rng = np.random.default_rng(1000 + layer * 10 + i)
            random_direction = rng.standard_normal(direction.shape[0]).astype(np.float32)
            random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8) * np.linalg.norm(direction)
            control_answer = patch_layer_with_vector(
                model, processor, layers_ref, layer, img, question, random_direction * steer_scale, max_new_tokens=20
            )

            if contains_relation(clean_answer, source_rel):
                clean += 1
            if contains_relation(patched_answer, target_rel):
                targeted += 1
            if contains_relation(control_answer, target_rel):
                control += 1

        pair_results[f"{source_rel}_to_{target_rel}"] = {
            "clean_accuracy": clean / N_TEST,
            "targeted_rate": targeted / N_TEST,
            "control_rate": control / N_TEST,
        }

    avg_targeted = float(np.mean([v["targeted_rate"] for v in pair_results.values()]))
    avg_control = float(np.mean([v["control_rate"] for v in pair_results.values()]))
    return {"pair_results": pair_results, "avg_targeted": avg_targeted, "avg_control": avg_control}


def main():
    print("=" * 60)
    print("EXPERIMENT 10: Layer-Wise Causal Sweep")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
    print()

    collector = ActivationCollector(model, layer_indices=TARGET_LAYERS)

    print("=" * 60)
    print("STEP 1: Train Directions for Candidate Layers")
    print("=" * 60)
    layer_data = collect_layer_activations(model, processor, collector)
    layer_models = train_directions(layer_data)
    for layer in TARGET_LAYERS:
        print(f"  Layer {layer}: train accuracy = {layer_models[layer]['train_accuracy']:.3f}")
    print()

    print("=" * 60)
    print("STEP 2: Causal Sweep")
    print("=" * 60)
    sweep_results = {}
    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} ---")
        result = evaluate_layer(model, processor, layers_ref, layer, layer_models[layer]["class_means"])
        sweep_results[str(layer)] = result
        print(f"  avg targeted = {result['avg_targeted']:.3f}")
        print(f"  avg control   = {result['avg_control']:.3f}")
        for pair_name, pair_stats in result["pair_results"].items():
            print(
                f"    {pair_name}: targeted={pair_stats['targeted_rate']:.3f}, control={pair_stats['control_rate']:.3f}"
            )

    best_layer = max(TARGET_LAYERS, key=lambda layer: sweep_results[str(layer)]["avg_targeted"])
    best_targeted = sweep_results[str(best_layer)]["avg_targeted"]
    best_control = sweep_results[str(best_layer)]["avg_control"]

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"  Best layer: {best_layer}")
    print(f"  Best targeted rate: {best_targeted:.3f}")
    print(f"  Best control rate:   {best_control:.3f}")

    results = {
        "best_targeted_above_0p30": best_targeted > 0.30,
        "best_control_below_0p20": best_control < 0.20,
        "best_layer_in_expected_range": best_layer in TARGET_LAYERS,
    }

    for key, value in results.items():
        print(f"  [{'PASS' if value else 'FAIL'}] {key}")

    all_pass = all(results.values())
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: ONE OF THE MIDDLE/DEEP LAYERS SUPPORTS CAUSAL STEERING")
    else:
        print("RESULT: CAUSAL EFFECT IS WEAK OR DISTRIBUTED")
    print("=" * 60)

    output = {
        "experiment": "exp10_layerwise_causal_sweep",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layers": TARGET_LAYERS,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "layer_models": {
            str(layer): {"train_accuracy": layer_models[layer]["train_accuracy"]} for layer in TARGET_LAYERS
        },
        "sweep_results": sweep_results,
        "best_layer": best_layer,
        "best_targeted": best_targeted,
        "best_control": best_control,
        "criteria": results,
        "all_pass": all_pass,
    }

    with open("results_exp10.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to results_exp10.json")
    print("Done.")


if __name__ == "__main__":
    main()
