"""Experiment 9: 4-Way Causal Intervention."""

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


TARGET_LAYER = 21
N_TRAIN = 16
N_TEST = 6
PAIRINGS = [("above", "below"), ("below", "above"), ("left", "right"), ("right", "left")]
COLOR_PAIRS = [("red", "blue"), ("green", "orange"), ("purple", "cyan"), ("red", "green"), ("blue", "orange")]


def collect_training_data(model, processor, collector):
    class_acts = {rel: [] for rel in RELATIONS}
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
            class_acts[rel].append(collector.get_last_token_activation(TARGET_LAYER))
            collector.detach()
    return class_acts


def train_directions(class_acts):
    X = []
    y = []
    for rel in RELATIONS:
        for act in class_acts[rel]:
            X.append(act)
            y.append(rel)
    X = np.array(X)
    y = np.array(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=3000, multi_class="multinomial")
    clf.fit(X_scaled, y)
    train_acc = clf.score(X_scaled, y)

    class_dirs = {}
    class_means = {}
    for idx, rel in enumerate(clf.classes_):
        vec = clf.coef_[idx]
        class_dirs[rel] = vec / (np.linalg.norm(vec) + 1e-8)
        class_means[rel] = np.stack(class_acts[rel]).mean(axis=0)

    return scaler, clf, class_dirs, class_means, train_acc


def evaluate_pair(model, processor, layers_ref, source_rel, target_rel, direction, scale):
    clean_hits = 0
    patched_hits = 0
    control_hits = 0
    for i in range(N_TEST):
        c1, c2 = COLOR_PAIRS[i % len(COLOR_PAIRS)]
        img = create_two_object_image(relation=source_rel, obj1_color=c1, obj2_color=c2)
        question = relation_question(source_rel, c1=c1, c2=c2)
        clean = run_generation(model, processor, img, question, max_new_tokens=20)
        patched = patch_layer_with_vector(
            model, processor, layers_ref, TARGET_LAYER, img, question, direction * scale, max_new_tokens=20
        )

        rng = np.random.default_rng(1234 + i)
        random_direction = rng.standard_normal(direction.shape[0]).astype(np.float32)
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8) * np.linalg.norm(direction)
        control = patch_layer_with_vector(
            model, processor, layers_ref, TARGET_LAYER, img, question, random_direction * scale, max_new_tokens=20
        )

        if contains_relation(clean, source_rel):
            clean_hits += 1
        if contains_relation(patched, target_rel):
            patched_hits += 1
        if contains_relation(control, target_rel):
            control_hits += 1

        print(f"  {source_rel}->{target_rel} | clean={clean!r} | patched={patched!r} | control={control!r}")

    return {
        "clean_accuracy": clean_hits / N_TEST,
        "patched_target_rate": patched_hits / N_TEST,
        "control_target_rate": control_hits / N_TEST,
    }


def main():
    print("=" * 60)
    print("EXPERIMENT 9: 4-Way Causal Intervention")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    t0 = time.time()
    processor, model, layers_ref, num_layers, hidden_dim = load_model(MODEL_NAME)
    print(f"Loaded in {time.time() - t0:.1f}s. {num_layers} layers, hidden_dim={hidden_dim}")
    print()

    collector = ActivationCollector(model, layer_indices=[TARGET_LAYER])

    print("=" * 60)
    print("STEP 1: Train 4-Way Relation Directions")
    print("=" * 60)
    class_acts = collect_training_data(model, processor, collector)
    scaler, clf, class_dirs, class_means, train_acc = train_directions(class_acts)
    print(f"Training accuracy: {train_acc:.3f}")
    print()

    print("=" * 60)
    print("STEP 2: Causal Steering Tests")
    print("=" * 60)

    steer_scale = 1.5
    pair_results = {}
    for source_rel, target_rel in PAIRINGS:
        # Use raw mean-difference vectors; normalization made patching too weak.
        direction = class_means[target_rel] - class_means[source_rel]
        print(f"\n--- Steering {source_rel} -> {target_rel} ---")
        pair_results[f"{source_rel}_to_{target_rel}"] = evaluate_pair(
            model, processor, layers_ref, source_rel, target_rel, direction, steer_scale
        )

    targeted_rates = [v["patched_target_rate"] for v in pair_results.values()]
    control_rates = [v["control_target_rate"] for v in pair_results.values()]
    avg_targeted = float(np.mean(targeted_rates))
    avg_control = float(np.mean(control_rates))

    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"  Average targeted steering rate: {avg_targeted:.3f}")
    print(f"  Average control steering rate:  {avg_control:.3f}")
    for key, value in pair_results.items():
        print(
            f"  {key}: targeted={value['patched_target_rate']:.3f}, control={value['control_target_rate']:.3f}"
        )

    results = {
        "avg_targeted_above_0p30": avg_targeted > 0.30,
        "avg_control_below_0p20": avg_control < 0.20,
        "three_or_more_pairs_above_0p50": sum(v["patched_target_rate"] > 0.50 for v in pair_results.values()) >= 3,
    }

    print()
    print(f"  [{'PASS' if results['avg_targeted_above_0p30'] else 'FAIL'}] Average targeted steering > 0.30")
    print(f"  [{'PASS' if results['avg_control_below_0p20'] else 'FAIL'}] Average control steering < 0.20")
    print(f"  [{'PASS' if results['three_or_more_pairs_above_0p50'] else 'FAIL'}] At least 3 pairings above 0.50")

    all_pass = all(results.values())
    print()
    print("=" * 60)
    if all_pass:
        print("RESULT: THE 4-WAY RELATION SPACE IS CAUSALLY USEFUL")
    else:
        print("RESULT: PARTIAL CAUSAL EFFECT")
    print("=" * 60)

    output = {
        "experiment": "exp9_4way_causal_intervention",
        "timestamp": datetime.now().isoformat(),
        "model": MODEL_NAME,
        "target_layer": TARGET_LAYER,
        "n_train": N_TRAIN,
        "n_test": N_TEST,
        "train_accuracy": float(train_acc),
        "pair_results": pair_results,
        "avg_targeted": avg_targeted,
        "avg_control": avg_control,
        "criteria": results,
        "all_pass": all_pass,
    }

    with open("results_exp9.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("\nResults saved to results_exp9.json")
    print("Done.")


if __name__ == "__main__":
    main()
