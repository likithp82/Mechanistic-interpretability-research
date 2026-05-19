"""Shared helpers for Day 5 experiments."""

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter

DAY1_DIR = Path(__file__).resolve().parent.parent / "day1"
if str(DAY1_DIR) not in sys.path:
    sys.path.insert(0, str(DAY1_DIR))

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from utils import create_two_object_image, format_prompt


MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
RELATIONS = ["above", "below", "left", "right"]


def load_model(model_name=MODEL_NAME):
    """Load processor and model, returning layers for hooking."""
    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, dtype=torch.float16, device_map="auto"
    )
    if hasattr(model.model, "language_model"):
        layers_ref = model.model.language_model.layers
    else:
        layers_ref = model.model.layers
    num_layers = len(layers_ref)
    hidden_dim = model.config.hidden_size
    return processor, model, layers_ref, num_layers, hidden_dim


def run_generation(model, processor, image, question, max_new_tokens=20):
    """Generate a short answer string for a single image-question pair."""
    inputs = processor(
        text=format_prompt(question), images=[image], return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    input_len = inputs["input_ids"].shape[1]
    return processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()


def patch_layer_with_vector(
    model,
    processor,
    layers_ref,
    layer_idx,
    image,
    question,
    vector,
    max_new_tokens=20,
):
    """Add a vector to the last-token residual stream at a chosen layer."""
    patch_tensor = torch.tensor(vector, dtype=torch.float16, device=model.device)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0].clone()
            hidden[0, -1, :] += patch_tensor
            return (hidden,) + output[1:]
        hidden = output.clone()
        hidden[0, -1, :] += patch_tensor
        return hidden

    handle = layers_ref[layer_idx].register_forward_hook(hook_fn)
    try:
        return run_generation(
            model, processor, image, question, max_new_tokens=max_new_tokens
        )
    finally:
        handle.remove()


def contains_relation(text, relation):
    return relation.lower() in text.lower()


def relation_question(relation, c1="red", c2="blue"):
    if relation in ("above", "below"):
        return f"Is the {c1} circle above or below the {c2} square?"
    if relation in ("left", "right"):
        return f"Is the {c1} circle to the left or right of the {c2} square?"
    raise ValueError(f"Unknown relation: {relation}")


def create_novel_objects_image(relation, size=448):
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

    triangle = [
        (x1, y1 - obj_size),
        (x1 - obj_size, y1 + obj_size),
        (x1 + obj_size, y1 + obj_size),
    ]
    draw.polygon(triangle, fill="green", outline="darkgreen", width=2)

    diamond = [
        (x2, y2 - obj_size),
        (x2 + obj_size, y2),
        (x2, y2 + obj_size),
        (x2 - obj_size, y2),
    ]
    draw.polygon(diamond, fill="orange", outline="darkorange", width=2)
    return img


def create_small_objects_image(relation, size=448):
    img = Image.new("RGB", (size, size), (245, 245, 240))
    draw = ImageDraw.Draw(img)
    obj_size = size // 12
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

    draw.ellipse(
        [x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
        fill="purple",
        outline="darkviolet",
        width=2,
    )
    draw.rectangle(
        [x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
        fill="brown",
        outline="saddlebrown",
        width=2,
    )
    return img


def create_three_objects_image(relation, size=448):
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

    draw.ellipse(
        [x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
        fill="red",
        outline="darkred",
        width=2,
    )
    draw.rectangle(
        [x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
        fill="blue",
        outline="darkblue",
        width=2,
    )

    if relation in ("above", "below"):
        dx = center + random.choice([-1, 1]) * random.randint(size // 4, size // 3)
        dy = random.randint(size // 4, 3 * size // 4)
    else:
        dx = random.randint(size // 4, 3 * size // 4)
        dy = center + random.choice([-1, 1]) * random.randint(size // 4, size // 3)
    dx = max(obj_size, min(size - obj_size, dx))
    dy = max(obj_size, min(size - obj_size, dy))
    draw.polygon(
        [
            (dx, dy - obj_size // 2),
            (dx + obj_size // 2, dy + obj_size // 2),
            (dx - obj_size // 2, dy + obj_size // 2),
        ],
        fill="gray",
        outline="darkgray",
        width=2,
    )
    return img


def create_hard_synthetic(relation, size=448):
    bg_r = random.randint(200, 255)
    bg_g = random.randint(200, 255)
    bg_b = random.randint(200, 255)
    img = Image.new("RGB", (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)
    obj1_size = random.randint(size // 8, size // 5)
    obj2_size = random.randint(size // 8, size // 5)
    margin = size // 6
    center = size // 2
    jitter = random.randint(-size // 12, size // 12)

    if relation == "above":
        x1, y1 = center + jitter, margin + obj1_size
        x2, y2 = center - jitter, size - margin - obj2_size
    elif relation == "below":
        x1, y1 = center + jitter, size - margin - obj1_size
        x2, y2 = center - jitter, margin + obj2_size
    elif relation == "left":
        x1, y1 = margin + obj1_size, center + jitter
        x2, y2 = size - margin - obj2_size, center - jitter
    elif relation == "right":
        x1, y1 = size - margin - obj1_size, center + jitter
        x2, y2 = margin + obj2_size, center - jitter

    c1 = random.choice(["red", "darkred", "orangered", "tomato"])
    c2 = random.choice(["blue", "darkblue", "navy", "royalblue"])
    draw.ellipse(
        [x1 - obj1_size, y1 - obj1_size, x1 + obj1_size, y1 + obj1_size],
        fill=c1,
        outline="black",
        width=2,
    )
    draw.rectangle(
        [x2 - obj2_size, y2 - obj2_size, x2 + obj2_size, y2 + obj2_size],
        fill=c2,
        outline="black",
        width=2,
    )
    return img


def create_cluttered(relation, size=448):
    img = Image.new("RGB", (size, size))
    pixels = img.load()
    base_r, base_g, base_b = random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)
    for y in range(size):
        for x in range(size):
            pixels[x, y] = (
                min(255, base_r + x // 4),
                min(255, base_g + y // 4),
                min(255, base_b + (x + y) // 8),
            )

    draw = ImageDraw.Draw(img)
    for _ in range(random.randint(3, 5)):
        dx = random.randint(50, size - 50)
        dy = random.randint(50, size - 50)
        ds = random.randint(15, 40)
        gray = random.randint(100, 180)
        shape_type = random.choice(["ellipse", "rectangle", "polygon"])
        if shape_type == "ellipse":
            draw.ellipse(
                [dx - ds, dy - ds, dx + ds, dy + ds],
                fill=(gray, gray, gray),
                outline=(gray - 30, gray - 30, gray - 30),
            )
        elif shape_type == "rectangle":
            draw.rectangle([dx - ds, dy - ds, dx + ds, dy + ds], fill=(gray, gray, gray))
        else:
            points = [
                (dx + int(ds * np.cos(a)), dy + int(ds * np.sin(a)))
                for a in np.linspace(0, 2 * np.pi, 5)[:-1]
            ]
            draw.polygon(points, fill=(gray, gray, gray))

    obj_size = random.randint(size // 7, size // 5)
    margin = size // 5
    if relation == "above":
        x1, y1 = size // 2 + random.randint(-30, 30), margin + obj_size
        x2, y2 = size // 2 + random.randint(-30, 30), size - margin - obj_size
    elif relation == "below":
        x1, y1 = size // 2 + random.randint(-30, 30), size - margin - obj_size
        x2, y2 = size // 2 + random.randint(-30, 30), margin + obj_size
    elif relation == "left":
        x1, y1 = margin + obj_size, size // 2 + random.randint(-30, 30)
        x2, y2 = size - margin - obj_size, size // 2 + random.randint(-30, 30)
    elif relation == "right":
        x1, y1 = size - margin - obj_size, size // 2 + random.randint(-30, 30)
        x2, y2 = margin + obj_size, size // 2 + random.randint(-30, 30)

    draw.ellipse(
        [x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
        fill="red",
        outline="darkred",
        width=3,
    )
    draw.rectangle(
        [x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
        fill="blue",
        outline="darkblue",
        width=3,
    )
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))


def create_adversarial(relation, size=448):
    img = Image.new("RGB", (size, size), (240, 235, 230))
    draw = ImageDraw.Draw(img)
    obj_size = size // 14
    center = size // 2
    offset = obj_size + random.randint(5, 20)

    if relation == "above":
        x1, y1 = center + random.randint(-50, 50), center - offset
        x2, y2 = center + random.randint(-50, 50), center + offset
    elif relation == "below":
        x1, y1 = center + random.randint(-50, 50), center + offset
        x2, y2 = center + random.randint(-50, 50), center - offset
    elif relation == "left":
        x1, y1 = center - offset, center + random.randint(-50, 50)
        x2, y2 = center + offset, center + random.randint(-50, 50)
    elif relation == "right":
        x1, y1 = center + offset, center + random.randint(-50, 50)
        x2, y2 = center - offset, center + random.randint(-50, 50)

    draw.ellipse(
        [x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
        fill=(180, 60, 60),
    )
    draw.rectangle(
        [x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
        fill=(60, 60, 180),
    )
    return img


def create_sky_ground_conflict_image(relation, size=448):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    half = size // 2
    if relation == "above":
        draw.rectangle([0, 0, size, half], fill=(120, 180, 255))
        draw.rectangle([0, half, size, size], fill=(120, 80, 40))
    else:
        draw.rectangle([0, 0, size, half], fill=(120, 80, 40))
        draw.rectangle([0, half, size, size], fill=(120, 180, 255))
    return img


def create_roof_floor_conflict_image(relation, size=448):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    half = size // 2
    if relation == "above":
        draw.rectangle([0, 0, size, half], fill=(110, 110, 110))
        draw.rectangle([0, half, size, size], fill=(200, 190, 170))
    else:
        draw.rectangle([0, 0, size, half], fill=(200, 190, 170))
        draw.rectangle([0, half, size, size], fill=(110, 110, 110))
    return img
