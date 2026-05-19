"""
Sample Data Generator for Experiments 6, 7, 8
==============================================
Generates sample images at different difficulty levels so the user can
visually confirm the stimuli before running the full experiments.

Difficulty Levels:
  Level 1: Current synthetic (white bg, fixed size, centered) — ALREADY TESTED
  Level 2: Hard synthetic (random bg color, varying sizes, slight jitter)
  Level 3: Noisy/cluttered (textured background, distractor shapes, size variation)
  Level 4: Adversarial (tiny objects, extreme positions, near-boundary)

Run: python generate_sample_data.py
     -> Outputs sample images to samples/ folder for visual inspection
"""

import sys
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "day1"))

output_dir = Path("samples")
output_dir.mkdir(exist_ok=True)

# ── Level 2: Hard Synthetic ─────────────────────────────────────

def create_hard_synthetic(relation, size=448):
    """
    Harder than basic synthetic:
    - Random pastel background color
    - Randomized object sizes (not uniform)
    - Slight position jitter (not perfectly centered)
    - Different shape combinations
    """
    # Random background
    bg_r = random.randint(200, 255)
    bg_g = random.randint(200, 255)
    bg_b = random.randint(200, 255)
    img = Image.new("RGB", (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Random sizes for objects (asymmetric)
    obj1_size = random.randint(size // 8, size // 5)
    obj2_size = random.randint(size // 8, size // 5)

    # Position with jitter
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

    # Draw obj1 (circle, random warm color)
    c1 = random.choice(["red", "darkred", "orangered", "tomato"])
    draw.ellipse([x1 - obj1_size, y1 - obj1_size, x1 + obj1_size, y1 + obj1_size],
                 fill=c1, outline="black", width=2)

    # Draw obj2 (square, random cool color)
    c2 = random.choice(["blue", "darkblue", "navy", "royalblue"])
    draw.rectangle([x2 - obj2_size, y2 - obj2_size, x2 + obj2_size, y2 + obj2_size],
                   fill=c2, outline="black", width=2)

    return img, c1, c2


# ── Level 3: Cluttered/Noisy ───────────────────────────────────

def create_cluttered(relation, size=448):
    """
    Cluttered scene:
    - Gradient or textured background
    - 3-5 distractor shapes (gray/muted)
    - Main objects with slight transparency effect
    - Gaussian noise overlay
    """
    # Create gradient background
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

    # Add distractor shapes
    n_distractors = random.randint(3, 5)
    for _ in range(n_distractors):
        dx = random.randint(50, size - 50)
        dy = random.randint(50, size - 50)
        ds = random.randint(15, 40)
        gray = random.randint(100, 180)
        shape_type = random.choice(["ellipse", "rectangle", "polygon"])
        if shape_type == "ellipse":
            draw.ellipse([dx - ds, dy - ds, dx + ds, dy + ds],
                        fill=(gray, gray, gray), outline=(gray - 30, gray - 30, gray - 30))
        elif shape_type == "rectangle":
            draw.rectangle([dx - ds, dy - ds, dx + ds, dy + ds],
                          fill=(gray, gray, gray))
        else:
            points = [(dx + int(ds * np.cos(a)), dy + int(ds * np.sin(a)))
                      for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
            draw.polygon(points, fill=(gray, gray, gray))

    # Main objects (larger, bright colors to stand out)
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

    # Bright red circle (obj1)
    draw.ellipse([x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
                 fill="red", outline="darkred", width=3)
    # Bright blue square (obj2)
    draw.rectangle([x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
                   fill="blue", outline="darkblue", width=3)

    # Add slight blur to simulate less-than-perfect image
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


# ── Level 4: Adversarial ───────────────────────────────────────

def create_adversarial(relation, size=448):
    """
    Adversarial cases designed to be hard:
    - Very small objects (harder to localize)
    - Objects near edges
    - Similar colors (low contrast)
    - Ambiguous spacing (barely satisfying the relation)
    """
    img = Image.new("RGB", (size, size), (240, 235, 230))  # Off-white
    draw = ImageDraw.Draw(img)

    # Small objects
    obj_size = size // 14  # Much smaller than usual

    center = size // 2
    # Minimal separation — barely above/below etc.
    offset = obj_size + random.randint(5, 20)  # Very close together

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

    # Low-contrast colors (dark red vs dark blue — hard to distinguish)
    draw.ellipse([x1 - obj_size, y1 - obj_size, x1 + obj_size, y1 + obj_size],
                 fill=(180, 60, 60))  # Muted red
    draw.rectangle([x2 - obj_size, y2 - obj_size, x2 + obj_size, y2 + obj_size],
                   fill=(60, 60, 180))  # Muted blue

    return img


# ── Generate Samples ────────────────────────────────────────────

print("Generating sample data for visual inspection...")
print(f"Output directory: {output_dir.resolve()}\n")

random.seed(42)  # Reproducible
np.random.seed(42)

relations = ["above", "below", "left", "right"]

# Generate 2 samples per level per relation
for level, (name, func) in enumerate([
    ("level2_hard_synthetic", create_hard_synthetic),
    ("level3_cluttered", create_cluttered),
    ("level4_adversarial", create_adversarial),
], start=2):
    print(f"--- {name} ---")
    for rel in relations:
        for i in range(2):
            img = func(rel) if level != 2 else func(rel)[0]
            fname = f"{name}_{rel}_{i+1}.png"
            img.save(output_dir / fname)
            print(f"  Saved: {fname}")
    print()

# Also generate the Experiment 8 train/test split samples
print("--- Experiment 8: Cross-generalization samples ---")
# Train set: red circle + blue square (what we've been using)
from utils import create_two_object_image

for rel in relations:
    img = create_two_object_image(relation=rel, obj1_color="red", obj2_color="blue")
    img.save(output_dir / f"exp8_train_{rel}.png")
    print(f"  Train: exp8_train_{rel}.png (red circle, blue square)")

# Test set: COMPLETELY different — green triangle + orange pentagon (new shapes/colors)
def create_novel_objects_image(relation, size=448):
    """Different shapes and colors for generalization test."""
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

    # Obj1: Green triangle
    tri_points = [
        (x1, y1 - obj_size),
        (x1 - obj_size, y1 + obj_size),
        (x1 + obj_size, y1 + obj_size),
    ]
    draw.polygon(tri_points, fill="green", outline="darkgreen", width=2)

    # Obj2: Orange diamond
    diamond_points = [
        (x2, y2 - obj_size),
        (x2 + obj_size, y2),
        (x2, y2 + obj_size),
        (x2 - obj_size, y2),
    ]
    draw.polygon(diamond_points, fill="orange", outline="darkorange", width=2)

    return img

for rel in relations:
    img = create_novel_objects_image(relation=rel)
    img.save(output_dir / f"exp8_test_{rel}.png")
    print(f"  Test:  exp8_test_{rel}.png (green triangle, orange diamond)")

print(f"\n{'='*60}")
print(f"DONE. Generated {len(list(output_dir.glob('*.png')))} sample images.")
print(f"Please inspect: {output_dir.resolve()}")
print(f"{'='*60}")
print("\nImage categories:")
print("  level2_* : Harder synthetic (random bg, varied sizes, jitter)")
print("  level3_* : Cluttered (gradient bg, distractor shapes, blur)")
print("  level4_* : Adversarial (tiny objects, low contrast, close together)")
print("  exp8_train_* : Standard training images (red circle + blue square)")
print("  exp8_test_* : Novel test images (green triangle + orange diamond)")
