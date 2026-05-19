"""
Utility functions for Day 1 experiments.
Creates synthetic images, formats prompts, extracts activations.
"""

import random
from PIL import Image, ImageDraw, ImageFont
import torch
import numpy as np


# ── Image Creation ──────────────────────────────────────────────

COLORS = ["red", "blue", "green", "orange", "purple", "cyan"]
SHAPE_NAMES = {"ellipse": "circle", "rectangle": "square", "polygon": "triangle"}


def create_two_object_image(
    relation="above", size=448, obj1_color="red", obj2_color="blue"
):
    """
    Create a synthetic image with two colored shapes in a specified spatial relation.

    Args:
        relation: One of "above", "below", "left", "right"
        size: Image dimensions (square)
        obj1_color: Color of object 1 (the subject)
        obj2_color: Color of object 2 (the reference)

    Returns:
        PIL.Image with two shapes
    """
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    obj_size = size // 5  # each object is ~20% of image
    margin = size // 8
    center = size // 2

    if relation == "above":
        # obj1 (subject) on top, obj2 (reference) on bottom
        x1, y1 = center, margin + obj_size // 2
        x2, y2 = center, size - margin - obj_size // 2
    elif relation == "below":
        x1, y1 = center, size - margin - obj_size // 2
        x2, y2 = center, margin + obj_size // 2
    elif relation == "left":
        x1, y1 = margin + obj_size // 2, center
        x2, y2 = size - margin - obj_size // 2, center
    elif relation == "right":
        x1, y1 = size - margin - obj_size // 2, center
        x2, y2 = margin + obj_size // 2, center
    else:
        raise ValueError(f"Unknown relation: {relation}")

    # Draw object 1 as a circle
    r = obj_size // 2
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=obj1_color)
    # Draw object 2 as a square
    draw.rectangle([x2 - r, y2 - r, x2 + r, y2 + r], fill=obj2_color)

    return img


def create_random_two_object_image(size=448):
    """
    Create an image with two shapes at random positions.
    Returns (image, obj1_info, obj2_info) where info includes
    color, shape, position.
    """
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)

    colors = random.sample(COLORS, 2)
    obj_size = size // 6
    margin = obj_size + 10

    # Random positions ensuring no overlap
    while True:
        x1 = random.randint(margin, size - margin)
        y1 = random.randint(margin, size - margin)
        x2 = random.randint(margin, size - margin)
        y2 = random.randint(margin, size - margin)
        dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
        if dist > obj_size * 2.5:
            break

    r = obj_size // 2
    draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=colors[0])
    draw.rectangle([x2 - r, y2 - r, x2 + r, y2 + r], fill=colors[1])

    obj1_info = {"color": colors[0], "shape": "circle", "x": x1, "y": y1}
    obj2_info = {"color": colors[1], "shape": "square", "x": x2, "y": y2}

    # Determine ground truth relations
    if y1 < y2:
        vertical_rel = "above"
    else:
        vertical_rel = "below"

    if x1 < x2:
        horizontal_rel = "to the left of"
    else:
        horizontal_rel = "to the right of"

    return img, obj1_info, obj2_info, vertical_rel, horizontal_rel


def create_single_object_image(x_grid, y_grid, grid_size=5, img_size=448, color="red"):
    """
    Create image with a single object at a grid position.
    Used for Experiment 3 (position encoding test).

    Args:
        x_grid: Grid x position (0 to grid_size-1)
        y_grid: Grid y position (0 to grid_size-1)
        grid_size: Number of grid positions per dimension
        img_size: Image pixel size
        color: Object color

    Returns:
        PIL.Image
    """
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    margin = img_size // 8
    usable = img_size - 2 * margin
    step = usable / (grid_size - 1) if grid_size > 1 else 0

    cx = int(margin + x_grid * step)
    cy = int(margin + y_grid * step)
    r = img_size // 12

    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    return img


# ── Prompt Formatting ───────────────────────────────────────────

def format_prompt(question, system_msg=None):
    """Format a question into Qwen2-VL chat format."""
    parts = []
    if system_msg:
        parts.append(f"<|im_start|>system\n{system_msg}<|im_end|>")
    parts.append(
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{question}<|im_end|>"
    )
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── Activation Extraction ──────────────────────────────────────

class ActivationCollector:
    """
    Collects residual stream activations from specified layers
    during a forward pass.
    """

    def __init__(self, model, layer_indices=None):
        """
        Args:
            model: Qwen2VLForConditionalGeneration model
            layer_indices: List of layer indices to hook. None = all layers.
        """
        self.model = model
        self.hooks = []
        self.activations = {}

        # Determine which layers to hook
        # Qwen2-VL: model.model.language_model.layers
        if hasattr(model.model, 'language_model'):
            self._layers = model.model.language_model.layers
        else:
            self._layers = model.model.layers
        num_layers = len(self._layers)
        if layer_indices is None:
            self.layer_indices = list(range(num_layers))
        else:
            self.layer_indices = [i for i in layer_indices if i < num_layers]

    def _make_hook(self, name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self.activations[name] = output[0].detach().cpu()
            else:
                self.activations[name] = output.detach().cpu()
        return hook_fn

    def attach(self):
        """Register forward hooks on target layers."""
        self.clear()
        for i in self.layer_indices:
            h = self._layers[i].register_forward_hook(
                self._make_hook(f"layer_{i}")
            )
            self.hooks.append(h)

    def detach(self):
        """Remove all hooks."""
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def clear(self):
        """Clear stored activations and hooks."""
        self.detach()
        self.activations = {}

    def get_last_token_activation(self, layer_idx):
        """Get activation at the last token position for a given layer."""
        key = f"layer_{layer_idx}"
        if key not in self.activations:
            raise KeyError(f"No activation for {key}. Did you run a forward pass?")
        return self.activations[key][0, -1, :].float().numpy()

    def get_image_token_mean_activation(self, layer_idx, input_ids, image_token_id: int):
        """Return mean of hidden states over image-pad token positions at layer_idx.

        Args:
            layer_idx:       Layer index to query.
            input_ids:       1-D or (1, seq_len) int tensor from the processor.
            image_token_id:  Token id for the image-pad token
                             (processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).
        Returns:
            float32 numpy array of shape (hidden_dim,).
        Raises:
            KeyError if layer was not hooked.
            ValueError if no image tokens are found in input_ids.
        """
        key = f"layer_{layer_idx}"
        if key not in self.activations:
            raise KeyError(f"No activation for {key}. Did you run a forward pass?")
        ids = input_ids.view(-1)  # flatten to 1-D
        mask = (ids == image_token_id)
        if not mask.any():
            raise ValueError(
                f"No image_pad tokens (id={image_token_id}) found in input_ids."
            )
        # activations shape: (1, seq_len, hidden_dim)
        hidden = self.activations[key][0]           # (seq_len, hidden_dim)
        img_acts = hidden[mask.cpu()]               # (n_img_tokens, hidden_dim)
        return img_acts.float().mean(dim=0).numpy()

    def get_token_activation(self, layer_idx, token_pos):
        """Get activation at a specific token position for a given layer."""
        key = f"layer_{layer_idx}"
        if key not in self.activations:
            raise KeyError(f"No activation for {key}.")
        seq_len = self.activations[key].shape[1]
        if token_pos >= seq_len:
            raise IndexError(f"Token pos {token_pos} >= seq_len {seq_len}")
        return self.activations[key][0, token_pos, :].float().numpy()

    def get_all_last_token(self):
        """Get last-token activations for all hooked layers. Returns dict."""
        result = {}
        for i in self.layer_indices:
            key = f"layer_{i}"
            if key in self.activations:
                result[i] = self.activations[key][0, -1, :].float().numpy()
        return result

    def validate(self):
        """
        Check all activations are valid.
        Returns (pass_bool, report_string).
        """
        issues = []
        if len(self.activations) == 0:
            return False, "No activations collected."

        expected = len(self.layer_indices)
        actual = len(self.activations)
        if actual < expected:
            issues.append(f"Expected {expected} layers, got {actual}")

        for name, act in self.activations.items():
            if torch.isnan(act).any():
                issues.append(f"{name}: contains NaN")
            if torch.isinf(act).any():
                issues.append(f"{name}: contains Inf")
            if act.ndim != 3:
                issues.append(f"{name}: expected 3D, got {act.ndim}D")

        if issues:
            return False, "\n".join(issues)
        return True, "All activations valid."


# ── Model Inference Helper ──────────────────────────────────────

def run_inference(model, processor, image, question, max_new_tokens=50):
    """
    Run model inference and return the generated text.

    Args:
        model: Qwen2VLForConditionalGeneration
        processor: AutoProcessor
        image: PIL.Image
        question: str
        max_new_tokens: int

    Returns:
        str: generated answer text
    """
    prompt = format_prompt(question)
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Decode only the new tokens
    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return answer.strip()


def run_inference_with_activations(
    model, processor, image, question, collector, max_new_tokens=1
):
    """
    Run a single forward pass (1 token generation) while collecting activations.

    Args:
        model, processor, image, question: as above
        collector: ActivationCollector (already attached)
        max_new_tokens: typically 1 for activation collection

    Returns:
        (answer_text, activations_dict)
    """
    collector.activations = {}  # reset

    prompt = format_prompt(question)
    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    input_len = inputs["input_ids"].shape[1]
    answer = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return answer.strip(), dict(collector.activations)
