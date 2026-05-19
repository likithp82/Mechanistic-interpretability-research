"""Build a real-image spatial-relation dataset from COCO-style annotations.

This script mines object pairs from real images, assigns robust 4-way
relations (above/below/left/right) using bounding-box centers, and
optionally writes label-preserving flipped variants.

Input expected:
- COCO images directory
- COCO annotations JSON (instances_*.json format)

Output:
- images/ (copied original and transformed images)
- manifest.jsonl (one row per (image, object-pair, relation))
- summary.json

Example:
python build_real_spatial_dataset.py \
  --images-dir /data/coco/train2017 \
  --annotations /data/coco/annotations/instances_train2017.json \
  --output-dir /data/spatial_real_dataset \
  --max-images 3000 \
  --max-pairs-per-image 8 \
  --transforms original,hflip,vflip
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


RELATIONS = ["above", "below", "left", "right"]


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class Ann:
    ann_id: int
    image_id: int
    category_id: int
    category_name: str
    box: Box


@dataclass
class Img:
    image_id: int
    file_name: str
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build real-image spatial relation dataset")
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-images", type=int, default=5000)
    p.add_argument("--max-pairs-per-image", type=int, default=8)
    p.add_argument("--min-box-area", type=float, default=900.0)
    p.add_argument("--axis-gap", type=float, default=18.0)
    p.add_argument("--dominance-ratio", type=float, default=1.25)
    p.add_argument("--allowed-categories", type=str, default="")
    p.add_argument("--transforms", type=str, default="original,hflip,vflip")
    p.add_argument("--copy-images", action="store_true", default=True)
    p.add_argument("--no-copy-images", dest="copy_images", action="store_false")
    return p.parse_args()


def load_coco(ann_path: Path) -> Tuple[Dict[int, Img], Dict[int, str], Dict[int, List[Ann]]]:
    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}
    images = {
        im["id"]: Img(
            image_id=im["id"],
            file_name=im["file_name"],
            width=im["width"],
            height=im["height"],
        )
        for im in data["images"]
    }

    by_image: Dict[int, List[Ann]] = {}
    for a in data["annotations"]:
        if a.get("iscrowd", 0) == 1:
            continue
        cat_id = a["category_id"]
        x, y, w, h = a["bbox"]
        ann = Ann(
            ann_id=a["id"],
            image_id=a["image_id"],
            category_id=cat_id,
            category_name=categories[cat_id],
            box=Box(x=x, y=y, w=w, h=h),
        )
        by_image.setdefault(a["image_id"], []).append(ann)

    return images, categories, by_image


def infer_relation(
    subj: Box,
    obj: Box,
    axis_gap: float,
    dominance_ratio: float,
) -> Optional[str]:
    dx = subj.cx - obj.cx
    dy = subj.cy - obj.cy

    adx = abs(dx)
    ady = abs(dy)

    if adx >= axis_gap and adx >= dominance_ratio * ady:
        return "left" if dx < 0 else "right"
    if ady >= axis_gap and ady >= dominance_ratio * adx:
        return "above" if dy < 0 else "below"
    return None


def flip_relation(rel: str, transform: str) -> str:
    if transform == "original":
        return rel
    if transform == "hflip":
        if rel == "left":
            return "right"
        if rel == "right":
            return "left"
        return rel
    if transform == "vflip":
        if rel == "above":
            return "below"
        if rel == "below":
            return "above"
        return rel
    raise ValueError(f"Unknown transform: {transform}")


def flip_box(box: Box, width: int, height: int, transform: str) -> Box:
    if transform == "original":
        return Box(box.x, box.y, box.w, box.h)
    if transform == "hflip":
        return Box(width - (box.x + box.w), box.y, box.w, box.h)
    if transform == "vflip":
        return Box(box.x, height - (box.y + box.h), box.w, box.h)
    raise ValueError(f"Unknown transform: {transform}")


def iter_candidate_pairs(
    anns: List[Ann],
    min_box_area: float,
    allowed_categories: Optional[set],
) -> Iterable[Tuple[Ann, Ann]]:
    filtered = []
    for a in anns:
        if a.box.area < min_box_area:
            continue
        if allowed_categories and a.category_name not in allowed_categories:
            continue
        filtered.append(a)

    for a, b in itertools.permutations(filtered, 2):
        if a.ann_id == b.ann_id:
            continue
        yield a, b


def load_and_transform_image(src_path: Path, dst_path: Path, transform: str) -> None:
    from PIL import Image

    with Image.open(src_path) as im:
        if transform == "original":
            out = im.copy()
        elif transform == "hflip":
            out = im.transpose(Image.FLIP_LEFT_RIGHT)
        elif transform == "vflip":
            out = im.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            raise ValueError(f"Unknown transform: {transform}")
        out.save(dst_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = args.output_dir
    img_out = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_out.mkdir(parents=True, exist_ok=True)

    transforms = [t.strip() for t in args.transforms.split(",") if t.strip()]
    for t in transforms:
        if t not in {"original", "hflip", "vflip"}:
            raise ValueError(f"Unsupported transform: {t}")

    allowed_categories = None
    if args.allowed_categories.strip():
        allowed_categories = {x.strip() for x in args.allowed_categories.split(",") if x.strip()}

    images, _, anns_by_image = load_coco(args.annotations)
    image_ids = list(anns_by_image.keys())
    random.shuffle(image_ids)
    if args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"

    rows_written = 0
    relation_counts: Dict[str, int] = {r: 0 for r in RELATIONS}
    transform_counts: Dict[str, int] = {t: 0 for t in transforms}

    with manifest_path.open("w", encoding="utf-8") as mf:
        for image_id in image_ids:
            img = images[image_id]
            src_path = args.images_dir / img.file_name
            if not src_path.exists():
                continue

            # Bug 2 fix: collect all candidate pairs, infer relations first,
            # then cap so we always get max_pairs_per_image valid pairs.
            all_candidates = list(
                iter_candidate_pairs(
                    anns_by_image[image_id],
                    min_box_area=args.min_box_area,
                    allowed_categories=allowed_categories,
                )
            )
            random.shuffle(all_candidates)

            valid_pairs: List[Tuple[Ann, Ann, str]] = []
            for subj_ann, obj_ann in all_candidates:
                rel = infer_relation(
                    subj_ann.box,
                    obj_ann.box,
                    axis_gap=args.axis_gap,
                    dominance_ratio=args.dominance_ratio,
                )
                if rel is not None:
                    valid_pairs.append((subj_ann, obj_ann, rel))
                if len(valid_pairs) >= args.max_pairs_per_image:
                    break

            for subj_ann, obj_ann, rel in valid_pairs:
                for t in transforms:
                    rel_t = flip_relation(rel, t)
                    subj_box_t = flip_box(subj_ann.box, img.width, img.height, t)
                    obj_box_t = flip_box(obj_ann.box, img.width, img.height, t)

                    stem = Path(img.file_name).stem
                    ext = Path(img.file_name).suffix or ".jpg"
                    out_name = f"{stem}__{image_id}__{subj_ann.ann_id}_{obj_ann.ann_id}__{t}{ext}"
                    dst_path = img_out / out_name

                    if args.copy_images and not dst_path.exists():
                        load_and_transform_image(src_path, dst_path, t)

                    base_record = {
                        "image_id": image_id,
                        "source_file": img.file_name,
                        "output_file": str(Path("images") / out_name),
                        "transform": t,
                        "true_relation": rel_t,
                        "subject": {
                            "ann_id": subj_ann.ann_id,
                            "category": subj_ann.category_name,
                            "bbox_xywh": [subj_box_t.x, subj_box_t.y, subj_box_t.w, subj_box_t.h],
                        },
                        "object": {
                            "ann_id": obj_ann.ann_id,
                            "category": obj_ann.category_name,
                            "bbox_xywh": [obj_box_t.x, obj_box_t.y, obj_box_t.w, obj_box_t.h],
                        },
                    }

                    # Bug 3 fix: explicit label field.
                    # Bug 1 fix: emit positive row + one foil (negative) row.
                    positive = {**base_record,
                                "relation": rel_t,
                                "label": 1,
                                "question": f"Is the {subj_ann.category_name} {rel_t} the {obj_ann.category_name}?"}
                    mf.write(json.dumps(positive) + "\n")
                    rows_written += 1
                    relation_counts[rel_t] += 1
                    transform_counts[t] += 1

                    # Foil: pick a wrong relation uniformly at random.
                    wrong_rels = [r for r in RELATIONS if r != rel_t]
                    foil_rel = random.choice(wrong_rels)
                    foil = {**base_record,
                            "relation": foil_rel,
                            "label": 0,
                            "question": f"Is the {subj_ann.category_name} {foil_rel} the {obj_ann.category_name}?"}
                    mf.write(json.dumps(foil) + "\n")
                    rows_written += 1
                    relation_counts[foil_rel] += 1
                    transform_counts[t] += 1

    summary = {
        "rows_written": rows_written,
        "num_images_considered": len(image_ids),
        "transforms": transforms,
        "relation_counts": relation_counts,
        "transform_counts": transform_counts,
        "params": {
            "max_images": args.max_images,
            "max_pairs_per_image": args.max_pairs_per_image,
            "min_box_area": args.min_box_area,
            "axis_gap": args.axis_gap,
            "dominance_ratio": args.dominance_ratio,
            "allowed_categories": sorted(allowed_categories) if allowed_categories else None,
        },
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print("REAL-IMAGE SPATIAL DATASET BUILD COMPLETE")
    print("=" * 60)
    print(f"Rows written: {rows_written}")
    print(f"Manifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print(f"Images:   {img_out}")


if __name__ == "__main__":
    main()
