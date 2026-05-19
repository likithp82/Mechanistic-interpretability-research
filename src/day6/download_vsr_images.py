#!/usr/bin/env python3
"""Download VSR image files referenced by VSR JSONL splits.

This script reads train/dev/test jsonl files, collects unique image URLs,
and downloads them into a local images directory with resume + retries.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VSR images from jsonl URL fields")
    parser.add_argument(
        "--vsr-root",
        required=True,
        help="Path to VSR dataset root containing train/dev/test jsonl",
    )
    parser.add_argument(
        "--images-dir",
        default=None,
        help="Output directory for downloaded images (default: <vsr-root>/images)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap for quick tests (0 means no limit)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="Curl retry count per image",
    )
    return parser.parse_args()


def iter_jsonl_paths(vsr_root: Path) -> Iterable[Path]:
    for name in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        path = vsr_root / name
        if path.exists():
            yield path


def collect_urls(vsr_root: Path) -> list[tuple[str, str]]:
    seen: Set[str] = set()
    pairs: list[tuple[str, str]] = []

    for path in iter_jsonl_paths(vsr_root):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                url = row.get("image_link")
                image_name = row.get("image")
                if not url or not image_name:
                    continue
                if image_name in seen:
                    continue
                seen.add(image_name)
                pairs.append((image_name, url))

    return pairs


def ensure_curl() -> None:
    if not shutil_which("curl"):
        raise RuntimeError("curl is required but not found in PATH")


def shutil_which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def download_one(image_name: str, url: str, out_dir: Path, retries: int) -> bool:
    out_path = out_dir / image_name

    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "curl",
        "-fL",
        "--continue-at",
        "-",
        "--retry",
        str(retries),
        "--retry-delay",
        "3",
        "--retry-all-errors",
        url,
        "-o",
        str(out_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True

    # Fallback for environments with TLS interception/cert mismatch.
    if url.startswith("https://"):
        http_url = "http://" + url[len("https://") :]
        cmd_http = cmd.copy()
        cmd_http[-3] = http_url
        result_http = subprocess.run(cmd_http, capture_output=True, text=True)
        if result_http.returncode == 0:
            return True

    if out_path.exists() and out_path.stat().st_size == 0:
        out_path.unlink(missing_ok=True)

    return False


def main() -> int:
    args = parse_args()
    vsr_root = Path(args.vsr_root).expanduser().resolve()
    if not vsr_root.exists():
        print(f"[error] VSR root does not exist: {vsr_root}")
        return 1

    images_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else (vsr_root / "images")
    ensure_curl()

    pairs = collect_urls(vsr_root)
    if not pairs:
        print("[error] No image URLs found in VSR jsonl files")
        return 1

    if args.max_images > 0:
        pairs = pairs[: args.max_images]

    total = len(pairs)
    success = 0
    failed = 0

    print(f"[info] Unique images to download: {total}")
    print(f"[info] Output directory: {images_dir}")

    for idx, (image_name, url) in enumerate(pairs, start=1):
        ok = download_one(image_name, url, images_dir, args.retries)
        if ok:
            success += 1
        else:
            failed += 1
        if idx % 200 == 0 or idx == total:
            print(f"[progress] {idx}/{total} done | success={success} failed={failed}")

    print("[done]")
    print(f"  downloaded: {success}")
    print(f"  failed    : {failed}")
    print(f"  images_dir: {images_dir}")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
