"""Download a tiny WIDERFACE subset for the censor-faces use case.

Pulls the validation split from a HuggingFace mirror, samples N images per
event, writes a YOLO-style data.yaml. Total download is small enough to
fit in a few minutes on a laptop connection.

Usage:
    python -m src.download_widerface --out data/widerface --train-images 80 --val-images 20
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import requests
from tqdm import tqdm

WIDERFACE_VAL_IMAGES_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/WIDER_val.zip"
WIDERFACE_ANNOTATIONS_URL = "https://huggingface.co/datasets/wider_face/resolve/main/data/wider_face_split.zip"


def stream_download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  cached: {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1 << 15):
                f.write(chunk)
                bar.update(len(chunk))


def parse_wider_annotations(annotations_txt: Path) -> Dict[str, List[Tuple[int, int, int, int]]]:
    """Parse wider_face_val_bbx_gt.txt into {image_relpath: [(x, y, w, h), ...]}.

    The format is:
        path/to/image.jpg
        N
        x y w h blur expression illum invalid occlusion pose
        ...
    """
    out: Dict[str, List[Tuple[int, int, int, int]]] = {}
    with annotations_txt.open() as f:
        lines = [ln.strip() for ln in f]
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        path = lines[i]
        n = int(lines[i + 1]) if lines[i + 1] else 0
        boxes: List[Tuple[int, int, int, int]] = []
        if n == 0:
            # Edge case: a "0" count is followed by one zero-padded line.
            i += 3
            out[path] = boxes
            continue
        for j in range(n):
            parts = lines[i + 2 + j].split()
            x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        out[path] = boxes
        i += 2 + n
    return out


def write_yolo_label(boxes: List[Tuple[int, int, int, int]], img_w: int, img_h: int, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h
        # class id 0 = face
        lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("data/widerface"))
    p.add_argument("--train-images", type=int, default=80)
    p.add_argument("--val-images", type=int, default=20)
    args = p.parse_args()

    out: Path = args.out
    cache = out / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    print("1/4 fetching WIDERFACE validation split (images + annotations)")
    images_zip = cache / "WIDER_val.zip"
    ann_zip = cache / "wider_face_split.zip"
    stream_download(WIDERFACE_VAL_IMAGES_URL, images_zip)
    stream_download(WIDERFACE_ANNOTATIONS_URL, ann_zip)

    print("2/4 extracting annotations")
    ann_dir = cache / "wider_face_split"
    if not ann_dir.exists():
        with zipfile.ZipFile(ann_zip) as zf:
            zf.extractall(cache)
    boxes = parse_wider_annotations(ann_dir / "wider_face_val_bbx_gt.txt")
    # Pick images that actually contain at least one face.
    candidates = [p for p, b in boxes.items() if len(b) >= 1 and len(b) <= 6]
    candidates.sort()  # deterministic
    needed = args.train_images + args.val_images
    if len(candidates) < needed:
        print(f"only {len(candidates)} candidate images, need {needed}", file=sys.stderr)
        return 1
    pick = candidates[: needed]
    train_pick, val_pick = pick[: args.train_images], pick[args.train_images :]

    print(f"3/4 extracting {needed} images and writing YOLO labels")
    images_root = cache / "WIDER_val" / "images"
    if not images_root.exists():
        # Selectively extract only the images we need.
        with zipfile.ZipFile(images_zip) as zf:
            wanted = {f"WIDER_val/images/{p}" for p in pick}
            for member in tqdm(zf.namelist(), unit="entry"):
                if member in wanted:
                    zf.extract(member, cache)
    # OpenCV import is lazy: only need it once we read images.
    import cv2

    # Roboflow YOLO layout: <dataset>/<split>/{images,labels}/. The upstream
    # rfdetr trainer requires this exact nesting, plus the val split named "valid".
    splits = {"train": train_pick, "valid": val_pick}
    for split, items in splits.items():
        for relpath in tqdm(items, desc=split):
            src = images_root / relpath
            if not src.exists():
                with zipfile.ZipFile(images_zip) as zf:
                    zf.extract(f"WIDER_val/images/{relpath}", cache)
            img = cv2.imread(str(src))
            if img is None:
                print(f"  skip unreadable: {relpath}", file=sys.stderr)
                continue
            h, w = img.shape[:2]
            stem = relpath.replace("/", "_").replace(".jpg", "")
            dst_img = out / split / "images" / f"{stem}.jpg"
            dst_lbl = out / split / "labels" / f"{stem}.txt"
            dst_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst_img)
            write_yolo_label(boxes[relpath], w, h, dst_lbl)

    print("4/4 writing data.yaml")
    yaml = out / "data.yaml"
    yaml.write_text(
        f"""path: {out.resolve()}
train: train/images
val: valid/images

names:
  0: face
"""
    )
    print(f"\nDone. Dataset at {out.resolve()}")
    print(f"  train: {len(train_pick)} images")
    print(f"  val:   {len(val_pick)} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
