"""VisDrone → YOLO format conversion. No torch dep."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


# VisDrone category ids 1..10; 0 (ignored-regions) and 11 (others) are skipped.
VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]
VISDRONE_ID_TO_YOLO: Dict[int, int] = {cid: cid - 1 for cid in range(1, 11)}


def visdrone_line_to_yolo(line: str, img_w: int, img_h: int) -> str | None:
    """Convert one VisDrone annotation line to a YOLO-format line.

    VisDrone format: `bbox_left,bbox_top,bbox_width,bbox_height,score,cat,trunc,occ`.
    Returns `"cls cx cy w h"` (normalized, 6 decimals), or None if the line
    should be dropped (ignored category, zero-size, malformed).
    """
    parts = [p.strip() for p in line.strip().split(",") if p.strip()]
    if len(parts) < 6:
        return None
    try:
        x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cat = int(parts[5])
    except ValueError:
        return None
    if cat not in VISDRONE_ID_TO_YOLO:
        return None
    if w <= 0 or h <= 0 or img_w <= 0 or img_h <= 0:
        return None
    cls = VISDRONE_ID_TO_YOLO[cat]
    cx = max(0.0, min(1.0, (x + w / 2) / img_w))
    cy = max(0.0, min(1.0, (y + h / 2) / img_h))
    bw = max(0.0, min(1.0, w / img_w))
    bh = max(0.0, min(1.0, h / img_h))
    if bw <= 0 or bh <= 0:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def image_wh(path: Path) -> Tuple[int, int]:
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def convert_raw_split(images_dir: Path, annotations_dir: Path, out_labels_dir: Path) -> dict:
    """Convert a raw VisDrone split (images/ + annotations/) to YOLO labels."""
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    images = sorted(p for p in images_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    summary = {"images": 0, "labels_written": 0, "skipped_images": 0, "skipped_lines": 0}
    for img in images:
        ann = annotations_dir / (img.stem + ".txt")
        if not ann.exists():
            summary["skipped_images"] += 1
            continue
        w, h = image_wh(img)
        kept: List[str] = []
        for line in ann.read_text().splitlines():
            y = visdrone_line_to_yolo(line, w, h)
            if y is None:
                summary["skipped_lines"] += 1
            else:
                kept.append(y)
        (out_labels_dir / (img.stem + ".txt")).write_text("\n".join(kept))
        summary["images"] += 1
        summary["labels_written"] += len(kept)
    return summary


def convert_fiftyone_dataset(root: Path, out_root: Path, split_ratio: float = 0.9) -> dict:
    """Convert Voxel51's FiftyOne-format VisDrone mirror to YOLO layout.

    The Voxel51/VisDrone2019-DET HF dataset ships a flat `data/` of images
    plus a `samples.json` with detections in FiftyOne format. We split
    deterministically and write the YOLO directory layout.
    """
    samples_file = root / "samples.json"
    data_dir = root / "data"
    if not samples_file.exists() or not data_dir.exists():
        raise FileNotFoundError(
            f"Expected {samples_file} and {data_dir}. Is this a Voxel51/FiftyOne "
            "snapshot? Fallback: pass --raw <path-to-unzipped-VisDrone2019-DET>."
        )

    samples = json.loads(samples_file.read_text()).get("samples", [])
    if not samples:
        raise RuntimeError("No samples in samples.json")

    summary = {"train": 0, "val": 0, "labels_written": 0, "skipped_lines": 0}
    for split in ("train", "val"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = int(len(samples) * split_ratio)
    for idx, s in enumerate(samples):
        split = "train" if idx < n_train else "val"
        rel = s.get("filepath", "").split("data/", 1)[-1]
        img_src = data_dir / rel
        if not img_src.exists():
            continue
        img_dst = out_root / "images" / split / img_src.name
        if not img_dst.exists():
            img_dst.symlink_to(img_src.resolve())

        w, h = image_wh(img_src)
        detections = (s.get("ground_truth") or {}).get("detections") or []
        lines: List[str] = []
        for det in detections:
            label = det.get("label", "").lower()
            if label not in VISDRONE_CLASSES:
                summary["skipped_lines"] += 1
                continue
            cls = VISDRONE_CLASSES.index(label)
            bb = det.get("bounding_box")  # [x, y, w, h] normalized
            if not bb or len(bb) != 4:
                summary["skipped_lines"] += 1
                continue
            bx, by, bw, bh = bb
            cx = max(0.0, min(1.0, bx + bw / 2))
            cy = max(0.0, min(1.0, by + bh / 2))
            bw = max(0.0, min(1.0, bw))
            bh = max(0.0, min(1.0, bh))
            if bw <= 0 or bh <= 0:
                summary["skipped_lines"] += 1
                continue
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        (out_root / "labels" / split / f"{img_src.stem}.txt").write_text("\n".join(lines))
        summary[split] += 1
        summary["labels_written"] += len(lines)

    return summary


def write_data_yaml(out_root: Path) -> Path:
    import yaml
    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(yaml.dump({
        "path": str(out_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(VISDRONE_CLASSES),
        "names": VISDRONE_CLASSES,
    }, sort_keys=False))
    return data_yaml
