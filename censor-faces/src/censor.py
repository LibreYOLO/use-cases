"""Detect faces and blur them.

Loads a trained checkpoint (or any LibreRFDETR weights), runs inference on an
image, and Gaussian-blurs every detected face. Pads each box outward to fully
cover the face including hair / chin edges.

Usage:
    python -m src.censor --weights weights/face.pt --image path/to/photo.jpg
    python -m src.censor --weights weights/face.pt --image in.jpg --out out.jpg --conf 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch

from libreyolo import LibreYOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def blur_boxes(img, boxes, pad: float = 0.15, kernel: int = 31) -> None:
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, int(x1) - px)
        y1 = max(0, int(y1) - py)
        x2 = min(w, int(x2) + px)
        y2 = min(h, int(y2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = img[y1:y2, x1:x2]
        # Kernel must be odd; scale slightly with face size for chunkier blur.
        k = max(kernel, (min(x2 - x1, y2 - y1) // 4) | 1)
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/face.pt"))
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing weights: {args.weights}. Run: python -m src.train", file=sys.stderr)
        return 1
    if not args.image.exists():
        print(f"missing image: {args.image}", file=sys.stderr)
        return 1

    out_path = args.out or args.image.with_name(args.image.stem + ".censored.jpg")

    print(f"loading {args.weights}")
    model = LibreYOLO(str(args.weights), nb_classes=1, device=args.device)
    print(f"running inference on {args.image}")
    results = model(image=str(args.image), conf=args.conf, save=False)

    # Normalize whatever shape the API hands back into a list of (x1,y1,x2,y2) ints.
    boxes_xyxy = []
    if isinstance(results, dict):
        for det in results.get("detections", []):
            x1, y1, x2, y2 = det["bbox"]
            boxes_xyxy.append((x1, y1, x2, y2))
    else:
        # Results object: .boxes.xyxy is a tensor (N, 4)
        b = results.boxes if hasattr(results, "boxes") else None
        if b is not None and hasattr(b, "xyxy"):
            for row in b.xyxy.cpu().tolist():
                boxes_xyxy.append(tuple(row))

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1
    print(f"{len(boxes_xyxy)} face(s) detected, blurring")
    blur_boxes(img, boxes_xyxy)
    cv2.imwrite(str(out_path), img)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
