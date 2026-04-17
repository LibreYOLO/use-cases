"""Download the pretrained face detector and censor an image. Zero training.

The "use it" path: skip the dataset download + training, just blur faces.

Usage:
    python -m src.use_pretrained --image path/to/photo.jpg
    python -m src.use_pretrained --image in.jpg --out out.jpg --conf 0.30
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# RF-DETR uses an MPS-unsupported op; fall back for that one op only.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
from huggingface_hub import hf_hub_download

from libreyolo import LibreYOLO

from src.censor import blur_boxes  # reuse the same blur logic as the build path

HF_REPO = "LibreYOLO/face-rfdetr-nano"
HF_FILE = "face.pt"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.image.exists():
        print(f"missing image: {args.image}", file=sys.stderr)
        return 1
    out_path = args.out or args.image.with_name(args.image.stem + ".censored.jpg")

    print(f"downloading {HF_REPO}/{HF_FILE} (cached on first run)")
    weights = hf_hub_download(repo_id=HF_REPO, filename=HF_FILE)

    print(f"loading on {args.device}")
    model = LibreYOLO(weights, nb_classes=1, device=args.device)

    print(f"detecting in {args.image}")
    results = model(str(args.image), conf=args.conf, save=False)
    boxes = []
    if hasattr(results, "boxes") and len(results.boxes.xyxy):
        for row in results.boxes.xyxy.cpu().tolist():
            boxes.append(tuple(row))

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"could not read image: {args.image}", file=sys.stderr)
        return 1
    print(f"{len(boxes)} face(s), blurring")
    blur_boxes(img, boxes)
    cv2.imwrite(str(out_path), img)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
