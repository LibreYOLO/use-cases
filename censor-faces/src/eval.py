"""Evaluate trained face detector on the WIDERFACE val split.

Usage:
    python -m src.eval --weights weights/face.pt --data data/widerface/data.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from libreyolo import LibreYOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/face.pt"))
    p.add_argument("--data", type=Path, default=Path("data/widerface/data.yaml"))
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing weights: {args.weights}. Run: python -m src.train", file=sys.stderr)
        return 1

    print(f"loading {args.weights}")
    model = LibreYOLO(str(args.weights), nb_classes=1, device=args.device)
    metrics = model.val(data=str(args.data), imgsz=args.imgsz, device=args.device)
    print("\nmetrics:")
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
