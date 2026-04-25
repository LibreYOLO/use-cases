"""Evaluate a VisDrone-finetuned LibreYOLO9 checkpoint.

Usage:
    python -m src.eval --weights weights/visdrone.pt --data data/visdrone/data.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from libreyolo import LibreYOLO  # noqa: E402


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/visdrone.pt"))
    p.add_argument("--data", type=Path, default=Path("data/visdrone/data.yaml"))
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing {args.weights}. Run src.train first.", file=sys.stderr)
        return 1
    if not args.data.exists():
        print(f"missing {args.data}. Run src.download_visdrone first.", file=sys.stderr)
        return 1

    print(f"device: {args.device}")
    model = LibreYOLO(str(args.weights), device=args.device)
    results = model.val(
        data=str(args.data), split="val", batch=args.batch,
        imgsz=args.imgsz, conf=0.001, iou=0.6,
    )

    def _fmt(v):
        try:
            return f"{float(v):.4f}"
        except (TypeError, ValueError):
            return str(v)

    print("\nValidation metrics:")
    for key in ("metrics/mAP50", "metrics/mAP50-95", "metrics/precision", "metrics/recall"):
        if key in results:
            print(f"  {key:<22} = {_fmt(results[key])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
