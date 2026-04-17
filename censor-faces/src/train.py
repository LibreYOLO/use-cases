"""Train LibreRFDETR Nano on the WIDERFACE subset.

Default config is the smoke-test scope: 2 epochs on 80 training images, batch 4,
input 384, MPS or CUDA if available. Plenty fast for a "did this learn anything"
gut check; not a finished face detector.

Usage:
    python -m src.train --data data/widerface/data.yaml --epochs 2
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# RF-DETR uses grid_sampler_2d_backward which is not yet implemented on MPS.
# This env var makes PyTorch fall back to CPU for that one op only. Set BEFORE
# importing torch so it takes effect.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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
    p.add_argument("--data", type=Path, default=Path("data/widerface"),
                   help="Dataset directory (Roboflow YOLO layout).")
    p.add_argument("--model", type=str, default="LibreRFDETRn.pt")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--imgsz", type=int, default=384)
    p.add_argument("--device", type=str, default=pick_device())
    p.add_argument("--out", type=Path, default=Path("weights/face.pt"))
    args = p.parse_args()

    if not (args.data / "data.yaml").exists():
        print(f"missing {args.data}/data.yaml. Run: python -m src.download_widerface", file=sys.stderr)
        return 1

    print(f"device: {args.device}")
    print(f"loading {args.model} (auto-downloads pretrained weights from HuggingFace on first run)")
    # nb_classes=1 because the dataset only has the 'face' class.
    model = LibreYOLO(args.model, nb_classes=1, device=args.device)

    print(f"training: {args.epochs} epochs, batch {args.batch}, imgsz {args.imgsz}")
    output_dir = Path("runs/train")
    try:
        model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project="runs/train",
            name="censor_faces",
            exist_ok=True,
        )
    except ZeroDivisionError:
        # Upstream rfdetr crashes in its end-of-training eval logger on tiny
        # validation sets. The trained checkpoints are already on disk by the
        # time this fires, so we just keep going.
        print("(upstream rfdetr eval logger crashed; checkpoints are saved, continuing)")

    candidates = [
        output_dir / "checkpoint_best_total.pth",
        output_dir / "checkpoint_best_ema.pth",
        output_dir / "checkpoint_best_regular.pth",
        output_dir / "checkpoint.pth",
    ]
    src = next((c for c in candidates if c.exists()), None)
    if src is None:
        print(f"no checkpoint found in {output_dir}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, args.out)
    print(f"\nweights ← {src.name} → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
