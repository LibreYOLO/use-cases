"""Fine-tune LibreYOLO9 on VisDrone.

Default config is a reasonable small-scale run: 20 epochs, batch 8, size 640,
MPS or CUDA if available. For a proper VisDrone result, bump to 50-100 epochs
on an A10G-class GPU.

Usage:
    python -m src.train --data data/visdrone/data.yaml --epochs 20
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from libreyolo import LibreYOLO9  # noqa: E402


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data/visdrone/data.yaml"))
    p.add_argument("--model", type=str, default="LibreYOLO9s.pt",
                   help="Size t/s/m/c — will auto-download from HuggingFace.")
    p.add_argument("--size", type=str, default="s", choices=["t", "s", "m", "c"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.005)
    p.add_argument("--device", type=str, default=pick_device())
    p.add_argument("--out", type=Path, default=Path("weights/visdrone.pt"))
    args = p.parse_args()

    if not args.data.exists():
        print(f"missing {args.data}. Run: python -m src.download_visdrone", file=sys.stderr)
        return 1

    print(f"device: {args.device}")
    print(f"loading {args.model} (auto-downloads COCO-pretrained weights on first run)")
    # nb_classes=10: VisDrone's 10 useful classes (category 0 and 11 dropped).
    model = LibreYOLO9(args.model, size=args.size, nb_classes=10, device=args.device)

    print(f"training: {args.epochs} epochs, batch {args.batch}, imgsz {args.imgsz}, lr0={args.lr0}")
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        project="runs/train",
        name="visdrone",
        exist_ok=True,
    )

    best = Path(results.get("best_checkpoint", ""))
    if not best.exists():
        print(f"no best checkpoint found in results", file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, args.out)
    print(f"\nweights ← {best.name} → {args.out}")
    print(f"best mAP50-95 = {results.get('best_mAP50_95')}  @ epoch {results.get('best_epoch')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
