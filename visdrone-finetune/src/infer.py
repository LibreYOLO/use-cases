"""Run VisDrone-finetuned detection on an image or video.

Usage:
    python -m src.infer --weights weights/visdrone.pt --source aerial.jpg
    python -m src.infer --weights weights/visdrone.pt --source drone.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from libreyolo import LibreYOLO  # noqa: E402


VISDRONE_CLASSES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/visdrone.pt"))
    p.add_argument("--source", type=Path, required=True, help="Image or video file.")
    p.add_argument("--out", type=Path, default=None,
                   help="Annotated output. Default: <source>.detected.<ext>")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing {args.weights}. Run src.train first.", file=sys.stderr)
        return 1
    if not args.source.exists():
        print(f"missing {args.source}", file=sys.stderr)
        return 1

    model = LibreYOLO(str(args.weights), device=args.device)
    is_video = args.source.suffix.lower() in (".mp4", ".avi", ".mkv", ".mov", ".webm")
    out = args.out or args.source.with_suffix(f".detected{args.source.suffix}")
    out.parent.mkdir(parents=True, exist_ok=True)

    if is_video:
        print(f"running detection on video {args.source} → {out}")
        for frame in model.track(
            source=str(args.source), track_conf=args.conf, iou=args.iou,
            imgsz=args.imgsz, save=True, output_path=str(out),
        ):
            pass  # model.track yields per-frame Results; we just drain it.
    else:
        print(f"running detection on image {args.source} → {out}")
        result = model(str(args.source), conf=args.conf, iou=args.iou,
                       imgsz=args.imgsz, save=True, output_path=str(out))
        res = result[0] if isinstance(result, list) else result
        print(f"  detections: {len(res.boxes)}")
        if len(res.boxes) > 0:
            for cls_id, conf in zip(res.boxes.cls.long().tolist(), res.boxes.conf.tolist()):
                label = VISDRONE_CLASSES[cls_id] if 0 <= cls_id < 10 else f"cls_{cls_id}"
                print(f"    {label:<18}  conf={conf:.3f}")

    print(f"\nannotated → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
