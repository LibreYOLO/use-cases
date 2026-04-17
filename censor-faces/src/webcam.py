"""Live webcam face censor.

Opens your default camera, runs the trained face detector on every frame,
and Gaussian-blurs every face it finds. Real-time. Press q or ESC to quit.

Usage:
    python -m src.webcam
    python -m src.webcam --weights weights/face-snapshot.pt --conf 0.30
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# RF-DETR uses an op MPS doesn't implement; fall back for that one op only.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch

from libreyolo import LibreYOLO


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def blur_boxes(frame, boxes, pad: float = 0.10) -> None:
    h, w = frame.shape[:2]
    for x1, y1, x2, y2 in boxes:
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * pad), int(bh * pad)
        x1 = max(0, int(x1) - px)
        y1 = max(0, int(y1) - py)
        x2 = min(w, int(x2) + px)
        y2 = min(h, int(y2) + py)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = frame[y1:y2, x1:x2]
        k = max(31, (min(x2 - x1, y2 - y1) // 4) | 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=Path("weights/face-snapshot.pt"))
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--device", type=str, default=pick_device())
    args = p.parse_args()

    if not args.weights.exists():
        print(f"missing weights: {args.weights}", file=sys.stderr)
        return 1

    print(f"loading {args.weights} on {args.device}")
    model = LibreYOLO(str(args.weights), nb_classes=1, device=args.device)

    print(f"opening camera index {args.camera}")
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"could not open camera {args.camera}", file=sys.stderr)
        return 1

    print("press q or ESC to quit")
    last = time.time()
    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            print("frame grab failed", file=sys.stderr)
            break

        # libreyolo expects RGB; OpenCV gives BGR.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb, conf=args.conf, save=False)

        boxes = []
        if hasattr(results, "boxes") and hasattr(results.boxes, "xyxy"):
            xyxy = results.boxes.xyxy.cpu().tolist() if len(results.boxes.xyxy) else []
            for row in xyxy:
                boxes.append(tuple(row))
        elif isinstance(results, list) and results:
            r0 = results[0]
            if hasattr(r0, "boxes") and len(r0.boxes.xyxy):
                for row in r0.boxes.xyxy.cpu().tolist():
                    boxes.append(tuple(row))

        blur_boxes(frame, boxes)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
        last = now
        cv2.putText(
            frame,
            f"{len(boxes)} face(s)  {fps:4.1f} fps  conf>={args.conf:.2f}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        cv2.imshow("censor faces (q to quit)", frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
