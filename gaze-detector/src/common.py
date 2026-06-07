"""Shared gaze pipeline: weights resolution, face detection, gaze test.

Both entry points (`src.gaze_console` and `src.overlay`) build on this so the
model-loading and "is this person looking at the screen?" logic lives in one
place.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from libreyolo import LibreL2CS
from libreyolo.models.l2cs.face import FaceBox


# Use-case root (the gaze-detector/ folder), so weights/ resolves the same way
# no matter which working directory you launch from.
ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "LibreL2CSr50.pt"

CAM_INDEX = 0
INFER_EVERY = 2  # run gaze every Nth frame to save CPU

# "Looking at screen" thresholds — tweak after running once.
# Looking directly at the camera produces pitch ≈ 0, yaw ≈ 0.
YAW_THRESHOLD_DEG = 20.0
PITCH_THRESHOLD_DEG = 15.0


def is_gazing(pitch_deg: float, yaw_deg: float) -> bool:
    """True when the head pose points at the screen within the thresholds."""
    return abs(yaw_deg) < YAW_THRESHOLD_DEG and abs(pitch_deg) < PITCH_THRESHOLD_DEG


def load_model(device: str = "cpu") -> LibreL2CS:
    """Load the L2CS gaze estimator.

    If the checkpoint is missing and ``libreyolo[gaze]`` is installed, LibreYOLO
    auto-downloads it from the official Gaze360 Google Drive (research /
    non-commercial use only — see the README).
    """
    return LibreL2CS(str(WEIGHTS), size="r50", device=device)


def make_haar_face_detector():
    """Return a callable(image_rgb) -> list[FaceBox] using OpenCV's Haar cascade.

    No extra weights needed — it ships with opencv-python. Good enough for one
    face up close; swap in a stronger detector if you need crowds or distance.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Failed to load Haar cascade at {cascade_path}")

    def detect(image_rgb: np.ndarray):
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60),
        )
        return [
            FaceBox(xyxy=(float(x), float(y), float(x + w), float(y + h)), score=0.99)
            for (x, y, w, h) in faces
        ]

    return detect


def pick_main_face(boxes_xyxy: np.ndarray) -> int:
    """Return the index of the largest face (usually the closest to the camera)."""
    areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
    return int(np.argmax(areas))


def to_pil(frame_bgr: np.ndarray) -> Image.Image:
    """OpenCV BGR frame -> RGB PIL image for the model."""
    return Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
