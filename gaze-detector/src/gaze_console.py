"""Webcam → face detect (OpenCV Haar) → L2CS gaze → console.

The simplest end-to-end check that the gaze pipeline works before any UI.
Run it, look at the screen, look away, and watch pitch/yaw in the terminal.

    python -m src.gaze_console

Press 'q' in the preview window to quit.
"""

from __future__ import annotations

import time

import cv2

from .common import (
    CAM_INDEX,
    INFER_EVERY,
    WEIGHTS,
    is_gazing,
    load_model,
    make_haar_face_detector,
    to_pil,
)


def main():
    print(f"Loading L2CS from {WEIGHTS} ...")
    model = load_model(device="cpu")
    face_detector = make_haar_face_detector()
    print("Model loaded. Opening webcam...")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {CAM_INDEX}")

    frame_idx = 0
    last_pitch = last_yaw = None
    t_last = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Webcam read failed.")
                break

            if frame_idx % INFER_EVERY == 0:
                pil = to_pil(frame_bgr)
                try:
                    result = model(pil, face_detector=face_detector)
                except Exception as e:
                    # No face detected -> LibreL2CS may raise; handle gracefully.
                    if "no faces" in str(e).lower():
                        last_pitch = last_yaw = None
                    else:
                        raise
                else:
                    if result.gaze is not None and len(result.gaze) > 0:
                        # Take the first face (largest, usually closest).
                        last_pitch = float(result.gaze.pitch_deg[0])
                        last_yaw = float(result.gaze.yaw_deg[0])
                    else:
                        last_pitch = last_yaw = None

                # Console output: one line per inference.
                now = time.time()
                fps = 1.0 / max(now - t_last, 1e-6) * INFER_EVERY
                t_last = now
                if last_pitch is None:
                    print(f"[fps {fps:5.1f}]  no face")
                else:
                    tag = "GAZING 👀" if is_gazing(last_pitch, last_yaw) else "       -"
                    print(
                        f"[fps {fps:5.1f}]  pitch={last_pitch:+6.1f}°  "
                        f"yaw={last_yaw:+6.1f}°   {tag}"
                    )

            # Tiny preview so you know it's running. Press q to quit.
            preview = frame_bgr.copy()
            if last_pitch is not None:
                gazing = is_gazing(last_pitch, last_yaw)
                color = (0, 255, 0) if gazing else (200, 200, 200)
                txt = f"p={last_pitch:+.0f}  y={last_yaw:+.0f}"
                cv2.putText(preview, txt, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if gazing:
                    cv2.putText(preview, "GAZING", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    cv2.rectangle(preview, (0, 0),
                                  (preview.shape[1]-1, preview.shape[0]-1),
                                  (0, 255, 0), 6)
            cv2.imshow("gaze-detector (q to quit)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
