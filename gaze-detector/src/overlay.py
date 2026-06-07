"""gaze-detector overlay — a silent background app.

Behavior:
- No visible UI while nobody is looking at the screen.
- The moment someone's gaze points at the screen, a fullscreen, always-on-top
  window shows a freeze-frame of their face (plus a caption and a sound).
- When they look away, the window hides and your original content is visible
  again.

A playful demo of "is a person looking at my screen right now?" built on
LibreYOLO's L2CS gaze estimator. Run it, then open something on screen and
have someone glance over your shoulder.

Run from the terminal:
    python -m src.overlay
Stop with Ctrl+C in that terminal (or press Esc on the overlay).
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from .common import (
    CAM_INDEX,
    INFER_EVERY,
    ROOT,
    is_gazing,
    load_model,
    make_haar_face_detector,
    pick_main_face,
)


CROP_PADDING = 0.6  # extra fraction of face box to include around the crop
HOLD_MS = 6000  # keep the image up at least this long after gaze ends
CAPTION = "I SEE YOU"

# Sound played on each new catch. Drop a short .wav/.mp3/.m4a here.
# Falls back to a system beep if the file is missing.
SOUND_FILE = ROOT / "sounds" / "catch.wav"


def _popen_silent(cmd):
    subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def play_sound():
    """Play the catch sound without blocking. Cross-platform, best-effort.

    macOS uses ``afplay``, Windows uses the stdlib ``winsound`` module, and
    Linux tries the usual CLI players (``paplay``/``aplay``/``ffplay``). If the
    .wav is missing or no player is available it falls back to a system sound /
    beep, and silently does nothing if even that fails — audio is never allowed
    to crash the overlay.
    """
    have_file = SOUND_FILE.exists()
    try:
        if sys.platform == "darwin":
            src = str(SOUND_FILE) if have_file else "/System/Library/Sounds/Glass.aiff"
            _popen_silent(["afplay", src])
        elif sys.platform == "win32":
            import winsound

            if have_file:
                winsound.PlaySound(
                    str(SOUND_FILE),
                    winsound.SND_FILENAME | winsound.SND_ASYNC,
                )
            else:
                winsound.MessageBeep()
        else:  # Linux / other POSIX
            if not have_file:
                # No bundled file and no portable system sound — best-effort beep.
                _popen_silent(["printf", "\a"])
                return
            for player in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
                try:
                    _popen_silent(player + [str(SOUND_FILE)])
                    return
                except FileNotFoundError:
                    continue
    except Exception:
        pass


def crop_with_padding(frame_rgb: np.ndarray, xyxy: tuple, pad: float) -> np.ndarray:
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad, bh * pad
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    return frame_rgb[y1:y2, x1:x2]


# -----------------------------------------------------------------------------
# Vision thread
# -----------------------------------------------------------------------------

def vision_loop(events: queue.Queue, stop_event: threading.Event):
    print("Loading L2CS ...")
    model = load_model(device="cpu")
    face_detector = make_haar_face_detector()
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {CAM_INDEX}")
    print("Vision running. Look at the screen to trigger. Ctrl+C to stop.")

    gazing_state = False
    frame_idx = 0

    try:
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                time.sleep(0.05)
                continue

            if frame_idx % INFER_EVERY != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)

            try:
                result = model(pil, face_detector=face_detector)
            except Exception:
                continue

            new_state = False
            face_crop = None
            if result.gaze is not None and len(result.gaze) > 0:
                xyxy = result.boxes.xyxy
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                idx = pick_main_face(xyxy)
                pitch = float(result.gaze.pitch_deg[idx])
                yaw = float(result.gaze.yaw_deg[idx])
                if is_gazing(pitch, yaw):
                    new_state = True
                    face_crop = crop_with_padding(
                        frame_rgb, tuple(xyxy[idx]), CROP_PADDING
                    )

            if new_state and not gazing_state:
                events.put(("on", face_crop))
                gazing_state = True
            elif not new_state and gazing_state:
                events.put(("off", None))
                gazing_state = False
    finally:
        cap.release()


# -----------------------------------------------------------------------------
# UI thread (main)
# -----------------------------------------------------------------------------

class Overlay:
    def __init__(self, events: queue.Queue, stop_event: threading.Event):
        self.events = events
        self.stop_event = stop_event
        self.root = tk.Tk()
        # Hide IMMEDIATELY before anything else — avoid the white-flash on macOS.
        self.root.withdraw()
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        self.root.title("gaze-detector")
        self.root.configure(bg="black")
        # IMPORTANT for macOS: don't use -fullscreen, it creates a new Space.
        # Use a borderless window sized to the screen instead — stays on the
        # current Space, on top of your actual content.
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)

        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{sw}x{sh}+0+0")
        self.sw, self.sh = sw, sh
        self.canvas = tk.Canvas(self.root, width=sw, height=sh,
                                bg="black", highlightthickness=0,
                                borderwidth=0)
        self.canvas.pack(expand=True, fill="both")
        self._photo = None  # keep ref so it's not GC'd
        self._image_id = None
        self._caption_shadow_id = None
        self._caption_id = None
        self._pending_hide = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", lambda e: self._on_close())
        self.root.after(50, self._poll)

    def _on_close(self):
        self.stop_event.set()
        self.root.after(50, self.root.destroy)

    def _fill_screen_crop(self, face_rgb: np.ndarray) -> Image.Image:
        """Center-crop face to screen aspect ratio, then scale to fill screen.

        Result is exactly (sw, sh) — no black bars.
        """
        img = Image.fromarray(face_rgb)
        iw, ih = img.size
        screen_aspect = self.sw / self.sh
        image_aspect = iw / ih
        if image_aspect > screen_aspect:
            # Image is wider than screen — crop sides
            new_w = int(ih * screen_aspect)
            x0 = (iw - new_w) // 2
            img = img.crop((x0, 0, x0 + new_w, ih))
        else:
            # Image is taller than screen — crop top/bottom
            new_h = int(iw / screen_aspect)
            y0 = (ih - new_h) // 2
            img = img.crop((0, y0, iw, y0 + new_h))
        return img.resize((self.sw, self.sh), Image.LANCZOS)

    def _show(self, face_rgb: np.ndarray):
        # Play the catch sound only when appearing fresh (not on a re-show
        # while already visible during continuous gaze).
        was_hidden = self._pending_hide is not None or self._image_id is None \
            or self.root.state() == "withdrawn"
        # Cancel any pending hide — we're visible again.
        if self._pending_hide is not None:
            self.root.after_cancel(self._pending_hide)
            self._pending_hide = None
        if was_hidden:
            play_sound()

        img = self._fill_screen_crop(face_rgb)
        self._photo = ImageTk.PhotoImage(img)

        if self._image_id is None:
            self._image_id = self.canvas.create_image(
                self.sw // 2, self.sh // 2, image=self._photo
            )
        else:
            self.canvas.itemconfigure(self._image_id, image=self._photo)

        # Caption — big, bottom-center, with a shadow for legibility.
        font_size = max(48, self.sh // 12)
        font = ("Helvetica", font_size, "bold")
        cy = self.sh - font_size  # near bottom
        if self._caption_id is None:
            self._caption_shadow_id = self.canvas.create_text(
                self.sw // 2 + 4, cy + 4,
                text=CAPTION, fill="black", font=font, anchor="s",
            )
            self._caption_id = self.canvas.create_text(
                self.sw // 2, cy,
                text=CAPTION, fill="white", font=font, anchor="s",
            )
        else:
            self.canvas.itemconfigure(self._caption_shadow_id, text=CAPTION)
            self.canvas.itemconfigure(self._caption_id, text=CAPTION)
            # Make sure caption is above the image
            self.canvas.tag_raise(self._caption_shadow_id)
            self.canvas.tag_raise(self._caption_id)

        self.root.deiconify()
        self.root.lift()
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 1.0)
        except tk.TclError:
            pass

    def _hide_now(self):
        self._pending_hide = None
        try:
            self.root.attributes("-alpha", 0.0)
        except tk.TclError:
            pass
        self.root.withdraw()

    def _schedule_hide(self):
        # Defer hiding so the image stays at least HOLD_MS.
        if self._pending_hide is not None:
            return  # already scheduled
        self._pending_hide = self.root.after(HOLD_MS, self._hide_now)

    def _poll(self):
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "on" and payload is not None:
                    self._show(payload)
                elif kind == "off":
                    self._schedule_hide()
        except queue.Empty:
            pass
        if self.stop_event.is_set():
            self.root.destroy()
            return
        self.root.after(50, self._poll)

    def run(self):
        self.root.mainloop()


def main():
    events: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    vt = threading.Thread(
        target=vision_loop, args=(events, stop_event), daemon=True
    )
    vt.start()
    try:
        Overlay(events, stop_event).run()
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()


if __name__ == "__main__":
    main()
