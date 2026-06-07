# Gaze Detector. "Is someone looking at my screen?" 100% MIT.

[![PyPI](https://img.shields.io/pypi/v/libreyolo?label=libreyolo)](https://pypi.org/project/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

Real-time gaze detection on your webcam with LibreYOLO's **L2CS** estimator. A face detector finds faces, L2CS regresses each head's gaze direction (pitch / yaw), and a simple threshold answers the only question that matters: *is this person looking at the screen right now?*

Two entry points share one pipeline (`src/common.py`):

- **`src.gaze_console`** — a diagnostic: prints live pitch / yaw and a `GAZING 👀` flag, with a small webcam preview. Run this first to sanity-check and tune the thresholds.
- **`src.overlay`** — the demo. A silent background app that stays invisible until a gaze lands on your screen, then slams a fullscreen, always-on-top freeze-frame of the onlooker's face up with an "I SEE YOU" caption (and a sound). Look away and it disappears. Great for catching shoulder-surfers.

## Run it (a few minutes)

```bash
git clone https://github.com/LibreYOLO/use-cases
cd use-cases/gaze-detector
pip install -r requirements.txt

python -m src.gaze_console   # look at the camera, then away; watch pitch/yaw
python -m src.overlay        # the "I SEE YOU" shoulder-surfer overlay
```

Stop either with `q` (console preview) or `Ctrl+C` / `Esc` (overlay).

### Weights — they auto-download

`pip install -r requirements.txt` pulls in `libreyolo[gaze]`, so the first run downloads the L2CS ResNet-50 Gaze360 checkpoint (~96 MB) from the authors' official Google Drive into `weights/`. No manual step.

If Google Drive throttles the download (it rate-limits popular files), grab `L2CSNet_gaze360.pkl` manually from the [L2CS authors' Drive](https://drive.google.com/file/d/18S956r4jnHtSeT8z8t3z8AoJZjVnNqPJ/view) and drop it at `weights/LibreL2CSr50.pt`.

> **License note:** the L2CS weights are trained on the [Gaze360](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md) dataset and are **research / non-commercial use only, no redistribution** — which is why they aren't committed here and aren't mirrored on the LibreYOLO HuggingFace org. The code in this folder is MIT; the weights are not.

## How it works

- **`src/common.py`** — loads `LibreL2CS`, builds an OpenCV Haar face detector (`make_haar_face_detector`), and defines `is_gazing(pitch, yaw)` plus the `YAW_THRESHOLD_DEG` / `PITCH_THRESHOLD_DEG` thresholds. Looking straight at the camera gives pitch ≈ 0, yaw ≈ 0.
- **`src/gaze_console.py`** — webcam loop → first face → gaze → one console line per inference. Use it to find thresholds that match your setup.
- **`src/overlay.py`** — webcam runs on a background thread; the main thread is a borderless, always-on-top Tk window that shows the onlooker's cropped face when `is_gazing` flips true and hides it (after a short hold) when they look away. `pick_main_face` picks the largest (closest) face when several are visible.

### Tuning

- **Sensitivity** — widen or narrow `YAW_THRESHOLD_DEG` / `PITCH_THRESHOLD_DEG` in `src/common.py`. Run `src.gaze_console` and read off the angles at which *you* count as "looking."
- **CPU** — `INFER_EVERY` runs gaze every Nth frame. Bump it on slower machines. The model loads on `cpu` by default; pass `device="mps"`/`"cuda"` in `load_model` if you have it.
- **The catch sound** — drop any short `.wav` at `sounds/catch.wav` (one is included). Missing file falls back to a macOS system sound.
- **The overlay** is macOS-tuned (`afplay` for sound; a borderless screen-sized window instead of `-fullscreen` to avoid spawning a new Space). The vision pipeline is cross-platform; the Tk overlay specifics may need tweaking on Linux/Windows.

### `bait/`

`bait/bank.html` is a fake online-banking page — open it fullscreen as the "sensitive content" you pretend to be reading while `src.overlay` waits for someone to peek. Pure demo dressing; nothing real.

## Attribution

L2CS gaze estimation: Abdelrahman et al. *"L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments."* Trained on the Gaze360 dataset (Kellnhofer et al., ICCV 2019). Dataset/weights license is research-only; the code here is MIT.
