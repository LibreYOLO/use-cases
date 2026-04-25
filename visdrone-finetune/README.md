# VisDrone Fine-Tune. Aerial object detection. 100% MIT.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LibreYOLO/use-cases/blob/main/visdrone-finetune/notebooks/pipeline.ipynb)
[![PyPI](https://img.shields.io/pypi/v/libreyolo?label=libreyolo)](https://pypi.org/project/libreyolo/)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

Fine-tune LibreYOLO9 on the [VisDrone2019-DET](http://aiskyeye.com/) aerial-imagery benchmark, end to end. Ten classes (pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor), ~6.5k train / 550 val images, top-down drone perspective. COCO-pretrained weights zero-shot are notably bad on this distribution — fine-tuning for even 20 epochs recovers a lot.

## Path 1: use it in the browser (zero install)

**Not yet available.** Needs ONNX-exported weights on HuggingFace. Planned once we have a finished VisDrone checkpoint to host. See [path 3](#path-3-build-it-under-an-hour) in the meantime.

## Path 2: use it in Python (once weights exist)

Once a `LibreYOLO/visdrone-yolo9s` HF repo is published, this will be the one-liner. For now, train your own (path 3) and use `src.infer` with your local checkpoint:

```bash
pip install -r requirements.txt

python -m src.infer --weights weights/visdrone.pt --source aerial.jpg
# writes aerial.detected.jpg
```

Works on any still image or video (`.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`). Videos also get per-frame tracking via LibreYOLO's `model.track()`.

## Path 3: build it (under an hour)

For when you want to learn the pipeline, fine-tune on a different aerial dataset, or just not trust someone else's weights.

```bash
git clone https://github.com/LibreYOLO/use-cases
cd use-cases/visdrone-finetune
pip install -r requirements.txt

python -m src.download_visdrone                  # ~1.5 GB from HF, 2-5 min
python -m src.train --epochs 20                  # ~30-45 min on MPS / CUDA
python -m src.eval                               # mAP on val
python -m src.infer --source my_drone_frame.jpg  # detect with your own weights
```

### The pipeline

- **`src/download_visdrone.py`** — snapshot Voxel51/VisDrone2019-DET from HuggingFace (FiftyOne format), convert to YOLO polygon/bbox layout (per-image `.txt` files, standard `images/{train,val}` + `labels/{train,val}` + `data.yaml`). Supports `--raw <path-to-unzipped-VisDrone>` for when you already have the official release.
- **`src/train.py`** — LibreYOLO9-s fine-tune (10 classes), COCO weights auto-downloaded, MPS / CUDA autodetect, default 20 epochs / batch 8 / imgsz 640. Writes best checkpoint to `weights/visdrone.pt`.
- **`src/eval.py`** — COCO-style mAP50 / mAP50-95 on the val split.
- **`src/infer.py`** — image OR video inference with the trained weights; videos get per-frame tracking via `model.track()`.

### Key VisDrone conversion details

The dataset's annotation format (`bbox_left,bbox_top,bbox_width,bbox_height,score,category,truncation,occlusion`) has two categories that **must be dropped**: `0` (ignored-regions) and `11` (others). Remaining 10 map cleanly to YOLO classes 0..9. `src/common.py` handles this + edge-of-image bbox clamping.

### Scaling the run

Defaults are a laptop-friendly demo. For a real VisDrone benchmark result:

- `--size m` or `--size c` for larger model.
- `--epochs 100`, `--batch 16`, `--imgsz 640` on an A10G or better.
- Consider adding `--lr0 0.01` with standard YOLOv9 schedule.

### Classes

`pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor`.

## Attribution

VisDrone2019-DET: Zhu et al. *"Vision Meets Drones: Past, Present and Future."* [arXiv:2001.06303](https://arxiv.org/abs/2001.06303). Dataset license applies to the imagery; our code here is MIT.
