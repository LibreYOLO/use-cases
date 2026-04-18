# Blur Faces. Zero install. 100% MIT.

[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://libreyolo.github.io/use-cases/blur-faces/)
[![HF model](https://img.shields.io/badge/%F0%9F%A4%97-face--rfdetr--nano-yellow)](https://huggingface.co/LibreYOLO/face-rfdetr-nano)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

Real-time face anonymization in any browser. Detects every face in your webcam feed and Gaussian-blurs it. Runs entirely client-side on an ONNX model fine-tuned from RF-DETR Nano.

## Try it in 60 seconds

**Option 1: live demo** → https://libreyolo.github.io/use-cases/blur-faces/

**Option 2: copy the file**
```bash
curl -O https://raw.githubusercontent.com/LibreYOLO/use-cases/main/blur-faces/index.html
open index.html   # or double-click it
```

The face detector (~108 MB) is pulled from [huggingface.co/LibreYOLO/face-rfdetr-nano](https://huggingface.co/LibreYOLO/face-rfdetr-nano) on first visit and cached by the browser.

## How it works

1. `getUserMedia` gets the webcam as an `HTMLVideoElement`.
2. Each frame is drawn to an `OffscreenCanvas`, resized to 384x384, ImageNet-normalized, fed to `onnxruntime-web`.
3. Output queries are decoded (sigmoid logits, cxcywh boxes) and filtered at confidence 0.30.
4. Every detected bbox is repainted from a Gaussian-blurred crop of the source, clipped to a hard rectangle so corners don't soften into an oval.

## Train your own detector

See the sister use case [censor-faces](../censor-faces/) for the full training pipeline (download WIDERFACE, fine-tune, evaluate, export to ONNX). Drop your own `face.onnx` in place of the HF URL and you have the same demo with your weights.

## FAQ

**Does it work offline?** After first load, yes. The model is cached by the browser. First visit needs internet to fetch `onnxruntime-web` from a CDN and the ONNX from HuggingFace.

**Why is the first load 108 MB?** That's the RF-DETR Nano ONNX. It runs fast once loaded (~10 FPS on WebGPU) but the initial download is real. Subsequent visits are instant.

**Is this face recognition?** No. It detects where faces are so it can blur them. It does not identify anyone.

**Can I use a smaller model?** Yes, if you train one. The `censor-faces` pipeline supports smaller backbones like YOLO9 Tiny or YOLOX Nano; export those to ONNX and swap the URL.

**Is this really MIT?** Yes. Code, model weights, and demo are all MIT licensed. The model was trained on WIDERFACE which has its own license terms for the training data.

## Credits

Detector: [LibreYOLO/face-rfdetr-nano](https://huggingface.co/LibreYOLO/face-rfdetr-nano). Runtime: [onnxruntime-web](https://www.npmjs.com/package/onnxruntime-web). Part of [LibreYOLO](https://github.com/LibreYOLO). MIT. No AGPL.
