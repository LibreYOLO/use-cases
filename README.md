# LibreYOLO Use Cases

**Real YOLO. Real places. 100% MIT.**

Use cases for the [LibreYOLO](https://github.com/LibreYOLO/libreyolo) family. Each folder is an independent demo you can fork.

## Catalog

| Use case | What it does | Live demo | Stack |
|---|---|---|---|
| [quickstart](quickstart/) | Detect objects in an image and export to ONNX, in 2 minutes | [Colab](https://colab.research.google.com/github/LibreYOLO/use-cases/blob/main/quickstart/predict.ipynb) | `libreyolo` (Python) |
| [chromium](chromium/) | Webcam object detection in Chrome / Edge | [Open](https://libreyolo.github.io/use-cases/chromium/) | `libreyolo-web` + WebGPU |
| [blur-people](blur-people/) | Real-time people blurring in the webcam feed | [Open](https://libreyolo.github.io/use-cases/blur-people/) | `libreyolo-web` + canvas blur |
| [blur-faces](blur-faces/) | Browser face blur, Python CLI, or train your own | [Open](https://libreyolo.github.io/use-cases/blur-faces/demo/) | `onnxruntime-web` + `libreyolo` (Python) |
| [gaze-detector](gaze-detector/) | "Is someone looking at my screen?" — webcam gaze detection + shoulder-surfer overlay | Local (Python) | `libreyolo[gaze]` + L2CS |

## Adding a new use case

Copy an existing folder, replace the contents. We'll extract a template the day three folders look identical.

## License

MIT.
