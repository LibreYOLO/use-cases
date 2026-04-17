# LibreYOLO Use Cases

**Real YOLO. Real places. 100% MIT.**

Use cases for the [LibreYOLO](https://github.com/LibreYOLO/libreyolo) family. Each folder is an independent demo you can fork.

## Catalog

| Use case | What it does | Live demo | Stack |
|---|---|---|---|
| [chromium](chromium/) | Webcam object detection in Chrome / Edge | [Open](https://libreyolo.github.io/use-cases/chromium/) | `libreyolo-web` + WebGPU |
| [blur-people](blur-people/) | Real-time people blurring in the webcam feed | [Open](https://libreyolo.github.io/use-cases/blur-people/) | `libreyolo-web` + canvas blur |
| [censor-faces](censor-faces/) | Train your own face detector and blur faces | [Colab](https://colab.research.google.com/github/LibreYOLO/use-cases/blob/main/censor-faces/notebooks/pipeline.ipynb) | `libreyolo` (Python) + RF-DETR Nano |

## Adding a new use case

Copy an existing folder, replace the contents. We'll extract a template the day three folders look identical.

## License

MIT.
