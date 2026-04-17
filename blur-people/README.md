# Blur People. Zero install. 100% MIT.

Real-time privacy filter: detects every person in your webcam feed and blurs them. Runs entirely in your browser.

[![npm](https://img.shields.io/npm/v/libreyolo-web?label=libreyolo-web)](https://www.npmjs.com/package/libreyolo-web)
[![Live demo](https://img.shields.io/badge/demo-live-brightgreen)](https://libreyolo.github.io/use-cases/blur-people/)
[![License](https://img.shields.io/badge/license-MIT-green)](../LICENSE)

## Try it in 60 seconds

**Option 1: live demo** -> https://libreyolo.github.io/use-cases/blur-people/

**Option 2: copy the file**
```bash
curl -O https://raw.githubusercontent.com/LibreYOLO/use-cases/main/blur-people/index.html
open index.html
```

The model auto-downloads on first run (~3.6 MB) and runs entirely in your browser. No server, no Python, no install.

## What it does

- Detects people in the webcam feed with `LibreYOLOXn`, a 3.6 MB COCO model.
- Filters detections to the `person` class, pads each bbox by 8%, then applies a Gaussian blur to that region on the output canvas.
- WebGPU acceleration when available, WASM fallback otherwise.
- 100% MIT.

## How it works

Two canvases, one blur step.

1. Each frame is drawn to an offscreen canvas.
2. The offscreen canvas is passed to `model.predict()`.
3. The same frame is drawn sharp to the visible canvas.
4. For every `person` detection, we clip the visible canvas to the padded bbox and redraw with `ctx.filter = 'blur(24px)'`. Only the clipped region is blurred; the rest stays sharp.

That's it. No per-pixel loop, no WASM side-car. The browser handles the blur kernel.

## Tuning

Three knobs at the top of `index.html`:

```js
const PERSON_CLASS_ID = 0;   // COCO person; leave as-is or switch to another class
const BLUR_RADIUS = 24;      // px. Higher = stronger blur.
const BBOX_PADDING = 0.08;   // 8% bbox expansion so edges aren't visible
```

## FAQ

**Does it detect faces specifically?** No. It blurs the whole person silhouette (the COCO `person` class). Face-only anonymization needs a face detector, tracked in the `blur-faces` folder.

**Does it work offline?** Yes after first load. The model and runtime are cached by the browser.

**Can I swap in my own model?** Yes. Replace `loadModel('LibreYOLOXn')` with `loadModel('./my_model.onnx', { modelFamily: 'yolox', inputSize: 640 })`. Any COCO-class-0 detector will work without other changes.

## Credits

Built on [libreyolo-web](https://github.com/LibreYOLO/libreyolo-web) by [LibreYOLO](https://github.com/LibreYOLO). MIT licensed.
