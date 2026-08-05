# RTMDet iOS Runtime

The production iOS segmentation asset is:

- `rtmdet-ins-m-raw-fp16.tflite`

It preserves the exact Android RTMDet tensor contract and math, with one iOS-only
Metal compatibility rewrite: four `RELU_0_TO_1` clamps are represented as equivalent
`MAXIMUM(x, 0)` + `MINIMUM(x, 1)` pairs. This lets the LiteRT 2.17 Metal delegate own
the complete graph. `scripts/rewrite_rtmdet_ios_metal_graph.py` reproducibly creates
the reviewed iOS payload from Android's reviewed source model. The file is stored with
Git LFS and is tagged as the `RTMDetModel` On-Demand Resource by `Furnit.xcodeproj`,
so it is not part of the initial app install.

Reviewed payloads:

- Android source SHA-256:
  `7edbd6692733d42a70344999aa5815762585c2a785b0e47cead4d786d4fb854d`
- iOS Metal graph SHA-256:
  `f13a4bf62e79284ae1b2f872c8ab7288767475fc2864af627c7dd79479bf1757`
- The ten named output tensors were bit-for-bit equal on the checked CPU reference
  frame after the rewrite.

## Runtime contract

- LiteRT/TensorFlow Lite 2.17.0 C runtime
- Mandatory iOS Metal delegate; no Core ML or CPU runtime fallback
- Explicit post-delegation execution-plan audit; loading fails if any CPU node remains
- Model creation, warm-up, inference, and destruction stay on one dedicated thread
- One zero-input Metal warm-up during model load, matching Android's first-frame preparation
- One shared `RTMDetLiteRuntime` instance for live Furniture Fit, Settings image
  scan, and the room-generation object anchor
- Input: `input`, float32 NHWC `[1, 640, 640, 3]`, raw BGR values in
  `0...255`; the graph owns normalization
- Preprocess: direct stretch to 640×640, matching the Android graph contract
- Outputs:
  - `cls_80`, `bbox_80`, `kernel_80`
  - `cls_40`, `bbox_40`, `kernel_40`
  - `cls_20`, `bbox_20`, `kernel_20`
  - `mask_feat`

`RTMDetLiteRuntime` uses LiteRT's documented tensor-copy API after each Metal
invocation, then physically converts NHWC outputs into contiguous NCHW storage exactly
as Android does. Both buffers are persistent, so this adds no per-frame output
allocation and avoids the retired custom-stride view over delegate memory. Callers
must consume the arrays while the dedicated runtime worker is occupied. Metal uses
Android-parity delegate options (`allow_precision_loss = true`, quantization support
enabled). Automatic delegate fallback is disabled, and a native execution-plan audit
requires zero remaining CPU nodes before the model is accepted.

## Distribution and licensing

The official static `TensorFlowLiteC.xcframework` and
`TensorFlowLiteCMetal.xcframework` are kept under `Vendor/LiteRT`. Complete
LiteRT license and attribution text is bundled at
`Furnit/Licenses/LITERT-LICENSE.txt` and is readable offline from Settings >
Licenses. RTMDet/MMDetection remains covered by the bundled Apache 2.0 text and
third-party notice.

The older Core ML exports and conversion scripts are retained only as experimental
history. They are not members of the production app target and are not runtime
fallbacks. See `Furnit/Views/FurnitureFit/README.md` and
`Furnit/diagrams/rtmdet-swift-flow.svg` for the active pipeline.
