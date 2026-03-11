# iOS vs Android: YOLO segmentation model

## Why the Android ONNX model is not used in Swift

- **Android** runs furniture segmentation with **ONNX Runtime**: it loads `yoloe-11l-seg-pf.onnx` (under `android/`) and runs it in Kotlin via `FurnitureFitManager.initializeOnnx()`.
- **iOS** runs the **same model** in **Core ML** form: `yoloe-11l-seg-pf.mlpackage` (ODR tag `YOLOEModel`), loaded by `YOLOEModelService` and used by `FurnitureFitView` / `FurnitureFitContainerView`.

So the **same YOLO model** (yoloe-11l-seg-pf) is used on both platforms, but in different formats:

| Platform | Format   | File / asset                         | Runtime      |
|----------|----------|--------------------------------------|--------------|
| Android  | ONNX     | `android/yoloe-11l-seg-pf.onnx`      | ONNX Runtime |
| iOS      | Core ML  | `yoloe-11l-seg-pf.mlpackage` (ODR)   | Core ML      |

The ONNX file is **not** copied into the Swift/Xcode project because:

1. **iOS does not run ONNX in this app** — it uses Apple’s Core ML stack only. There is no ONNX Runtime dependency in the Furnit iOS target.
2. **Same model, two exports** — From the same source (e.g. `yoloe-11l-seg-pf.pt`), the project uses:
   - an ONNX export for Android,
   - a Core ML export for iOS (`yoloe-11l-seg-pf.mlpackage`).
3. **Copying the .onnx into the app bundle would not run it** unless you add ONNX Runtime for iOS and implement an ONNX-based inference path in Swift.

So “YOLO 26 ONNX” (or the segmentation ONNX used on Android) is not “in Swift” because Swift uses the **Core ML** version of that model, not the ONNX file.

## YOLO 11l vs 26l

- **yoloe-11l-seg-pf** (11-layer) is what both Android (ONNX) and iOS (Core ML) use for segmentation; it has correct segmentation heads.
- **yoloe-26l-seg-pf** (26-layer) had a broken Core ML export (see `docs/YOLOE_COREML_FIX.md`); iOS does **not** use 26l. Android’s default segmentation asset is also the 11l ONNX (`yoloe-11l-seg-pf.onnx`).

## If you want to run the exact ONNX on iOS

To use the same `.onnx` file on iOS you would need to:

1. Add **ONNX Runtime for iOS** (e.g. CocoaPod or SPM).
2. Bundle `yoloe-11l-seg-pf.onnx` in the app (e.g. copy from `android/` into the Xcode project and add it to the app target).
3. Implement a Swift (or C++) inference path that loads the ONNX file and runs it with ONNX Runtime, and plug that into the FurnitureFit pipeline instead of (or alongside) the Core ML path.

Until then, parity is “same model (11l), different format per platform (ONNX on Android, Core ML on iOS).”
