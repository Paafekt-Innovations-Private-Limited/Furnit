# Furnit Project Context

Furnit is a cross-platform image-to-3D room generation and furniture-fitment project.

- **iOS app:** `Furnit/` (Swift, SwiftUI/UIKit, Core ML, MetalSplatter).
- **Android app:** `android/` (Kotlin, ExecuTorch, NCNN/ONNX support, Vulkan-focused SHARP path).
- **Shared docs:** `docs/`, `Furnit/docs/`, `android/docs/`.
- **Model/export scripts:** `scripts/`, `android/pyfiles/`, `android/scripts/`.

## Current iOS Architecture

### RTMDet / Furniture Fit

The active iOS Furniture Fit "brain" path is RTMDet-Ins-m through Core ML.

Important files:

- `Furnit/Services/OnDevice/RTMDetModelService.swift` — model loading.
- `Furnit/Services/OnDevice/RTMDetImageInference.swift` — image/live inference, raw-head decode, confidence-first NMS, mask-head execution, mask-affinity grouping, cached mask rebuilds.
- `Furnit/Views/FurnitureFit/FurnitureFitView.swift` — live overlay, selection, segmentation, cutout display, pinch/pan gesture ownership.
- `Furnit/Views/SettingsFurnitureFitImageScanView.swift` — Settings still-image diagnostic path.
- `Furnit/Views/FurnitureFit/README.md` — main RTMDet/Furniture Fit pipeline documentation.
- `Furnit/diagrams/rtmdet-swift-flow.svg` — visual flow.

Current behavior:

- Core ML image input is preferred; BGR mean/std normalization is expected inside the model graph.
- Raw outputs are `cls/bbox/kernel` at 80/40/20 plus `mask_feat`.
- NMS is class-aware and confidence-first; `maxDetectionCount` may be `nil` for no artificial cap.
- Object-piece fusion is class-agnostic mask-affinity grouping, not chair-specific logic.
- Settings image scan should mirror the live RTMDet path: uncapped detections, fused instance masks, and pixel-level RGBA union.
- In room viewers, the room layer also owns pinch zoom; when a segmented cutout is visible, Furniture Fit must capture two-finger touches so pinch scales the furniture cluster instead of the room camera.

### SHARP Room Flow

The active iOS SHARP path generates Gaussian splat rooms and renders them with MetalSplatter.

Important files:

- `Furnit/Services/OnDevice/SHARPService.swift` — Core ML SHARP inference, PLY write path, `.splatcache` generation.
- `Furnit/Views/Components/GaussianSplatView.swift` — MetalSplatter render and `.splatcache` load path.
- `Furnit/Views/SharpRoomView.swift` — SHARP room viewer, Furniture Fit overlay host, room dimensions/measurement UI.
- `Furnit/Models/USDZModelManager.swift` — saved room artifacts, including related `.splatcache` files.
- `Furnit/diagrams/sharp-swift-flow.svg` — visual flow.

Current behavior:

- SHARP saves `Room_*_classic.ply` plus sidecars and a binary `.splatcache`.
- `.splatcache` is the preferred fast path for parsed/encoded splat data; PLY parsing is fallback.
- Room/furniture size and overlay scale details live in `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`.

## Current Android Architecture

Android SHARP work is centered on ExecuTorch debug flavors.

Important files/docs:

- `android/README.md` — Android overview, model staging, logs.
- `android/docs/TEST_AND_SETTINGS.md` — run/settings checklist.
- `android/docs/EXECUTORCH_VULKAN_KNOWN_GOOD_FLOW.md` — Vulkan known-good flow.
- `android/app/src/main/java/com/furnit/android/` — app source.
- `android/app/src/main/cpp/` — native SHARP/ExecuTorch integration.

Default agent compile check is both ExecuTorch flavors:

```bash
cd android && ./gradlew :app:compileEtCpuDebugKotlin :app:compileEtVulkanDebugKotlin --no-daemon
```

## Build And Test Rules

Follow repository rules before running builds/tests.

- `CLAUDE.md` requires compile before concluding code changes.
- `.cursor/rules/model-usage-preference.mdc` says mechanical execution such as `xcodebuild`, `./gradlew`, and git mechanics should use Claude Sonnet when available.
- `.cursor/rules/ios-build-preference.mdc` defines the preferred iOS compile command:

```bash
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

- Do not use Simulator destinations unless explicitly requested.
- Do not pass `-derivedDataPath` into the repo unless explicitly requested.
- `.cursor/rules/android-build-preference.mdc` defines the preferred Android compile command shown above.

## Xcode Cache Safety

The user has an external SSD-backed Xcode cache setup. Be conservative.

- See `.cursor/rules/xcode-cache-safety.mdc`.
- Diagnose Xcode/SwiftPM cache issues by reading/listing only.
- Do not create, delete, replace, or repair symlinks/folders under Xcode/SwiftPM cache paths unless the user explicitly approves that repair in the current conversation.
- If `xcodebuild` fails because of DerivedData, SwiftPM, SourcePackages, permissions, symlinks, missing external volume, or disk space, stop and report the exact error.

## Documentation Map

Start here when updating docs:

- `README.md` — top-level documentation index.
- `Furnit/docs/README.md` — iOS docs and diagram index.
- `Furnit/Views/FurnitureFit/README.md` — RTMDet/Furniture Fit pipeline.
- `docs/RTMDET_IOS_SWIFT_SPIKE.md` — RTMDet Core ML loader/export and Swift postprocess status.
- `docs/IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md` — room/furniture sizing and overlay scaling.
- `Furnit/CHANGELOG.md` — recent iOS changes.
- `android/README.md` and `android/docs/` — Android-specific docs.

When code behavior changes, update the nearest detailed doc plus any affected diagram/index.

## Practical Debug Notes

- A screenshot of the live RTMDet overlay is not the same as the original camera frame; re-scanning a screenshot can produce different scores/detections.
- If segmented cutout pinch does not work in the main room flow, first verify whether `FurnitureFitContainerView.handlePinch(_:)` logs appear. If not, debug hit-testing/gesture ownership before changing scale math.
- Python RTMDet probes are useful for export inspection, but Swift/Core ML app tests are the source of truth for current image-input behavior.
- For RTMDet class labels, `Furnit/Models/RTMDet/rtmdet-coco80-classes.json` is standard COCO-80 (`56 == chair`).
