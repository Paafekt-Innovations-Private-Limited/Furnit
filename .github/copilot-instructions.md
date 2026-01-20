# Copilot / AI Agent Instructions for Furnit

Purpose: Help AI coding agents make safe, focused, and correct changes in this iOS/AR codebase.

- **Big picture:** This is a Swift iOS app that combines AR/RealityKit, CoreML models and Metal shaders. Key runtime flow: camera feed -> segmentation (CoreML / `Services/*`) -> `AR/ObjectSegmentationProcessor.swift` -> RealityKit placement (`RealityKitObjectPlacementManager.swift` / `Models/USDZModelManager.swift`). Entry points: `Furnit/FurnitApp.swift` and `Furnit.xcodeproj` / workspace.

- **Key directories / files to inspect before making changes:**
  - `AR/` — AR helpers, camera manager, segmentation processor, `Shaders.metal`.
  - `Services/` — network clients and model wrappers (e.g., `U2NetSegmentationManager.swift`, `Stable3DAPIClient.swift`).
  - `Models/` — app data models and managers (`USDZModel.swift`, `USDZModelManager.swift`).
  - `Assets.xcassets/` — image/3D assets and color sets.
  - Top-level ML assets: `DeepLabV3.mlmodel`, `u2net.mlmodelc/`, `*.mlpackage/` folders (`FastSAM*`, `MobileSAM*`, `best.mlpackage`). These are consumed by `Services/*` managers.

- **Architectural patterns to follow:**
  - "Manager" classes handle lifecycle and resource ownership (e.g., `RealityKitCameraManager`, `USDZModelManager`). Follow existing initialization and teardown patterns in `Utilities/` and `AR/`.
  - `Services/` wrap external APIs and CoreML invocations; keep them small and focused. Examples: `U2NetSegmentationManager.swift` performs CoreML inference; `Stable3DAPIClient.swift` performs network I/O.
  - UI is mixed SwiftUI and UIKit-like helpers: check `Views/` and `Authentication/` (e.g., `LoginView.swift`) for conventions.

- **Data flow examples (search to confirm):**
  - Camera image -> `ARKitCameraManager` -> `ObjectSegmentationProcessor` -> segmentation manager (e.g., `U2NetSegmentationManager`) -> segmentation mask -> `RealityKitObjectPlacementManager`.

- **Build / run / debug workflows:**
  - Preferred: open the Xcode workspace/project in Xcode (run on a physical device for AR):
    - `open Furnit.xcodeproj` or open the workspace in Xcode UI.
  - CLI build: use `xcodebuild -workspace Furnit.xcworkspace -scheme <scheme> -configuration Debug` (replace `<scheme>` with the app scheme from Xcode). Prefer validating in Xcode first to resolve code signing and entitlements for AR/camera.
  - For Metal/AR debugging use Xcode's GPU Frame Capture (Product > Capture GPU Frame) and the Metal debugger.

- **Project-specific conventions:**
  - Prefer clear, descriptive identifiers (see `CLAUDE.md`: "Always use descriptive variable names").
  - Compile locally before finalizing changes (again from `CLAUDE.md`).
  - Use existing naming: `*Manager.swift` for lifecycle, `*Processor.swift` for data pipelines, `*View.swift` for UI components.
  - Many ML assets live at repo root and are referenced directly — do not move or rename `.mlpackage`, `.mlmodel`, or `.mlmodelc` files without updating corresponding managers in `Services/`.

- **Integration points & external deps:**
  - CoreML models: `DeepLabV3.mlmodel`, `u2net.mlmodelc/`, `*.mlpackage/` — used by `Services/*` classes.
  - RealityKit / ARKit: `AR/` folder and `RealityKit*` helpers; AR features require device testing.
  - Metal: `AR/Shaders.metal` — shader changes require recompilation and testing on-device.
  - External network calls: `Services/Stable3DAPIClient.swift` — check for API keys or endpoints in code or project settings before modifying.

- **Search patterns & quick examples:**
  - Find segmentation entry points: `rg "U2NetSegmentationManager|ObjectSegmentationProcessor|ARKitCameraManager"`
  - Find models and mlpackages: `ls -1 *.mlmodel* *.mlpackage*` at repo root.
  - Trace 3D model handling: open `Models/USDZModelManager.swift` and `RealityKitObjectPlacementManager.swift`.

- **When editing:**
  - Keep changes minimal and focused; update the nearest `*Manager` or `*Processor` for behavior changes.
  - Run a local build in Xcode and test on-device if the change touches AR, CoreML inference, or Metal shaders.
  - If you touch model files or add a new `.mlpackage`, document where it's referenced and ensure `Services/` loads the correct resource path.

- **Oddities to note:**
  - The repo contains some unusually named files (for example under `Services/`) that may be experimental or stray — confirm with the owner before deleting or renaming.

Merge notes: preserve the two bullets in `CLAUDE.md` (descriptive names, compile before concluding) — they are included above.

If any of these items are unclear or you want me to add CI commands, the expected Xcode scheme name, or known device requirements (iOS version / device models), tell me and I'll iterate.
