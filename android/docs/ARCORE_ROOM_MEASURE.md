# ARCore distance measure (beta)

## What it does

- **AR distance measure** is implemented by `ArMeasureActivity` (two taps on a plane → metric length). It is not linked from the Sharp room top bar anymore (AR-assisted FurnitureFit uses ARCore in the brain flow instead). You can still start `ArMeasureActivity` from code or a debug entry if needed.
- Tap **two points** on a detected horizontal or vertical plane; the app shows **distance in meters** and an optional overlay line.
- **Done** returns to the viewer and offers to **calibrate** displayed room size: pick which SHARP axis (width / height / depth) your segment matches. The app stores an isotropic **`arDisplayScale`** in `room_meta.json` (raw SHARP bbox stays unchanged on disk).

## Requirements

- Device with **Google Play Services for AR** (ARCore).
- **Camera** permission.

## Testing

1. Launch `ArMeasureActivity` (e.g. from a test hook) → move phone until planes appear → tap two corners of a real edge.
2. Tap **Done** → choose calibration axis if values look reasonable.
3. Re-open the room: title and WebView fallback dims should reflect `arDisplayScale`; home list dims use the same scale via `ModelManager`.

## Implementation notes

- Dependency: `com.google.ar:core:1.45.0` in `app/build.gradle`.
- Manifest: `com.google.ar.core` = **optional** so the APK installs on non-AR devices.
- Rendering: `GLSurfaceView` + `ArBackgroundRenderer` (camera texture) + `ArMeasureOverlayView` (projected anchors).
