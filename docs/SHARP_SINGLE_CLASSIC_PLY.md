# SHARP room export: single PLY on iOS (`_classic.ply`)

## What changed (iOS)

Previously, on-device SHARP room creation wrote **three** large binary PLY files under the temp `SHARPModels` folder for each generation:

| File | Role |
|------|------|
| `Room_<timestamp>.ply` | “Base” uchar RGB, same vertex frame as internal `rows` (no Y/Z flip). |
| `Room_<timestamp>_classic.ply` | Uchar RGB with Y/Z flip — **the frame used by MetalSplatter / `GaussianSplatView`**. |
| `Room_<timestamp>_3dgs.ply` | SuperSplat-style layout (`f_dc_*`, normals, larger payload) for external viewers. |

The in-app viewer already resolved and loaded **`_classic.ply`** when present (`SharpRoomView` → `viewerPlyURL`), while `SHARPGenerationResult.plyURL` and some logs still pointed at the base file, which was easy to misread in Console.

**Current behavior:** `SHARPService.writePLY` writes **only** `Room_<timestamp>_classic.ply`. `generateGaussians` returns that URL for `SHARPGenerationResult.plyURL`, `GenerationStatus.completed(fileURL:)`, thumbnail/sidecar paths, and logging.

### Code locations (iOS)

- `Furnit/Services/OnDevice/SHARPService.swift` — `writePLY` return type `(classic: URL, aabbWidth, aabbHeight, aabbDepth)`; single `writeStandardPLY(..., flipYZ: true)`.
- `Furnit/Views/SharpRoomView.swift` — if `plyURL` already ends with `_classic.ply`, `viewerPlyURL` / `classicPlyURL` use it directly (no synthetic `*_classic_classic.ply` path).

### Tradeoffs

- **Removed:** On-device `_3dgs.ply` export. To support SuperSplat or similar again, reintroduce that write behind a debug flag or settings toggle.
- **Saved:** Roughly two large disk writes per room (base + 3DGS) and corresponding I/O time.

---

## Android: same “three PLY” problem?

**No.** The ExecuTorch INT8 path does **not** mirror the old iOS triple export.

- Kotlin: `ExecutorchInt8Sharp.writePly` creates **one** file per room folder: `room.ply`.
- Native: `writePlyBinaryNative` writes that single file (3DGS-style header with `f_dc_*` / `f_rest_*`), which matches what the **WebView + SparkJS** viewer expects on Android.
- `StreamingResult` exposes both `plyFile` and `classicPlyFile` for API compatibility with older layers; they **reference the same path** (`StreamingResult(plyFile, plyFile, …)` in `ExecutorchInt8Sharp.kt`).

So there was nothing to delete on Android for parity with this iOS cleanup; behavior is already “one PLY per generation” for the active pipeline.

For more Android detail, see `android/docs/EXECUTORCH_INT8_SHARP.md` (initialization and `StreamingResult`).

---

## Quick verification (iOS)

After generating a room from a photo, the temp folder should contain:

- `Room_<ts>_classic.ply` (only large PLY)
- `Room_<ts>_thumbnail.jpg` and sidecars (e.g. EXIF) as before

Logs should **not** mention separate “Saved PLY” / “3DGS PLY” paths for the same timestamp.
