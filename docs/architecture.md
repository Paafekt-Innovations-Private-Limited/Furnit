# Architecture — entry point

Furnit has one iOS app and one Android app with shared product semantics but separate
native implementations. The authoritative path-to-code inventory is
[`architecture/CODE_MAP.md`](architecture/CODE_MAP.md); decision history starts in
[`adr/`](adr/).

## End-to-end flows

| Flow | iOS | Android |
|---|---|---|
| Authentication | `LoginView` → `AuthenticationManager` → `OTPVerificationView` → Firebase Auth | `LoginActivity` → `AuthenticationManager` → `OTPVerificationActivity` → Firebase Auth |
| AI room creation | GeoCalib/Depth Anything/RTMDet → version-5 single-surface projective USDZ → exact preview → Save promotes the same artifact (device-unconfirmed) | Immediate method picker → EXIF-aware decode → metric measurement → one continuous version-5 projective GLB → exact preview → byte-identical Save promotion; flat fallback only on generation failure (device-unconfirmed) |
| Manual room creation | Boundary editor → `SinglePhotoRoomReconstructor` → textured mesh/USDZ path | `RoomBoundaryActivity` → five-plane textured GLB → `GLBRoomActivity` |
| Furniture Fit | Android-equivalent RTMDet FP16 graph variant → mandatory audited full Metal (`cpuNodes=0`) → class-aware NMS → mask affinity → transparent overlays | RTMDet FP16 LiteRT (GPU or XNNPACK CPU) → boxes-only identify or mask segmentation → transparent overlays |
| Saved room | Private `Documents/SavedRooms` | Private `files/rooms` (excluded from backup/transfer) |
| Legal disclosure | Bundled third-party notice + Apache text opened offline by `LicensesView` | Bundled notices plus Apache, LiteRT/Caffe, ONNX Runtime, Three.js, and reCAPTCHA texts opened offline by `LicensesActivity` |

## Deep dives

| Document | Covers |
|---|---|
| [`architecture/CODE_MAP.md`](architecture/CODE_MAP.md) | Modules, owning files, generated assets, and build configuration |
| [`../Furnit/docs/README.md`](../Furnit/docs/README.md) | iOS room/Furniture Fit docs and diagrams |
| [`../Furnit/Views/FurnitureFit/README.md`](../Furnit/Views/FurnitureFit/README.md) | iOS RTMDet and overlay behavior |
| [`../android/docs/README.md`](../android/docs/README.md) | Android implementation/runbook index |
| [`../android/docs/ANDROID_ROOM_CREATION.md`](../android/docs/ANDROID_ROOM_CREATION.md) | Android room-generation flow |
| [`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md) | Android Firebase/Play identity and login behavior |
| [`IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md`](IOS_ROOM_FURNITURE_DIMENSIONS_AND_OVERLAY.md) | iOS sizing, ratios, and overlay ownership |
| [`ROOM_3D_APPROACHES.md`](ROOM_3D_APPROACHES.md) | Research chronology and current iOS production approach |

## Invariants

- Raw room photos/scans are not uploaded as a condition of core functionality.
- Saved rooms remain local unless the user explicitly shares or exports them.
- Manual OTP entry remains a fallback even when automatic verification/autofill is
  available.
- Android store identity is `com.paafekt.android`; the Kotlin namespace is an internal
  implementation detail and intentionally differs.
- Model and viewer implementations can differ by platform; preserve observable flow,
  metadata, and disclosure parity.
- Preserve offline license/notice access on both platforms. Android release builds
  must continue to pass `verifyLegalAssets` before `preBuild`.

## Historical and deferred work

Do not infer current architecture from a dated release checklist, deleted prototype,
or experiment report. Use [`deferred/README.md`](deferred/README.md) for lifecycle
rules and [`deferred/CONFIRMED_DEFERRED.md`](deferred/CONFIRMED_DEFERRED.md) for the
evidence-backed backlog.
