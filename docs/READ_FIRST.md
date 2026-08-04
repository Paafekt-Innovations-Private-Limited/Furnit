# Furnit read-first context

This is the compact active context for contributors and coding agents. It records
settled facts that should not be reconstructed from chat history or stale audits.
Last verified: 2026-08-05.

## Settled product and ownership facts

- **Mobile App operator/controller and store publisher:** Paafekt Innovations Private
  Limited (India).
- **Affiliate:** Paafekt Inc. (United States). It is a valid affiliated entity, but it
  is not the mobile App operator/controller and should not be described as jointly
  operating the App.
- Certificate subject/organization text does not choose the Firebase app, Play
  package, App Store provider, or privacy-policy controller. Do not replace a valid
  key merely because old certificate metadata names the US affiliate.
- The repository copy of the privacy policy is [`privacy.html`](privacy.html).
  Editing it does not deploy `https://paafekt.com/privacy`; website deployment is a
  separate operation.

## Android production identity and authentication

- Android is released on Google Play as `com.paafekt.android`.
- Kotlin code intentionally remains under namespace `com.furnit.android`.
- Firebase project: `paafektprod`.
- Active Firebase Android app:
  `1:613415224058:android:8d0a97fe4990e559a13f43`.
- On 2026-08-03, the production Play build failed Phone Auth because Firebase did not
  have the Play App Signing SHA-256. The installed APK's Play SHA-1/SHA-256 were
  extracted, added to the active Firebase app, and verified live. The repair script is
  [`../android/scripts/fix_firebase_android_auth.sh`](../android/scripts/fix_firebase_android_auth.sh).
- Full operational details and fingerprints:
  [`../android/docs/AUTHENTICATION.md`](../android/docs/AUTHENTICATION.md).

Do not diagnose the original production error as a package-name or upload-key problem
unless fresh evidence shows that. The error explicitly identified a valid Play
Integrity token with no matching Firebase SHA-256; the missing Play certificate was
the resolved cause.

## Android login behavior

- Country preselection prefers the mobile-network country, then the SIM country, then
  the effective app/device locale. The picker remains the manual override; IP lookup
  and phone/SMS runtime permissions are not used for this choice.
- The six OTP fields retain manual entry and declare Android positional autofill hints
  `smsOTPCode1` through `smsOTPCode6` on Android 8+. The change is included in the
  signed version 1.2 / code 5 artifact; Play publication of that artifact has not been
  re-confirmed in this repository.
- Do not start SMS User Consent in parallel with Firebase's SMS Retriever path without
  revalidating the Firebase Auth interaction. Manual entry must always remain usable.

## Active platform architecture

### iOS

- Swift/SwiftUI/UIKit app under [`../Furnit/`](../Furnit/).
- Default Photo → 3D is two-phase: an immediate flat-photo preview, then GeoCalib +
  Depth Anything V2 Metric Indoor + RTMDet measurement and textured USDZ export on
  first save.
- The manual boundary path uses `SinglePhotoRoomReconstructor` and
  `SyntheticDepthEstimator`, then opens `MeshRoomView`.
- Furniture Fit uses RTMDet Core ML, mask-affinity grouping, transparent cutouts, and
  room-viewer brain/full-video modes.

### Android

- Kotlin app under [`../android/`](../android/), target/compile SDK 36, arm64 runtime.
- Photo → 3D shows the AI/manual picker before heavy decode work. The default AI
  preview is an optimized flat full-photo GLB; manual setup produces a textured
  five-plane GLB.
- Depth Anything and GeoCalib ONNX models plus one RTMDet FP16 LiteRT model ship in install-time
  Play Asset Delivery packs and are loaded through Android's `AssetManager`.
- `GLBRoomActivity` hosts the Three.js/WebView room plus the inline RTMDet brain and
  full-video segmentation experience.

Use [`architecture.md`](architecture.md) and
[`architecture/CODE_MAP.md`](architecture/CODE_MAP.md) for owning files and deeper
links.

## License and attribution packaging

- Version 1.2 source on both platforms exposes a readable third-party notice and the
  complete Apache License 2.0 text offline under Settings → Licenses.
- Android additionally bundles LiteRT 1.4.2's complete combined license and required
  Caffe/BSD attribution. `verifyLegalAssets` is a `preBuild` dependency.
- Notices identify the exact **Depth Anything V2 Metric Indoor Small** checkpoint and
  Paafekt's Core ML, ONNX, and LiteRT/TFLite model-format conversions.
- The unresolved Hypersim CC-BY-SA training-data question remains a lawyer-priority
  tail risk; packaging attribution does not decide that interpretation. See
  [`MODEL_LICENSE_AUDIT.md`](MODEL_LICENSE_AUDIT.md).

## Documentation lifecycle

- [`README.md`](README.md) is the complete documentation index.
- Detailed platform docs stay beside code in
  [`../Furnit/docs/`](../Furnit/docs/) and
  [`../android/docs/`](../android/docs/).
- Dated audits and experiments are evidence, not automatic descriptions of current
  behavior. Historical classifications and confirmed future work live under
  [`deferred/`](deferred/).
- When code changes, update the nearest owning doc plus any central status/index whose
  claim changed.

## Validation defaults

```bash
# Android, from android/
./gradlew :app:assembleDebug --no-daemon

# iOS, from repository root
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

Read [`../CLAUDE.md`](../CLAUDE.md) and `.cursor/rules/` before builds. Do not alter
external Xcode cache paths or signing material without explicit authorization.
