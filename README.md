# Furnit

Image-to-3D room generation and furniture fitment across iOS and Android.

## Start here

- [`docs/READ_FIRST.md`](docs/READ_FIRST.md) — compact, current context and settled production facts.
- [`docs/README.md`](docs/README.md) — complete documentation index and lifecycle classification.
- [`docs/overview.md`](docs/overview.md) — product and platform overview.
- [`docs/architecture.md`](docs/architecture.md) — architecture entry point and code map.
- [`AGENTS.md`](AGENTS.md) — contributor/agent reading order and safety rules.

## Platform documentation

| Platform | Code | Documentation |
|---|---|---|
| iOS | [`Furnit/`](Furnit/) | [`Furnit/docs/README.md`](Furnit/docs/README.md) |
| Android | [`android/`](android/) | [`android/docs/README.md`](android/docs/README.md) |

The apps share the Paafekt product and Firebase phone-number sign-in but use
platform-native room-generation and rendering pipelines. See
[`docs/architecture/CODE_MAP.md`](docs/architecture/CODE_MAP.md) before changing a
cross-platform behavior.

## Current release note

Android is released on Google Play as `com.paafekt.android`. The Play App Signing
fingerprints required by Firebase Phone Auth were registered and live-verified on
2026-08-03. Signed Android version 1.2 / code 5 and iOS version 1.2 / build 87 were
prepared with offline license and attribution packaging on 2026-08-05. Store upload,
review, and publication state is recorded conservatively in
[`docs/store-submission-status.md`](docs/store-submission-status.md); authentication
details remain in [`android/docs/AUTHENTICATION.md`](android/docs/AUTHENTICATION.md).

## Build checks

```bash
# Android
cd android && ./gradlew :app:assembleDebug --no-daemon

# iOS
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

See [`CLAUDE.md`](CLAUDE.md) and `.cursor/rules/` before running builds on this
machine, especially for the external Xcode cache constraints.
