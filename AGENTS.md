# AGENTS.md — read this first

Furnit contains active documentation, dated release/audit records, and experimental
history. Any coding agent changing this repository must read active context in this
order:

1. [`docs/READ_FIRST.md`](docs/READ_FIRST.md)
2. [`docs/architecture.md`](docs/architecture.md)
3. The platform index that owns the change:
   [`Furnit/docs/README.md`](Furnit/docs/README.md) for iOS or
   [`android/docs/README.md`](android/docs/README.md) for Android
4. The nearest README/doc beside the code being changed
5. [`docs/deferred/README.md`](docs/deferred/README.md) before reviving a stub,
   compatibility branch, or historical approach

Hard rules:

- Code, build configuration, and verified artifacts outrank dated documentation.
- Keep detailed platform docs beside their owning code; update the central indexes
  when adding, moving, or superseding a document.
- Do not present a dated audit, experiment, or submission checklist as current state.
- Paafekt Innovations Private Limited (India) is the mobile App operator/controller
  and the Apple/Google publisher. Paafekt Inc. (United States) is an affiliate only.
- Android production identity is `com.paafekt.android`; its Kotlin namespace remains
  `com.furnit.android`. Do not register or document the namespace as the Play package.
- Do not replace or recreate signing keys because certificate organization metadata
  says `Paafekt Inc.`. Key metadata does not choose the legal operator or Firebase app.
- Preserve manual OTP entry when changing automatic verification or autofill.
- Never commit keystores, signing properties, tokens, or credentials.
- Record only confirmed deferred/unused findings. Search hits alone do not prove that
  a fallback, persisted format, reflection target, test symbol, or compatibility API
  is dead.

Validation defaults:

```bash
# Android (from android/)
./gradlew :app:assembleDebug --no-daemon

# iOS (from repository root)
xcodebuild -project "Furnit.xcodeproj" -scheme "Furnit" -configuration Debug \
  -sdk iphoneos -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO COMPILER_INDEX_STORE_ENABLE=NO ENABLE_PREVIEWS=NO \
  -jobs 2 build
```

Follow [`CLAUDE.md`](CLAUDE.md) and `.cursor/rules/` for environment-specific build
and Xcode cache constraints. When behavior changes, update the closest owning doc and
any central status/index that would otherwise become misleading.
