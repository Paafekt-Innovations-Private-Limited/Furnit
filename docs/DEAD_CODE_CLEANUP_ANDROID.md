# Android dead-code cleanup

Branch history: `cleanup/dead-code-android` (merged PR #72), then follow-up commits on `main`.

Method: Lint `UnusedResources` + codebase grep (`R.string.*`, `@string/*`, `@color/*`), validated against manifest/XML/DI/reflection traps. **Deletion only** — no refactors.

## Verification

```bash
cd android && ./gradlew :app:assembleDebug :app:compileDebugAndroidTestKotlin --no-daemon
```

- **assembleDebug** + **compileDebugAndroidTestKotlin**: passed after all cleanup commits.
- **Instrumented tests on device**: not run in CI agent; smoke-test room viewer flows locally before release.

## Tools run

| Tool | Result |
|------|--------|
| `./gradlew :app:lintDebug` | `UnusedResources` warnings used as candidate list; build fails on pre-existing `NewApi` errors (unrelated). |
| Codebase grep | Cross-checked Lint string/color candidates against `R.string.*` / `@string/*` / `@color/*` in `.kt`, `.java`, `.xml`. |
| `./gradlew :app:dependencies --configuration debugRuntimeClasspath` | Gradle dependency audit (see below). |
| detekt / ktlint | Not configured in this repo. |
| R8 `-printusage` | Not run (`minifyEnabled false` in release). |

Lint report (when generated): `android/app/build/intermediates/lint_intermediate_text_report/debug/lintReportDebug/lint-results-debug.txt`

---

## Removed — pass 1 (PR #72)

### Unused drawables
`ic_square_split_2x2`, `ic_arrow_*`, `ic_brain`, `ic_save` — zero Kotlin/XML refs.

### Template colors
`purple_*`, `teal_*` in `colors.xml` — default Material leftovers.

### Dead GLB immersive Kotlin
D-pad overlay, unused toolbar button factories, `JoystickView` + tests, hidden `topBar` / floating brain chrome (preview save moved to summoned toolbar).

---

## Removed — pass 2 (main follow-up)

### Paafekt XML color tokens (`paafekt_colors.xml`)

Removed XML entries duplicated in `PaafektColors.kt` with **no** `@color/` or `R.color` references:

| Removed | Kept in `PaafektColors.kt` |
|---------|---------------------------|
| `paafekt_surface_hi` | `surfaceHi` |
| `paafekt_hairline` | `hairline` |
| `paafekt_success` | `success` |
| `paafekt_danger` | `danger` |
| `paafekt_viewer_capsule` | `viewerCapsule` |

Theme wiring (`themes.xml`) still uses the remaining `@color/paafekt_*` entries that are referenced.

### Unused `strings.xml` entries (186 names × 14 locale files)

Removed **2,539** `<string>` lines across `values/` and `values-*` after grep proved **zero** references to each name in Kotlin, Java, or XML (excluding the string files themselves).

Categories removed (unused on Android today):

- Explore / favorites / profile tab placeholders (`explore_*`, `favorites_*`, `profile_*` except strings still referenced elsewhere)
- Legacy camera picker copy (`camera_*` wide-angle flow not wired)
- Unused boundary labels (`boundary_title`, `boundary_ceiling`, …) while `boundary_back`, `boundary_adjust`, etc. remain
- Stale help/email and accessibility strings not referenced in layouts or code
- Orphan GLB toasts (`glb_room_saved`, …) superseded by other save strings
- Other Lint-flagged strings with no `R.string.<name>` or `@string/<name>` hit

**Not removed:** any string referenced from `.kt`, `.java`, or layout/menu XML; FAQ/help strings still used by `HelpActivity` / settings; all manifest-linked copy.

### Gradle dependency audit

| Dependency | Verdict |
|------------|---------|
| `org.jetbrains.kotlin:kotlin-stdlib` (explicit) | **Removed** — redundant with Kotlin Android plugin / transitive stdlib. |
| `androidx.core:core-ktx` | Keep — used throughout. |
| `androidx.exifinterface:exifinterface` | Keep — `PhotoOrientation.kt`. |
| `androidx.appcompat`, `material`, `activity-ktx` | Keep — Activities/themes. |
| `androidx.fragment:fragment-ktx` | Keep — `FurnitureFitFragment`, `FragmentActivity`. |
| `androidx.webkit:webkit` | Keep — GLB `WebViewAssetLoader`. |
| CameraX (`camera-*`) | Keep — Furniture Fit / inline brain camera. |
| `lifecycle-runtime-ktx`, `kotlinx-coroutines-android` | Keep — coroutines in viewers/services. |
| `onnxruntime-android` | Keep — on-device inference. |
| `sceneview` | Keep — `ModelDetailActivity` GLB rendering. |
| `com.google.ar:core` | Keep — `ArMeasureActivity`, AR sizing. |
| Firebase Auth BOM | Keep — `AuthenticationManager`. |
| Test deps (`junit`, `androidx.test.*`) | Keep — instrumented tests. |

---

## Still flagged / out of scope

| Item | Reason |
|------|--------|
| Pre-existing Lint `NewApi` errors (`Bitmap.Config.HARDWARE`, etc.) | Fix separately; blocks clean `lintDebug` gate. |
| `fragment-ktx` without direct `viewModels` usage | Retained; low risk, required by fragment hosting pattern. |
| ExecuTorch / multi-flavor-only symbols | Not analyzed per variant. |
| R8 usage report | Deferred until `minifyEnabled true`. |
| detekt in CI | Recommended follow-up. |

---

## Recommended follow-ups

1. On-device smoke: GLB room + Model detail — summon chrome, toolbar, Fit, Capture, preview save.
2. Fix Lint `NewApi` issues for CI gating.
3. Add detekt for ongoing unused-private detection.
