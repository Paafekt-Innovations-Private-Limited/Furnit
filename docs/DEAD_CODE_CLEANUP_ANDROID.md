# Android dead-code cleanup

Branch: `cleanup/dead-code-android`  
Method: Lint `UnusedResources` + manual grep, validated against manifest/XML/DI/reflection traps. **Deletion only** — no refactors.

## Verification after each commit

```bash
cd android && ./gradlew :app:assembleDebug --no-daemon
```

- **assembleDebug:** passed after all four commits on this branch.
- **Unit tests:** no `src/test` JVM tests in `:app`.
- **Instrumented tests:** `connectedDebugAndroidTest` not run here (requires device/emulator). **On-device smoke test** (room open → summon chrome → Fit/Capture → back) should be run locally before merging.

## Tools run

| Tool | Result |
|------|--------|
| `./gradlew :app:lintDebug` | Report generated; build exits non-zero due to pre-existing `NewApi` errors (unrelated). `UnusedResources` warnings used as candidate list. |
| Android Studio Inspect Code | Not run in CI agent |
| detekt | Not configured in this repo |
| ktlint | Not configured in this repo |
| R8 `-printusage` | Not run (deps cleanup deferred) |

Lint report path: `android/app/build/intermediates/lint_intermediate_text_report/debug/lintReportDebug/lint-results-debug.txt`

---

## Removed (by commit)

### 1. `826b637` — Unused drawables (Lint `UnusedResources` + zero grep hits)

| Resource | Reason |
|----------|--------|
| `ic_square_split_2x2.xml` | Audit-confirmed; no Kotlin/XML refs |
| `ic_arrow_down.xml` | No refs |
| `ic_arrow_left.xml` | No refs |
| `ic_arrow_right.xml` | No refs |
| `ic_arrow_up.xml` | No refs |
| `ic_brain.xml` | Superseded by `ic_ai` PNG; no refs |
| `ic_save.xml` | Superseded by `ic_download`; no refs |

### 2. `5928bb2` — Unused template colors

| Resource | Reason |
|----------|--------|
| `purple_200`, `purple_500`, `purple_700`, `teal_200`, `teal_700` in `colors.xml` | Default Material template; no `@color/` references. Kept `ic_launcher_background`. |

### 3. `5a4ab72` — Dead GLB immersive leftovers (Kotlin)

Removed from `GLBRoomActivity.kt` (never called or superseded):

- `cameraDpadOverlay` field + empty init
- `createCameraDPadOverlay()`, `createDpadCircleButton()`
- `nudgeCameraLeft/Right/Up/Down()` (only used by removed d-pad)
- `createBottomIconButton()`, `createToolbarIconButton()`, `createToolbarTextButton()`
- `toolbarCapsuleDrawable()`, `toolbarCircleDrawable()`

**Kept:** `recenterCamera()` — used by summoned toolbar viewfinder action.

### 4. `3183855` — Unused private field

- `bottomControlsInnerColumn` in `GLBRoomActivity` — orphaned after immersive bottom chrome refactor.

---

## Flagged — do not delete (human review)

| Item | Reason |
|------|--------|
| `JoystickView.kt` + `JoystickViewTest.kt` / `FurnitureFitControlsTest` joystick cases | Public custom `View`; still referenced by **androidTest**. UI no longer inflates it — candidate for a follow-up that removes view **and** updates tests together. |
| `GLBRoomActivity.createTopBar()` + hidden `topBar` | Still constructed (visibility `GONE`); contains `trailingArSizingButton` wiring partially duplicated by summoned toolbar. **Viewer/immersive path** — consolidate in a dedicated refactor, not blind delete. |
| `paafekt_colors.xml` entries: `paafekt_surface_hi`, `paafekt_hairline`, `paafekt_success`, `paafekt_danger`, `paafekt_viewer_capsule` | Lint-unused in XML; mirrored in `PaafektColors.kt` as design tokens. Keep for theme parity / future XML layouts. |
| **~400+ `strings.xml` entries** flagged `UnusedResources` | Many are used from Kotlin via `R.string.*`, localized FAQ copy, or planned flows Lint does not trace. **Do not bulk-delete.** |
| Manifest-declared activities/services | All retained; not part of this pass. |
| ExecuTorch / flavor-specific code | Not analyzed per variant; no flavor-only symbols removed. |
| Gradle dependencies | Deferred to last; not touched. |

---

## Not found

- No `*.bak` assets under `android/`
- No orphaned d-pad/joystick **layout** XML (joystick was a custom view only)

---

## Recommended follow-ups (out of scope)

1. On-device smoke: GLB room + Model detail — resting summon, full summoned toolbar, Fit, Capture, hide.
2. Remove `JoystickView` + androidTest together after confirming product no longer needs programmatic joystick tests.
3. Delete or inline `GLBRoomActivity.createTopBar()` once AR sizing / preview save are fully on summoned toolbar only.
4. Add detekt to `android/` for ongoing unused-private detection.
5. Fix pre-existing Lint `NewApi` errors so `lintDebug` can gate CI.
