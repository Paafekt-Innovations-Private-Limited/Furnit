# Confirmed deferred and cleanup ledger

Last reviewed: 2026-08-03. This is a documentation ledger, not an authorization to
change runtime behavior or delete code.

| Area | Confirmed state and evidence | Safe future decision |
|---|---|---|
| Android login country preselection | `android/.../auth/CountryCode.kt` uses `Locale.getDefault().country` only. Physical location, SIM, and network country are not consulted, so `en-GB` can default to `+44` in India. | Keep locale-only behavior, add an explicit remembered/default-country preference, or re-evaluate a privacy-compliant telephony signal. Preserve manual selection. |
| Android boundary result callback | `SinglePhotoRoomActivity.boundaryActivityLauncher` handles `RESULT_OK` and contains a TODO to read boundaries. `RoomBoundaryActivity` directly launches the viewer and calls `finish()` without `setResult(RESULT_OK)`, so the result branch is not currently reached. | Either remove the unused result callback/contract after flow testing or make `RoomBoundaryActivity` return a documented result and let the caller own navigation. |
| Android vanishing-point gravity | `roomreconstruction/ScaleEstimator.kt` defines `VanishingPointGravity.refine`, which returns the input leveling rotation with confidence `0.0` and debug value `vp_refiner_unimplemented_using_input_gravity`. | Implement and validate a real refiner, or remove the stub after proving no persisted/debug contract depends on it. Current GeoCalib/capture/fallback gravity remains active. |
| Android viewer FOV parameter | `utils/RoomBoundaryManager.getOptimalCameraPosition` suppresses `horizontalFovDegrees` as `UNUSED_PARAMETER` and documents it as retained for API compatibility. | Keep compatibility until callers/contracts are audited; otherwise remove it in a deliberate API cleanup. |
| iOS manual room back-wall color | `SinglePhotoRoomReconstructor.build3DRoom` computes `backWallColor`, skips the back wall so the camera can look in from outside, then discards the value with `_ = backWallColor`. | Stop computing the unused color, or restore a back-wall design only with camera/viewer validation. Do not add a wall just to eliminate a warning. |
| Android lint gate | `docs/DEAD_CODE_CLEANUP_ANDROID.md` records pre-existing Lint `NewApi` failures that prevent a clean `lintDebug` gate. | Fix the concrete API guards, rerun full Lint, and promote it to CI only after the baseline is clean. |
| Android R8 usage report | Release minification is active, but the cleanup audit has not yet added/archived an R8 `-printusage` report. | Add a scoped `-printusage` output for a future audit and triage it before deleting anything reflection/resource related. |
| Android static analysis | detekt/ktlint is not configured; the cleanup audit recommends detekt for ongoing unused-private detection. | Add configuration and a non-destructive baseline before enforcing CI. |
| Archived experimental ML guidance | `.cursor/rules/furnit-ml.mdc` explicitly describes archived SHARP/ExecuTorch Vulkan work; current Android room/Furniture Fit paths use ONNX Runtime and flat-photo GLB generation. | Leave archived unless that experiment is deliberately resumed; do not use it as active implementation guidance. |

## Historical documents

Historical classifications live in [`README.md`](README.md). They remain at their
original paths for link stability and should receive dated addenda instead of silent
claims that they describe current production.
