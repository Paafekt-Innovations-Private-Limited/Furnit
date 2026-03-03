# Logcat analysis (22:46 run, 03-01)

## Pipeline timing

| Phase | Time (ms) | Time (s) |
|-------|-----------|----------|
| Part1+2 load | 32 | 0.03 |
| Part1+2 (1x patches 5×5) | 30,848 | 30.8 |
| Part1+2 (0.5x patches 3×3) | 11,484 | 11.5 |
| Part1+2 (0.25x patch) | 1,269 | 1.3 |
| **Part1+2 total** | **~43,601** | **~43.6** |
| Part3 (image encoder) | 1,355 | 1.4 |
| Part4a (chunked decoder) | 3,505 | 3.5 |
| **Part4b forward (FP32)** | **138,294** | **138.3** |
| writePly | 2,820 | 2.8 |
| **Total pipeline** | **190,463** | **190.5** |

- **Part4b share of total:** 138.3 / 190.5 ≈ **72.6%**
- Part1+2: ~22.9% | Part3 + Part4a + writePly: ~4.0%

---

## Duplicate SharpRoomActivity (still occurring)

- **One** `Viewer open: starting SharpRoomActivity` at 22:50:16.685 → app starts the viewer once.
- **Two onCreates:**
  - First: 22:50:16.751 `this=160203589` → `existing=false samePath=false` (sets ref, full load).
  - Second: 22:50:17.848 `this=202957278` → `existing=false samePath=false` (ref null; full load again).
- **No** `onNewIntent` log → with `singleTask`, the second intent should have been delivered to the first instance via `onNewIntent`; the fact that a second `onCreate` runs means either (1) the build does not have `singleTask` in the manifest, or (2) the second start uses a different task/affinity so a new instance is created.

**Result:** Two PLY copies (~293 MB each), two WebView inits.

---

## Summary

| Metric | Value |
|--------|--------|
| Total pipeline | 190.5 s |
| Part4b | 138.3 s (72.6%) |
| Viewer | Duplicate still present; verify manifest has `launchMode="singleTask"` and do a clean rebuild. |
