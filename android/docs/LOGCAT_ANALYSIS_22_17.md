# Logcat analysis (22:17 run, 03-01)

## Pipeline timing

| Phase | Time (ms) | Time (s) |
|-------|-----------|----------|
| Part1+2 load | 18 | 0.02 |
| Part1+2 (1x patches 5×5) | 30,721 | 30.7 |
| Part1+2 (0.5x patches 3×3) | 11,495 | 11.5 |
| Part1+2 (0.25x patch) | 1,279 | 1.3 |
| **Part1+2 total** | **~43,513** | **~43.5** |
| Part3 (image encoder) | 1,360 | 1.4 |
| Part4a (chunked decoder) | 3,547 | 3.5 |
| **Part4b forward (FP32)** | **138,277** | **138.3** |
| writePly | 2,567 | 2.6 |
| **Total pipeline** | **190,237** | **190.2** |

- **Part4b share of total:** 138.3 / 190.2 ≈ **73%**
- Part1+2 (patch encoder): ~23%
- Part3 + Part4a + writePly: ~4%

---

## Duplicate SharpRoomActivity (still occurring)

- **Single viewer start from app:** One `Viewer open: starting SharpRoomActivity` at 22:21:09.295.
- **Two onCreates:** 
  - First: 22:21:09.341 `this=4507580` → `existing=false samePath=false` (sets ref, does full load).
  - Second: 22:21:10.268 `this=202957278` → `existing=false samePath=false` (ref is null; does full load again).
- **No** `Viewer open debounced` → the second `startActivity(SharpRoomActivity)` is **not** going through `SinglePhotoRoomActivity.openSharpRoomWithResult`, so it’s coming from another code path or the same intent is being used to create a second instance before the first is “on top.”

So we still get:
- Two PLY copies (~293 MB each).
- Two WebView inits and two full viewer setups.

---

## Interpretation

1. **Part4b** is the main latency target (FP32 CPU; Vulkan/FP16 would be the next step).
2. **Duplicate viewer** is not from our debounce path (static + posting didn’t stop the second activity). Either:
   - Another component starts `SharpRoomActivity` with the same room (e.g. list/refresh or another activity), or
   - The same `startActivity` leads the system to create a second instance because the first isn’t resumed yet (singleTop doesn’t reuse).

---

## Recommended next steps

1. **Find the second caller:** Add a single `android.util.Log.d("SharpService", "startActivity(SharpRoomActivity) from " + Thread.currentThread().stackTrace[3].toString())` (or log a tag) in **every** place that calls `startActivity(Intent(..., SharpRoomActivity::class.java))` (SinglePhotoRoomActivity, ContentActivity, ModelDetailActivity, GLBRoomActivity). Re-run and check which caller appears twice in logcat.
2. **Done:** `SharpRoomActivity` is now `singleTask` in the manifest so the second intent goes to `onNewIntent` on the existing instance. Rebuild and run.
3. **Keep in-Activity guard:** Leave the duplicate-instance guard and diagnostic log in `SharpRoomActivity` so that if the second start is from our code path in the future, we’ll at least skip the second PLY copy/load (once the first instance is considered “current” and the ref isn’t cleared before the second runs).
