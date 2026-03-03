# Logcat analysis (21:03 run, 03-01)

## Pipeline timing

| Phase | Time |
|-------|------|
| Part1+2 (1x + 0.5x + 0.25x patches) | ~44 s |
| Part3 (image encoder) | ~1.4 s |
| Part4a (chunked decoder) | ~3.5 s |
| **Part4b forward (FP32)** | **142.3 s** |
| writePly | ~2.9 s |
| **Total pipeline** | **194.7 s** |

Part4b is ~73% of total time. No crash.

---

## Duplicate SharpRoomActivity (still occurring)

- **First open:** 21:11:00.288 – Opening SharpRoomActivity, Copied PLY, Loading PLY file.
- **Second open:** 21:11:01.255 – same again (~1 s later).

So the viewer still starts twice and the ~293 MB PLY is copied twice.

### Why debounce might not show in your log

1. **Log filter** – Debounce logs use tag `SinglePhotoRoom`, but your filter is  
   `ExecutorchInt8Sharp:D SharpService:D SharpRoomActivity:D`, so those messages are dropped.
2. **Build** – If the app wasn’t rebuilt after adding the debounce, the second start would still run.

### Change made

- **SharpService-tag logs** were added so they appear with your current filter:
  - `SharpService: Viewer open: starting SharpRoomActivity path=...` – when we actually start the viewer.
  - `SharpService: Viewer open debounced: skip duplicate (same path, Xms ago)` – when we skip the second start.

### What to do next

1. **Rebuild and reinstall**  
   `./gradlew installDebug` (or Run from Android Studio).

2. **Run the same flow** and capture logcat with your usual filter:
   ```bash
   adb -s 53181JEBF16055 logcat -s ExecutorchInt8Sharp:D SharpService:D SharpRoomActivity:D -v time
   ```

3. **Check for:**
   - **One** `Viewer open: starting SharpRoomActivity` and **one** `Viewer open debounced: skip duplicate`  
     → Debounce is working; only one viewer start.
   - **Two** `Viewer open: starting SharpRoomActivity`  
     → Second start isn’t going through the debounce (e.g. old build or another caller).
   - **One** `Viewer open: starting SharpRoomActivity` and **one** “Opening SharpRoomActivity” (SharpRoomActivity tag)  
     → Single viewer start; duplicate is fixed.

If you still see two “Opening SharpRoomActivity” lines after a clean rebuild, say so and we can add a guard inside SharpRoomActivity (e.g. finish() if the same room is already visible).
