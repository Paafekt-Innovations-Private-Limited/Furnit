# Part1 Vulkan performance (Furnit)

Focus **only Part1** (`sharp_split_part1.pte`) until forward time is acceptable. Registration and “keep Module alive” are necessary but **not sufficient** if each `forward()` still costs many minutes: that usually means **per-inference** cost (layout churn, attention on Vulkan, repacking), not a missing warmup.

## On-device: three timed forwards (same PID)

1. Install/run the app; **do not force-stop** (same process).
2. **Settings → Developer → Benchmark 3×** (or Warmup first if you want load+2× warmup done separately).
3. Capture logs:

```bash
adb logcat -d | grep P1_BENCH
adb logcat -d | grep PART1_RUN
adb logcat -d | grep PART1_GOLDEN
adb logcat -d | grep PART1_ARTIFACT
```

Lines include:

- `P1_BENCH session_ensure_ms=…` — includes **load + 2× warmup** only when the session was cold.
- `P1_BENCH timed_forward 1/3 duration_ms=…` (and 2/3, 3/3).
- `P1_BENCH summary duration_ms=[a,b,c] ratio_2_over_1=… ratio_3_over_1=…`

**Read results:**

- **Run 1 huge, runs 2–3 much smaller** → first forward pays pipeline/shader; later work reuses caches.
- **All three similar and huge** → steady-state execution is slow (graph/partition/layout), not “missing second warmup.”

Compare with **portable Part1** (`--part12-only-portable` or `--part1-only` + `portable`) for a fast CPU baseline on the same patch.

## Export-side levers (Part1 only)

Script: `android/export_sharp_executorch_split4.py` (`--part1-only`, `--backend vulkan|portable`).

1. **Fewer Vulkan layout transitions**  
   Re-export and compare exporter output: fewer “Inserting transition(s)” / buffer↔texture repacks is better. Large delegated graphs with many `view`/`permute`/`linear`/`bmm`/`softmax` boundaries often bounce **WIDTH_PACKED** vs **CHANNELS_PACKED**.

2. **Partition size tuning**  
   Try **smaller** Vulkan regions (more CPU fallback) vs **maximal** Vulkan — “more Vulkan” is not always faster if transitions dominate. Compare `[Partition] Vulkan strings in .pte` and real device times.

3. **Split the model (future export work)**  
   Part1a (early blocks) vs Part1b (attention-heavy) as separate `.pte` files lets you see which half burns time on Vulkan.

4. **One canonical artifact**  
   For tuning, stick to one Part1 file name/path (`sharp_split_part1.pte`) so the app cache and driver behavior stay comparable.

5. **Release + ExecuTorch version**  
   Build native/ExecuTorch **Release** for timing. Vulkan backend evolves quickly — newer ExecuTorch + re-export may change transition counts and perf.

## App behavior (already in Furnit)

- **Persistent `Module`** for Part1 test path; **double warmup** after load; **Run** reuses cache.
- **Release Part1 cache** in Settings if you swap `.pte` on disk.

See also: `android/docs/EXECUTORCH_VULKAN_REGISTRATION.md` (registration / load order).
