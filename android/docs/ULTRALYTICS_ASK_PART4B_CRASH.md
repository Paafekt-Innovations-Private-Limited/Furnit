# Ask Ultralytics: Part4b crash during forward

Use this when the app crashes during Part4b inference (after Part4a chunks complete). Copy the block below into Ultralytics.

---

## Paste this into Ultralytics

```
We're running a SHARP 3D Gaussian splatting pipeline on Android with ExecuTorch INT8 for Part1+2, Part3, and Part4a; Part4b is FP32 (sharp_split_part4b.pte) on XNNPACK/CPU. The app crashes during the Part4b forward call—no exception in our Kotlin code; the process dies (likely native crash in ExecuTorch or XNNPACK).

**When it crashes:**
- Right after "Part4b forward: 7 inputs tokens, img, latent0, latent1, x0Feat, x1Feat, x2Feat" and before any "[TIMING] Part4b forward" or "[PLY]" log.
- Part4a completes successfully (chunked decoder, combinedTokens size=590848). Part4b is loaded (sharp_split_part4b.pte, FP32). Crash happens on the first Part4b execute/forward.

**Setup:**
- Device: Android (e.g. 53181JEBF16055).
- Part4b: single .pte, FP32, 7 inputs (tokens, img, latent0, latent1, x0Feat, x1Feat, x2Feat), output ~1.2M Gaussians.
- We use ExecuTorch's standard load/execute path; no Vulkan (pip build).

**Questions:**
1. What are the most likely causes of a native crash in Part4b forward on Android (OOM, XNNPACK op unsupported, alignment, thread count, input layout)?
2. How should we capture a useful stack trace (adb logcat with debug tags, tombstone, or ExecuTorch verbose logging) to narrow it down?
3. Any known issues or workarounds for large decoder forwards (e.g. splitting the run, different memory method, or backend flags) that avoid crashes on mid-range devices?
```

---

## Optional: add device and build info

```
Device: [your device model / API level]
ExecuTorch: pip package version [e.g. 0.2.0]
Part4b .pte: exported with executorch with XNNPACK; FP32; single module forward.
```
