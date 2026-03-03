# Ask Ultralytics: Part4b timing (FP32) and duplicate SharpRoomActivity

Use the text below in Ultralytics (Ask AI or forum) to get advice on reducing Part4b latency and avoiding duplicate viewer start.

---

## Paste this into Ultralytics

```
We're running a SHARP 3D Gaussian splatting pipeline on Android with ExecuTorch. Part4b (decoder + Gaussian heads) runs on CPU (XNNPACK); we don't have Vulkan in our pip ExecuTorch build. On device we see:

**Part4b timing (FP32):**
- Part4b forward: ~142 s (single run, 7 inputs, ~1.2M Gaussians output)
- Total pipeline: ~196 s (Part1+2 ~44 s, Part3 ~1.5 s, Part4a ~3.7 s, writePly ~3.1 s)
- Part4b is 72% of total time

We exported Part4b as FP32 with XNNPACK (Vulkan partition not available in pip package; Part4b FP16 export failed with "Input type float and bias type c10::Half" mixed-precision error). What do you recommend to reduce Part4b latency on Android when Vulkan isn't in the pip build? Options we're considering: (1) Build ExecuTorch from source with Vulkan and re-export Part4b for GPU. (2) Fix the FP16 export (which layers to keep FP32 to avoid the mixed-precision error). (3) Any other deployment tweaks (thread count, memory layout, or backend flags) that help CPU decoder throughput.

**Separate Android UX question:** After inference finishes we start a single Activity (SharpRoomActivity) to show the 3D room in a WebView. Logcat shows the Activity opening twice in quick succession (two "Opening SharpRoomActivity with PLY", two "Copied PLY to internal storage") with a "stopBrainDetection()" in between—so one instance seems to replace the other and we copy the ~280 MB PLY twice. We start the viewer once from our service (VIEWER_FEED intent). What's a reliable way to avoid this duplicate start (e.g. launchMode singleTask/singleTop, or not re-starting if the same room is already visible) so we only load the PLY once?
```

---

## Short version (if character limit)

```
1) ExecuTorch Part4b on Android CPU: ~142 s for one forward (72% of pipeline). Vulkan not in pip build; FP16 export failed (float/half mix). Best way to reduce Part4b latency—build ExecuTorch with Vulkan, fix FP16 export, or other CPU/deployment tweaks?

2) SharpRoomActivity starts twice after inference (two PLY copies, two WebView inits). We launch once with VIEWER_FEED. How to avoid duplicate start (singleTask/singleTop or same-room check)?
```
