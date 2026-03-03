# SHARP ExecuTorch pipeline timing analysis

## Inference timing (example run 23:53:22–23:56:43)

| Phase | Duration | % of total |
|-------|----------|------------|
| Part1+2 load | 18 ms | — |
| 1x patches (5×5) | **31.99 s** | 15.9% |
| 0.5x patches (3×3) | **12.40 s** | 6.2% |
| 0.25x patch (35th) | **1.41 s** | 0.7% |
| Part3 (image encoder) | **1.52 s** | 0.8% |
| Part4a chunks (512+65) | **3.72 s** | 1.9% |
| **Part4b FP32** | **146.31 s** | **72.9%** |
| writePly | **2.59 s** | 1.3% |
| **Total pipeline** | **200.88 s (3 m 21 s)** | 100% |

## Viewer timing (this run)

| Event | Time | Delta |
|-------|------|-------|
| startActivity | 23:56:43.998 | — |
| First onCreate | 23:56:44.052 | +54 ms |
| Second onCreate (recreated) | 23:56:45.031 | +1.0 s |
| Copied PLY | 23:56:45.504 | +0.5 s from 2nd onCreate |
| WebView load / SparkJS init | 23:56:46.313 | +0.8 s |
| SplatMesh onLoad (PLY decoded) | 23:56:49.109 | +2.8 s |
| autoFrameRoom (fallback) | 23:56:49.725 | +0.6 s |
| **WebGL viewer reported loaded** | 23:56:49.773 | **4742 ms since onCreate** |

So **onCreate → viewer loaded** for the visible instance is **~4.7 s**.

## Duplicate activity

Two onCreates still occur (first then recreated second; both see `existing=false`). We only finish the second when another *live* instance exists, so the recreated instance proceeds and shows the room. Both do PLY copy + load; the one that stays on screen is the second.

## Summary

- **Part4b FP32** is ~73% of total inference time (~146 s).
- **Total pipeline** ~200 s (~3 m 21 s); **viewer ready** ~4.7 s after that activity’s onCreate.
- 1,179,648 Gaussians; PLY ~293 MB; writePly ~2.6 s.
