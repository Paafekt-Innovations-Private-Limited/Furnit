# SHARP split `.pte` basenames

**Kotlin source of truth:** `SharpExecuTorchSplitModelNames`  
(`android/app/src/main/java/com/furnit/android/services/SharpExecuTorchSplitModelNames.kt`)

**Native literals must match:**

- `android/app/src/main/cpp/sharp_executorch_full.cpp` — CPU / portable Part1–4
- `android/app/src/main/cpp/sharp_executorch_full_vulkan.cpp` — Vulkan + hybrid Part3–4
- `android/app/src/main/cpp/sharp_executorch_full_common.cpp` — tiled / fine-split Part4b

**Before renaming an export:** grep this object and the three C++ files for the old basename, then update all occurrences.

**Vulkan stems:** `VULKAN_RESOLVE_STEM_PART1` … `PART4_1280` — C++ `resolveVulkanSplitPte(dir, stem)` builds `{stem}_vulkan_fp32.pte` / `_fp16.pte`.
