package com.furnit.android.services

/**
 * Canonical basenames for the **CPU / XNNPACK** split SHARP pipeline (C++ `sharp_executorch_full`).
 *
 * **Reference set (Mar 2026 v2):** `/Volumes/LaCie/march10th2026/v2` — six required files plus **optional** single Part4b
 * (`PART4B_INT8` / `PART4B_FP16` / `PART4B_FP32`) for clearer output than tiled `PART4B_TILE_B2` / `PART4B_TILE_B4` alone.
 * Native code loads these exact strings
 * (`sharp_executorch_full.cpp`, `sharp_executorch_full_common.cpp`); keep names in sync when exporting.
 *
 * This object is the single Kotlin-side source for [ExecutorchInt8Sharp] `findFile(...)` checks.
 * C++ literals must match (search repo for the same basename before renaming).
 */
object SharpExecuTorchSplitModelNames {
    const val PART1_INT8 = "sharp_split_part1_int8.pte"
    const val PART2_INT8 = "sharp_split_part2_int8.pte"
    const val PART3_INT8 = "sharp_split_part3_int8.pte"
    const val PART4A_CHUNK_512 = "sharp_split_part4a_chunk_512.pte"
    const val PART4A_CHUNK_65 = "sharp_split_part4a_chunk_65.pte"

    /** Single-decoder Part4b (optional if tiled path is used). Pick order: INT8 → FP16 → FP32 (matches C++). */
    const val PART4B_INT8 = "sharp_split_part4b_int8.pte"
    const val PART4B_FP16 = "sharp_split_part4b_fp16.pte"
    const val PART4B_FP32 = "sharp_split_part4b.pte"

    /** Batched tiled Part4b variants; checked before tile_00 sequential path in C++. */
    const val PART4B_TILE_B2 = "sharp_split_part4b_tile_b2.pte"
    const val PART4B_TILE_B4 = "sharp_split_part4b_tile_b4.pte"
    const val PART4B_TILE_00 = "sharp_split_part4b_tile_00.pte"
    const val PART4B_TILE_FULL = "sharp_split_part4b_tile_full.pte"
    const val PART4B_TILE_B2_STAGE_A_VULKAN = "sharp_split_part4b_tile_b2_stage_a_vulkan.pte"
    const val PART4B_TILE_B2_INIT_BASE = "sharp_split_part4b_tile_b2_init_base.pte"
    const val PART4B_TILE_B2_RAW_HEADS_VULKAN = "sharp_split_part4b_tile_b2_raw_heads_vulkan.pte"
    const val PART4B_TILE_B2_COMPOSE = "sharp_split_part4b_tile_b2_compose.pte"
    const val PART4B_TILE_00_STAGE_A_VULKAN = "sharp_split_part4b_tile_00_stage_a_vulkan.pte"
    const val PART4B_TILE_00_INIT_BASE = "sharp_split_part4b_tile_00_init_base.pte"
    const val PART4B_TILE_00_RAW_HEADS_VULKAN = "sharp_split_part4b_tile_00_raw_heads_vulkan.pte"
    const val PART4B_TILE_00_COMPOSE = "sharp_split_part4b_tile_00_compose.pte"

    const val PART1_B4_INT8 = "sharp_split_part1_b4_int8.pte"
    const val PART2_B4_INT8 = "sharp_split_part2_b4_int8.pte"
    const val PART1_FP32 = "sharp_split_part1.pte"
    const val PART1_FP16 = "sharp_split_part1_fp16.pte"
    const val PART2_FP32 = "sharp_split_part2.pte"
    const val PART2_FP16 = "sharp_split_part2_fp16.pte"
    const val PART3_FP16 = "sharp_split_part3_fp16.pte"
    const val PART3_FP32 = "sharp_split_part3.pte"
}
