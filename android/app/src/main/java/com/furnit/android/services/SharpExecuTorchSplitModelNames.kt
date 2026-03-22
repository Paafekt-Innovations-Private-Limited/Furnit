package com.furnit.android.services

/**
 * **Single catalog of SHARP split `sharp_split_*.pte` basenames** used by Kotlin and expected by native code.
 *
 * Keep in sync with string literals in:
 * `android/app/src/main/cpp/sharp_executorch_full.cpp` (CPU Part1–4 / portable),
 * `sharp_executorch_full_vulkan.cpp` (Vulkan Part1–4 + hybrid Part3–4),
 * `sharp_executorch_full_common.cpp` (tiled / fine-split Part4b).
 *
 * Before renaming an export on disk, grep this object **and** those C++ files for the old basename.
 *
 * Vulkan Part1/2/3 resolution in C++ also uses `{stem}_vulkan_fp32.pte` / `{stem}_vulkan_fp16.pte` via
 * `resolveVulkanSplitPte` (stems below + optional `sharp_split_part1_b2` / `sharp_split_part2_b2`).
 */
object SharpExecuTorchSplitModelNames {

    /** Passed to Kotlin `orderedVulkanPrecisionNames` / C++ `resolveVulkanSplitPte(dir, stem)`. */
    const val VULKAN_RESOLVE_STEM_PART1 = "sharp_split_part1"
    const val VULKAN_RESOLVE_STEM_PART2 = "sharp_split_part2"
    const val VULKAN_RESOLVE_STEM_PART3 = "sharp_split_part3"
    const val VULKAN_RESOLVE_STEM_PART3_1280 = "sharp_split_part3_1280"
    const val VULKAN_RESOLVE_STEM_PART4_1280 = "sharp_split_part4_1280"

    // --- CPU / portable Part1–2 (hybrid sidecars + CPU-stable path) — sharp_executorch_full.cpp moduleCacheLoadPart12Cpu ---
    const val PART1_INT8 = "sharp_split_part1_int8.pte"
    const val PART2_INT8 = "sharp_split_part2_int8.pte"
    const val PART1_B4_INT8 = "sharp_split_part1_b4_int8.pte"
    const val PART2_B4_INT8 = "sharp_split_part2_b4_int8.pte"
    const val PART1_FP32 = "sharp_split_part1.pte"
    const val PART1_FP16 = "sharp_split_part1_fp16.pte"
    const val PART2_FP32 = "sharp_split_part2.pte"
    const val PART2_FP16 = "sharp_split_part2_fp16.pte"
    const val PART1_B4_FP32 = "sharp_split_part1_b4.pte"
    const val PART2_B4_FP32 = "sharp_split_part2_b4.pte"

    // --- CPU Part3–4 (CPU-stable / XNNPACK) — sharp_executorch_full.cpp ---
    const val PART3_INT8 = "sharp_split_part3_int8.pte"
    const val PART4A_CHUNK_512 = "sharp_split_part4a_chunk_512.pte"
    const val PART4A_CHUNK_65 = "sharp_split_part4a_chunk_65.pte"

    /** Single-decoder Part4b (optional if tiled path is used). Pick order in C++: FP32 → FP16 → INT8 (see cpp). */
    const val PART4B_INT8 = "sharp_split_part4b_int8.pte"
    const val PART4B_FP16 = "sharp_split_part4b_fp16.pte"
    const val PART4B_FP32 = "sharp_split_part4b.pte"
    const val PART3_FP16 = "sharp_split_part3_fp16.pte"
    const val PART3_FP32 = "sharp_split_part3.pte"

    // --- Vulkan Part3–4 (etVulkan + hybrid) — sharp_executorch_full_vulkan.cpp ---
    const val PART1_VULKAN_FP32 = "sharp_split_part1_vulkan_fp32.pte"
    const val PART1_VULKAN_FP16 = "sharp_split_part1_vulkan_fp16.pte"
    const val PART2_VULKAN_FP32 = "sharp_split_part2_vulkan_fp32.pte"
    const val PART2_VULKAN_FP16 = "sharp_split_part2_vulkan_fp16.pte"
    const val PART3_VULKAN_FP32 = "sharp_split_part3_vulkan_fp32.pte"
    const val PART3_VULKAN_FP16 = "sharp_split_part3_vulkan_fp16.pte"
    const val PART3_1280_VULKAN_FP32 = "sharp_split_part3_1280_vulkan_fp32.pte"
    const val PART3_1280_VULKAN_FP16 = "sharp_split_part3_1280_vulkan_fp16.pte"
    const val PART4_1280_VULKAN_FP32 = "sharp_split_part4_1280_vulkan_fp32.pte"
    const val PART4_1280_VULKAN_FP16 = "sharp_split_part4_1280_vulkan_fp16.pte"
    const val PART4A_CHUNK_512_VULKAN = "sharp_split_part4a_chunk_512_vulkan.pte"
    const val PART4A_CHUNK_65_VULKAN = "sharp_split_part4a_chunk_65_vulkan.pte"
    /** Legacy filename; **not** loaded by `sharp_executorch_full_vulkan.cpp` (tiled Part4b only). Kept for tooling / old docs. */
    const val PART4B_VULKAN = "sharp_split_part4b_vulkan.pte"

    // --- Tiled / fine-split Part4b — sharp_executorch_full_common.cpp + Kotlin routing ---
    const val PART4B_TILE_B2 = "sharp_split_part4b_tile_b2.pte"
    const val PART4B_TILE_B4 = "sharp_split_part4b_tile_b4.pte"
    const val PART4B_TILE_00 = "sharp_split_part4b_tile_00.pte"
    const val PART4B_TILE_FULL = "sharp_split_part4b_tile_full.pte"
    const val PART4B_TILE_B2_STAGE_A_VULKAN = "sharp_split_part4b_tile_b2_stage_a_vulkan.pte"
    const val PART4B_TILE_B2_INIT_BASE = "sharp_split_part4b_tile_b2_init_base.pte"
    const val PART4B_TILE_B2_RAW_HEADS_VULKAN = "sharp_split_part4b_tile_b2_raw_heads_vulkan.pte"
    const val PART4B_TILE_B2_COMPOSE = "sharp_split_part4b_tile_b2_compose.pte"
    const val PART4B_TILE_B2_STAGE_PRE_VULKAN = "sharp_split_part4b_tile_b2_stage_pre_vulkan.pte"
    const val PART4B_TILE_B2_DECODER_HEAD = "sharp_split_part4b_tile_b2_decoder_head.pte"
    const val PART4B_TILE_00_STAGE_A_VULKAN = "sharp_split_part4b_tile_00_stage_a_vulkan.pte"
    const val PART4B_TILE_00_INIT_BASE = "sharp_split_part4b_tile_00_init_base.pte"
    const val PART4B_TILE_00_RAW_HEADS_VULKAN = "sharp_split_part4b_tile_00_raw_heads_vulkan.pte"
    const val PART4B_TILE_00_COMPOSE = "sharp_split_part4b_tile_00_compose.pte"
    const val PART4B_TILE_00_STAGE_PRE_VULKAN = "sharp_split_part4b_tile_00_stage_pre_vulkan.pte"
    const val PART4B_TILE_00_DECODER_HEAD = "sharp_split_part4b_tile_00_decoder_head.pte"
    /** Fine-split / Part4-only test slices (see `Part4OnlyTest`; optional for main C++ path). */
    const val PART4B_TILE_00_DECODER_ONLY = "sharp_split_part4b_tile_00_decoder_only.pte"
    const val PART4B_TILE_00_DISPARITY_HEAD = "sharp_split_part4b_tile_00_disparity_head.pte"
    const val PART4B_TILE_00_DECODER_SEED = "sharp_split_part4b_tile_00_decoder_seed.pte"
    const val PART4B_TILE_00_DECODER_MERGE_X1 = "sharp_split_part4b_tile_00_decoder_merge_x1.pte"
    const val PART4B_TILE_00_DECODER_MERGE_X0 = "sharp_split_part4b_tile_00_decoder_merge_x0.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT1 = "sharp_split_part4b_tile_00_decoder_merge_latent1.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT0 = "sharp_split_part4b_tile_00_decoder_merge_latent0.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE = "sharp_split_part4b_tile_00_decoder_merge_latent0_prefuse.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE = "sharp_split_part4b_tile_00_decoder_merge_latent0_postfuse.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE_PORTABLE =
        "sharp_split_part4b_tile_00_decoder_merge_latent0_prefuse_portable.pte"
    const val PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE_PORTABLE =
        "sharp_split_part4b_tile_00_decoder_merge_latent0_postfuse_portable.pte"
    const val PART4B_TILE_00_DECODER_HEAD_PORTABLE = "sharp_split_part4b_tile_00_decoder_head_portable.pte"

    /** Legacy 4×4 grid single-tile decoders: tile_00 … tile_15 (Kotlin assets / optional paths). */
    val PART4B_LEGACY_TILE_GRID_PTES: List<String> = (0 until 16).map { idx ->
        String.format("sharp_split_part4b_tile_%02d.pte", idx)
    }

    /** Minimal Vulkan split set for sanity checks / small APK bundles. */
    val VULKAN_SPLIT_CORE_PTES: Array<String> = arrayOf(
        PART1_VULKAN_FP32,
        PART2_VULKAN_FP32,
        PART3_VULKAN_FP32,
        PART4B_VULKAN,
    )

    /** Optional `.pte` names that Gradle may copy into `assets/models_cpuvulkan_hybrid` for local testing. */
    val ASSET_OFFLOADABLE_VULKAN_PTES: Array<String> = arrayOf(
        PART1_VULKAN_FP32,
        PART2_VULKAN_FP32,
        PART3_VULKAN_FP32,
        PART3_1280_VULKAN_FP32,
        PART1_VULKAN_FP16,
        PART2_VULKAN_FP16,
        PART3_VULKAN_FP16,
        PART3_1280_VULKAN_FP16,
        PART4_1280_VULKAN_FP32,
        PART4_1280_VULKAN_FP16,
        PART4A_CHUNK_512_VULKAN,
        PART4A_CHUNK_65_VULKAN,
        PART4B_VULKAN,
        PART4B_TILE_B2,
        PART4B_TILE_FULL,
        *PART4B_LEGACY_TILE_GRID_PTES.toTypedArray(),
    )
}
