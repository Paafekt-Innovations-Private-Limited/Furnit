/**
 * SHARP full pipeline — Vulkan Part1+2 (optional hybrid Part12-on-CPU) + Vulkan Part3–4 .pte names.
 *
 * Native stage boundaries (wall-clock; see [HYBRID_TIMING] summary log):
 *  - Downsample (optional): multiscale half/quarter images for 0.5×/0.25× patches.
 *  - Part1+2: patch encoder (Vulkan or CPU); 25-only Vulkan uses two-pass or optional interleaved path.
 *  - Early encoder scratch release: frees patch/temp buffers after Part1+2 (before Part3/4a).
 *  - Part3 / Part4a / Part4b: sequential ExecuTorch stages (runtime is not thread-safe across threads).
 *  - Part4b: tiled / batched Gaussian decode (tile_00 loads on the pipeline thread only).
 */
#include "sharp_executorch_full_internal.h"
#if defined(EXECUTORCH_HAS_ETDUMP)
#include <executorch/devtools/etdump/etdump_flatcc.h>
#endif
#include <thread>

namespace {

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

/**
 * Prefer explicit FP32 Vulkan exports first, then fall back to FP16.
 *
 * The current Android Vulkan AAR build shipped in this app does not bundle all FP16 conversion shaders
 * (for example view_convert_buffer_float_half), so FP16-first resolution can abort during delegate init.
 * Keep runtime resolution FP32-first until the app ships a Vulkan runtime that fully supports these shaders.
 */
std::string resolveVulkanSplitPte(const std::string& dir, const char* stem) {
    const std::string candidates[] = {
        std::string(stem) + "_vulkan_fp32.pte",
        std::string(stem) + "_vulkan_fp16.pte",
    };
    for (const auto& name : candidates) {
        std::string full = pathJoin(dir, name);
        if (fileExists(full)) {
            return full;
        }
    }
    return {};
}

}  // namespace

bool moduleCacheLoadPart12Vulkan(ModuleCache& cache, const std::string& dir) {
    std::string p1vk = resolveVulkanSplitPte(dir, "sharp_split_part1");
    std::string p2vk = resolveVulkanSplitPte(dir, "sharp_split_part2");
    if (p1vk.empty() || p2vk.empty()) {
        LOGE("ModuleCache: Vulkan Part1/2 path resolve failed under %s (expected *_vulkan_fp16.pte or *_vulkan_fp32.pte)",
             dir.c_str());
        return false;
    }
    LOGD("ModuleCache: loading Vulkan Part1=%s Part2=%s", p1vk.c_str(), p2vk.c_str());
    cache.mod1 = std::make_unique<ETModule>(p1vk, ETModule::LoadMode::Mmap);
    cache.mod2 = std::make_unique<ETModule>(p2vk, ETModule::LoadMode::Mmap);
    if (cache.mod1->load() == Error::Ok && cache.mod2->load() == Error::Ok) {
        LOGD("ModuleCache: Part1+Part2 Vulkan split loaded");
        std::string p1b2 = resolveVulkanSplitPte(dir, "sharp_split_part1_b2");
        std::string p2b2 = resolveVulkanSplitPte(dir, "sharp_split_part2_b2");
        if (!p1b2.empty() && !p2b2.empty()) {
            cache.mod1_b2 = std::make_unique<ETModule>(p1b2, ETModule::LoadMode::Mmap);
            cache.mod2_b2 = std::make_unique<ETModule>(p2b2, ETModule::LoadMode::Mmap);
            if (cache.mod1_b2->load() == Error::Ok && cache.mod2_b2->load() == Error::Ok) {
                LOGD("ModuleCache: Part1+Part2 batch=2 Vulkan split loaded");
            } else {
                LOGW("ModuleCache: Vulkan batch=2 load failed p1=%s p2=%s; using single-patch path",
                     p1b2.c_str(), p2b2.c_str());
                cache.mod1_b2.reset();
                cache.mod2_b2.reset();
            }
        }
        return true;
    }
    LOGE("ModuleCache: Vulkan Part1/2 load failed p1=%s p2=%s", p1vk.c_str(), p2vk.c_str());
    cache.mod1.reset();
    cache.mod2.reset();
    LOGE("ModuleCache: Vulkan Part1/2 unavailable (expected *_vulkan_fp16.pte or *_vulkan_fp32.pte)");
    return false;
}

jfloatArray runSharpFullPipeline_Vulkan(
        JNIEnv* env,
        jobject thiz,
        jstring modelDirPath,
        jfloatArray imageNCHW,
        jint maxGaussians,
        jboolean preferSinglePart4b,
        jboolean part12OnCpu,
        jboolean part12ForceSinglePatch,
        jboolean part12_25Only,
        jint part1MaxPatches1x,
        jint part1MaxPatches05x,
        jint part12Chunk1x,
        jint part12Chunk05x,
        jint part12YieldMsBetweenChunks,
        jboolean swapTileNdcXY,
        jobject progressReporter,
        jstring etdumpOutputPath,
        jboolean hybridInterleavePart12,
        jlong hybridInterleaveMinAvailMemBytes) {
    if (!modelDirPath || !imageNCHW) {
        LOGE("runFullPipelineInt8: null modelDir or image");
        return nullptr;
    }
    jmethodID reportProgressMethodId = nullptr;
    if (progressReporter) {
        jclass clazz = env->GetObjectClass(progressReporter);
        if (clazz) {
            reportProgressMethodId = env->GetMethodID(clazz, "reportProgressFromNative", "(FLjava/lang/String;)V");
            env->DeleteLocalRef(clazz);
        }
    }
    const int maxG = static_cast<int>(maxGaussians);
    // JNI keeps preferSinglePart4b for the shared CPU/Vulkan signature; Vulkan Part4b is always tiled-first here.
    (void)preferSinglePart4b;
    const bool swapTileNdc = (swapTileNdcXY == JNI_TRUE);
    const bool useVulkanBackend = true;
    const bool part12Cpu = (part12OnCpu == JNI_TRUE);
    const bool useVulkanForPart12 = !part12Cpu;

    ensureRuntimeInit();
    clearLastMonodepthCapture();
    const long long t0 = nowMs();
    const bool wantInterleavePart12 = (hybridInterleavePart12 == JNI_TRUE);
    const jlong interleaveMinAvailBytes = hybridInterleaveMinAvailMemBytes;
    const long memAvailKbGate = sharpDeviceMemAvailableKb();
    const bool interleaveMemOk =
            interleaveMinAvailBytes <= 0 ||
            (memAvailKbGate >= 0 &&
             (static_cast<unsigned long long>(memAvailKbGate) * 1024ull >=
              static_cast<unsigned long long>(interleaveMinAvailBytes)));

    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string modelDir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (modelDir.empty()) { LOGE("empty model dir"); return nullptr; }

    // Resolve Vulkan Part3/Part4a paths once (overlap worker + sequential fallback).
    const std::string vkPathPart3 = resolveVulkanSplitPte(modelDir, "sharp_split_part3");
    const std::string vkPath4a512 = pathJoin(modelDir, "sharp_split_part4a_chunk_512_vulkan.pte");
    const std::string vkPath4a65 = pathJoin(modelDir, "sharp_split_part4a_chunk_65_vulkan.pte");

    std::string etdumpPath;
    if (etdumpOutputPath) {
        const char* ep = env->GetStringUTFChars(etdumpOutputPath, nullptr);
        if (ep) {
            etdumpPath = ep;
            env->ReleaseStringUTFChars(etdumpOutputPath, ep);
        }
    }

    const int limit1x = (part1MaxPatches1x > 0) ? std::min(25, part1MaxPatches1x) : 25;
    // When in test mode (limit1x < 25), allow 0 at 0.5x (minimal 1+0+1). Otherwise 0 means full 9.
    const int limit05x = (part1MaxPatches1x > 0 && part1MaxPatches05x == 0)
            ? 0
            : ((part1MaxPatches05x > 0) ? std::min(9, part1MaxPatches05x) : 9);
    const bool skip05xAnd025x = (part12_25Only == JNI_TRUE);
    const bool needMultiscalePatchPrep = !skip05xAnd025x;
    const int chunk1x = (part12Chunk1x > 0) ? std::min(25, part12Chunk1x) : 0;
    const int chunk05x = (part12Chunk05x > 0) ? std::min(9, part12Chunk05x) : 0;
    const int yieldMs = (part12YieldMsBetweenChunks > 0) ? part12YieldMsBetweenChunks : 0;
    LOGD("runFullPipelineInt8: modelDir=%s useVulkan=%d part12OnCpu=%d part1Limits 1x=%d 0.5x=%d chunk1x=%d chunk05x=%d yieldMs=%d preferSingleP4b=%d (Vulkan ignores; tiled-first) swapTileNdc=%d",
         modelDir.c_str(), useVulkanBackend ? 1 : 0, part12Cpu ? 1 : 0, limit1x, limit05x, chunk1x, chunk05x, yieldMs,
         0, swapTileNdc ? 1 : 0);

    // Full image: 1536x1536 or 1280x1280 NCHW. Part1 is fed 384x384 patches only.
    jsize imageLen = env->GetArrayLength(imageNCHW);
    int imageSize = IMAGE_SIZE;
    int halfSize = HALF_SIZE;
    int stride1x = STRIDE_1X;
    int stride05x = STRIDE_05X;
    if (imageLen == 3 * IMAGE_SIZE_1280 * IMAGE_SIZE_1280) {
        imageSize = IMAGE_SIZE_1280;
        halfSize = IMAGE_SIZE_1280 / 2;
        stride1x = (IMAGE_SIZE_1280 - PATCH_SIZE) / 4;
        stride05x = (halfSize - PATCH_SIZE) / 2;
        LOGD("Using 1280x1280 input (reduced memory); need Part3/Part4 exported for 1280");
    } else if (imageLen != 3 * IMAGE_SIZE * IMAGE_SIZE) {
        LOGE("bad image length %d (expected %d or %d for 1536/1280 NCHW)", (int)imageLen, 3 * IMAGE_SIZE * IMAGE_SIZE, 3 * IMAGE_SIZE_1280 * IMAGE_SIZE_1280);
        return nullptr;
    }
    // Allocate workspace (reused across calls; only first call mallocs)
    if (!g_workspace.allocate()) {
        LOGE("workspace alloc failed");
        return nullptr;
    }
    g_workspace.zero();
    logProcessMemory("after workspace allocate");

    jfloat* imageData = env->GetFloatArrayElements(imageNCHW, nullptr);
    if (!imageData) return nullptr;

    // ── Downsample for 0.5x and 0.25x patches ───────────────────────────────
    if (needMultiscalePatchPrep) {
        long long tDown = nowMs();
        downsample2x(imageData, imageSize, imageSize, 3, g_workspace.halfImg);
        downsample4x(imageData, imageSize, imageSize, 3, g_workspace.quarterImg);
        LOGD("Downsample 2x+4x: %lldms", nowMs() - tDown);
    } else {
        LOGD("Skipped 0.5x+0.25x image downsample prep (part12_25_only)");
    }

    // ── Part1 + Part2 (singleton cache) ──────────────────────────────────────
    // When part12OnCpu we load portable Part1+2; only require Vulkan Part1 file when useVulkanForPart12.
    if (useVulkanForPart12) {
        std::string path1_vk = resolveVulkanSplitPte(modelDir, "sharp_split_part1");
        if (path1_vk.empty()) {
            LOGE("Part1 Vulkan .pte not found under %s (try sharp_split_part1_vulkan_fp32.pte or _fp16.pte)",
                 modelDir.c_str());
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
    }

    long long tLoad12 = nowMs();
    if (!g_moduleCache.ensureLoaded(modelDir, useVulkanForPart12)) {
        LOGE("Failed to load Part1+Part2");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGD("Part1+Part2 ready (cache): %lldms", nowMs() - tLoad12);

    const long long tPart12WallStart = nowMs();
    float* ws_patch = g_workspace.patchBuf;
    float* ws_tokens = g_workspace.tokensCopy;
    float* ws_temp = g_workspace.tempSpatial;
    const size_t patchSz = (size_t)(3 * PATCH_SIZE * PATCH_SIZE);
    const size_t tokenSliceSz = (size_t)TOKENS_577 * FEATURE_DIM;
    const size_t featSliceSz = (size_t)FEATURE_DIM * SPATIAL_HW;
    ETModule* m1 = g_moduleCache.mod1.get();
    ETModule* m2 = g_moduleCache.mod2.get();
    ETModule* m1_b4 = g_moduleCache.mod1_b4.get();
    ETModule* m2_b4 = g_moduleCache.mod2_b4.get();
    ETModule* m1_b2 = g_moduleCache.mod1_b2.get();
    ETModule* m2_b2 = g_moduleCache.mod2_b2.get();
    const bool part12ForceSingle = (part12ForceSinglePatch == JNI_TRUE);
    bool usedTwoPassVulkan25Only = false;
    bool usedInterleavedVulkan25Only = false;

    // Vulkan 25-only: optional interleaved Part1→Part2 per patch (keeps Part1+Part2 resident; memory-gated).
    // Default: two-pass (all Part1, release Part1, all Part2) for lower peak encoder memory.
    if (useVulkanForPart12 && skip05xAnd025x && limit05x == 0 && wantInterleavePart12 && interleaveMemOk) {
        usedInterleavedVulkan25Only = true;
        LOGD("[HYBRID] Vulkan 25-only: interleaved Part1→Part2 (MemAvailable~%ld KB, gate bytes=%lld)",
             (long)memAvailKbGate, (long long)interleaveMinAvailBytes);
        const int totalPart12ProgressUnits = limit1x + limit05x + 1;
        long long t1x = nowMs();
        if (progressReporter && reportProgressMethodId) {
            reportProgress(env, progressReporter, reportProgressMethodId, 0.21f,
                           "Part 1+2 (Vulkan interleaved): starting…");
        }
        for (int patchIdx = 0; patchIdx < limit1x; ++patchIdx) {
            const int ii = patchIdx / GRID_1X;
            const int jj = patchIdx % GRID_1X;
            cropNCHW(imageData, imageSize, imageSize, 3,
                     ii * stride1x, jj * stride1x, PATCH_SIZE, PATCH_SIZE,
                     ws_patch);
            auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            auto out1 = m1->forward(in1);
            if (!out1.ok() || out1->size() < 2) {
                LOGE("Part1 fail 1x interleaved (%d,%d)", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const auto& tokT = (*out1)[0].toTensor();
            const auto& spaT = (*out1)[1].toTensor();
            const float* tokPtr = tokT.const_data_ptr<float>();
            const float* spaPtr = spaT.const_data_ptr<float>();
            if (!tokPtr || !spaPtr) {
                LOGE("Part1 1x interleaved (%d,%d): null ptr", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            reshapeToSpatial(spaPtr, spaT.numel(), ws_temp);
            mergeCrop(g_workspace.latent0, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            reshapeToSpatial(tokPtr, tokT.numel(), ws_temp);
            mergeCrop(g_workspace.latent1, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            std::memcpy(ws_tokens, tokPtr, tokenSliceSz * sizeof(float));
            auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokInput};
            auto out2 = m2->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 fail 1x interleaved (%d,%d)", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            if (!featPtr) {
                LOGE("Part2 1x interleaved (%d,%d): feat null", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
            mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            if (progressReporter && reportProgressMethodId) {
                const int patchesDone = patchIdx + 1;
                const float p = (patchesDone / (float)totalPart12ProgressUnits) * 0.5f;
                char buf[96];
                snprintf(buf, sizeof(buf), "Part 1+2 interleaved: patch %d/%d…", patchesDone, limit1x);
                reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
            }
        }
        LOGD("1x Part1+2 interleaved (%d patches): %lldms", limit1x, nowMs() - t1x);
        downsample2x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x1Feat);
        downsample4x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x2Feat);
        LOGD("Skipped 0.5x+0.25x (part12_25_only); filled x1Feat,x2Feat from downsampled x0Feat");
    }

    // In the common Vulkan 25-only path, avoid keeping Part1 and Part2 live across each patch.
    // Run all Part1 patches first, cache token outputs, release Part1 modules, then run Part2.
    if (!usedInterleavedVulkan25Only && useVulkanForPart12 && skip05xAnd025x && limit05x == 0) {
        usedTwoPassVulkan25Only = true;
        const int totalPart12ProgressUnits = limit1x + limit05x + 1;
        long long t1x = nowMs();
        std::vector<float> cachedTokens((size_t)limit1x * tokenSliceSz);
        // Part1 runs all patches before Part2; without callbacks the UI stays at Kotlin's ~20% until Part1
        // finishes — first Vulkan forward often blocks on shader compile (minutes on some devices).
        if (progressReporter && reportProgressMethodId) {
            reportProgress(env, progressReporter, reportProgressMethodId, 0.21f,
                           "Part 1 (Vulkan): starting — first GPU forward may compile shaders (slow)…");
        }
        for (int patchIdx = 0; patchIdx < limit1x; ++patchIdx) {
            const int ii = patchIdx / GRID_1X;
            const int jj = patchIdx % GRID_1X;
            cropNCHW(imageData, imageSize, imageSize, 3,
                     ii * stride1x, jj * stride1x, PATCH_SIZE, PATCH_SIZE,
                     ws_patch);
            auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            const auto& inTensor = in1[0].toTensor();
            const int64_t inputNumel = inTensor.numel();
            const int64_t expectedPart1Numel = (int64_t)1 * 3 * PATCH_SIZE * PATCH_SIZE;
            if (inputNumel != expectedPart1Numel || inTensor.dim() != 4) {
                LOGE("Part1 input shape check failed: numel=%lld (expected %lld) dim=%d (expected 4)",
                     (long long)inputNumel, (long long)expectedPart1Numel, (int)inTensor.dim());
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            if (ii == 0 && jj == 0) {
                const size_t inputNbytes = (size_t)inputNumel * sizeof(float);
                LOGD("BEFORE_PART1_FORWARD 1x (0,0) useCache=1 inputNumel=%lld nbytes=%zu dims=[1,3,%d,%d] OK",
                     (long long)inputNumel, inputNbytes, PATCH_SIZE, PATCH_SIZE);
                LOGD("BEFORE_PART1_FORWARD input[0..2]=%.6g %.6g %.6g", (double)ws_patch[0], (double)ws_patch[1], (double)ws_patch[2]);
                LOGD("Part1 1x (0,0): calling m1->forward (Vulkan GPU — crash here = timeout/device-lost)");
            }
            auto out1 = m1->forward(in1);
            if (ii == 0 && jj == 0) {
                LOGD("AFTER_PART1_FORWARD 1x (0,0) status=%s", out1.ok() ? "ok" : "fail");
                if (!out1.ok()) LOGD("AFTER_PART1_FORWARD forward_error=%d", static_cast<int>(out1.error()));
                else if (!out1->empty()) LOGD("AFTER_PART1_FORWARD output_size=%zu out0_numel=%lld", out1->size(), (long long)(*out1)[0].toTensor().numel());
            }
            if (!out1.ok()) {
                int err = static_cast<int>(out1.error());
                LOGE("Part1 fail 1x (%d,%d): forward error %d %s. CPU path: use Part1/Part2 exported with --backend portable (no Vulkan).", ii, jj, err, executorchErrorStr(err));
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            if (out1->size() < 2) {
                LOGE("Part1 fail 1x (%d,%d): expected >=2 outputs, got %zu", ii, jj, out1->size());
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const auto& tokT = (*out1)[0].toTensor();
            const auto& spaT = (*out1)[1].toTensor();
            const float* tokPtr = tokT.const_data_ptr<float>();
            const float* spaPtr = spaT.const_data_ptr<float>();
            if (!tokPtr || !spaPtr) {
                LOGE("Part1 1x (%d,%d): token/spatial output ptr null (Vulkan?)", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            reshapeToSpatial(spaPtr, spaT.numel(), ws_temp);
            mergeCrop(g_workspace.latent0, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            reshapeToSpatial(tokPtr, tokT.numel(), ws_temp);
            mergeCrop(g_workspace.latent1, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            std::memcpy(cachedTokens.data() + (size_t)patchIdx * tokenSliceSz, tokPtr, tokenSliceSz * sizeof(float));
            if (progressReporter && reportProgressMethodId) {
                const int patchesDone = patchIdx + 1;
                const float p = (patchesDone / (float)totalPart12ProgressUnits) * 0.25f;
                char buf[96];
                snprintf(buf, sizeof(buf), "Part 1 (Vulkan): patch %d/%d…", patchesDone, limit1x);
                reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
            }
        }
        LOGD("1x Part1 pass (%d): %lldms", limit1x, nowMs() - t1x);

        g_moduleCache.mod1.reset();
        g_moduleCache.mod1_b2.reset();
        g_moduleCache.mod1_b4.reset();
        LOGD("Released Vulkan Part1 modules before Part2 pass to reduce encoder overlap");
        ETModule* m2_only = g_moduleCache.mod2.get();
        if (!m2_only) {
            LOGE("Part2 unavailable after releasing Part1 modules");
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }

        long long t2_1x = nowMs();
        for (int patchIdx = 0; patchIdx < limit1x; ++patchIdx) {
            const int ii = patchIdx / GRID_1X;
            const int jj = patchIdx % GRID_1X;
            const float* tokSlice = cachedTokens.data() + (size_t)patchIdx * tokenSliceSz;
            std::memcpy(ws_tokens, tokSlice, tokenSliceSz * sizeof(float));
            auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokInput};
            auto out2 = m2_only->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 fail 1x (%d,%d)", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            if (!featPtr) {
                LOGE("Part2 1x (%d,%d): feat output ptr null (Vulkan?)", ii, jj);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
            mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            if (progressReporter && reportProgressMethodId) {
                const int patchesDone = patchIdx + 1;
                const float p = 0.25f + (patchesDone / (float)totalPart12ProgressUnits) * 0.25f;
                char buf[96];
                snprintf(buf, sizeof(buf), "Part 2 (Vulkan): patch %d/%d…", patchesDone, limit1x);
                reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
            }
        }
        LOGD("1x Part2 pass (%d): %lldms", limit1x, nowMs() - t2_1x);
        LOGD("1x patches (%d): %lldms", limit1x, nowMs() - t1x);
        downsample2x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x1Feat);
        downsample4x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x2Feat);
        LOGD("Skipped 0.5x+0.25x (part12_25_only); filled x1Feat,x2Feat from downsampled x0Feat");
    }

    if (!usedTwoPassVulkan25Only && !usedInterleavedVulkan25Only) {
    // ── 1x patches (5x5 = 25): batch-2 Vulkan FP16 or batch-4 INT8 when available ─
    long long t1x = nowMs();
    int batchSize1x = 1;
    ETModule* m1_batch = nullptr;
    ETModule* m2_batch = nullptr;
    // Vulkan batch-2 triggers write_attribute outside boundary in ExecuTorch Tensor.cpp; use single-patch.
    if (!part12ForceSingle && false && useVulkanForPart12 && m1_b2 && m2_b2) {
        batchSize1x = PATCH_BATCH_2;
        m1_batch = m1_b2;
        m2_batch = m2_b2;
    } else if (!part12ForceSingle && !useVulkanForPart12 && m1_b4 && m2_b4) {
        batchSize1x = PATCH_BATCH;
        m1_batch = m1_b4;
        m2_batch = m2_b4;
    }

    for (int start = 0; start < limit1x; start += batchSize1x) {
        const int n = std::min(batchSize1x, limit1x - start);
        const bool useBatch = (n == batchSize1x && m1_batch && m2_batch);
        bool batchOk = false;

        if (useBatch) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_1X;
                const int jj = (start + b) % GRID_1X;
                cropNCHW(imageData, imageSize, imageSize, 3,
                         ii * stride1x, jj * stride1x, PATCH_SIZE, PATCH_SIZE,
                         g_workspace.patchBuf4 + b * patchSz);
            }
            auto pTensor = from_blob(g_workspace.patchBuf4, {n, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            if (start == 0) LOGD("1x batch-%d: Part1 forward (start=0)", n);
            const size_t needTokenNumel = (size_t)n * tokenSliceSz;
            const size_t needFeatNumel = (size_t)n * featSliceSz;
            bool batchTokensReady = false;
            {
                auto out1 = m1_batch->forward(in1);
                if (out1.ok() && out1->size() >= 2) {
                    const auto& tokT = (*out1)[0].toTensor();
                    const auto& spaT = (*out1)[1].toTensor();
                    const float* tokBase = tokT.const_data_ptr<float>();
                    const float* spaBase = spaT.const_data_ptr<float>();
                    if (tokBase && spaBase && tokT.numel() >= (int64_t)needTokenNumel && spaT.numel() >= (int64_t)needTokenNumel) {
                        for (int b = 0; b < n; ++b) {
                            const int ii = (start + b) / GRID_1X;
                            const int jj = (start + b) % GRID_1X;
                            reshapeToSpatial(spaBase + b * tokenSliceSz, tokenSliceSz, ws_temp);
                            mergeCrop(g_workspace.latent0, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                            reshapeToSpatial(tokBase + b * tokenSliceSz, tokenSliceSz, ws_temp);
                            mergeCrop(g_workspace.latent1, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                        }
                        std::memcpy(g_workspace.tokensCopy4, tokBase, n * tokenSliceSz * sizeof(float));
                        batchTokensReady = true;
                    }
                }
            }
            if (batchTokensReady) {
                auto tokInput = from_blob(g_workspace.tokensCopy4, {n, TOKENS_577, FEATURE_DIM});
                std::vector<EValue> in2 = {*tokInput};
                auto out2 = m2_batch->forward(in2);
                if (out2.ok() && !out2->empty()) {
                    const auto& featT = (*out2)[0].toTensor();
                    const float* featBase = featT.const_data_ptr<float>();
                    if (featBase && featT.numel() >= (int64_t)needFeatNumel) {
                        for (int b = 0; b < n; ++b) {
                            const int ii = (start + b) / GRID_1X;
                            const int jj = (start + b) % GRID_1X;
                            reshapeToSpatial(featBase + b * featSliceSz, featSliceSz, ws_temp);
                            mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                        }
                        batchOk = true;
                    }
                }
            }
            if (batchOk && progressReporter && reportProgressMethodId) {
                const int totalPart1Patches = limit1x + limit05x + 1;
                const int patchesDone = start + n;
                float p = (patchesDone / (float)totalPart1Patches) * 0.5f;
                char buf[64];
                snprintf(buf, sizeof(buf), "Part 1+2: patch %d/%d…", patchesDone, totalPart1Patches);
                reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
            }
            if (!batchOk) {
                LOGD("Part1/Part2 batch-%d 1x start=%d: forward failed or bad shape, using single-patch", n, start);
            }
        }
        if (!batchOk) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_1X;
                const int jj = (start + b) % GRID_1X;
                cropNCHW(imageData, imageSize, imageSize, 3,
                         ii * stride1x, jj * stride1x, PATCH_SIZE, PATCH_SIZE,
                         ws_patch);
                auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
                std::vector<EValue> in1 = {*pTensor};
                // Part1 expects [1, 3, 384, 384]; we crop from 1536x1536, never feed full res.
                const auto& inTensor = in1[0].toTensor();
                const int64_t inputNumel = inTensor.numel();
                const int64_t expectedPart1Numel = (int64_t)1 * 3 * PATCH_SIZE * PATCH_SIZE;
                if (inputNumel != expectedPart1Numel || inTensor.dim() != 4) {
                    LOGE("Part1 input shape check failed: numel=%lld (expected %lld) dim=%d (expected 4)",
                         (long long)inputNumel, (long long)expectedPart1Numel, (int)inTensor.dim());
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const size_t inputNbytes = (size_t)inputNumel * sizeof(float);
                if (ii == 0 && jj == 0) {
                    LOGD("BEFORE_PART1_FORWARD 1x (0,0) useCache=1 inputNumel=%lld nbytes=%zu dims=[1,3,%d,%d] OK",
                         (long long)inputNumel, inputNbytes, PATCH_SIZE, PATCH_SIZE);
                    LOGD("BEFORE_PART1_FORWARD input[0..2]=%.6g %.6g %.6g", (double)ws_patch[0], (double)ws_patch[1], (double)ws_patch[2]);
                    LOGD("Part1 1x (0,0): calling m1->forward (%s)",
                         useVulkanForPart12 ? "Vulkan GPU — crash here = timeout/device-lost" : "CPU/XNNPACK INT8");
                }
                bool part1Prepared = false;
                {
                    auto out1 = m1->forward(in1);
                    if (ii == 0 && jj == 0) {
                        LOGD("AFTER_PART1_FORWARD 1x (0,0) status=%s", out1.ok() ? "ok" : "fail");
                        if (!out1.ok()) LOGD("AFTER_PART1_FORWARD forward_error=%d", static_cast<int>(out1.error()));
                        else if (!out1->empty()) LOGD("AFTER_PART1_FORWARD output_size=%zu out0_numel=%lld", out1->size(), (long long)(*out1)[0].toTensor().numel());
                    }
                    if (!out1.ok()) {
                        int err = static_cast<int>(out1.error());
                        LOGE("Part1 fail 1x (%d,%d): forward error %d %s. CPU path: use Part1/Part2 exported with --backend portable (no Vulkan).", ii, jj, err, executorchErrorStr(err));
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    if (out1->size() < 2) {
                        LOGE("Part1 fail 1x (%d,%d): expected >=2 outputs, got %zu", ii, jj, out1->size());
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    const auto& tokT = (*out1)[0].toTensor();
                    const auto& spaT = (*out1)[1].toTensor();
                    const float* tokPtr = tokT.const_data_ptr<float>();
                    const float* spaPtr = spaT.const_data_ptr<float>();
                    if (!tokPtr || !spaPtr) {
                        LOGE("Part1 1x (%d,%d): token/spatial output ptr null (Vulkan?)", ii, jj);
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    reshapeToSpatial(spaPtr, spaT.numel(), ws_temp);
                    mergeCrop(g_workspace.latent0, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                    reshapeToSpatial(tokPtr, tokT.numel(), ws_temp);
                    mergeCrop(g_workspace.latent1, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                    std::memcpy(ws_tokens, tokPtr, tokenSliceSz * sizeof(float));
                    part1Prepared = true;
                }
                if (!part1Prepared) {
                    LOGE("Part1 1x (%d,%d): did not produce tokens for Part2", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
                std::vector<EValue> in2 = {*tokInput};
                auto out2 = m2->forward(in2);
                if (!out2.ok() || out2->empty()) {
                    LOGE("Part2 fail 1x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
                if (!featPtr) {
                    LOGE("Part2 1x (%d,%d): feat output ptr null (Vulkan?)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
                mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                if (progressReporter && reportProgressMethodId) {
                    const int totalPart1Patches = limit1x + limit05x + 1;
                    const int patchesDone = ii * GRID_1X + jj + 1;
                    float p = (patchesDone / (float)totalPart1Patches) * 0.5f;
                    char buf[64];
                    snprintf(buf, sizeof(buf), "Part 1+2: patch %d/%d…", patchesDone, totalPart1Patches);
                    reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
                }
            }
        }
        const int patchesDone1x = start + n;
        if (chunk1x > 0 && yieldMs > 0 && patchesDone1x < limit1x && patchesDone1x % chunk1x == 0) {
            LOGD("Part1+2 chunk yield: %d ms after %d 1x patches", yieldMs, patchesDone1x);
            usleep(static_cast<unsigned>(yieldMs) * 1000);
        }
    }
    LOGD("1x patches (%d): %lldms", limit1x, nowMs() - t1x);

    // ── 0.5x patches (3x3 = 9): batch-2 Vulkan FP16 or batch-4 INT8 when available ─
    long long t05x = nowMs();
    int batchSize05x = 1;
    if (!part12ForceSingle && false && useVulkanForPart12 && m1_b2 && m2_b2) batchSize05x = PATCH_BATCH_2;  // Vulkan batch-2: write_attribute boundary error
    else if (!part12ForceSingle && !useVulkanForPart12 && m1_b4 && m2_b4) batchSize05x = PATCH_BATCH;
    ETModule* m1_batch_05 = (batchSize05x == 2) ? m1_b2 : ((batchSize05x == 4) ? m1_b4 : nullptr);
    ETModule* m2_batch_05 = (batchSize05x == 2) ? m2_b2 : ((batchSize05x == 4) ? m2_b4 : nullptr);

    for (int start = 0; start < limit05x; start += batchSize05x) {
        const int n = std::min(batchSize05x, limit05x - start);
        const bool useBatch = (n == batchSize05x && m1_batch_05 && m2_batch_05);

        if (useBatch) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_05X;
                const int jj = (start + b) % GRID_05X;
                cropNCHW(g_workspace.halfImg, halfSize, halfSize, 3,
                         ii * stride05x, jj * stride05x, PATCH_SIZE, PATCH_SIZE,
                         g_workspace.patchBuf4 + b * patchSz);
            }
            auto pTensor = from_blob(g_workspace.patchBuf4, {n, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            auto out1 = m1_batch_05->forward(in1);
            if (!out1.ok() || out1->empty()) {
                LOGE("Part1 batch-%d fail 0.5x start=%d", n, start);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* tokBase = (*out1)[0].toTensor().const_data_ptr<float>();
            std::memcpy(g_workspace.tokensCopy4, tokBase, n * tokenSliceSz * sizeof(float));
            auto tokInput = from_blob(g_workspace.tokensCopy4, {n, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokInput};
            auto out2 = m2_batch_05->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 batch-%d fail 0.5x start=%d", n, start);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featBase = (*out2)[0].toTensor().const_data_ptr<float>();
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_05X;
                const int jj = (start + b) % GRID_05X;
                reshapeToSpatial(featBase + b * featSliceSz, featSliceSz, ws_temp);
                mergeCrop(g_workspace.x1Feat, M_05X, ws_temp, jj, ii, GRID_05X, PADDING_05X, ROW_OFFS_05X, COL_OFFS_05X);
            }
            if (progressReporter && reportProgressMethodId) {
                const int totalPart1Patches = limit1x + limit05x + 1;
                const int patchesDone = limit1x + start + n;
                float p = (patchesDone / (float)totalPart1Patches) * 0.5f;
                char buf[64];
                snprintf(buf, sizeof(buf), "Part 1+2: patch %d/%d…", patchesDone, totalPart1Patches);
                reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
            }
        } else {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_05X;
                const int jj = (start + b) % GRID_05X;
                cropNCHW(g_workspace.halfImg, halfSize, halfSize, 3,
                         ii * stride05x, jj * stride05x, PATCH_SIZE, PATCH_SIZE,
                         ws_patch);
                auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
                std::vector<EValue> in1 = {*pTensor};
                bool part1Prepared = false;
                {
                    auto out1 = m1->forward(in1);
                    if (!out1.ok()) {
                        int err = static_cast<int>(out1.error());
                        LOGE("Part1 fail 0.5x (%d,%d): forward error %d %s", ii, jj, err, executorchErrorStr(err));
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    if (out1->size() < 1) {
                        LOGE("Part1 fail 0.5x (%d,%d): expected >=1 output, got %zu", ii, jj, out1->size());
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    const float* tokPtr = (*out1)[0].toTensor().const_data_ptr<float>();
                    if (!tokPtr) {
                        LOGE("Part1 0.5x (%d,%d): token output ptr null (Vulkan?)", ii, jj);
                        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                        return nullptr;
                    }
                    std::memcpy(ws_tokens, tokPtr, tokenSliceSz * sizeof(float));
                    part1Prepared = true;
                }
                if (!part1Prepared) {
                    LOGE("Part1 0.5x (%d,%d): did not produce tokens for Part2", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
                std::vector<EValue> in2 = {*tokInput};
                auto out2 = m2->forward(in2);
                if (!out2.ok() || out2->empty()) {
                    LOGE("Part2 fail 0.5x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
                if (!featPtr) {
                    LOGE("Part2 0.5x (%d,%d): feat output ptr null (Vulkan?)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
                mergeCrop(g_workspace.x1Feat, M_05X, ws_temp, jj, ii, GRID_05X, PADDING_05X, ROW_OFFS_05X, COL_OFFS_05X);
                if (progressReporter && reportProgressMethodId) {
                    const int totalPart1Patches = limit1x + limit05x + 1;
                    const int patchesDone = limit1x + ii * GRID_05X + jj + 1;
                    float p = (patchesDone / (float)totalPart1Patches) * 0.5f;
                    char buf[64];
                    snprintf(buf, sizeof(buf), "Part 1+2: patch %d/%d…", patchesDone, totalPart1Patches);
                    reportProgress(env, progressReporter, reportProgressMethodId, p, buf);
                }
            }
        }
        const int patchesDone05x = start + n;
        if (chunk05x > 0 && yieldMs > 0 && patchesDone05x < limit05x && patchesDone05x % chunk05x == 0) {
            LOGD("Part1+2 chunk yield: %d ms after %d 0.5x patches", yieldMs, patchesDone05x);
            usleep(static_cast<unsigned>(yieldMs) * 1000);
        }
    }
    LOGD("0.5x patches (%d): %lldms", limit05x, nowMs() - t05x);

    // ── 0.25x patch (1) ─────────────────────────────────────────────────────
    if (!skip05xAnd025x) {
    long long t025 = nowMs();
    auto qTensor = from_blob(g_workspace.quarterImg, {1, 3, PATCH_SIZE, PATCH_SIZE});
    std::vector<EValue> inQ = {*qTensor};
    bool qPrepared = false;
    {
        auto outQ = g_moduleCache.mod1->forward(inQ);
        if (!outQ.ok()) {
            int err = static_cast<int>(outQ.error());
            LOGE("Part1 fail 0.25x: forward error %d %s", err, executorchErrorStr(err));
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
        if (outQ->size() < 1) {
            LOGE("Part1 fail 0.25x: expected >=1 output, got %zu", outQ->size());
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
        const float* qTokPtr = (*outQ)[0].toTensor().const_data_ptr<float>();
        if (!qTokPtr) {
            LOGE("Part1 0.25x: token output ptr null (Vulkan?)");
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
        std::memcpy(ws_tokens, qTokPtr, (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float));
        qPrepared = true;
    }
    if (!qPrepared) {
        LOGE("Part1 0.25x: did not produce tokens for Part2");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto qTokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
    std::vector<EValue> inQ2 = {*qTokInput};
    auto outQ2 = g_moduleCache.mod2->forward(inQ2);
    if (!outQ2.ok() || outQ2->empty()) {
        LOGE("Part2 fail 0.25x");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* qFeatPtr = (*outQ2)[0].toTensor().const_data_ptr<float>();
    if (!qFeatPtr) {
        LOGE("Part2 0.25x: feat output ptr null (Vulkan?)");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    if (progressReporter && reportProgressMethodId) {
        const int totalPart1PatchesDone = limit1x + limit05x + 1;
        char p12DoneBuf[96];
        snprintf(p12DoneBuf, sizeof(p12DoneBuf), "Part 1+2: %d/%d. Part 3…", totalPart1PatchesDone, totalPart1PatchesDone);
        reportProgress(env, progressReporter, reportProgressMethodId, 0.5f, p12DoneBuf);
    }
    size_t qFeatLen = (*outQ2)[0].toTensor().numel();
    reshapeToSpatial(qFeatPtr, qFeatLen, g_workspace.x2Feat);
    LOGD("0.25x patch: %lldms. All %d patches: %lldms", nowMs() - t025, limit1x + limit05x + 1, nowMs() - t1x);
    } else {
        downsample2x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x1Feat);
        downsample4x(g_workspace.x0Feat, M_1X, M_1X, FEATURE_DIM, g_workspace.x2Feat);
        LOGD("Skipped 0.5x+0.25x (part12_25_only); filled x1Feat,x2Feat from downsampled x0Feat");
    }
    }

    const long long tPart12WallMs = nowMs() - tPart12WallStart;
    LOGD("[TIMING] Part1+2 wall: %lldms twoPass=%d interleaved=%d",
         tPart12WallMs,
         usedTwoPassVulkan25Only ? 1 : 0,
         usedInterleavedVulkan25Only ? 1 : 0);
    g_workspace.releaseEncoderScratch();
    logProcessMemory("after Part1+2: early encoder scratch release");

    // ── Part3 + Part4a paths (fixed names per export; no .pte delegate scanning) ──
    long long tP3 = nowMs();
    std::string path3;
    if (useVulkanBackend) {
        path3 = vkPathPart3;
        if (path3.empty()) {
            LOGE("Part3 Vulkan .pte not found (expected sharp_split_part3_vulkan_fp32.pte or _fp16.pte)");
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
    } else {
        path3 = pathJoin(modelDir, "sharp_split_part3_int8.pte");
        std::ifstream ft(path3);
        if (!ft.good()) {
            path3 = pathJoin(modelDir, "sharp_split_part3_fp16.pte");
            ft.open(path3);
        }
        if (!ft.good()) {
            path3 = pathJoin(modelDir, "sharp_split_part3.pte");
        }
    }
    const std::string path4a512 = useVulkanBackend ? vkPath4a512 : pathJoin(modelDir, "sharp_split_part4a_chunk_512.pte");
    const std::string path4a65 = useVulkanBackend ? vkPath4a65 : pathJoin(modelDir, "sharp_split_part4a_chunk_65.pte");
    // Vulkan: Part4b is **tiled-only** (batched → full tiled in common.cpp). No sharp_split_part4b_vulkan.pte path.
    // CPU (non-Vulkan) branch below: single portable Part4b .pte.
    std::string path4bSingle;
    if (!useVulkanBackend) {
        // Prefer FP32 Part4b for clean output; INT8 can be foggy.
        const char* cpuPart4bNames[] = {
                "sharp_split_part4b.pte",       // FP32 — clean results
                "sharp_split_part4b_fp16.pte",
                "sharp_split_part4b_int8.pte",
        };
        for (const char* name : cpuPart4bNames) {
            std::string candidate = pathJoin(modelDir, name);
            std::ifstream fc(candidate);
            if (fc.good()) {
                path4bSingle = std::move(candidate);
                break;
            }
        }
    }
    LOGI("SHARP_pte_paths useVulkan=%d dir=%s\n  Part3=%s\n  Part4a512=%s\n  Part4a65=%s\n  Part4b_single=%s",
         useVulkanBackend ? 1 : 0,
         modelDir.c_str(),
         path3.c_str(),
         path4a512.c_str(),
         path4a65.c_str(),
         useVulkanBackend ? "(Vulkan tiled-only)" : (path4bSingle.empty() ? "(none yet / tiled path)" : path4bSingle.c_str()));
    LOGD("[MEM] combinedTokens planned ~%zuMB image=%dx%d", ((size_t)TOKENS_577 * FEATURE_DIM * sizeof(float)) / (1024 * 1024), imageSize, imageSize);

    std::vector<float> combinedTokens;
    size_t imgTokensNumel = 0;

    logProcessMemory("before Part3 load");
    auto mod3 = std::make_unique<ETModule>(path3, ETModule::LoadMode::Mmap);
    if (mod3->load() != Error::Ok) {
        LOGE("Part3 load fail: %s", path3.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    if (progressReporter && reportProgressMethodId) {
        reportProgress(env, progressReporter, reportProgressMethodId, 0.52f, "Part 3: preparing encoder…");
    }
    auto fullImgTensor = from_blob(imageData, {1, 3, imageSize, imageSize});
    std::vector<EValue> in3 = {*fullImgTensor};
    auto out3 = mod3->forward(in3);
    if (!out3.ok() || out3->empty()) {
        if (!out3.ok()) {
            int err = static_cast<int>(out3.error());
            LOGE("Part3 forward fail: error %d %s", err, executorchErrorStr(err));
        } else {
            LOGE("Part3 forward fail: empty outputs");
        }
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* imgTokensPtr = (*out3)[0].toTensor().const_data_ptr<float>();
    imgTokensNumel = (size_t)(*out3)[0].toTensor().numel();
    combinedTokens.resize(imgTokensNumel);
    std::memcpy(combinedTokens.data(), imgTokensPtr, imgTokensNumel * sizeof(float));
    mod3.reset();
    const long long part3Ms = nowMs() - tP3;
    LOGD("Part3: %lldms (sequential, numel=%zu)", part3Ms, imgTokensNumel);
    logProcessMemory("after Part3 reset");
    if (progressReporter && reportProgressMethodId) {
        reportProgress(env, progressReporter, reportProgressMethodId, 0.56f, "Part 3: encoding room…");
    }

    // ── Part4a chunk 512 + 65 ─────────────────────────────────────────────────
    long long tP4a = nowMs();
    if (progressReporter && reportProgressMethodId) {
        const char* msg = useVulkanBackend
            ? "Part 4a (1/2): ViT 512-token chunk…"
            : "Part 4a (1/2): ViT 512 on CPU…";
        reportProgress(env, progressReporter, reportProgressMethodId, 0.62f, msg);
    }
    if (!useVulkanBackend) {
        LOGI("Part4a/512: portable CPU/XNNPACK — forward may take longer than Part3; timing follows in log.");
        LOGW("Part4a/512: CPU INT8 ViT is often many minutes on phone SoCs. For ~2–3 min total, use etVulkan build + "
             "Vulkan Part3/4 .pte in models_cpuvulkan_hybrid (logcat showed 400s+ on one device before OMP tuning).");
    }
    logProcessMemory("before Part4a/512 load");
    long long tLoad512 = nowMs();
    auto mod4a512 = std::make_unique<ETModule>(path4a512, ETModule::LoadMode::Mmap);
    if (mod4a512->load() != Error::Ok) {
        LOGE("Part4a/512 load fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGI("Part4a/512: mmap+load OK in %lldms; forward() file=%s", nowMs() - tLoad512, path4a512.c_str());
    auto c512Tensor = from_blob(combinedTokens.data(), {1, 512, FEATURE_DIM});
    std::vector<EValue> in4a512 = {*c512Tensor};
    long long tFwd512 = nowMs();
    auto out4a512 = [&]() {
        if (!useVulkanBackend) {
            std::atomic<bool> hbKeep{true};
            std::thread hb(cpuForwardHeartbeatLoop, &hbKeep, "Part4a/512", 10);
            auto r = mod4a512->forward(in4a512);
            hbKeep.store(false, std::memory_order_release);
            if (hb.joinable()) {
                hb.join();
            }
            return r;
        }
        return mod4a512->forward(in4a512);
    }();
    LOGI("Part4a/512: forward finished in %lldms", nowMs() - tFwd512);
    if (!out4a512.ok() || out4a512->empty()) {
        if (!out4a512.ok()) {
            int err = static_cast<int>(out4a512.error());
            LOGE("Part4a/512 forward fail: error %d %s", err, executorchErrorStr(err));
        } else {
            LOGE("Part4a/512 forward fail: empty outputs");
        }
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out512Ptr = (*out4a512)[0].toTensor().const_data_ptr<float>();
    size_t out512Len = (*out4a512)[0].toTensor().numel();
    std::memcpy(combinedTokens.data(), out512Ptr, out512Len * sizeof(float));
    mod4a512.reset();
    LOGD("Part4a/512: %lldms (out=%zu)", nowMs() - tP4a, out512Len);
    logProcessMemory("after Part4a/512 reset");
    if (progressReporter && reportProgressMethodId) {
        reportProgress(env, progressReporter, reportProgressMethodId, 0.645f, "Part 4a (1/2): ViT 512 done…");
    }

    // ── Part4a chunk 65 ──────────────────────────────────────────────────────
    long long tP4a65 = nowMs();
    if (progressReporter && reportProgressMethodId) {
        reportProgress(env, progressReporter, reportProgressMethodId, 0.67f, "Part 4a (2/2): ViT 65-token chunk…");
    }
    if (!useVulkanBackend) {
        LOGI("Part4a/65: portable CPU/XNNPACK…");
    }
    logProcessMemory("before Part4a/65 load");
    long long tLoad65 = nowMs();
    auto mod4a65 = std::make_unique<ETModule>(path4a65, ETModule::LoadMode::Mmap);
    if (mod4a65->load() != Error::Ok) {
        LOGE("Part4a/65 load fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGI("Part4a/65: load OK in %lldms; forward() file=%s", nowMs() - tLoad65, path4a65.c_str());
    auto c65Tensor = from_blob(combinedTokens.data() + 512 * FEATURE_DIM, {1, 65, FEATURE_DIM});
    std::vector<EValue> in4a65 = {*c65Tensor};
    long long tFwd65 = nowMs();
    auto out4a65 = [&]() {
        if (!useVulkanBackend) {
            std::atomic<bool> hbKeep{true};
            std::thread hb(cpuForwardHeartbeatLoop, &hbKeep, "Part4a/65", 10);
            auto r = mod4a65->forward(in4a65);
            hbKeep.store(false, std::memory_order_release);
            if (hb.joinable()) {
                hb.join();
            }
            return r;
        }
        return mod4a65->forward(in4a65);
    }();
    LOGI("Part4a/65: forward finished in %lldms", nowMs() - tFwd65);
    if (!out4a65.ok() || out4a65->empty()) {
        if (!out4a65.ok()) {
            int err = static_cast<int>(out4a65.error());
            LOGE("Part4a/65 forward fail: error %d %s", err, executorchErrorStr(err));
        } else {
            LOGE("Part4a/65 forward fail: empty outputs");
        }
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out65Ptr = (*out4a65)[0].toTensor().const_data_ptr<float>();
    size_t out65Len = (*out4a65)[0].toTensor().numel();
    std::memcpy(combinedTokens.data() + 512 * FEATURE_DIM, out65Ptr, out65Len * sizeof(float));
    mod4a65.reset();
    LOGD("Part4a/65: %lldms. Part4a total: %lldms", nowMs() - tP4a65, nowMs() - tP4a);
    logProcessMemory("after Part4a/65 reset");
    const long long part4aMs = nowMs() - tP4a;

    LOGI(
            "[HYBRID_TIMING] interleaveP12=%d earlyEncoderScratch=1 "
            "part12_ms=%lld part3_ms=%lld part4a_ms=%lld total_ms=%lld availKb=%ld",
            (wantInterleavePart12 && interleaveMemOk) ? 1 : 0,
            (long long)tPart12WallMs,
            (long long)part3Ms,
            (long long)part4aMs,
            (long long)(nowMs() - t0),
            (long)memAvailKbGate);

    // ── Part4b: Vulkan — batched tiled → sequential tiled only (no single-decoder .pte)
    g_moduleCache.release();
    LOGD("Released Part1+Part2 cache before Part4b to free memory");
    logProcessMemory("before Part4b routing");

    std::vector<float> tiledGaussians;
    if (runPart4bBatchedTiledPipeline(modelDir,
                                      imageData,
                                      g_workspace.latent0,
                                      g_workspace.latent1,
                                      g_workspace.x0Feat,
                                      g_workspace.x1Feat,
                                      g_workspace.x2Feat,
                                      combinedTokens.data(),
                                      imgTokensNumel,
                                      swapTileNdc,
                                      tiledGaussians,
                                      env,
                                      progressReporter,
                                      reportProgressMethodId)) {
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        pruneGaussiansByOpacity(tiledGaussians, maxG);
        const int numFloats = static_cast<int>(tiledGaussians.size());
        if (numFloats <= 0 || (numFloats % PARAMS_PER_GAUSSIAN) != 0) {
            LOGE("Batched tiled Part4b output invalid: %d", numFloats);
            return nullptr;
        }
        jfloatArray jResult = env->NewFloatArray(numFloats);
        if (!jResult) { LOGE("result array alloc fail (batched tiled)"); return nullptr; }
        env->SetFloatArrayRegion(jResult, 0, numFloats, tiledGaussians.data());
        LOGD("JNI RETURN: size=%d validated (batched tiled)", numFloats);
        return jResult;
    }

    tiledGaussians.clear();
    if (runPart4bTiledFullPipeline(modelDir,
                                   imageData,
                                   g_workspace.latent0,
                                   g_workspace.latent1,
                                   g_workspace.x0Feat,
                                   g_workspace.x1Feat,
                                   g_workspace.x2Feat,
                                   combinedTokens.data(),
                                   imgTokensNumel,
                                   swapTileNdc,
                                   tiledGaussians,
                                   env,
                                   progressReporter,
                                   reportProgressMethodId)) {
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        pruneGaussiansByOpacity(tiledGaussians, maxG);
        const int numFloats = static_cast<int>(tiledGaussians.size());
        if (numFloats <= 0 || (numFloats % PARAMS_PER_GAUSSIAN) != 0) {
            LOGE("Tiled Part4b output invalid: %d", numFloats);
            return nullptr;
        }
        jfloatArray jResult = env->NewFloatArray(numFloats);
        if (!jResult) { LOGE("result array alloc fail (tiled)"); return nullptr; }
        env->SetFloatArrayRegion(jResult, 0, numFloats, tiledGaussians.data());
        LOGD("JNI RETURN: size=%d validated (tiled)", numFloats);
        return jResult;
    }

    if (useVulkanBackend) {
        LOGE("Part4b Vulkan: batched tiled and full tiled pipelines failed; tiled Part4b .pte set is missing or incomplete "
             "(see sharp_executorch_full_common.cpp: tile_b2/tile_b4, split tile_b2/tile_00, fine-split tile_00, …).");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    long long tP4b = nowMs();
    std::string path4b = path4bSingle;
    if (path4b.empty()) {
        LOGE("Part4b single: no sharp_split_part4b_int8/fp16/.pte in %s", modelDir.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    if (progressReporter && reportProgressMethodId) {
        reportProgress(env, progressReporter, reportProgressMethodId, 0.75f, "Part 4b: Gaussian decoder on CPU…");
    }
    LOGI("Part4b single load: %s", path4b.c_str());
    logProcessMemory("Part4b single fallback: before module load");
    long long tLoad4b = nowMs();
#if defined(EXECUTORCH_HAS_ETDUMP)
    std::unique_ptr<ETModule> mod4b = (!etdumpPath.empty())
        ? std::make_unique<ETModule>(path4b, ETModule::LoadMode::Mmap, std::make_unique<executorch::etdump::ETDumpGen>())
        : std::make_unique<ETModule>(path4b, ETModule::LoadMode::Mmap);
#else
    auto mod4b = std::make_unique<ETModule>(path4b, ETModule::LoadMode::Mmap);
#endif
    if (mod4b->load() != Error::Ok) {
        LOGE("Part4b load fail: %s", path4b.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGI("Part4b: load OK in %lldms; forward(7 inputs) starting…", nowMs() - tLoad4b);
    logProcessMemory("Part4b single fallback: after module load");

    auto tokens4b = from_blob(combinedTokens.data(), {1, TOKENS_577, FEATURE_DIM});
    auto img4b    = from_blob(imageData, {1, 3, imageSize, imageSize});
    auto lat0_4b  = from_blob(g_workspace.latent0, {1, FEATURE_DIM, M_1X, M_1X});
    auto lat1_4b  = from_blob(g_workspace.latent1, {1, FEATURE_DIM, M_1X, M_1X});
    auto x0_4b    = from_blob(g_workspace.x0Feat, {1, FEATURE_DIM, M_1X, M_1X});
    auto x1_4b    = from_blob(g_workspace.x1Feat, {1, FEATURE_DIM, M_05X, M_05X});
    auto x2_4b    = from_blob(g_workspace.x2Feat, {1, FEATURE_DIM, SPATIAL_SIZE, SPATIAL_SIZE});

    std::vector<EValue> in4b;
    in4b.reserve(7);
    in4b.push_back(*tokens4b);
    in4b.push_back(*img4b);
    in4b.push_back(*lat0_4b);
    in4b.push_back(*lat1_4b);
    in4b.push_back(*x0_4b);
    in4b.push_back(*x1_4b);
    in4b.push_back(*x2_4b);

    long long tFwd4b = nowMs();
    LOGI("Part4b forward running (single decoder)…");
    auto out4b = mod4b->forward(in4b);
    LOGI("Part4b forward finished in %lldms", nowMs() - tFwd4b);
    logProcessMemory("Part4b single fallback: after forward");
#if defined(EXECUTORCH_HAS_ETDUMP)
    if (!etdumpPath.empty()) {
        if (auto* etdump = dynamic_cast<executorch::etdump::ETDumpGen*>(mod4b->event_tracer())) {
            executorch::etdump::ETDumpResult result = etdump->get_etdump_data();
            if (result.buf != nullptr && result.size > 0) {
                std::ofstream f(etdumpPath, std::ios::binary);
                if (f) {
                    f.write(static_cast<const char*>(result.buf), static_cast<std::streamsize>(result.size));
                    LOGI("ETDump written to %s (%zu bytes)", etdumpPath.c_str(), result.size);
                }
                free(result.buf);
            }
        }
    }
#endif

    if (!out4b.ok() || out4b->empty()) {
        if (!out4b.ok()) {
            int err = static_cast<int>(out4b.error());
            LOGE("Part4b forward fail: error %d %s (path=%s)", err, executorchErrorStr(err), path4b.c_str());
        } else {
            LOGE("Part4b forward fail: empty outputs (path=%s)", path4b.c_str());
        }
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    const auto& gaussianTensor = (*out4b)[0].toTensor();
    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);

    if (gaussianTensor.dim() != 3) {
        LOGE("Part4b AOT shape: dim=%ld expected 3 [1,N,14]", (long)gaussianTensor.dim());
        return nullptr;
    }
    const int64_t rawNumel = gaussianTensor.numel();
    if (rawNumel <= 0 || (rawNumel % PARAMS_PER_GAUSSIAN) != 0) {
        LOGE("Part4b output invalid: numel=%lld", (long long)rawNumel);
        return nullptr;
    }
    const int64_t nG = rawNumel / PARAMS_PER_GAUSSIAN;
    if (nG > 2000000) {
        LOGE("Part4b AOT shape: N=%lld unreasonably large", (long long)nG);
        return nullptr;
    }
    const int validatedNumFloats = static_cast<int>(nG) * PARAMS_PER_GAUSSIAN;

    const float* gaussianPtr = gaussianTensor.const_data_ptr<float>();
    if (!gaussianPtr) {
        LOGE("Part4b output tensor data null (Vulkan backend may not expose CPU-visible buffer on this device)");
        return nullptr;
    }
    std::vector<float> safeCopy;
    try {
        safeCopy.resize(static_cast<size_t>(validatedNumFloats));
    } catch (const std::bad_alloc&) {
        LOGE("Part4b safeCopy alloc fail: %d floats", validatedNumFloats);
        return nullptr;
    }
    std::memcpy(safeCopy.data(), gaussianPtr, (size_t)validatedNumFloats * sizeof(float));

    std::vector<float> prunedBuf;
    const int finalGaussians = pruneGaussiansFromPtr(
        safeCopy.data(), static_cast<int>(nG), maxG, prunedBuf);
    const int numFloats = finalGaussians * PARAMS_PER_GAUSSIAN;

    long long tTotal = nowMs() - t0;
    LOGD("Part4b (single): %lldms. TOTAL pipeline: %lldms. Gaussians=%d",
         nowMs() - tP4b, tTotal, numFloats / PARAMS_PER_GAUSSIAN);

    jfloatArray jResult = env->NewFloatArray(numFloats);
    if (!jResult) { LOGE("result array alloc fail"); return nullptr; }
    env->SetFloatArrayRegion(jResult, 0, numFloats, prunedBuf.data());
    LOGD("JNI RETURN: size=%d validated", numFloats);
    return jResult;
}
