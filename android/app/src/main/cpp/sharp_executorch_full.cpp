/**
 * ExecuTorch C++ full INT8 pipeline: Part1, Part2, Part3, Part4a, Part4b (single).
 *
 * Optimized for ARM64/Android:
 *  - NEON SIMD for downsample2x/4x
 *  - __restrict + pointer arithmetic in hot loops (mergeCrop, reshapeToSpatial)
 *  - memcpy row-blit in mergeCrop instead of element-by-element
 *  - 32-byte aligned pre-allocated workspace (cache-line friendly)
 *  - Singleton module cache: Part1/Part2 stay loaded across calls (eliminates ~400ms mmap I/O)
 *  - Part3/Part4a/Part4b loaded on demand and released to keep RSS bounded
 */

#include <jni.h>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <chrono>
#include <android/log.h>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define HAS_NEON 1
#else
#define HAS_NEON 0
#endif

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

#define TAG "ExecTorchFull"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using ETModule = executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr int IMAGE_SIZE = 1536;
static constexpr int PATCH_SIZE = 384;
static constexpr int FEATURE_DIM = 1024;
static constexpr int SPATIAL_SIZE = 24;
static constexpr int M_1X = 96;
static constexpr int M_05X = 48;
static constexpr int GRID_1X = 5;
static constexpr int GRID_05X = 3;
static constexpr int PADDING_1X = 3;
static constexpr int PADDING_05X = 6;
static constexpr int PARAMS_PER_GAUSSIAN = 14;
static constexpr int TOKENS_577 = 577;
static constexpr int HALF_SIZE = IMAGE_SIZE / 2;   // 768
static constexpr int STRIDE_1X = (IMAGE_SIZE - PATCH_SIZE) / 4;  // 288
static constexpr int STRIDE_05X = (HALF_SIZE - PATCH_SIZE) / 2;  // 192
static constexpr int SPATIAL_HW = SPATIAL_SIZE * SPATIAL_SIZE;    // 576

// ── Aligned allocation helper ────────────────────────────────────────────────
static constexpr size_t ALIGN = 32;

static float* alignedAlloc(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, ALIGN, count * sizeof(float)) != 0) return nullptr;
    return static_cast<float*>(ptr);
}

static void alignedFree(float* p) { free(p); }

// ── Pre-allocated workspace (single allocation, reused every call) ───────────
struct Workspace {
    float* latent0   = nullptr;  // FEATURE_DIM * M_1X * M_1X
    float* latent1   = nullptr;
    float* x0Feat    = nullptr;
    float* x1Feat    = nullptr;  // FEATURE_DIM * M_05X * M_05X
    float* x2Feat    = nullptr;  // FEATURE_DIM * SPATIAL_HW
    float* tempSpatial = nullptr;
    float* halfImg   = nullptr;  // 3 * HALF_SIZE * HALF_SIZE
    float* quarterImg = nullptr; // 3 * PATCH_SIZE * PATCH_SIZE
    float* patchBuf  = nullptr;  // 3 * PATCH_SIZE * PATCH_SIZE
    float* tokensCopy = nullptr; // TOKENS_577 * FEATURE_DIM
    bool allocated = false;

    bool allocate() {
        if (allocated) return true;
        const size_t feat96 = FEATURE_DIM * M_1X * M_1X;
        const size_t feat48 = FEATURE_DIM * M_05X * M_05X;
        const size_t feat24 = FEATURE_DIM * SPATIAL_HW;
        const size_t patchSz = 3 * PATCH_SIZE * PATCH_SIZE;
        const size_t halfSz = 3 * HALF_SIZE * HALF_SIZE;
        const size_t tokensSz = TOKENS_577 * FEATURE_DIM;

        latent0    = alignedAlloc(feat96);
        latent1    = alignedAlloc(feat96);
        x0Feat     = alignedAlloc(feat96);
        x1Feat     = alignedAlloc(feat48);
        x2Feat     = alignedAlloc(feat24);
        tempSpatial = alignedAlloc(feat24);
        halfImg    = alignedAlloc(halfSz);
        quarterImg = alignedAlloc(patchSz);
        patchBuf   = alignedAlloc(patchSz);
        tokensCopy = alignedAlloc(tokensSz);

        allocated = latent0 && latent1 && x0Feat && x1Feat && x2Feat &&
                    tempSpatial && halfImg && quarterImg && patchBuf && tokensCopy;
        if (!allocated) release();
        return allocated;
    }

    void zero() {
        if (!allocated) return;
        const size_t feat96 = FEATURE_DIM * M_1X * M_1X;
        const size_t feat48 = FEATURE_DIM * M_05X * M_05X;
        const size_t feat24 = FEATURE_DIM * SPATIAL_HW;
        std::memset(latent0, 0, feat96 * sizeof(float));
        std::memset(latent1, 0, feat96 * sizeof(float));
        std::memset(x0Feat,  0, feat96 * sizeof(float));
        std::memset(x1Feat,  0, feat48 * sizeof(float));
        std::memset(x2Feat,  0, feat24 * sizeof(float));
    }

    void release() {
        alignedFree(latent0);   alignedFree(latent1);   alignedFree(x0Feat);
        alignedFree(x1Feat);    alignedFree(x2Feat);    alignedFree(tempSpatial);
        alignedFree(halfImg);   alignedFree(quarterImg); alignedFree(patchBuf);
        alignedFree(tokensCopy);
        latent0 = latent1 = x0Feat = x1Feat = x2Feat = tempSpatial = nullptr;
        halfImg = quarterImg = patchBuf = tokensCopy = nullptr;
        allocated = false;
    }
};

// ── Singleton module cache ───────────────────────────────────────────────────
struct ModuleCache {
    std::unique_ptr<ETModule> mod1, mod2;
    std::string modelDir;
    std::mutex mu;

    bool ensureLoaded(const std::string& dir) {
        std::lock_guard<std::mutex> lock(mu);
        if (mod1 && mod2 && modelDir == dir) return true;
        mod1.reset(); mod2.reset();
        modelDir = dir;
        std::string p1 = dir + "/sharp_split_part1_int8.pte";
        std::string p2 = dir + "/sharp_split_part2_int8.pte";
        mod1 = std::make_unique<ETModule>(p1, ETModule::LoadMode::Mmap);
        if (mod1->load() != Error::Ok) { LOGE("Cache: Part1 load fail"); mod1.reset(); return false; }
        mod2 = std::make_unique<ETModule>(p2, ETModule::LoadMode::Mmap);
        if (mod2->load() != Error::Ok) { LOGE("Cache: Part2 load fail"); mod2.reset(); mod1.reset(); return false; }
        LOGD("ModuleCache: Part1+Part2 loaded (kept alive across calls)");
        return true;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mu);
        mod1.reset(); mod2.reset(); modelDir.clear();
        LOGD("ModuleCache: released Part1+Part2");
    }
};

static ModuleCache g_moduleCache;
static Workspace g_workspace;
static bool g_runtime_initialized = false;

static void ensureRuntimeInit() {
    if (!g_runtime_initialized) {
        executorch::runtime::runtime_init();
        g_runtime_initialized = true;
    }
}

// ── Precomputed offsets (compile-time known) ─────────────────────────────────
static constexpr int ROW_OFFS_1X[5] = {0, 21, 39, 57, 75};
static constexpr int COL_OFFS_1X[5] = {0, 21, 39, 57, 75};
static constexpr int ROW_OFFS_05X[3] = {0, 18, 30};
static constexpr int COL_OFFS_05X[3] = {0, 18, 30};

// ── NEON-optimized downsample2x ──────────────────────────────────────────────
static void downsample2x(const float* __restrict src, int H, int W, int C,
                         float* __restrict dst) {
    const int outH = H / 2;
    const int outW = W / 2;
    const int srcHW = H * W;
    const int dstHW = outH * outW;

    for (int c = 0; c < C; ++c) {
        const float* sp = src + c * srcHW;
        float* dp = dst + c * dstHW;
        for (int y = 0; y < outH; ++y) {
            const float* row0 = sp + (2 * y) * W;
            const float* row1 = row0 + W;
            float* out = dp + y * outW;
            int x = 0;
#if HAS_NEON
            const float32x4_t quarter = vdupq_n_f32(0.25f);
            for (; x + 3 < outW; x += 4) {
                float32x4x2_t r0 = vld2q_f32(row0 + 2 * x);
                float32x4x2_t r1 = vld2q_f32(row1 + 2 * x);
                float32x4_t sum = vaddq_f32(vaddq_f32(r0.val[0], r0.val[1]),
                                            vaddq_f32(r1.val[0], r1.val[1]));
                vst1q_f32(out + x, vmulq_f32(sum, quarter));
            }
#endif
            for (; x < outW; ++x) {
                out[x] = (row0[2*x] + row0[2*x+1] + row1[2*x] + row1[2*x+1]) * 0.25f;
            }
        }
    }
}

// ── NEON-optimized downsample4x ──────────────────────────────────────────────
static void downsample4x(const float* __restrict src, int H, int W, int C,
                         float* __restrict dst) {
    const int outH = H / 4;
    const int outW = W / 4;
    const int srcHW = H * W;
    const int dstHW = outH * outW;
    const float inv16 = 1.0f / 16.0f;

    for (int c = 0; c < C; ++c) {
        const float* sp = src + c * srcHW;
        float* dp = dst + c * dstHW;
        for (int y = 0; y < outH; ++y) {
            float* out = dp + y * outW;
            for (int x = 0; x < outW; ++x) {
                const float* base = sp + (4 * y) * W + 4 * x;
                float sum = 0;
#if HAS_NEON
                float32x4_t acc = vdupq_n_f32(0.0f);
                acc = vaddq_f32(acc, vld1q_f32(base));
                acc = vaddq_f32(acc, vld1q_f32(base + W));
                acc = vaddq_f32(acc, vld1q_f32(base + 2 * W));
                acc = vaddq_f32(acc, vld1q_f32(base + 3 * W));
                sum = vaddvq_f32(acc);
#else
                for (int dy = 0; dy < 4; ++dy)
                    for (int dx = 0; dx < 4; ++dx)
                        sum += base[dy * W + dx];
#endif
                out[x] = sum * inv16;
            }
        }
    }
}

// ── Optimized cropNCHW: __restrict + row-blit ────────────────────────────────
static void cropNCHW(const float* __restrict src, int srcH, int srcW, int C,
                     int startY, int startX, int cropH, int cropW,
                     float* __restrict dst) {
    const int srcHW = srcH * srcW;
    const int dstHW = cropH * cropW;
    const size_t rowBytes = cropW * sizeof(float);
    for (int c = 0; c < C; ++c) {
        const float* srcPlane = src + c * srcHW + startY * srcW + startX;
        float* dstPlane = dst + c * dstHW;
        for (int y = 0; y < cropH; ++y) {
            std::memcpy(dstPlane + y * cropW, srcPlane + y * srcW, rowBytes);
        }
    }
}

// ── Optimized reshapeToSpatial: pointer arithmetic, no repeated multiply ─────
static void reshapeToSpatial(const float* __restrict tokens, size_t tokenLen,
                             float* __restrict out) {
    const bool hasCls = (tokenLen >= (size_t)((SPATIAL_HW + 1) * FEATURE_DIM));
    const float* tokenBase = tokens + (hasCls ? FEATURE_DIM : 0);
    for (int pos = 0; pos < SPATIAL_HW; ++pos) {
        const float* src = tokenBase + pos * FEATURE_DIM;
        for (int c = 0; c < FEATURE_DIM; ++c) {
            out[c * SPATIAL_HW + pos] = src[c];
        }
    }
}

// ── Optimized mergeCrop: memcpy row-blit instead of element-by-element ───────
static void mergeCrop(float* __restrict out, int outW,
                      const float* __restrict patch,
                      int gI, int gJ, int gS, int pad,
                      const int* rowOff, const int* colOff) {
    const int cT = (gJ != 0) ? pad : 0;
    const int cB = (gJ != gS - 1) ? pad : 0;
    const int cL = (gI != 0) ? pad : 0;
    const int cR = (gI != gS - 1) ? pad : 0;
    const int cH = SPATIAL_SIZE - cT - cB;
    const int cW = SPATIAL_SIZE - cL - cR;
    const int outHW = outW * outW;
    const int rowStart = rowOff[gJ];
    const int colStart = colOff[gI];
    const size_t rowBytes = cW * sizeof(float);

    for (int c = 0; c < FEATURE_DIM; ++c) {
        const float* patchRow = patch + c * SPATIAL_HW + cT * SPATIAL_SIZE + cL;
        float* outRow = out + c * outHW + rowStart * outW + colStart;
        for (int dy = 0; dy < cH; ++dy) {
            std::memcpy(outRow + dy * outW, patchRow + dy * SPATIAL_SIZE, rowBytes);
        }
    }
}

static std::string pathJoin(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

// ── Timing helper ────────────────────────────────────────────────────────────
static long long nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ── JNI: preload Part1+Part2 (call from Kotlin on init for warm start) ───────
extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_preloadCppModules(
        JNIEnv* env, jobject, jstring modelDirPath) {
    ensureRuntimeInit();
    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string dir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (dir.empty()) return JNI_FALSE;
    return g_moduleCache.ensureLoaded(dir) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_releaseCppModules(
        JNIEnv*, jobject) {
    g_moduleCache.release();
    g_workspace.release();
    LOGD("Released module cache + workspace");
}

// ── JNI: full pipeline ───────────────────────────────────────────────────────
JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_runFullPipelineInt8Native(
        JNIEnv* env,
        jobject,
        jstring modelDirPath,
        jfloatArray imageNCHW) {

    if (!modelDirPath || !imageNCHW) {
        LOGE("runFullPipelineInt8: null modelDir or image");
        return nullptr;
    }

    ensureRuntimeInit();
    const long long t0 = nowMs();

    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string modelDir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (modelDir.empty()) { LOGE("empty model dir"); return nullptr; }

    LOGD("runFullPipelineInt8: modelDir=%s", modelDir.c_str());

    jsize imageLen = env->GetArrayLength(imageNCHW);
    if (imageLen != 3 * IMAGE_SIZE * IMAGE_SIZE) {
        LOGE("bad image length %d", (int)imageLen);
        return nullptr;
    }

    // Allocate workspace (reused across calls; only first call mallocs)
    if (!g_workspace.allocate()) {
        LOGE("workspace alloc failed");
        return nullptr;
    }
    g_workspace.zero();

    jfloat* imageData = env->GetFloatArrayElements(imageNCHW, nullptr);
    if (!imageData) return nullptr;

    // ── Downsample for 0.5x and 0.25x patches ───────────────────────────────
    long long tDown = nowMs();
    downsample2x(imageData, IMAGE_SIZE, IMAGE_SIZE, 3, g_workspace.halfImg);
    downsample4x(imageData, IMAGE_SIZE, IMAGE_SIZE, 3, g_workspace.quarterImg);
    LOGD("Downsample 2x+4x: %lldms", nowMs() - tDown);

    // ── Part1 + Part2 (singleton cache) ──────────────────────────────────────
    std::string path1check = pathJoin(modelDir, "sharp_split_part1_int8.pte");
    {
        std::ifstream f1(path1check);
        if (!f1.good()) {
            LOGE("Part1 not found: %s", path1check.c_str());
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
    }

    long long tLoad12 = nowMs();
    if (!g_moduleCache.ensureLoaded(modelDir)) {
        LOGE("Failed to load Part1+Part2");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGD("Part1+Part2 ready (cache): %lldms", nowMs() - tLoad12);

    float* ws_patch = g_workspace.patchBuf;
    float* ws_tokens = g_workspace.tokensCopy;
    float* ws_temp = g_workspace.tempSpatial;

    // ── 1x patches (5x5 = 25) ───────────────────────────────────────────────
    long long t1x = nowMs();
    for (int i = 0; i < GRID_1X; ++i) {
        for (int j = 0; j < GRID_1X; ++j) {
            cropNCHW(imageData, IMAGE_SIZE, IMAGE_SIZE, 3,
                     i * STRIDE_1X, j * STRIDE_1X, PATCH_SIZE, PATCH_SIZE,
                     ws_patch);
            auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            auto out1 = g_moduleCache.mod1->forward(in1);
            if (!out1.ok() || out1->size() < 2) {
                LOGE("Part1 fail 1x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const auto& tokT = (*out1)[0].toTensor();
            const auto& spaT = (*out1)[1].toTensor();
            const float* tokPtr = tokT.const_data_ptr<float>();
            const float* spaPtr = spaT.const_data_ptr<float>();
            size_t tokLen = tokT.numel();
            size_t spaLen = spaT.numel();

            reshapeToSpatial(spaPtr, spaLen, ws_temp);
            mergeCrop(g_workspace.latent0, M_1X, ws_temp, j, i, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);

            reshapeToSpatial(tokPtr, tokLen, ws_temp);
            mergeCrop(g_workspace.latent1, M_1X, ws_temp, j, i, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);

            std::memcpy(ws_tokens, tokPtr, tokLen * sizeof(float));
            auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokInput};
            auto out2 = g_moduleCache.mod2->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 fail 1x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            size_t featLen = (*out2)[0].toTensor().numel();
            reshapeToSpatial(featPtr, featLen, ws_temp);
            mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, j, i, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
        }
    }
    LOGD("1x patches (25): %lldms", nowMs() - t1x);

    // ── 0.5x patches (3x3 = 9) ──────────────────────────────────────────────
    long long t05x = nowMs();
    for (int i = 0; i < GRID_05X; ++i) {
        for (int j = 0; j < GRID_05X; ++j) {
            cropNCHW(g_workspace.halfImg, HALF_SIZE, HALF_SIZE, 3,
                     i * STRIDE_05X, j * STRIDE_05X, PATCH_SIZE, PATCH_SIZE,
                     ws_patch);
            auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            auto out1 = g_moduleCache.mod1->forward(in1);
            if (!out1.ok() || out1->size() < 1) {
                LOGE("Part1 fail 0.5x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* tokPtr = (*out1)[0].toTensor().const_data_ptr<float>();
            std::memcpy(ws_tokens, tokPtr, (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float));
            auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokInput};
            auto out2 = g_moduleCache.mod2->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 fail 0.5x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            size_t featLen = (*out2)[0].toTensor().numel();
            reshapeToSpatial(featPtr, featLen, ws_temp);
            mergeCrop(g_workspace.x1Feat, M_05X, ws_temp, j, i, GRID_05X, PADDING_05X, ROW_OFFS_05X, COL_OFFS_05X);
        }
    }
    LOGD("0.5x patches (9): %lldms", nowMs() - t05x);

    // ── 0.25x patch (1) ─────────────────────────────────────────────────────
    long long t025 = nowMs();
    auto qTensor = from_blob(g_workspace.quarterImg, {1, 3, PATCH_SIZE, PATCH_SIZE});
    std::vector<EValue> inQ = {*qTensor};
    auto outQ = g_moduleCache.mod1->forward(inQ);
    if (!outQ.ok() || outQ->size() < 1) {
        LOGE("Part1 fail 0.25x");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* qTokPtr = (*outQ)[0].toTensor().const_data_ptr<float>();
    std::memcpy(ws_tokens, qTokPtr, (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float));
    auto qTokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
    std::vector<EValue> inQ2 = {*qTokInput};
    auto outQ2 = g_moduleCache.mod2->forward(inQ2);
    if (!outQ2.ok() || outQ2->empty()) {
        LOGE("Part2 fail 0.25x");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* qFeatPtr = (*outQ2)[0].toTensor().const_data_ptr<float>();
    size_t qFeatLen = (*outQ2)[0].toTensor().numel();
    reshapeToSpatial(qFeatPtr, qFeatLen, g_workspace.x2Feat);
    LOGD("0.25x patch: %lldms. All 35 patches: %lldms", nowMs() - t025, nowMs() - t1x);

    // ── Part3 (load, run, copy, release) ─────────────────────────────────────
    long long tP3 = nowMs();
    std::string path3 = pathJoin(modelDir, "sharp_split_part3_int8.pte");
    auto mod3 = std::make_unique<ETModule>(path3, ETModule::LoadMode::Mmap);
    if (mod3->load() != Error::Ok) {
        LOGE("Part3 load fail: %s", path3.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto fullImgTensor = from_blob(imageData, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
    std::vector<EValue> in3 = {*fullImgTensor};
    auto out3 = mod3->forward(in3);
    if (!out3.ok() || out3->empty()) {
        LOGE("Part3 forward fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* imgTokensPtr = (*out3)[0].toTensor().const_data_ptr<float>();
    size_t imgTokensNumel = (*out3)[0].toTensor().numel();

    std::vector<float> combinedTokens(imgTokensNumel);
    std::memcpy(combinedTokens.data(), imgTokensPtr, imgTokensNumel * sizeof(float));
    mod3.reset();
    LOGD("Part3: %lldms (numel=%zu)", nowMs() - tP3, imgTokensNumel);

    // ── Part4a chunk 512 ─────────────────────────────────────────────────────
    long long tP4a = nowMs();
    std::string path4a512 = pathJoin(modelDir, "sharp_split_part4a_chunk_512.pte");
    auto mod4a512 = std::make_unique<ETModule>(path4a512, ETModule::LoadMode::Mmap);
    if (mod4a512->load() != Error::Ok) {
        LOGE("Part4a/512 load fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto c512Tensor = from_blob(combinedTokens.data(), {1, 512, FEATURE_DIM});
    std::vector<EValue> in4a512 = {*c512Tensor};
    auto out4a512 = mod4a512->forward(in4a512);
    if (!out4a512.ok() || out4a512->empty()) {
        LOGE("Part4a/512 forward fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out512Ptr = (*out4a512)[0].toTensor().const_data_ptr<float>();
    size_t out512Len = (*out4a512)[0].toTensor().numel();
    std::memcpy(combinedTokens.data(), out512Ptr, out512Len * sizeof(float));
    mod4a512.reset();
    LOGD("Part4a/512: %lldms (out=%zu)", nowMs() - tP4a, out512Len);

    // ── Part4a chunk 65 ──────────────────────────────────────────────────────
    long long tP4a65 = nowMs();
    std::string path4a65 = pathJoin(modelDir, "sharp_split_part4a_chunk_65.pte");
    auto mod4a65 = std::make_unique<ETModule>(path4a65, ETModule::LoadMode::Mmap);
    if (mod4a65->load() != Error::Ok) {
        LOGE("Part4a/65 load fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto c65Tensor = from_blob(combinedTokens.data() + 512 * FEATURE_DIM, {1, 65, FEATURE_DIM});
    std::vector<EValue> in4a65 = {*c65Tensor};
    auto out4a65 = mod4a65->forward(in4a65);
    if (!out4a65.ok() || out4a65->empty()) {
        LOGE("Part4a/65 forward fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out65Ptr = (*out4a65)[0].toTensor().const_data_ptr<float>();
    size_t out65Len = (*out4a65)[0].toTensor().numel();
    std::memcpy(combinedTokens.data() + 512 * FEATURE_DIM, out65Ptr, out65Len * sizeof(float));
    mod4a65.reset();
    LOGD("Part4a/65: %lldms. Part4a total: %lldms", nowMs() - tP4a65, nowMs() - tP4a);

    // ── Part4b (single, load + run + release) ────────────────────────────────
    // Release Part1+Part2 cache before Part4b to free ~1GB RSS; re-cached on next pipeline call
    g_moduleCache.release();
    LOGD("Released Part1+Part2 cache before Part4b to free memory");

    long long tP4b = nowMs();
    std::string path4b = pathJoin(modelDir, "sharp_split_part4b.pte");
    auto mod4b = std::make_unique<ETModule>(path4b, ETModule::LoadMode::Mmap);
    if (mod4b->load() != Error::Ok) {
        LOGE("Part4b load fail: %s", path4b.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    auto tokens4b = from_blob(combinedTokens.data(), {1, TOKENS_577, FEATURE_DIM});
    auto img4b    = from_blob(imageData, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
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

    LOGD("Part4b forward starting...");
    auto out4b = mod4b->forward(in4b);

    if (!out4b.ok() || out4b->empty()) {
        LOGE("Part4b forward fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    const auto& gaussianTensor = (*out4b)[0].toTensor();
    const float* gaussianPtr = gaussianTensor.const_data_ptr<float>();
    const int numFloats = static_cast<int>(gaussianTensor.numel());

    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);

    if (numFloats <= 0 || (numFloats % PARAMS_PER_GAUSSIAN) != 0) {
        LOGE("Part4b output invalid: %d", numFloats);
        return nullptr;
    }

    long long tTotal = nowMs() - t0;
    LOGD("Part4b: %lldms. TOTAL pipeline: %lldms. Gaussians=%d",
         nowMs() - tP4b, tTotal, numFloats / PARAMS_PER_GAUSSIAN);

    jfloatArray jResult = env->NewFloatArray(numFloats);
    if (!jResult) { LOGE("result array alloc fail"); return nullptr; }
    env->SetFloatArrayRegion(jResult, 0, numFloats, gaussianPtr);
    return jResult;
}

} // extern "C"
