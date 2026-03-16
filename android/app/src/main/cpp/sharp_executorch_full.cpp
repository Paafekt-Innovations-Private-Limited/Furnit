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
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
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

#define TAG "sharp_executorch_full"
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
static constexpr int PATCH_BATCH = 4;   // batch-4 Part1/Part2 (INT8, non-Vulkan only)
static constexpr int PATCH_BATCH_2 = 2; // batch-2 for Vulkan FP16 (95% success, no INT8 staging crash)
static constexpr int HALF_SIZE = IMAGE_SIZE / 2;   // 768
static constexpr int STRIDE_1X = (IMAGE_SIZE - PATCH_SIZE) / 4;  // 288
static constexpr int STRIDE_05X = (HALF_SIZE - PATCH_SIZE) / 2;  // 192
static constexpr int SPATIAL_HW = SPATIAL_SIZE * SPATIAL_SIZE;    // 576

static std::string pathJoin(const std::string& dir, const std::string& name);

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
    float* patchBuf4 = nullptr;  // PATCH_BATCH * 3 * PATCH_SIZE * PATCH_SIZE
    float* tokensCopy = nullptr;  // TOKENS_577 * FEATURE_DIM
    float* tokensCopy4 = nullptr; // PATCH_BATCH * TOKENS_577 * FEATURE_DIM (for Part2_b4 input)
    bool allocated = false;

    bool allocate() {
        if (allocated) return true;
        const size_t feat96 = FEATURE_DIM * M_1X * M_1X;
        const size_t feat48 = FEATURE_DIM * M_05X * M_05X;
        const size_t feat24 = FEATURE_DIM * SPATIAL_HW;
        const size_t patchSz = 3 * PATCH_SIZE * PATCH_SIZE;
        const size_t patchSz4 = (size_t)PATCH_BATCH * patchSz;
        const size_t halfSz = 3 * HALF_SIZE * HALF_SIZE;
        const size_t tokensSz = TOKENS_577 * FEATURE_DIM;
        const size_t tokensSz4 = (size_t)PATCH_BATCH * tokensSz;

        latent0    = alignedAlloc(feat96);
        latent1    = alignedAlloc(feat96);
        x0Feat     = alignedAlloc(feat96);
        x1Feat     = alignedAlloc(feat48);
        x2Feat     = alignedAlloc(feat24);
        tempSpatial = alignedAlloc(feat24);
        halfImg    = alignedAlloc(halfSz);
        quarterImg = alignedAlloc(patchSz);
        patchBuf   = alignedAlloc(patchSz);
        patchBuf4  = alignedAlloc(patchSz4);
        tokensCopy = alignedAlloc(tokensSz);
        tokensCopy4 = alignedAlloc(tokensSz4);

        allocated = latent0 && latent1 && x0Feat && x1Feat && x2Feat &&
                    tempSpatial && halfImg && quarterImg && patchBuf && patchBuf4 &&
                    tokensCopy && tokensCopy4;
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
        alignedFree(patchBuf4); alignedFree(tokensCopy); alignedFree(tokensCopy4);
        latent0 = latent1 = x0Feat = x1Feat = x2Feat = tempSpatial = nullptr;
        halfImg = quarterImg = patchBuf = patchBuf4 = tokensCopy = tokensCopy4 = nullptr;
        allocated = false;
    }
};

// ── Vulkan input validation (power-of-2 alignment; can catch staging bugs) ───
static bool validateVulkanInputs(const std::vector<EValue>& inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (!inputs[i].isTensor()) continue;
        const auto& t = inputs[i].toTensor();
        if (t.dim() == 0) {
            LOGE("VULKAN VALID: input[%zu] dim=0", i);
            return false;
        }
        const int64_t n = t.numel();
        if (n <= 0 || (n % 4 != 0)) {
            LOGE("VULKAN VALID: input[%zu] numel=%lld (need >0 and %%4==0)", i, (long long)n);
            return false;
        }
    }
    return true;
}

// ── Singleton module cache ───────────────────────────────────────────────────
struct ModuleCache {
    std::unique_ptr<ETModule> mod1, mod2;
    std::unique_ptr<ETModule> mod1_b4, mod2_b4;  // batch=4 INT8 (non-Vulkan only)
    std::unique_ptr<ETModule> mod1_b2, mod2_b2;  // batch=2 Vulkan FP16
    std::string modelDir;
    bool lastUseVulkan = false;
    std::mutex mu;

    bool ensureLoaded(const std::string& dir, bool useVulkan) {
        std::lock_guard<std::mutex> lock(mu);
        if (mod1 && mod2 && modelDir == dir && lastUseVulkan == useVulkan) return true;
        mod1.reset(); mod2.reset(); mod1_b4.reset(); mod2_b4.reset(); mod1_b2.reset(); mod2_b2.reset();
        modelDir = dir;
        lastUseVulkan = useVulkan;

        if (useVulkan) {
            // Prefer Vulkan FP16 Part1/Part2 (avoids INT8 staging crash).
            std::string p1vk = pathJoin(dir, "sharp_split_part1_vulkan_fp16.pte");
            std::string p2vk = pathJoin(dir, "sharp_split_part2_vulkan_fp16.pte");
            std::ifstream f1(p1vk), f2(p2vk);
            if (f1.good() && f2.good()) {
                f1.close(); f2.close();
                mod1 = std::make_unique<ETModule>(p1vk, ETModule::LoadMode::Mmap);
                mod2 = std::make_unique<ETModule>(p2vk, ETModule::LoadMode::Mmap);
                if (mod1->load() == Error::Ok && mod2->load() == Error::Ok) {
                    LOGD("ModuleCache: Part1+Part2 Vulkan FP16 loaded");
                    std::string p1b2 = pathJoin(dir, "sharp_split_part1_b2_vulkan_fp16.pte");
                    std::string p2b2 = pathJoin(dir, "sharp_split_part2_b2_vulkan_fp16.pte");
                    std::ifstream b1(p1b2), b2(p2b2);
                    if (b1.good() && b2.good()) {
                        b1.close(); b2.close();
                        mod1_b2 = std::make_unique<ETModule>(p1b2, ETModule::LoadMode::Mmap);
                        mod2_b2 = std::make_unique<ETModule>(p2b2, ETModule::LoadMode::Mmap);
                        if (mod1_b2->load() == Error::Ok && mod2_b2->load() == Error::Ok)
                            LOGD("ModuleCache: Part1+Part2 batch=2 Vulkan FP16 loaded");
                        else
                            mod1_b2.reset(), mod2_b2.reset();
                    }
                    return true;
                }
                mod1.reset(); mod2.reset();
            }
            // Fallback: INT8 single-patch only (no B4 under Vulkan – crashes).
            LOGD("ModuleCache: Vulkan FP16 not found, falling back to INT8 single-patch");
        }

        std::string p1 = pathJoin(dir, "sharp_split_part1_int8.pte");
        std::string p2 = pathJoin(dir, "sharp_split_part2_int8.pte");
        mod1 = std::make_unique<ETModule>(p1, ETModule::LoadMode::Mmap);
        if (mod1->load() != Error::Ok) { LOGE("Cache: Part1 load fail"); mod1.reset(); return false; }
        mod2 = std::make_unique<ETModule>(p2, ETModule::LoadMode::Mmap);
        if (mod2->load() != Error::Ok) { LOGE("Cache: Part2 load fail"); mod2.reset(); mod1.reset(); return false; }
        LOGD("ModuleCache: Part1+Part2 loaded (kept alive across calls)");

        if (!useVulkan) {
            std::string p1b4 = pathJoin(dir, "sharp_split_part1_b4_int8.pte");
            std::string p2b4 = pathJoin(dir, "sharp_split_part2_b4_int8.pte");
            std::ifstream f1(p1b4), f2(p2b4);
            if (f1.good() && f2.good()) {
                f1.close(); f2.close();
                mod1_b4 = std::make_unique<ETModule>(p1b4, ETModule::LoadMode::Mmap);
                mod2_b4 = std::make_unique<ETModule>(p2b4, ETModule::LoadMode::Mmap);
                if (mod1_b4->load() == Error::Ok && mod2_b4->load() == Error::Ok)
                    LOGD("ModuleCache: Part1+Part2 batch=4 loaded (patch batching enabled)");
                else
                    mod1_b4.reset(), mod2_b4.reset();
            }
        }
        return true;
    }

    void release() {
        std::lock_guard<std::mutex> lock(mu);
        mod1.reset(); mod2.reset(); mod1_b4.reset(); mod2_b4.reset(); mod1_b2.reset(); mod2_b2.reset();
        modelDir.clear();
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

/** Prefer INT8 Part4b when present (e.g. sharp_split_part4b_int8.pte), else use FP32 (sharp_split_part4b.pte). */
static std::string tryInt8ThenFp32(const std::string& modelDir, const std::string& fp32Name) {
    size_t dot = fp32Name.rfind('.');
    std::string int8Name = (dot != std::string::npos)
        ? fp32Name.substr(0, dot) + "_int8" + fp32Name.substr(dot)
        : fp32Name + "_int8";
    std::string int8Path = pathJoin(modelDir, int8Name);
    std::ifstream f(int8Path);
    if (f.good()) {
        f.close();
        return int8Path;
    }
    return pathJoin(modelDir, fp32Name);
}

// ── Helper: BATCHED tiled Part4b (batch=4, 4 forward calls for 16 tiles) ─────
// Loads sharp_split_part4b_tile_b4.pte (exported with batch=4 static shape).
// Crops 4 tiles at a time, stacks along batch dim, one forward, split output.
static bool runPart4bBatchedTiledPipeline(
        const std::string& modelDir,
        const float* __restrict imageData,   // [3,1536,1536] NCHW
        const float* __restrict latent0,     // [1024,96,96]
        const float* __restrict latent1,     // [1024,96,96]
        const float* __restrict x0Feat,      // [1024,96,96]
        const float* __restrict x1Feat,      // [1024,48,48]
        const float* __restrict x2Feat,      // [1024,24,24]
        const float* __restrict combinedTokens, size_t tokensNumel,
        std::vector<float>& outGaussians) {

    const int GRID = 4;
    const int NUM_TILES = GRID * GRID;
    const int BATCH = 4;
    const int NUM_BATCHES = NUM_TILES / BATCH;
    const int imgH = IMAGE_SIZE, imgW = IMAGE_SIZE, imgC = 3;
    const int imgTileH = imgH / GRID, imgTileW = imgW / GRID;
    const int lat96H = 96, lat96W = 96;
    const int x1H = 48, x1W = 48;
    const int x2H = 24, x2W = 24;
    const int xLowH = 24, xLowW = 24, xLowC = 1024;

    std::string modelPath = modelDir + "/sharp_split_part4b_tile_b4.pte";
    {
        std::ifstream f(modelPath);
        if (!f.good()) {
            LOGD("runPart4bBatchedTiledPipeline: tile_b4.pte not found, skipping batched path");
            return false;
        }
    }

    if (tokensNumel < (size_t)((SPATIAL_HW + 1) * FEATURE_DIM)) {
        LOGE("runPart4bBatchedTiledPipeline: combinedTokens too small");
        return false;
    }
    std::vector<float> xLowres(xLowC * xLowH * xLowW);
    for (int row = 0; row < xLowH; ++row) {
        for (int col = 0; col < xLowW; ++col) {
            const int srcTokenIdx = 1 + row * xLowW + col;
            const size_t srcBase = (size_t)srcTokenIdx * FEATURE_DIM;
            const int dstIndexBase = row * xLowW + col;
            for (int ch = 0; ch < xLowC; ++ch) {
                xLowres[ch * xLowH * xLowW + dstIndexBase] = combinedTokens[srcBase + ch];
            }
        }
    }

    auto cropTile = [GRID](const float* __restrict src, int C, int H, int W,
                           int tileRow, int tileCol, float* __restrict dst) {
        const int tileH = H / GRID;
        const int tileW = W / GRID;
        const int srcHW = H * W;
        const int dstHW = tileH * tileW;
        const size_t rowBytes = tileW * sizeof(float);
        for (int c = 0; c < C; ++c) {
            const float* srcPlane = src + c * srcHW + tileRow * tileH * W + tileCol * tileW;
            float* dstPlane = dst + c * dstHW;
            for (int y = 0; y < tileH; ++y) {
                std::memcpy(dstPlane + y * tileW, srcPlane + y * W, rowBytes);
            }
        }
    };

    auto correctNDC = [GRID](float* data, int numGaussians, int tileRow, int tileCol) {
        const float ndcOffsetX = (2.0f * tileCol + 1.0f - GRID) / GRID;
        const float ndcOffsetY = (2.0f * tileRow + 1.0f - GRID) / GRID;
        const float invGrid = 1.0f / GRID;
        for (int g = 0; g < numGaussians; ++g) {
            float* base = data + g * PARAMS_PER_GAUSSIAN;
            const float posZ = base[2];
            base[0] = base[0] * invGrid + posZ * ndcOffsetX;
            base[1] = base[1] * invGrid + posZ * ndcOffsetY;
            base[4] *= invGrid;
            base[5] *= invGrid;
            base[6] *= invGrid;
        }
    };

    ensureRuntimeInit();
    auto module = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
    if (module->load() != Error::Ok) {
        LOGE("runPart4bBatchedTiledPipeline: failed to load %s", modelPath.c_str());
        return false;
    }
    LOGD("runPart4bBatchedTiledPipeline: loaded %s (batch=%d)", modelPath.c_str(), BATCH);

    const int imgCropSize  = imgC * imgTileH * imgTileW;
    const int lat96CropSz  = 1024 * 24 * 24;
    const int x1CropSz     = 1024 * 12 * 12;
    const int x2CropSz     = 1024 * 6 * 6;

    std::vector<float> batchImg(BATCH * imgCropSize);
    std::vector<float> batchLat0(BATCH * lat96CropSz), batchLat1(BATCH * lat96CropSz), batchX0(BATCH * lat96CropSz);
    std::vector<float> batchX1(BATCH * x1CropSz), batchX2(BATCH * x2CropSz), batchXl(BATCH * x2CropSz);

    outGaussians.clear();
    int floatsPerTile = 0;

    auto tStart = std::chrono::steady_clock::now();
    for (int b = 0; b < NUM_BATCHES; ++b) {
        auto batchStart = std::chrono::steady_clock::now();

        for (int i = 0; i < BATCH; ++i) {
            const int t = b * BATCH + i;
            const int tileRow = t / GRID;
            const int tileCol = t % GRID;

            cropTile(imageData, imgC, imgH, imgW, tileRow, tileCol, batchImg.data() + i * imgCropSize);
            cropTile(latent0, 1024, lat96H, lat96W, tileRow, tileCol, batchLat0.data() + i * lat96CropSz);
            cropTile(latent1, 1024, lat96H, lat96W, tileRow, tileCol, batchLat1.data() + i * lat96CropSz);
            cropTile(x0Feat,  1024, lat96H, lat96W, tileRow, tileCol, batchX0.data() + i * lat96CropSz);
            cropTile(x1Feat,  1024, x1H,    x1W,    tileRow, tileCol, batchX1.data() + i * x1CropSz);
            cropTile(x2Feat,  1024, x2H,    x2W,    tileRow, tileCol, batchX2.data() + i * x2CropSz);
            cropTile(xLowres.data(), 1024, xLowH, xLowW, tileRow, tileCol, batchXl.data() + i * x2CropSz);
        }

        auto imgT  = from_blob(batchImg.data(),  {BATCH, 3, imgTileH, imgTileW});
        auto l0T   = from_blob(batchLat0.data(), {BATCH, 1024, 24, 24});
        auto l1T   = from_blob(batchLat1.data(), {BATCH, 1024, 24, 24});
        auto x0T   = from_blob(batchX0.data(),   {BATCH, 1024, 24, 24});
        auto x1T   = from_blob(batchX1.data(),   {BATCH, 1024, 12, 12});
        auto x2T   = from_blob(batchX2.data(),   {BATCH, 1024, 6, 6});
        auto xlT   = from_blob(batchXl.data(),   {BATCH, 1024, 6, 6});

        std::vector<EValue> inputs;
        inputs.reserve(7);
        inputs.emplace_back(*imgT);
        inputs.emplace_back(*l0T);
        inputs.emplace_back(*l1T);
        inputs.emplace_back(*x0T);
        inputs.emplace_back(*x1T);
        inputs.emplace_back(*x2T);
        inputs.emplace_back(*xlT);

        auto result = module->forward(inputs);
        if (!result.ok() || result->empty() || !(*result)[0].isTensor()) {
            LOGE("runPart4bBatchedTiledPipeline: batch %d forward failed", b);
            return false;
        }

        const auto& outTensor = (*result)[0].toTensor();
        const float* outData = outTensor.const_data_ptr<float>();
        const int totalFloats = static_cast<int>(outTensor.numel());
        const int floatsPerBatchItem = totalFloats / BATCH;
        const int gaussiansPerTile = floatsPerBatchItem / PARAMS_PER_GAUSSIAN;

        if (b == 0) {
            floatsPerTile = floatsPerBatchItem;
            outGaussians.resize(NUM_TILES * floatsPerTile);
        } else if (floatsPerBatchItem != floatsPerTile) {
            LOGE("runPart4bBatchedTiledPipeline: inconsistent batch output size");
            return false;
        }

        for (int i = 0; i < BATCH; ++i) {
            const int t = b * BATCH + i;
            const int tileRow = t / GRID;
            const int tileCol = t % GRID;
            float* dst = outGaussians.data() + t * floatsPerTile;
            std::memcpy(dst, outData + i * floatsPerBatchItem, floatsPerTile * sizeof(float));
            correctNDC(dst, gaussiansPerTile, tileRow, tileCol);
        }

        auto batchEnd = std::chrono::steady_clock::now();
        long long batchMs = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchStart).count();
        LOGD("Part4b batch %d/%d (tiles %d-%d): %lldms", b + 1, NUM_BATCHES,
             b * BATCH + 1, (b + 1) * BATCH, (long long)batchMs);
    }

    auto tEnd = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    LOGD("[TIMING] Part4b batched tiled (batch=%d, %d calls): %lldms, %d Gaussians",
         BATCH, NUM_BATCHES, (long long)ms,
         (int)(outGaussians.size() / PARAMS_PER_GAUSSIAN));
    return true;
}


// ── Helper: tiled Part4b inside full pipeline (uses same export as Kotlin path) ───
// Build 24x24x1024 low-res feature map from combinedTokens (CLS + 576 spatial),
// then run a 4x4 grid of tiles, correcting NDC per tile and concatenating outputs.
static bool runPart4bTiledFullPipeline(
        const std::string& modelDir,
        const float* __restrict imageData,   // [3,1536,1536] NCHW
        const float* __restrict latent0,     // [1024,96,96]
        const float* __restrict latent1,     // [1024,96,96]
        const float* __restrict x0Feat,      // [1024,96,96]
        const float* __restrict x1Feat,      // [1024,48,48]
        const float* __restrict x2Feat,      // [1024,24,24]
        const float* __restrict combinedTokens, size_t tokensNumel,
        std::vector<float>& outGaussians) {

    const int GRID = 4;
    const int NUM_TILES = GRID * GRID;
    const int imgH = IMAGE_SIZE, imgW = IMAGE_SIZE, imgC = 3;
    const int imgTileH = imgH / GRID, imgTileW = imgW / GRID;
    const int lat96H = 96, lat96W = 96;
    const int x1H = 48, x1W = 48;
    const int x2H = 24, x2W = 24;
    const int xLowH = 24, xLowW = 24, xLowC = 1024;

    // Choose tile model: prefer INT8 then FP32; prefer tile_full then tile_00
    std::string modelPath;
    const std::string tileFullFp32 = "sharp_split_part4b_tile_full.pte";
    const std::string tile00Fp32  = "sharp_split_part4b_tile_00.pte";
    std::string candidate = tryInt8ThenFp32(modelDir, tileFullFp32);
    std::ifstream f1(candidate);
    if (f1.good()) {
        f1.close();
        modelPath = candidate;
        LOGD("runPart4bTiledFullPipeline: using %s", modelPath.c_str());
    } else {
        f1.close();
        candidate = tryInt8ThenFp32(modelDir, tile00Fp32);
        std::ifstream f2(candidate);
        if (!f2.good()) {
            LOGD("runPart4bTiledFullPipeline: no tile_full or tile_00 (INT8 or FP32) found, skipping tiled path");
            return false;
        }
        f2.close();
        modelPath = candidate;
        LOGD("runPart4bTiledFullPipeline: using %s", modelPath.c_str());
    }

    // Build xLowres from combinedTokens: [CLS + 576 tokens, 1024] -> [1024,24,24] NCHW.
    if (tokensNumel < (size_t)((SPATIAL_HW + 1) * FEATURE_DIM)) {
        LOGE("runPart4bTiledFullPipeline: combinedTokens too small for xLowres");
        return false;
    }
    std::vector<float> xLowres(xLowC * xLowH * xLowW);
    for (int row = 0; row < xLowH; ++row) {
        for (int col = 0; col < xLowW; ++col) {
            const int srcTokenIdx = 1 + row * xLowW + col; // skip CLS
            const size_t srcBase = (size_t)srcTokenIdx * FEATURE_DIM;
            const int dstIndexBase = row * xLowW + col;
            for (int ch = 0; ch < xLowC; ++ch) {
                xLowres[ch * xLowH * xLowW + dstIndexBase] = combinedTokens[srcBase + ch];
            }
        }
    }

    // Local crop helper: tile (tileRow,tileCol) from [C,H,W]
    auto cropTileNCHWLocal = [GRID](const float* __restrict src, int C, int H, int W,
                                    int tileRow, int tileCol, float* __restrict dst) {
        const int tileH = H / GRID;
        const int tileW = W / GRID;
        const int srcHW = H * W;
        const int dstHW = tileH * tileW;
        const size_t rowBytes = tileW * sizeof(float);
        for (int c = 0; c < C; ++c) {
            const float* srcPlane = src + c * srcHW + tileRow * tileH * W + tileCol * tileW;
            float* dstPlane = dst + c * dstHW;
            for (int y = 0; y < tileH; ++y) {
                std::memcpy(dstPlane + y * tileW, srcPlane + y * W, rowBytes);
            }
        }
    };

    // NDC correction (same as tiles path), swapXY currently false.
    auto correctNDC = [GRID](float* data, int numGaussians, int tileRow, int tileCol, bool swapXY) {
        const float ndcOffsetX = (2.0f * tileCol + 1.0f - GRID) / GRID;
        const float ndcOffsetY = (2.0f * tileRow + 1.0f - GRID) / GRID;
        const float invGrid = 1.0f / GRID;
        const float offX = swapXY ? ndcOffsetY : ndcOffsetX;
        const float offY = swapXY ? ndcOffsetX : ndcOffsetY;
        for (int g = 0; g < numGaussians; ++g) {
            float* base = data + g * PARAMS_PER_GAUSSIAN;
            const float posZ = base[2];
            base[0] = base[0] * invGrid + posZ * offX;
            base[1] = base[1] * invGrid + posZ * offY;
            base[4] *= invGrid;
            base[5] *= invGrid;
            base[6] *= invGrid;
        }
    };

    ensureRuntimeInit();
    auto module = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
    if (module->load() != Error::Ok) {
        LOGE("runPart4bTiledFullPipeline: failed to load tile model: %s", modelPath.c_str());
        return false;
    }

    // Per-tile scratch buffers
    const int imgCropSize  = imgC * imgTileH * imgTileW;
    const int lat96CropSz  = 1024 * 24 * 24;
    const int x1CropSz     = 1024 * 12 * 12;
    const int x2CropSz     = 1024 * 6 * 6;
    std::vector<float> imgCrop(imgCropSize);
    std::vector<float> lat0Crop(lat96CropSz), lat1Crop(lat96CropSz), x0Crop(lat96CropSz);
    std::vector<float> x1Crop(x1CropSz), x2Crop(x2CropSz), xlCrop(x2CropSz);

    outGaussians.clear();
    int floatsPerTile = 0;

    auto tStart = std::chrono::steady_clock::now();
    for (int t = 0; t < NUM_TILES; ++t) {
        const int tileRow = t / GRID;
        const int tileCol = t % GRID;
        auto tileStart = std::chrono::steady_clock::now();

        cropTileNCHWLocal(imageData, imgC, imgH, imgW, tileRow, tileCol, imgCrop.data());
        cropTileNCHWLocal(latent0, 1024, lat96H, lat96W, tileRow, tileCol, lat0Crop.data());
        cropTileNCHWLocal(latent1, 1024, lat96H, lat96W, tileRow, tileCol, lat1Crop.data());
        cropTileNCHWLocal(x0Feat,  1024, lat96H, lat96W, tileRow, tileCol, x0Crop.data());
        cropTileNCHWLocal(x1Feat,  1024, x1H,    x1W,    tileRow, tileCol, x1Crop.data());
        cropTileNCHWLocal(x2Feat,  1024, x2H,    x2W,    tileRow, tileCol, x2Crop.data());
        cropTileNCHWLocal(xLowres.data(), 1024, xLowH, xLowW, tileRow, tileCol, xlCrop.data());

        auto imgTensor  = from_blob(imgCrop.data(),  {1, 3, imgTileH, imgTileW});
        auto lat0Tensor = from_blob(lat0Crop.data(), {1, 1024, 24, 24});
        auto lat1Tensor = from_blob(lat1Crop.data(), {1, 1024, 24, 24});
        auto x0Tensor   = from_blob(x0Crop.data(),   {1, 1024, 24, 24});
        auto x1Tensor   = from_blob(x1Crop.data(),   {1, 1024, 12, 12});
        auto x2Tensor   = from_blob(x2Crop.data(),   {1, 1024, 6, 6});
        auto xlTensor   = from_blob(xlCrop.data(),   {1, 1024, 6, 6});

        std::vector<EValue> inputs;
        inputs.reserve(7);
        inputs.emplace_back(*imgTensor);
        inputs.emplace_back(*lat0Tensor);
        inputs.emplace_back(*lat1Tensor);
        inputs.emplace_back(*x0Tensor);
        inputs.emplace_back(*x1Tensor);
        inputs.emplace_back(*x2Tensor);
        inputs.emplace_back(*xlTensor);

        auto result = module->forward(inputs);
        if (!result.ok() || result->empty() || !(*result)[0].isTensor()) {
            LOGE("runPart4bTiledFullPipeline: tile %d forward failed", t);
            return false;
        }

        const auto& outTensor = (*result)[0].toTensor();
        const float* outData = outTensor.const_data_ptr<float>();
        const int numFloats = static_cast<int>(outTensor.numel());
        const int numGaussians = numFloats / PARAMS_PER_GAUSSIAN;

        if (t == 0) {
            floatsPerTile = numFloats;
            outGaussians.resize(NUM_TILES * floatsPerTile);
        } else if (numFloats != floatsPerTile) {
            LOGE("runPart4bTiledFullPipeline: inconsistent tile output size");
            return false;
        }

        float* dst = outGaussians.data() + t * floatsPerTile;
        std::memcpy(dst, outData, numFloats * sizeof(float));
        correctNDC(dst, numGaussians, tileRow, tileCol, /*swapXY=*/false);

        auto tileEnd = std::chrono::steady_clock::now();
        long long tileMs = std::chrono::duration_cast<std::chrono::milliseconds>(tileEnd - tileStart).count();
        LOGD("Part4b tile %d/%d: %lldms", t + 1, NUM_TILES, (long long)tileMs);
    }

    auto tEnd = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    LOGD("[TIMING] Part4b tiled (full pipeline C++): %lldms, %d Gaussians", (long long)ms,
         (int)(outGaussians.size() / PARAMS_PER_GAUSSIAN));
    return true;
}

// ── Gaussian pruning: keep top-N by opacity ─────────────────────────────────
// Opacity is at offset 3 in [x,y,z,opacity,sx,sy,sz,rw,rx,ry,rz,c0,c1,c2].
// Uses partial_sort to avoid full sort cost.
static void pruneGaussiansByOpacity(std::vector<float>& gaussians, int maxGaussians) {
    const int totalGaussians = static_cast<int>(gaussians.size()) / PARAMS_PER_GAUSSIAN;
    if (maxGaussians <= 0 || totalGaussians <= maxGaussians) return;

    auto tStart = std::chrono::steady_clock::now();

    std::vector<int> indices(totalGaussians);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + maxGaussians, indices.end(),
        [&gaussians](int a, int b) {
            return gaussians[a * PARAMS_PER_GAUSSIAN + 3] > gaussians[b * PARAMS_PER_GAUSSIAN + 3];
        });

    std::vector<float> pruned(maxGaussians * PARAMS_PER_GAUSSIAN);
    for (int i = 0; i < maxGaussians; ++i) {
        std::memcpy(pruned.data() + i * PARAMS_PER_GAUSSIAN,
                     gaussians.data() + indices[i] * PARAMS_PER_GAUSSIAN,
                     PARAMS_PER_GAUSSIAN * sizeof(float));
    }
    gaussians = std::move(pruned);

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tStart).count();
    LOGD("Pruned Gaussians: %d -> %d (top by opacity, %lldms)", totalGaussians, maxGaussians, (long long)ms);
}

// Prune a raw float pointer + count (for single Part4b path that doesn't use vector).
// Returns new count. Caller must use prunedOut.
static int pruneGaussiansFromPtr(const float* src, int totalGaussians, int maxGaussians,
                                  std::vector<float>& prunedOut) {
    if (maxGaussians <= 0 || totalGaussians <= maxGaussians) {
        prunedOut.assign(src, src + totalGaussians * PARAMS_PER_GAUSSIAN);
        return totalGaussians;
    }

    auto tStart = std::chrono::steady_clock::now();

    std::vector<int> indices(totalGaussians);
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + maxGaussians, indices.end(),
        [src](int a, int b) {
            return src[a * PARAMS_PER_GAUSSIAN + 3] > src[b * PARAMS_PER_GAUSSIAN + 3];
        });

    prunedOut.resize(maxGaussians * PARAMS_PER_GAUSSIAN);
    for (int i = 0; i < maxGaussians; ++i) {
        std::memcpy(prunedOut.data() + i * PARAMS_PER_GAUSSIAN,
                     src + indices[i] * PARAMS_PER_GAUSSIAN,
                     PARAMS_PER_GAUSSIAN * sizeof(float));
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - tStart).count();
    LOGD("Pruned Gaussians: %d -> %d (top by opacity, %lldms)", totalGaussians, maxGaussians, (long long)ms);
    return maxGaussians;
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
    return g_moduleCache.ensureLoaded(dir, true) ? JNI_TRUE : JNI_FALSE;  // preload Vulkan path when available
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
        jfloatArray imageNCHW,
        jint maxGaussians,
        jboolean preferSinglePart4b,
        jboolean useVulkan,
        jboolean useB4Param) {

    if (!modelDirPath || !imageNCHW) {
        LOGE("runFullPipelineInt8: null modelDir or image");
        return nullptr;
    }
    const int maxG = static_cast<int>(maxGaussians);
    const bool useSinglePart4bOnly = (preferSinglePart4b == JNI_TRUE);
    const bool useVulkanBackend = (useVulkan == JNI_TRUE);
    // Disable batch-4 when Vulkan: Part1_b4/Part2_b4 forward often crashes on Vulkan runtime.
    const bool useB4 = (useB4Param == JNI_TRUE) && !useVulkanBackend;

    ensureRuntimeInit();
    const long long t0 = nowMs();

    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string modelDir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (modelDir.empty()) { LOGE("empty model dir"); return nullptr; }

    LOGD("runFullPipelineInt8: modelDir=%s useVulkan=%d", modelDir.c_str(), useVulkanBackend ? 1 : 0);

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
    std::string path1_int8 = pathJoin(modelDir, "sharp_split_part1_int8.pte");
    std::string path1_vk = pathJoin(modelDir, "sharp_split_part1_vulkan_fp16.pte");
    {
        std::ifstream f1(path1_int8), f2(path1_vk);
        if (!f1.good() && !f2.good()) {
            LOGE("Part1 not found: %s or %s", path1_int8.c_str(), path1_vk.c_str());
            env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
            return nullptr;
        }
    }

    long long tLoad12 = nowMs();
    if (!g_moduleCache.ensureLoaded(modelDir, useVulkanBackend)) {
        LOGE("Failed to load Part1+Part2");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGD("Part1+Part2 ready (cache): %lldms", nowMs() - tLoad12);

    float* ws_patch = g_workspace.patchBuf;
    float* ws_tokens = g_workspace.tokensCopy;
    float* ws_temp = g_workspace.tempSpatial;
    const size_t patchSz = (size_t)(3 * PATCH_SIZE * PATCH_SIZE);
    const size_t tokenSliceSz = (size_t)TOKENS_577 * FEATURE_DIM;
    const size_t featSliceSz = (size_t)FEATURE_DIM * SPATIAL_HW;

    // ── 1x patches (5x5 = 25): batch-2 Vulkan FP16 or batch-4 INT8 when available ─
    long long t1x = nowMs();
    ETModule* m1 = g_moduleCache.mod1.get();
    ETModule* m2 = g_moduleCache.mod2.get();
    ETModule* m1_b4 = g_moduleCache.mod1_b4.get();
    ETModule* m2_b4 = g_moduleCache.mod2_b4.get();
    ETModule* m1_b2 = g_moduleCache.mod1_b2.get();
    ETModule* m2_b2 = g_moduleCache.mod2_b2.get();

    int batchSize1x = 1;
    ETModule* m1_batch = nullptr;
    ETModule* m2_batch = nullptr;
    if (useVulkanBackend && m1_b2 && m2_b2) {
        batchSize1x = PATCH_BATCH_2;
        m1_batch = m1_b2;
        m2_batch = m2_b2;
    } else if (!useVulkanBackend && m1_b4 && m2_b4 && useB4) {
        batchSize1x = PATCH_BATCH;
        m1_batch = m1_b4;
        m2_batch = m2_b4;
    }

    for (int start = 0; start < 25; start += batchSize1x) {
        const int n = std::min(batchSize1x, 25 - start);
        const bool useBatch = (n == batchSize1x && m1_batch && m2_batch);
        bool batchOk = false;

        if (useBatch) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_1X;
                const int jj = (start + b) % GRID_1X;
                cropNCHW(imageData, IMAGE_SIZE, IMAGE_SIZE, 3,
                         ii * STRIDE_1X, jj * STRIDE_1X, PATCH_SIZE, PATCH_SIZE,
                         g_workspace.patchBuf4 + b * patchSz);
            }
            auto pTensor = from_blob(g_workspace.patchBuf4, {n, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            if (useVulkanBackend && validateVulkanInputs(in1)) { /* optional guard */ }
            if (start == 0) LOGD("1x batch-%d: Part1 forward (start=0)", n);
            auto out1 = m1_batch->forward(in1);
            const size_t needTokenNumel = (size_t)n * tokenSliceSz;
            const size_t needFeatNumel = (size_t)n * featSliceSz;
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
                    auto tokInput = from_blob(g_workspace.tokensCopy4, {n, TOKENS_577, FEATURE_DIM});
                    std::vector<EValue> in2 = {*tokInput};
                    if (useVulkanBackend && validateVulkanInputs(in2)) { /* optional guard */ }
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
            }
            if (!batchOk) {
                LOGD("Part1/Part2 batch-%d 1x start=%d: forward failed or bad shape, using single-patch", n, start);
            }
        }

        if (!batchOk) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_1X;
                const int jj = (start + b) % GRID_1X;
                cropNCHW(imageData, IMAGE_SIZE, IMAGE_SIZE, 3,
                         ii * STRIDE_1X, jj * STRIDE_1X, PATCH_SIZE, PATCH_SIZE,
                         ws_patch);
                auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
                std::vector<EValue> in1 = {*pTensor};
                if (useVulkanBackend && !validateVulkanInputs(in1)) { /* log already in validate */ }
                auto out1 = m1->forward(in1);
                if (!out1.ok() || out1->size() < 2) {
                    LOGE("Part1 fail 1x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const auto& tokT = (*out1)[0].toTensor();
                const auto& spaT = (*out1)[1].toTensor();
                const float* tokPtr = tokT.const_data_ptr<float>();
                const float* spaPtr = spaT.const_data_ptr<float>();
                reshapeToSpatial(spaPtr, spaT.numel(), ws_temp);
                mergeCrop(g_workspace.latent0, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                reshapeToSpatial(tokPtr, tokT.numel(), ws_temp);
                mergeCrop(g_workspace.latent1, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
                std::memcpy(ws_tokens, tokPtr, tokenSliceSz * sizeof(float));
                auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
                std::vector<EValue> in2 = {*tokInput};
                auto out2 = m2->forward(in2);
                if (!out2.ok() || out2->empty()) {
                    LOGE("Part2 fail 1x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
                reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
                mergeCrop(g_workspace.x0Feat, M_1X, ws_temp, jj, ii, GRID_1X, PADDING_1X, ROW_OFFS_1X, COL_OFFS_1X);
            }
        }
    }
    LOGD("1x patches (25): %lldms", nowMs() - t1x);

    // ── 0.5x patches (3x3 = 9): batch-2 Vulkan FP16 or batch-4 INT8 when available ─
    long long t05x = nowMs();
    int batchSize05x = 1;
    if (useVulkanBackend && m1_b2 && m2_b2) batchSize05x = PATCH_BATCH_2;
    else if (!useVulkanBackend && m1_b4 && m2_b4 && useB4) batchSize05x = PATCH_BATCH;
    ETModule* m1_batch_05 = (batchSize05x == 2) ? m1_b2 : ((batchSize05x == 4) ? m1_b4 : nullptr);
    ETModule* m2_batch_05 = (batchSize05x == 2) ? m2_b2 : ((batchSize05x == 4) ? m2_b4 : nullptr);

    for (int start = 0; start < 9; start += batchSize05x) {
        const int n = std::min(batchSize05x, 9 - start);
        const bool useBatch = (n == batchSize05x && m1_batch_05 && m2_batch_05);

        if (useBatch) {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_05X;
                const int jj = (start + b) % GRID_05X;
                cropNCHW(g_workspace.halfImg, HALF_SIZE, HALF_SIZE, 3,
                         ii * STRIDE_05X, jj * STRIDE_05X, PATCH_SIZE, PATCH_SIZE,
                         g_workspace.patchBuf4 + b * patchSz);
            }
            auto pTensor = from_blob(g_workspace.patchBuf4, {n, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*pTensor};
            if (useVulkanBackend && validateVulkanInputs(in1)) { /* optional */ }
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
            if (useVulkanBackend && validateVulkanInputs(in2)) { /* optional */ }
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
        } else {
            for (int b = 0; b < n; ++b) {
                const int ii = (start + b) / GRID_05X;
                const int jj = (start + b) % GRID_05X;
                cropNCHW(g_workspace.halfImg, HALF_SIZE, HALF_SIZE, 3,
                         ii * STRIDE_05X, jj * STRIDE_05X, PATCH_SIZE, PATCH_SIZE,
                         ws_patch);
                auto pTensor = from_blob(ws_patch, {1, 3, PATCH_SIZE, PATCH_SIZE});
                std::vector<EValue> in1 = {*pTensor};
                auto out1 = m1->forward(in1);
                if (!out1.ok() || out1->size() < 1) {
                    LOGE("Part1 fail 0.5x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const float* tokPtr = (*out1)[0].toTensor().const_data_ptr<float>();
                std::memcpy(ws_tokens, tokPtr, tokenSliceSz * sizeof(float));
                auto tokInput = from_blob(ws_tokens, {1, TOKENS_577, FEATURE_DIM});
                std::vector<EValue> in2 = {*tokInput};
                auto out2 = m2->forward(in2);
                if (!out2.ok() || out2->empty()) {
                    LOGE("Part2 fail 0.5x (%d,%d)", ii, jj);
                    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                    return nullptr;
                }
                const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
                reshapeToSpatial(featPtr, (*out2)[0].toTensor().numel(), ws_temp);
                mergeCrop(g_workspace.x1Feat, M_05X, ws_temp, jj, ii, GRID_05X, PADDING_05X, ROW_OFFS_05X, COL_OFFS_05X);
            }
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
    std::string path3 = useVulkanBackend
        ? pathJoin(modelDir, "sharp_split_part3_vulkan_fp16.pte")
        : pathJoin(modelDir, "sharp_split_part3_int8.pte");
    { std::ifstream f(path3); if (!f.good() && useVulkanBackend) path3 = pathJoin(modelDir, "sharp_split_part3_int8.pte"); }
    auto mod3 = std::make_unique<ETModule>(path3, ETModule::LoadMode::Mmap);
    if (mod3->load() != Error::Ok) {
        LOGE("Part3 load fail: %s", path3.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto fullImgTensor = from_blob(imageData, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
    std::vector<EValue> in3 = {*fullImgTensor};
    if (useVulkanBackend && !validateVulkanInputs(in3)) { /* optional */ }
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

    // ── Part4b: try batched tiled → sequential tiled → single ─────────────────
    // Release Part1+Part2 cache before Part4b to free ~1GB RSS; cache will be re-created on next call.
    g_moduleCache.release();
    LOGD("Released Part1+Part2 cache before Part4b to free memory");

    // When Stable mode (prefer single Part4b) is ON, skip tiled paths to avoid tile-boundary patches.
    std::vector<float> tiledGaussians;
    if (!useSinglePart4bOnly && runPart4bBatchedTiledPipeline(modelDir,
                                      imageData,
                                      g_workspace.latent0,
                                      g_workspace.latent1,
                                      g_workspace.x0Feat,
                                      g_workspace.x1Feat,
                                      g_workspace.x2Feat,
                                      combinedTokens.data(),
                                      imgTokensNumel,
                                      tiledGaussians)) {
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

    // 4b(b): sequential tiled path using tile_full (or tile_00) if present (skip when Stable mode).
    if (!useSinglePart4bOnly && runPart4bTiledFullPipeline(modelDir,
                                   imageData,
                                   g_workspace.latent0,
                                   g_workspace.latent1,
                                   g_workspace.x0Feat,
                                   g_workspace.x1Feat,
                                   g_workspace.x2Feat,
                                   combinedTokens.data(),
                                   imgTokensNumel,
                                   tiledGaussians)) {
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

    // 4b(c): single full Part4b as fallback.
    // Prefer FP16, then FP32, then INT8 (INT8 single can crash on some runtimes).
    long long tP4b = nowMs();
    std::string path4b;
    bool singleIsInt8Only = false;
    {
        std::string fp16Path = pathJoin(modelDir, "sharp_split_part4b_fp16.pte");
        std::ifstream f(fp16Path);
        if (f.good()) {
            f.close();
            path4b = fp16Path;
            LOGD("Part4b single: FP16 (sharp_split_part4b_fp16.pte)");
        }
    }
    if (path4b.empty()) {
        std::string fp32Path = pathJoin(modelDir, "sharp_split_part4b.pte");
        std::ifstream f(fp32Path);
        if (f.good()) {
            f.close();
            path4b = fp32Path;
            LOGD("Part4b single: FP32 (sharp_split_part4b.pte)");
        }
    }
    if (path4b.empty()) {
        std::string int8Path = pathJoin(modelDir, "sharp_split_part4b_int8.pte");
        std::ifstream f(int8Path);
        if (f.good()) {
            f.close();
            path4b = int8Path;
            singleIsInt8Only = true;
            LOGD("Part4b single: INT8 (sharp_split_part4b_int8.pte)");
        }
    }
    // When Stable mode requested but only INT8 single is available, use tiled path instead to avoid crash.
    if (useSinglePart4bOnly && singleIsInt8Only &&
        (runPart4bBatchedTiledPipeline(modelDir, imageData,
                                      g_workspace.latent0, g_workspace.latent1,
                                      g_workspace.x0Feat, g_workspace.x1Feat, g_workspace.x2Feat,
                                      combinedTokens.data(), imgTokensNumel, tiledGaussians) ||
         runPart4bTiledFullPipeline(modelDir, imageData,
                                   g_workspace.latent0, g_workspace.latent1,
                                   g_workspace.x0Feat, g_workspace.x1Feat, g_workspace.x2Feat,
                                   combinedTokens.data(), imgTokensNumel, tiledGaussians))) {
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        pruneGaussiansByOpacity(tiledGaussians, maxG);
        const int numFloats = static_cast<int>(tiledGaussians.size());
        if (numFloats > 0 && (numFloats % PARAMS_PER_GAUSSIAN) == 0) {
            jfloatArray jResult = env->NewFloatArray(numFloats);
            if (jResult) {
                env->SetFloatArrayRegion(jResult, 0, numFloats, tiledGaussians.data());
                LOGD("JNI RETURN: size=%d validated (Stable+INT8 tiled)", numFloats);
                return jResult;
            }
        }
    }
    if (path4b.empty()) {
        LOGE("Part4b single: no suitable .pte found in %s", modelDir.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
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

    LOGD("Part4b forward starting (single fallback)...");
    auto out4b = mod4b->forward(in4b);

    if (!out4b.ok() || out4b->empty()) {
        LOGE("Part4b forward fail");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    const auto& gaussianTensor = (*out4b)[0].toTensor();
    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);

    // AOT/Vulkan: never trust tensor metadata for size. Validate then use validated count only.
    if (gaussianTensor.dim() != 3) {
        LOGE("Part4b AOT shape: dim=%d expected 3 [1,N,14]", gaussianTensor.dim());
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

    // CRITICAL: copy using validated size only (ignore possibly corrupted numel for copy length).
    const float* gaussianPtr = gaussianTensor.const_data_ptr<float>();
    std::vector<float> safeCopy(validatedNumFloats);
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

} // extern "C"
