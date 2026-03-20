#include "sharp_executorch_full_internal.h"

// ExecuTorch Error enum values are hex-sized (see executorch/runtime/core/error.h): e.g. 18 = 0x12.
const char* executorchErrorStr(int err) {
    switch (err) {
        case 0x12: return "InvalidArgument(0x12)"; // 18 decimal — bad inputs / shapes vs export
        case 0x14: return "OperatorMissing(0x14)"; // 20 decimal
        case 0x20: return "NotFound(0x20)";         // 32 decimal
        case 0x32: return "DelegateInvalidHandle(0x32)"; // 50 decimal
        default: return "";
    }
}

namespace {
constexpr size_t kAlign = 32;
float* alignedAlloc(size_t count) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, kAlign, count * sizeof(float)) != 0) return nullptr;
    return static_cast<float*>(ptr);
}
void alignedFree(float* p) { free(p); }
}

bool Workspace::allocate() {
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

void Workspace::zero() {
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

void Workspace::release() {
        alignedFree(latent0);   alignedFree(latent1);   alignedFree(x0Feat);
        alignedFree(x1Feat);    alignedFree(x2Feat);    alignedFree(tempSpatial);
        alignedFree(halfImg);   alignedFree(quarterImg); alignedFree(patchBuf);
        alignedFree(patchBuf4); alignedFree(tokensCopy); alignedFree(tokensCopy4);
        latent0 = latent1 = x0Feat = x1Feat = x2Feat = tempSpatial = nullptr;
        halfImg = quarterImg = patchBuf = patchBuf4 = tokensCopy = tokensCopy4 = nullptr;
        allocated = false;
    }


bool ModuleCache::ensureLoaded(const std::string& dir, bool useVulkan) {
    std::lock_guard<std::mutex> lock(mu);
    if (mod1 && mod2 && modelDir == dir && lastUseVulkan == useVulkan) {
        LOGD("ModuleCache: using cached Part1+Part2 (no reload)");
        return true;
    }
    mod1.reset(); mod2.reset(); mod1_b4.reset(); mod2_b4.reset(); mod1_b2.reset(); mod2_b2.reset();
    modelDir = dir;
    lastUseVulkan = useVulkan;
    if (useVulkan) {
        return moduleCacheLoadPart12Vulkan(*this, dir);
    }
    return moduleCacheLoadPart12Cpu(*this, dir);
}

void ModuleCache::release() {
        std::lock_guard<std::mutex> lock(mu);
        mod1.reset(); mod2.reset(); mod1_b4.reset(); mod2_b4.reset(); mod1_b2.reset(); mod2_b2.reset();
        modelDir.clear();
        LOGD("ModuleCache: released Part1+Part2");
    }


ModuleCache g_moduleCache;
Workspace g_workspace;
bool g_runtime_initialized = false;

void configureCpuInferenceThreadsOnce() {
    static std::once_flag once;
    std::call_once(once, []() {
        long online = sysconf(_SC_NPROCESSORS_ONLN);
        if (online < 1) {
            online = 4;
        }
        long useThreads = online;
        if (useThreads > 8) {
            useThreads = 8;
        }
        char buf[32];
        snprintf(buf, sizeof(buf), "%ld", useThreads);
        setenv("OMP_NUM_THREADS", buf, 1);
        setenv("MKL_NUM_THREADS", buf, 1);
        setenv("OPENBLAS_NUM_THREADS", buf, 1);
        setenv("VECLIB_MAXIMUM_THREADS", buf, 1);
        LOGI("CPU inference: OMP_NUM_THREADS=%s (processors online=%ld)", buf, online);
    });
}

void ensureRuntimeInit() {
    if (!g_runtime_initialized) {
        configureCpuInferenceThreadsOnce();
        executorch::runtime::runtime_init();
        g_runtime_initialized = true;
    }
}

void downsample2x(const float* __restrict src, int H, int W, int C,
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
void downsample4x(const float* __restrict src, int H, int W, int C,
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
void cropNCHW(const float* __restrict src, int srcH, int srcW, int C,
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
void reshapeToSpatial(const float* __restrict tokens, size_t tokenLen,
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
void mergeCrop(float* __restrict out, int outW,
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

std::string pathJoin(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

// ── Helper: BATCHED tiled Part4b (batch=4, 4 forward calls for 16 tiles) ─────
// Loads sharp_split_part4b_tile_b4.pte (exported with batch=4 static shape).
// Crops 4 tiles at a time, stacks along batch dim, one forward, split output.
bool runPart4bBatchedTiledPipeline(
        const std::string& modelDir,
        const float* __restrict imageData,   // [3,1536,1536] NCHW
        const float* __restrict latent0,     // [1024,96,96]
        const float* __restrict latent1,     // [1024,96,96]
        const float* __restrict x0Feat,      // [1024,96,96]
        const float* __restrict x1Feat,      // [1024,48,48]
        const float* __restrict x2Feat,      // [1024,24,24]
        const float* __restrict combinedTokens, size_t tokensNumel,
        bool swapTileNdcXY,
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
            correctNDC(dst, gaussiansPerTile, tileRow, tileCol, swapTileNdcXY);
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
bool runPart4bTiledFullPipeline(
        const std::string& modelDir,
        const float* __restrict imageData,   // [3,1536,1536] NCHW
        const float* __restrict latent0,     // [1024,96,96]
        const float* __restrict latent1,     // [1024,96,96]
        const float* __restrict x0Feat,      // [1024,96,96]
        const float* __restrict x1Feat,      // [1024,48,48]
        const float* __restrict x2Feat,      // [1024,24,24]
        const float* __restrict combinedTokens, size_t tokensNumel,
        bool swapTileNdcXY,
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
    std::string candidate = pathJoin(modelDir, tileFullFp32);
    std::ifstream f1(candidate);
    if (f1.good()) {
        f1.close();
        modelPath = candidate;
        LOGD("runPart4bTiledFullPipeline: using %s quant_hint=%s swap_ndc=%d",
             modelPath.c_str(),
             (modelPath.find("int8") != std::string::npos) ? "int8" : "fp32",
             swapTileNdcXY ? 1 : 0);
    } else {
        f1.close();
        candidate = pathJoin(modelDir, tile00Fp32);
        std::ifstream f2(candidate);
        if (!f2.good()) {
            LOGD("runPart4bTiledFullPipeline: no tile_full or tile_00 found, skipping tiled path");
            return false;
        }
        f2.close();
        modelPath = candidate;
        LOGD("runPart4bTiledFullPipeline: using %s quant_hint=%s swap_ndc=%d",
             modelPath.c_str(),
             (modelPath.find("int8") != std::string::npos) ? "int8" : "fp32",
             swapTileNdcXY ? 1 : 0);
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
        correctNDC(dst, numGaussians, tileRow, tileCol, swapTileNdcXY);

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
void pruneGaussiansByOpacity(std::vector<float>& gaussians, int maxGaussians) {
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
int pruneGaussiansFromPtr(const float* src, int totalGaussians, int maxGaussians,
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
long long nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// While CPU/XNNPACK forward() runs (can be many minutes), log periodically so adb logcat shows life.
void cpuForwardHeartbeatLoop(std::atomic<bool>* keepGoing, const char* stageTag, int intervalSec) {
    int elapsed = 0;
    while (keepGoing->load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::seconds(intervalSec));
        if (!keepGoing->load(std::memory_order_acquire)) {
            break;
        }
        elapsed += intervalSec;
        LOGI("%s: XNNPACK forward still running (%ds elapsed) — if this increments, not deadlocked in app code", stageTag, elapsed);
    }
}

// Report progress to Kotlin so UI does not appear stuck at 20%. Part1+2 = 35 patches ≈ 50% of pipeline.
void reportProgress(JNIEnv* env, jobject reporter, jmethodID methodId, float progress, const char* message) {
    if (!reporter || !methodId || !env) return;
    jstring jmsg = env->NewStringUTF(message ? message : "");
    if (jmsg) {
        env->CallVoidMethod(reporter, methodId, static_cast<jfloat>(progress), jmsg);
        env->DeleteLocalRef(jmsg);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
}

// ── JNI: preload Part1+Part2 (call from Kotlin on init for warm start) ───────
extern "C" {

JNIEXPORT jboolean JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_preloadCppModules(
        JNIEnv* env, jobject, jstring modelDirPath, jboolean useVulkanForPart12) {
    ensureRuntimeInit();
    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string dir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (dir.empty()) return JNI_FALSE;
    return g_moduleCache.ensureLoaded(dir, (useVulkanForPart12 == JNI_TRUE)) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_releaseCppModules(
        JNIEnv*, jobject) {
    g_moduleCache.release();
    g_workspace.release();
    LOGD("Released module cache + workspace");
}


// ── JNI: full pipeline (dispatches to CPU-only or Vulkan TU) ─────────────────
JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_runFullPipelineInt8Native(
        JNIEnv* env,
        jobject thiz,
        jstring modelDirPath,
        jfloatArray imageNCHW,
        jint maxGaussians,
        jboolean preferSinglePart4b,
        jboolean useVulkan,
        jboolean part12OnCpu,
        jboolean part12ForceSinglePatch,
        jboolean part12_25Only,
        jint part1MaxPatches1x,
        jint part1MaxPatches05x,
        jint part12Chunk1x,
        jint part12Chunk05x,
        jint part12YieldMsBetweenChunks,
        jboolean swapTileNdcXY,
        jobject progressReporter) {
    if (useVulkan == JNI_TRUE) {
        return runSharpFullPipeline_Vulkan(
                env, thiz, modelDirPath, imageNCHW, maxGaussians, preferSinglePart4b, part12OnCpu,
                part12ForceSinglePatch, part12_25Only,
                part1MaxPatches1x, part1MaxPatches05x, part12Chunk1x, part12Chunk05x,
                part12YieldMsBetweenChunks, swapTileNdcXY, progressReporter);
    }
    return runSharpFullPipeline_Cpu(
            env, thiz, modelDirPath, imageNCHW, maxGaussians, preferSinglePart4b,
            part12ForceSinglePatch, part12_25Only,
            part1MaxPatches1x, part1MaxPatches05x, part12Chunk1x, part12Chunk05x,
            part12YieldMsBetweenChunks, swapTileNdcXY, progressReporter);
}

/** Set OMP thread env as soon as this .so loads — before Java/ExecuTorch runs any other native init. */
JNIEXPORT jint JNICALL JNI_OnLoad(JavaVM*, void*) {
    configureCpuInferenceThreadsOnce();
    return JNI_VERSION_1_6;
}

} // extern "C"
