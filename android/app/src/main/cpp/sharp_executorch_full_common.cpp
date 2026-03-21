#include "sharp_executorch_full_internal.h"

static std::atomic<int> g_sharp_exec_verbose{0};

bool sharpExecNativeVerboseLogsEnabled() {
    return g_sharp_exec_verbose.load(std::memory_order_relaxed) != 0;
}

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

long readStatusKb(const char* key) {
    std::ifstream f("/proc/self/status");
    if (!f.good()) return -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            long valueKb = -1;
            if (sscanf(line.c_str() + strlen(key), "%ld", &valueKb) == 1) {
                return valueKb;
            }
        }
    }
    return -1;
}

long readMemAvailableKb() {
    std::ifstream f("/proc/meminfo");
    if (!f.good()) return -1;
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) == 0) {
            long valueKb = -1;
            if (sscanf(line.c_str() + strlen("MemAvailable:"), "%ld", &valueKb) == 1) {
                return valueKb;
            }
        }
    }
    return -1;
}

size_t workspaceBytesEstimate(const Workspace& ws) {
    size_t total = 0;
    if (ws.latent0) total += (size_t)FEATURE_DIM * M_1X * M_1X * sizeof(float);
    if (ws.latent1) total += (size_t)FEATURE_DIM * M_1X * M_1X * sizeof(float);
    if (ws.x0Feat) total += (size_t)FEATURE_DIM * M_1X * M_1X * sizeof(float);
    if (ws.x1Feat) total += (size_t)FEATURE_DIM * M_05X * M_05X * sizeof(float);
    if (ws.x2Feat) total += (size_t)FEATURE_DIM * SPATIAL_HW * sizeof(float);
    if (ws.tempSpatial) total += (size_t)FEATURE_DIM * SPATIAL_HW * sizeof(float);
    if (ws.halfImg) total += (size_t)3 * HALF_SIZE * HALF_SIZE * sizeof(float);
    if (ws.quarterImg) total += (size_t)3 * PATCH_SIZE * PATCH_SIZE * sizeof(float);
    if (ws.patchBuf) total += (size_t)3 * PATCH_SIZE * PATCH_SIZE * sizeof(float);
    if (ws.patchBuf4) total += (size_t)PATCH_BATCH * 3 * PATCH_SIZE * PATCH_SIZE * sizeof(float);
    if (ws.tokensCopy) total += (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float);
    if (ws.tokensCopy4) total += (size_t)PATCH_BATCH * TOKENS_577 * FEATURE_DIM * sizeof(float);
    return total;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

struct OwnedTensor {
    std::vector<float> data;
    std::vector<executorch::aten::SizesType> sizes;
};

bool copyTensorToOwned(const executorch::runtime::etensor::Tensor& tensor,
                       OwnedTensor& dst,
                       const char* label) {
    const float* src = tensor.const_data_ptr<float>();
    if (!src) {
        LOGE("%s: tensor data null", label);
        return false;
    }
    const ssize_t numel = tensor.numel();
    if (numel <= 0) {
        LOGE("%s: tensor numel invalid: %ld", label, (long)numel);
        return false;
    }
    dst.sizes.assign(tensor.sizes().begin(), tensor.sizes().end());
    dst.data.resize(static_cast<size_t>(numel));
    std::memcpy(dst.data.data(), src, static_cast<size_t>(numel) * sizeof(float));
    return true;
}

bool copyOutputTensor(const std::vector<EValue>& outputs,
                      size_t index,
                      OwnedTensor& dst,
                      const char* label) {
    if (index >= outputs.size() || !outputs[index].isTensor()) {
        LOGE("%s: missing tensor output at index %zu", label, index);
        return false;
    }
    return copyTensorToOwned(outputs[index].toTensor(), dst, label);
}

bool validateOwnedTensorShape(const OwnedTensor& tensor,
                              std::initializer_list<executorch::aten::SizesType> expected,
                              const char* label) {
    if (tensor.sizes.size() != expected.size()) {
        LOGE("%s: rank mismatch got=%zu expected=%zu", label, tensor.sizes.size(), expected.size());
        return false;
    }
    size_t dim = 0;
    for (const auto expectedDim : expected) {
        if (tensor.sizes[dim] != expectedDim) {
            LOGE("%s: dim[%zu] mismatch got=%ld expected=%ld",
                 label,
                 dim,
                 static_cast<long>(tensor.sizes[dim]),
                 static_cast<long>(expectedDim));
            return false;
        }
        ++dim;
    }
    return true;
}

template <typename ForwardCallable>
auto runForwardWithHeartbeat(ForwardCallable forwardCallable,
                             const std::string& stageTag,
                             int intervalSec = 15) {
    std::atomic<bool> keepGoing(true);
    std::mutex heartbeatMu;
    std::condition_variable heartbeatCv;
    const long long tStart = nowMs();
    std::thread heartbeat([&keepGoing, &heartbeatMu, &heartbeatCv, stageTag, intervalSec]() {
        int elapsed = 0;
        while (keepGoing.load(std::memory_order_acquire)) {
            std::unique_lock<std::mutex> lock(heartbeatMu);
            if (heartbeatCv.wait_for(
                    lock,
                    std::chrono::seconds(intervalSec),
                    [&keepGoing]() { return !keepGoing.load(std::memory_order_acquire); })) {
                break;
            }
            elapsed += intervalSec;
            LOGI("%s: forward still running (%ds elapsed)", stageTag.c_str(), elapsed);
        }
    });

    auto stopHeartbeat = [&keepGoing, &heartbeat, &heartbeatCv]() {
        keepGoing.store(false, std::memory_order_release);
        heartbeatCv.notify_all();
        if (heartbeat.joinable()) {
            heartbeat.join();
        }
    };

    try {
        auto result = forwardCallable();
        stopHeartbeat();
        LOGI("%s: forward finished in %lldms", stageTag.c_str(), nowMs() - tStart);
        return result;
    } catch (...) {
        stopHeartbeat();
        throw;
    }
}
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

void Workspace::releaseEncoderScratch() {
        alignedFree(tempSpatial); tempSpatial = nullptr;
        alignedFree(halfImg); halfImg = nullptr;
        alignedFree(quarterImg); quarterImg = nullptr;
        alignedFree(patchBuf); patchBuf = nullptr;
        alignedFree(patchBuf4); patchBuf4 = nullptr;
        alignedFree(tokensCopy); tokensCopy = nullptr;
        alignedFree(tokensCopy4); tokensCopy4 = nullptr;
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

std::string processMemorySummary() {
    const long rssKb = readStatusKb("VmRSS:");
    const long hwmKb = readStatusKb("VmHWM:");
    const long vmSizeKb = readStatusKb("VmSize:");
    const long availKb = readMemAvailableKb();
    const size_t workspaceBytes = workspaceBytesEstimate(g_workspace);
    char buf[256];
    snprintf(
            buf,
            sizeof(buf),
            "rss=%ldMB hwm=%ldMB vmsize=%ldMB avail=%ldMB workspace_est=%zuMB",
            rssKb >= 0 ? rssKb / 1024 : -1L,
            hwmKb >= 0 ? hwmKb / 1024 : -1L,
            vmSizeKb >= 0 ? vmSizeKb / 1024 : -1L,
            availKb >= 0 ? availKb / 1024 : -1L,
            workspaceBytes / (1024 * 1024));
    return std::string(buf);
}

void logProcessMemory(const char* stage) {
    LOGD("[MEM] %s %s", stage, processMemorySummary().c_str());
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

// ── Helper: BATCHED tiled Part4b (batch=2 or batch=4 static exports) ─────────
// Loads sharp_split_part4b_tile_b2.pte or sharp_split_part4b_tile_b4.pte.
// Crops tiles in small batches, stacks along batch dim, one forward, split output.
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
    const int imgH = IMAGE_SIZE, imgW = IMAGE_SIZE, imgC = 3;
    const int imgTileH = imgH / GRID, imgTileW = imgW / GRID;
    const int lat96H = 96, lat96W = 96;
    const int x1H = 48, x1W = 48;
    const int x2H = 24, x2W = 24;
    const int xLowH = 24, xLowW = 24, xLowC = 1024;

    const std::string fineTile00StagePre = pathJoin(modelDir, "sharp_split_part4b_tile_00_stage_pre_vulkan.pte");
    const std::string fineTile00DecoderHead = pathJoin(modelDir, "sharp_split_part4b_tile_00_decoder_head.pte");
    const std::string fineTile00InitBase = pathJoin(modelDir, "sharp_split_part4b_tile_00_init_base.pte");
    const std::string fineTile00RawHeads = pathJoin(modelDir, "sharp_split_part4b_tile_00_raw_heads_vulkan.pte");
    const std::string fineTile00Compose = pathJoin(modelDir, "sharp_split_part4b_tile_00_compose.pte");
    const bool hasFineSplitTile00 =
            fileExists(fineTile00StagePre) &&
            fileExists(fineTile00DecoderHead) &&
            fileExists(fineTile00InitBase) &&
            fileExists(fineTile00RawHeads) &&
            fileExists(fineTile00Compose);
    if (hasFineSplitTile00) {
        LOGD("runPart4bBatchedTiledPipeline: fine split tile_00 artifacts present, preferring sequential tile_00 path");
        return false;
    }

    const std::string splitB2StagePre = pathJoin(modelDir, "sharp_split_part4b_tile_b2_stage_pre_vulkan.pte");
    const std::string splitB2DecoderHead = pathJoin(modelDir, "sharp_split_part4b_tile_b2_decoder_head.pte");
    const std::string splitB2StageA = pathJoin(modelDir, "sharp_split_part4b_tile_b2_stage_a_vulkan.pte");
    const std::string splitB2InitBase = pathJoin(modelDir, "sharp_split_part4b_tile_b2_init_base.pte");
    const std::string splitB2RawHeads = pathJoin(modelDir, "sharp_split_part4b_tile_b2_raw_heads_vulkan.pte");
    const std::string splitB2Compose = pathJoin(modelDir, "sharp_split_part4b_tile_b2_compose.pte");
    const bool hasFineSplitTileB2 =
            fileExists(splitB2StagePre) &&
            fileExists(splitB2DecoderHead) &&
            fileExists(splitB2InitBase) &&
            fileExists(splitB2RawHeads) &&
            fileExists(splitB2Compose);
    const bool hasSplitTileB2 =
            fileExists(splitB2StageA) &&
            fileExists(splitB2InitBase) &&
            fileExists(splitB2RawHeads) &&
            fileExists(splitB2Compose);

    std::string modelPath;
    int batch = 0;
    bool useFineSplitTileB2 = false;
    bool useSplitTileB2 = false;
    if (hasFineSplitTileB2) {
        batch = 2;
        useFineSplitTileB2 = true;
    } else if (hasSplitTileB2) {
        batch = 2;
        useSplitTileB2 = true;
    } else {
        const std::string pathB2 = modelDir + "/sharp_split_part4b_tile_b2.pte";
        const std::string pathB4 = modelDir + "/sharp_split_part4b_tile_b4.pte";
        std::ifstream f2(pathB2);
        if (f2.good()) {
            modelPath = pathB2;
            batch = 2;
        } else {
            std::ifstream f4(pathB4);
            if (f4.good()) {
                modelPath = pathB4;
                batch = 4;
            } else {
                LOGD("runPart4bBatchedTiledPipeline: no tile_b2/tile_b4 found, skipping batched path");
                return false;
            }
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
    logProcessMemory("Part4b batched tiled: before module load");
    std::unique_ptr<ETModule> module;
    std::unique_ptr<ETModule> splitStagePreModule;
    std::unique_ptr<ETModule> splitDecoderHeadModule;
    std::unique_ptr<ETModule> splitStageAModule;
    std::unique_ptr<ETModule> splitInitBaseModule;
    std::unique_ptr<ETModule> splitRawHeadsModule;
    std::unique_ptr<ETModule> splitComposeModule;
    if (useFineSplitTileB2) {
        splitStagePreModule = std::make_unique<ETModule>(splitB2StagePre, ETModule::LoadMode::Mmap);
        splitDecoderHeadModule = std::make_unique<ETModule>(splitB2DecoderHead, ETModule::LoadMode::Mmap);
        splitInitBaseModule = std::make_unique<ETModule>(splitB2InitBase, ETModule::LoadMode::Mmap);
        splitRawHeadsModule = std::make_unique<ETModule>(splitB2RawHeads, ETModule::LoadMode::Mmap);
        splitComposeModule = std::make_unique<ETModule>(splitB2Compose, ETModule::LoadMode::Mmap);
        const bool loaded =
                splitStagePreModule->load() == Error::Ok &&
                splitDecoderHeadModule->load() == Error::Ok &&
                splitInitBaseModule->load() == Error::Ok &&
                splitRawHeadsModule->load() == Error::Ok &&
                splitComposeModule->load() == Error::Ok;
        if (!loaded) {
            LOGW("runPart4bBatchedTiledPipeline: fine split tile_b2 load failed, falling back to split tile_b2");
            splitStagePreModule.reset();
            splitDecoderHeadModule.reset();
            splitInitBaseModule.reset();
            splitRawHeadsModule.reset();
            splitComposeModule.reset();
            useFineSplitTileB2 = false;
            useSplitTileB2 = hasSplitTileB2;
            if (!useSplitTileB2) {
                const std::string pathB2 = modelDir + "/sharp_split_part4b_tile_b2.pte";
                const std::string pathB4 = modelDir + "/sharp_split_part4b_tile_b4.pte";
                std::ifstream f2(pathB2);
                if (f2.good()) {
                    modelPath = pathB2;
                    batch = 2;
                } else {
                    std::ifstream f4(pathB4);
                    if (!f4.good()) {
                        LOGE("runPart4bBatchedTiledPipeline: no fallback tile_b2/tile_b4 available after fine split load failure");
                        return false;
                    }
                    modelPath = pathB4;
                    batch = 4;
                }
            }
        } else {
            LOGD("runPart4bBatchedTiledPipeline: loaded fine split tile_b2 modules (batch=%d)", batch);
        }
    }
    if (useSplitTileB2 && !useFineSplitTileB2) {
        splitStageAModule = std::make_unique<ETModule>(splitB2StageA, ETModule::LoadMode::Mmap);
        splitInitBaseModule = std::make_unique<ETModule>(splitB2InitBase, ETModule::LoadMode::Mmap);
        splitRawHeadsModule = std::make_unique<ETModule>(splitB2RawHeads, ETModule::LoadMode::Mmap);
        splitComposeModule = std::make_unique<ETModule>(splitB2Compose, ETModule::LoadMode::Mmap);
        const bool loaded =
                splitStageAModule->load() == Error::Ok &&
                splitInitBaseModule->load() == Error::Ok &&
                splitRawHeadsModule->load() == Error::Ok &&
                splitComposeModule->load() == Error::Ok;
        if (!loaded) {
            LOGW("runPart4bBatchedTiledPipeline: split tile_b2 load failed, falling back to legacy tile_b2/tile_b4");
            splitStageAModule.reset();
            splitInitBaseModule.reset();
            splitRawHeadsModule.reset();
            splitComposeModule.reset();
            useSplitTileB2 = false;
            const std::string pathB2 = modelDir + "/sharp_split_part4b_tile_b2.pte";
            const std::string pathB4 = modelDir + "/sharp_split_part4b_tile_b4.pte";
            std::ifstream f2(pathB2);
            if (f2.good()) {
                modelPath = pathB2;
                batch = 2;
            } else {
                std::ifstream f4(pathB4);
                if (!f4.good()) {
                    LOGE("runPart4bBatchedTiledPipeline: no legacy tile_b2/tile_b4 fallback available after split load failure");
                    return false;
                }
                modelPath = pathB4;
                batch = 4;
            }
        } else {
            LOGD("runPart4bBatchedTiledPipeline: loaded split tile_b2 modules (batch=%d)", batch);
        }
    }
    if (!useFineSplitTileB2 && !useSplitTileB2) {
        module = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
        if (module->load() != Error::Ok) {
            LOGE("runPart4bBatchedTiledPipeline: failed to load %s", modelPath.c_str());
            return false;
        }
        LOGD("runPart4bBatchedTiledPipeline: loaded %s (batch=%d)", modelPath.c_str(), batch);
    }
    logProcessMemory("Part4b batched tiled: after module load");
    const int numBatches = NUM_TILES / batch;

    const int imgCropSize  = imgC * imgTileH * imgTileW;
    const int lat96CropSz  = 1024 * 24 * 24;
    const int x1CropSz     = 1024 * 12 * 12;
    const int x2CropSz     = 1024 * 6 * 6;

    std::vector<float> batchImg(batch * imgCropSize);
    std::vector<float> batchLat0(batch * lat96CropSz), batchLat1(batch * lat96CropSz), batchX0(batch * lat96CropSz);
    std::vector<float> batchX1(batch * x1CropSz), batchX2(batch * x2CropSz), batchXl(batch * x2CropSz);

    outGaussians.clear();
    int floatsPerTile = 0;

    auto tStart = std::chrono::steady_clock::now();
    for (int b = 0; b < numBatches; ++b) {
        auto batchStart = std::chrono::steady_clock::now();
        LOGD("runPart4bBatchedTiledPipeline: batch %d/%d begin mode=%s batch=%d",
             b + 1, numBatches, useFineSplitTileB2 ? "fine_split_tile_b2" : (useSplitTileB2 ? "split_tile_b2" : "legacy"), batch);

        for (int i = 0; i < batch; ++i) {
            const int t = b * batch + i;
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

        auto imgT  = from_blob(batchImg.data(),  {batch, 3, imgTileH, imgTileW});
        auto l0T   = from_blob(batchLat0.data(), {batch, 1024, 24, 24});
        auto l1T   = from_blob(batchLat1.data(), {batch, 1024, 24, 24});
        auto x0T   = from_blob(batchX0.data(),   {batch, 1024, 24, 24});
        auto x1T   = from_blob(batchX1.data(),   {batch, 1024, 12, 12});
        auto x2T   = from_blob(batchX2.data(),   {batch, 1024, 6, 6});
        auto xlT   = from_blob(batchXl.data(),   {batch, 1024, 6, 6});

        std::vector<EValue> inputs;
        inputs.reserve(7);
        inputs.emplace_back(*imgT);
        inputs.emplace_back(*l0T);
        inputs.emplace_back(*l1T);
        inputs.emplace_back(*x0T);
        inputs.emplace_back(*x1T);
        inputs.emplace_back(*x2T);
        inputs.emplace_back(*xlT);

        const float* outData = nullptr;
        int totalFloats = 0;
        std::vector<float> packedBatch;
        if (useFineSplitTileB2) {
            const std::string batchLabel =
                    "runPart4bBatchedTiledPipeline: batch " + std::to_string(b + 1) + "/" +
                    std::to_string(numBatches) + " fine split tile_b2";
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 stage_pre begin",
                 b + 1, numBatches);
            auto stagePreResult = runForwardWithHeartbeat(
                    [&]() { return splitStagePreModule->forward(inputs); },
                    batchLabel + " stage_pre");
            if (!stagePreResult.ok() || stagePreResult->size() < 5) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 stage_pre failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 stage_pre end outputs=%zu",
                 b + 1, numBatches, stagePreResult->size());

            OwnedTensor latent0Up;
            OwnedTensor latent1Up;
            OwnedTensor x0Up;
            OwnedTensor x1Up;
            OwnedTensor xFused;
            if (!copyOutputTensor(*stagePreResult, 0, latent0Up, "fine split tile_b2 latent0_up") ||
                !copyOutputTensor(*stagePreResult, 1, latent1Up, "fine split tile_b2 latent1_up") ||
                !copyOutputTensor(*stagePreResult, 2, x0Up, "fine split tile_b2 x0_up") ||
                !copyOutputTensor(*stagePreResult, 3, x1Up, "fine split tile_b2 x1_up") ||
                !copyOutputTensor(*stagePreResult, 4, xFused, "fine split tile_b2 x_fused")) {
                return false;
            }

            auto latent0UpTensor = from_blob(latent0Up.data.data(), latent0Up.sizes);
            auto latent1UpTensor = from_blob(latent1Up.data.data(), latent1Up.sizes);
            auto x0UpTensor = from_blob(x0Up.data.data(), x0Up.sizes);
            auto x1UpTensor = from_blob(x1Up.data.data(), x1Up.sizes);
            auto xFusedTensor = from_blob(xFused.data.data(), xFused.sizes);

            std::vector<EValue> decoderHeadInputs;
            decoderHeadInputs.reserve(5);
            decoderHeadInputs.emplace_back(*latent0UpTensor);
            decoderHeadInputs.emplace_back(*latent1UpTensor);
            decoderHeadInputs.emplace_back(*x0UpTensor);
            decoderHeadInputs.emplace_back(*x1UpTensor);
            decoderHeadInputs.emplace_back(*xFusedTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 decoder_head begin",
                 b + 1, numBatches);
            auto decoderHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitDecoderHeadModule->forward(decoderHeadInputs); },
                    batchLabel + " decoder_head");
            if (!decoderHeadResult.ok() || decoderHeadResult->size() < 2) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 decoder_head failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 decoder_head end outputs=%zu",
                 b + 1, numBatches, decoderHeadResult->size());

            OwnedTensor disparity;
            OwnedTensor decoderFeatures;
            if (!copyOutputTensor(*decoderHeadResult, 0, disparity, "fine split tile_b2 disparity") ||
                !copyOutputTensor(*decoderHeadResult, 1, decoderFeatures, "fine split tile_b2 decoder_features")) {
                return false;
            }

            auto disparityTensor = from_blob(disparity.data.data(), disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgT);
            initInputs.emplace_back(*disparityTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 init_base begin",
                 b + 1, numBatches);
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    batchLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 init_base failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 init_base end outputs=%zu",
                 b + 1, numBatches, initResult->size());

            OwnedTensor featureInput;
            if (!copyOutputTensor(*initResult, 0, featureInput, "fine split tile_b2 feature_input")) {
                return false;
            }

            auto featureInputTensor = from_blob(featureInput.data.data(), featureInput.sizes);
            auto decoderFeaturesTensor = from_blob(decoderFeatures.data.data(), decoderFeatures.sizes);

            std::vector<EValue> rawHeadInputs;
            rawHeadInputs.reserve(7);
            rawHeadInputs.emplace_back(*featureInputTensor);
            rawHeadInputs.emplace_back(*latent0UpTensor);
            rawHeadInputs.emplace_back(*latent1UpTensor);
            rawHeadInputs.emplace_back(*x0UpTensor);
            rawHeadInputs.emplace_back(*x1UpTensor);
            rawHeadInputs.emplace_back(*xFusedTensor);
            rawHeadInputs.emplace_back(*decoderFeaturesTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 raw_heads begin",
                 b + 1, numBatches);
            auto rawHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitRawHeadsModule->forward(rawHeadInputs); },
                    batchLabel + " raw_heads");
            if (!rawHeadResult.ok() || rawHeadResult->size() < 2) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 raw_heads failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 raw_heads end outputs=%zu",
                 b + 1, numBatches, rawHeadResult->size());

            OwnedTensor geometryRaw;
            OwnedTensor textureRaw;
            OwnedTensor meanXNdc;
            OwnedTensor meanYNdc;
            OwnedTensor meanInverseZNdc;
            OwnedTensor scales;
            OwnedTensor quaternions;
            OwnedTensor colors;
            OwnedTensor opacities;
            OwnedTensor globalScale;
            if (!copyOutputTensor(*rawHeadResult, 0, geometryRaw, "fine split tile_b2 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, textureRaw, "fine split tile_b2 texture_raw") ||
                !copyOutputTensor(*initResult, 1, meanXNdc, "fine split tile_b2 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, meanYNdc, "fine split tile_b2 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, meanInverseZNdc, "fine split tile_b2 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scales, "fine split tile_b2 scales") ||
                !copyOutputTensor(*initResult, 5, quaternions, "fine split tile_b2 quaternions") ||
                !copyOutputTensor(*initResult, 6, colors, "fine split tile_b2 colors") ||
                !copyOutputTensor(*initResult, 7, opacities, "fine split tile_b2 opacities") ||
                !copyOutputTensor(*initResult, 8, globalScale, "fine split tile_b2 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(geometryRaw.data.data(), geometryRaw.sizes);
            auto textureRawTensor = from_blob(textureRaw.data.data(), textureRaw.sizes);
            auto meanXNdcTensor = from_blob(meanXNdc.data.data(), meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(meanYNdc.data.data(), meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(meanInverseZNdc.data.data(), meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scales.data.data(), scales.sizes);
            auto quaternionsTensor = from_blob(quaternions.data.data(), quaternions.sizes);
            auto colorsTensor = from_blob(colors.data.data(), colors.sizes);
            auto opacitiesTensor = from_blob(opacities.data.data(), opacities.sizes);
            auto globalScaleTensor = from_blob(globalScale.data.data(), globalScale.sizes);

            std::vector<EValue> composeInputs;
            composeInputs.reserve(10);
            composeInputs.emplace_back(*geometryRawTensor);
            composeInputs.emplace_back(*textureRawTensor);
            composeInputs.emplace_back(*meanXNdcTensor);
            composeInputs.emplace_back(*meanYNdcTensor);
            composeInputs.emplace_back(*meanInverseZNdcTensor);
            composeInputs.emplace_back(*scalesTensor);
            composeInputs.emplace_back(*quaternionsTensor);
            composeInputs.emplace_back(*colorsTensor);
            composeInputs.emplace_back(*opacitiesTensor);
            composeInputs.emplace_back(*globalScaleTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 compose begin",
                 b + 1, numBatches);
            auto composeResult = runForwardWithHeartbeat(
                    [&]() { return splitComposeModule->forward(composeInputs); },
                    batchLabel + " compose");
            if (!composeResult.ok() || composeResult->empty() || !(*composeResult)[0].isTensor()) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 compose failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 compose end",
                 b + 1, numBatches);

            const auto& outTensor = (*composeResult)[0].toTensor();
            const float* packedPtr = outTensor.const_data_ptr<float>();
            if (!packedPtr) {
                LOGE("runPart4bBatchedTiledPipeline: fine split tile_b2 compose returned null data on batch %d", b);
                return false;
            }
            totalFloats = static_cast<int>(outTensor.numel());
            packedBatch.resize(static_cast<size_t>(totalFloats));
            std::memcpy(packedBatch.data(), packedPtr, static_cast<size_t>(totalFloats) * sizeof(float));
            outData = packedBatch.data();
        } else if (useSplitTileB2) {
            const std::string batchLabel =
                    "runPart4bBatchedTiledPipeline: batch " + std::to_string(b + 1) + "/" +
                    std::to_string(numBatches) + " split tile_b2";
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 stage_a begin",
                 b + 1, numBatches);
            auto stageAResult = runForwardWithHeartbeat(
                    [&]() { return splitStageAModule->forward(inputs); },
                    batchLabel + " stage_a");
            if (!stageAResult.ok() || stageAResult->size() < 7) {
                LOGE("runPart4bBatchedTiledPipeline: split tile_b2 stage_a failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 stage_a end outputs=%zu",
                 b + 1, numBatches, stageAResult->size());

            OwnedTensor disparity;
            if (!copyOutputTensor(*stageAResult, 0, disparity, "split tile_b2 disparity")) {
                return false;
            }
            auto disparityTensor = from_blob(disparity.data.data(), disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgT);
            initInputs.emplace_back(*disparityTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 init_base begin",
                 b + 1, numBatches);
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    batchLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bBatchedTiledPipeline: split tile_b2 init_base failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 init_base end outputs=%zu",
                 b + 1, numBatches, initResult->size());

            OwnedTensor featureInput;
            OwnedTensor latent0Up;
            OwnedTensor latent1Up;
            OwnedTensor x0Up;
            OwnedTensor x1Up;
            OwnedTensor xFused;
            OwnedTensor decoderFeatures;
            if (!copyOutputTensor(*initResult, 0, featureInput, "split tile_b2 feature_input") ||
                !copyOutputTensor(*stageAResult, 1, latent0Up, "split tile_b2 latent0_up") ||
                !copyOutputTensor(*stageAResult, 2, latent1Up, "split tile_b2 latent1_up") ||
                !copyOutputTensor(*stageAResult, 3, x0Up, "split tile_b2 x0_up") ||
                !copyOutputTensor(*stageAResult, 4, x1Up, "split tile_b2 x1_up") ||
                !copyOutputTensor(*stageAResult, 5, xFused, "split tile_b2 x_fused") ||
                !copyOutputTensor(*stageAResult, 6, decoderFeatures, "split tile_b2 decoder_features")) {
                return false;
            }

            auto featureInputTensor = from_blob(featureInput.data.data(), featureInput.sizes);
            auto latent0UpTensor = from_blob(latent0Up.data.data(), latent0Up.sizes);
            auto latent1UpTensor = from_blob(latent1Up.data.data(), latent1Up.sizes);
            auto x0UpTensor = from_blob(x0Up.data.data(), x0Up.sizes);
            auto x1UpTensor = from_blob(x1Up.data.data(), x1Up.sizes);
            auto xFusedTensor = from_blob(xFused.data.data(), xFused.sizes);
            auto decoderFeaturesTensor = from_blob(decoderFeatures.data.data(), decoderFeatures.sizes);

            std::vector<EValue> rawHeadInputs;
            rawHeadInputs.reserve(7);
            rawHeadInputs.emplace_back(*featureInputTensor);
            rawHeadInputs.emplace_back(*latent0UpTensor);
            rawHeadInputs.emplace_back(*latent1UpTensor);
            rawHeadInputs.emplace_back(*x0UpTensor);
            rawHeadInputs.emplace_back(*x1UpTensor);
            rawHeadInputs.emplace_back(*xFusedTensor);
            rawHeadInputs.emplace_back(*decoderFeaturesTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 raw_heads begin",
                 b + 1, numBatches);
            auto rawHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitRawHeadsModule->forward(rawHeadInputs); },
                    batchLabel + " raw_heads");
            if (!rawHeadResult.ok() || rawHeadResult->size() < 2) {
                LOGE("runPart4bBatchedTiledPipeline: split tile_b2 raw_heads failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 raw_heads end outputs=%zu",
                 b + 1, numBatches, rawHeadResult->size());

            OwnedTensor geometryRaw;
            OwnedTensor textureRaw;
            OwnedTensor meanXNdc;
            OwnedTensor meanYNdc;
            OwnedTensor meanInverseZNdc;
            OwnedTensor scales;
            OwnedTensor quaternions;
            OwnedTensor colors;
            OwnedTensor opacities;
            OwnedTensor globalScale;
            if (!copyOutputTensor(*rawHeadResult, 0, geometryRaw, "split tile_b2 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, textureRaw, "split tile_b2 texture_raw") ||
                !copyOutputTensor(*initResult, 1, meanXNdc, "split tile_b2 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, meanYNdc, "split tile_b2 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, meanInverseZNdc, "split tile_b2 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scales, "split tile_b2 scales") ||
                !copyOutputTensor(*initResult, 5, quaternions, "split tile_b2 quaternions") ||
                !copyOutputTensor(*initResult, 6, colors, "split tile_b2 colors") ||
                !copyOutputTensor(*initResult, 7, opacities, "split tile_b2 opacities") ||
                !copyOutputTensor(*initResult, 8, globalScale, "split tile_b2 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(geometryRaw.data.data(), geometryRaw.sizes);
            auto textureRawTensor = from_blob(textureRaw.data.data(), textureRaw.sizes);
            auto meanXNdcTensor = from_blob(meanXNdc.data.data(), meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(meanYNdc.data.data(), meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(meanInverseZNdc.data.data(), meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scales.data.data(), scales.sizes);
            auto quaternionsTensor = from_blob(quaternions.data.data(), quaternions.sizes);
            auto colorsTensor = from_blob(colors.data.data(), colors.sizes);
            auto opacitiesTensor = from_blob(opacities.data.data(), opacities.sizes);
            auto globalScaleTensor = from_blob(globalScale.data.data(), globalScale.sizes);

            std::vector<EValue> composeInputs;
            composeInputs.reserve(10);
            composeInputs.emplace_back(*geometryRawTensor);
            composeInputs.emplace_back(*textureRawTensor);
            composeInputs.emplace_back(*meanXNdcTensor);
            composeInputs.emplace_back(*meanYNdcTensor);
            composeInputs.emplace_back(*meanInverseZNdcTensor);
            composeInputs.emplace_back(*scalesTensor);
            composeInputs.emplace_back(*quaternionsTensor);
            composeInputs.emplace_back(*colorsTensor);
            composeInputs.emplace_back(*opacitiesTensor);
            composeInputs.emplace_back(*globalScaleTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 compose begin",
                 b + 1, numBatches);
            auto composeResult = runForwardWithHeartbeat(
                    [&]() { return splitComposeModule->forward(composeInputs); },
                    batchLabel + " compose");
            if (!composeResult.ok() || composeResult->empty() || !(*composeResult)[0].isTensor()) {
                LOGE("runPart4bBatchedTiledPipeline: split tile_b2 compose failed on batch %d", b);
                return false;
            }
            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 compose end",
                 b + 1, numBatches);

            const auto& outTensor = (*composeResult)[0].toTensor();
            const float* packedPtr = outTensor.const_data_ptr<float>();
            if (!packedPtr) {
                LOGE("runPart4bBatchedTiledPipeline: split tile_b2 compose returned null data on batch %d", b);
                return false;
            }
            totalFloats = static_cast<int>(outTensor.numel());
            packedBatch.resize(static_cast<size_t>(totalFloats));
            std::memcpy(packedBatch.data(), packedPtr, static_cast<size_t>(totalFloats) * sizeof(float));
            outData = packedBatch.data();
        } else {
            const std::string batchLabel =
                    "runPart4bBatchedTiledPipeline: batch " + std::to_string(b + 1) + "/" +
                    std::to_string(numBatches) + " legacy";
            auto result = runForwardWithHeartbeat(
                    [&]() { return module->forward(inputs); },
                    batchLabel + " forward");
            if (!result.ok() || result->empty() || !(*result)[0].isTensor()) {
                LOGE("runPart4bBatchedTiledPipeline: batch %d forward failed", b);
                return false;
            }

            const auto& outTensor = (*result)[0].toTensor();
            outData = outTensor.const_data_ptr<float>();
            totalFloats = static_cast<int>(outTensor.numel());
        }
        if ((totalFloats % batch) != 0) {
            LOGE("runPart4bBatchedTiledPipeline: output size %d not divisible by batch=%d", totalFloats, batch);
            return false;
        }
        const int floatsPerBatchItem = totalFloats / batch;
        const int gaussiansPerTile = floatsPerBatchItem / PARAMS_PER_GAUSSIAN;

        if (b == 0) {
            floatsPerTile = floatsPerBatchItem;
            outGaussians.resize(NUM_TILES * floatsPerTile);
        } else if (floatsPerBatchItem != floatsPerTile) {
            LOGE("runPart4bBatchedTiledPipeline: inconsistent batch output size");
            return false;
        }

        for (int i = 0; i < batch; ++i) {
            const int t = b * batch + i;
            const int tileRow = t / GRID;
            const int tileCol = t % GRID;
            float* dst = outGaussians.data() + t * floatsPerTile;
            std::memcpy(dst, outData + i * floatsPerBatchItem, floatsPerTile * sizeof(float));
            correctNDC(dst, gaussiansPerTile, tileRow, tileCol, swapTileNdcXY);
        }

        auto batchEnd = std::chrono::steady_clock::now();
        long long batchMs = std::chrono::duration_cast<std::chrono::milliseconds>(batchEnd - batchStart).count();
        LOGD("Part4b batch %d/%d (tiles %d-%d): %lldms", b + 1, numBatches,
             b * batch + 1, (b + 1) * batch, (long long)batchMs);
    }

    auto tEnd = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart).count();
    LOGD("[TIMING] Part4b batched tiled (batch=%d, %d calls): %lldms, %d Gaussians",
         batch, numBatches, (long long)ms,
         (int)(outGaussians.size() / PARAMS_PER_GAUSSIAN));
    logProcessMemory("Part4b batched tiled: completed");
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

    // Choose tile model: prefer the fine-split tile_00 export first, then split tile_00, then legacy tile_00, then tile_full.
    // tile_full is known to abort in current Vulkan delegate init on some devices.
    const std::string fineSplitStagePre = pathJoin(modelDir, "sharp_split_part4b_tile_00_stage_pre_vulkan.pte");
    const std::string fineSplitDecoderHead = pathJoin(modelDir, "sharp_split_part4b_tile_00_decoder_head.pte");
    const std::string splitStageA = pathJoin(modelDir, "sharp_split_part4b_tile_00_stage_a_vulkan.pte");
    const std::string splitInitBase = pathJoin(modelDir, "sharp_split_part4b_tile_00_init_base.pte");
    const std::string splitRawHeads = pathJoin(modelDir, "sharp_split_part4b_tile_00_raw_heads_vulkan.pte");
    const std::string splitCompose = pathJoin(modelDir, "sharp_split_part4b_tile_00_compose.pte");
    const bool hasFineSplitTile00 =
            fileExists(fineSplitStagePre) &&
            fileExists(fineSplitDecoderHead) &&
            fileExists(splitInitBase) &&
            fileExists(splitRawHeads) &&
            fileExists(splitCompose);
    const bool hasSplitTile00 =
            fileExists(splitStageA) &&
            fileExists(splitInitBase) &&
            fileExists(splitRawHeads) &&
            fileExists(splitCompose);
    std::string modelPath;
    const std::string tileFullFp32 = "sharp_split_part4b_tile_full.pte";
    const std::string tile00Fp32  = "sharp_split_part4b_tile_00.pte";
    const auto chooseLegacyTileModel = [&]() -> std::string {
        std::string candidate = pathJoin(modelDir, tile00Fp32);
        std::ifstream f1(candidate);
        if (f1.good()) {
            f1.close();
            return candidate;
        }
        f1.close();
        candidate = pathJoin(modelDir, tileFullFp32);
        std::ifstream f2(candidate);
        if (f2.good()) {
            f2.close();
            return candidate;
        }
        f2.close();
        return std::string();
    };
    if (hasFineSplitTile00) {
        LOGD("runPart4bTiledFullPipeline: using fine split tile_00 path stage_pre=%s decoder_head=%s init_base=%s raw_heads=%s compose=%s swap_ndc=%d",
             fineSplitStagePre.c_str(), fineSplitDecoderHead.c_str(), splitInitBase.c_str(), splitRawHeads.c_str(), splitCompose.c_str(),
             swapTileNdcXY ? 1 : 0);
    } else if (hasSplitTile00) {
        LOGD("runPart4bTiledFullPipeline: using split tile_00 path stage_a=%s init_base=%s raw_heads=%s compose=%s swap_ndc=%d",
             splitStageA.c_str(), splitInitBase.c_str(), splitRawHeads.c_str(), splitCompose.c_str(),
             swapTileNdcXY ? 1 : 0);
    } else {
        modelPath = chooseLegacyTileModel();
        if (modelPath.empty()) {
            LOGD("runPart4bTiledFullPipeline: no split tile_00 / tile_00 / tile_full found, skipping tiled path");
            return false;
        }
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
    std::unique_ptr<ETModule> module;
    std::unique_ptr<ETModule> splitStagePreModule;
    std::unique_ptr<ETModule> splitDecoderHeadModule;
    std::unique_ptr<ETModule> splitStageAModule;
    std::unique_ptr<ETModule> splitInitBaseModule;
    std::unique_ptr<ETModule> splitRawHeadsModule;
    std::unique_ptr<ETModule> splitComposeModule;
    bool useFineSplitTile00 = false;
    bool useSplitTile00 = false;

    logProcessMemory("Part4b tiled seq: before module load");
    if (hasFineSplitTile00) {
        splitStagePreModule = std::make_unique<ETModule>(fineSplitStagePre, ETModule::LoadMode::Mmap);
        splitDecoderHeadModule = std::make_unique<ETModule>(fineSplitDecoderHead, ETModule::LoadMode::Mmap);
        splitInitBaseModule = std::make_unique<ETModule>(splitInitBase, ETModule::LoadMode::Mmap);
        splitRawHeadsModule = std::make_unique<ETModule>(splitRawHeads, ETModule::LoadMode::Mmap);
        splitComposeModule = std::make_unique<ETModule>(splitCompose, ETModule::LoadMode::Mmap);
        const bool loaded =
                splitStagePreModule->load() == Error::Ok &&
                splitDecoderHeadModule->load() == Error::Ok &&
                splitInitBaseModule->load() == Error::Ok &&
                splitRawHeadsModule->load() == Error::Ok &&
                splitComposeModule->load() == Error::Ok;
        if (loaded) {
            useFineSplitTile00 = true;
            LOGD("runPart4bTiledFullPipeline: loaded fine split tile_00 modules");
        } else {
            LOGW("runPart4bTiledFullPipeline: fine split tile_00 load failed, falling back to split tile_00 / legacy tile path");
            splitStagePreModule.reset();
            splitDecoderHeadModule.reset();
            splitInitBaseModule.reset();
            splitRawHeadsModule.reset();
            splitComposeModule.reset();
        }
    }
    if (!useFineSplitTile00 && hasSplitTile00) {
        splitStageAModule = std::make_unique<ETModule>(splitStageA, ETModule::LoadMode::Mmap);
        splitInitBaseModule = std::make_unique<ETModule>(splitInitBase, ETModule::LoadMode::Mmap);
        splitRawHeadsModule = std::make_unique<ETModule>(splitRawHeads, ETModule::LoadMode::Mmap);
        splitComposeModule = std::make_unique<ETModule>(splitCompose, ETModule::LoadMode::Mmap);
        const bool loaded =
                splitStageAModule->load() == Error::Ok &&
                splitInitBaseModule->load() == Error::Ok &&
                splitRawHeadsModule->load() == Error::Ok &&
                splitComposeModule->load() == Error::Ok;
        if (loaded) {
            useSplitTile00 = true;
            LOGD("runPart4bTiledFullPipeline: loaded split tile_00 modules");
        } else {
            LOGW("runPart4bTiledFullPipeline: split tile_00 load failed, falling back to legacy tile path");
            splitStageAModule.reset();
            splitInitBaseModule.reset();
            splitRawHeadsModule.reset();
            splitComposeModule.reset();
            modelPath = chooseLegacyTileModel();
            if (modelPath.empty()) {
                LOGE("runPart4bTiledFullPipeline: no legacy tile_00 / tile_full fallback available after split load failure");
                return false;
            }
        }
    }
    if (!useFineSplitTile00 && !useSplitTile00) {
        module = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
        if (module->load() != Error::Ok) {
            LOGE("runPart4bTiledFullPipeline: failed to load tile model: %s", modelPath.c_str());
            return false;
        }
    }
    logProcessMemory("Part4b tiled seq: after module load");

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

        int numFloats = 0;
        int numGaussians = 0;
        const float* outData = nullptr;
        std::vector<float> packedTile;
        if (useFineSplitTile00) {
            const std::string tileLabel =
                    "runPart4bTiledFullPipeline: tile " + std::to_string(t + 1) + "/" +
                    std::to_string(NUM_TILES) + " fine split tile_00";
            std::vector<EValue> stagePreInputs;
            stagePreInputs.reserve(7);
            stagePreInputs.emplace_back(*imgTensor);
            stagePreInputs.emplace_back(*lat0Tensor);
            stagePreInputs.emplace_back(*lat1Tensor);
            stagePreInputs.emplace_back(*x0Tensor);
            stagePreInputs.emplace_back(*x1Tensor);
            stagePreInputs.emplace_back(*x2Tensor);
            stagePreInputs.emplace_back(*xlTensor);

            LOGD("%s stage_pre begin", tileLabel.c_str());
            auto stagePreResult = runForwardWithHeartbeat(
                    [&]() { return splitStagePreModule->forward(stagePreInputs); },
                    tileLabel + " stage_pre");
            if (!stagePreResult.ok() || stagePreResult->size() < 5) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 stage_pre failed on tile %d", t);
                return false;
            }

            OwnedTensor latent0Up;
            OwnedTensor latent1Up;
            OwnedTensor x0Up;
            OwnedTensor x1Up;
            OwnedTensor xFused;
            if (!copyOutputTensor(*stagePreResult, 0, latent0Up, "fine split tile_00 latent0_up") ||
                !copyOutputTensor(*stagePreResult, 1, latent1Up, "fine split tile_00 latent1_up") ||
                !copyOutputTensor(*stagePreResult, 2, x0Up, "fine split tile_00 x0_up") ||
                !copyOutputTensor(*stagePreResult, 3, x1Up, "fine split tile_00 x1_up") ||
                !copyOutputTensor(*stagePreResult, 4, xFused, "fine split tile_00 x_fused")) {
                return false;
            }
            if (!validateOwnedTensorShape(latent0Up, {1, 256, 192, 192}, "fine split tile_00 latent0_up") ||
                !validateOwnedTensorShape(latent1Up, {1, 256, 96, 96}, "fine split tile_00 latent1_up") ||
                !validateOwnedTensorShape(x0Up, {1, 512, 48, 48}, "fine split tile_00 x0_up") ||
                !validateOwnedTensorShape(x1Up, {1, 1024, 24, 24}, "fine split tile_00 x1_up") ||
                !validateOwnedTensorShape(xFused, {1, 1024, 12, 12}, "fine split tile_00 x_fused")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 stage_pre produced unexpected shapes on tile %d", t);
                return false;
            }

            auto latent0UpTensor = from_blob(latent0Up.data.data(), latent0Up.sizes);
            auto latent1UpTensor = from_blob(latent1Up.data.data(), latent1Up.sizes);
            auto x0UpTensor = from_blob(x0Up.data.data(), x0Up.sizes);
            auto x1UpTensor = from_blob(x1Up.data.data(), x1Up.sizes);
            auto xFusedTensor = from_blob(xFused.data.data(), xFused.sizes);

            std::vector<EValue> decoderHeadInputs;
            decoderHeadInputs.reserve(5);
            decoderHeadInputs.emplace_back(*latent0UpTensor);
            decoderHeadInputs.emplace_back(*latent1UpTensor);
            decoderHeadInputs.emplace_back(*x0UpTensor);
            decoderHeadInputs.emplace_back(*x1UpTensor);
            decoderHeadInputs.emplace_back(*xFusedTensor);

            LOGD("%s decoder_head begin", tileLabel.c_str());
            auto decoderHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitDecoderHeadModule->forward(decoderHeadInputs); },
                    tileLabel + " decoder_head");
            if (!decoderHeadResult.ok() || decoderHeadResult->size() < 2) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 decoder_head failed on tile %d", t);
                return false;
            }

            OwnedTensor disparity;
            OwnedTensor decoderFeatures;
            if (!copyOutputTensor(*decoderHeadResult, 0, disparity, "fine split tile_00 disparity") ||
                !copyOutputTensor(*decoderHeadResult, 1, decoderFeatures, "fine split tile_00 decoder_features")) {
                return false;
            }
            if (!validateOwnedTensorShape(disparity, {1, 2, 384, 384}, "fine split tile_00 disparity") ||
                !validateOwnedTensorShape(decoderFeatures, {1, 256, 192, 192}, "fine split tile_00 decoder_features")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 decoder_head produced unexpected shapes on tile %d", t);
                return false;
            }

            auto disparityTensor = from_blob(disparity.data.data(), disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgTensor);
            initInputs.emplace_back(*disparityTensor);

            LOGD("%s init_base begin", tileLabel.c_str());
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    tileLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 init_base failed on tile %d", t);
                return false;
            }

            OwnedTensor featureInput;
            if (!copyOutputTensor(*initResult, 0, featureInput, "fine split tile_00 feature_input")) {
                return false;
            }
            if (!validateOwnedTensorShape(featureInput, {1, 5, 384, 384}, "fine split tile_00 feature_input")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 init_base produced unexpected feature_input shape on tile %d", t);
                return false;
            }

            auto featureInputTensor = from_blob(featureInput.data.data(), featureInput.sizes);
            auto decoderFeaturesTensor = from_blob(decoderFeatures.data.data(), decoderFeatures.sizes);

            std::vector<EValue> rawHeadInputs;
            rawHeadInputs.reserve(7);
            rawHeadInputs.emplace_back(*featureInputTensor);
            rawHeadInputs.emplace_back(*latent0UpTensor);
            rawHeadInputs.emplace_back(*latent1UpTensor);
            rawHeadInputs.emplace_back(*x0UpTensor);
            rawHeadInputs.emplace_back(*x1UpTensor);
            rawHeadInputs.emplace_back(*xFusedTensor);
            rawHeadInputs.emplace_back(*decoderFeaturesTensor);

            LOGD("%s raw_heads begin", tileLabel.c_str());
            auto rawHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitRawHeadsModule->forward(rawHeadInputs); },
                    tileLabel + " raw_heads");
            if (!rawHeadResult.ok() || rawHeadResult->size() < 2) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 raw_heads failed on tile %d", t);
                return false;
            }

            OwnedTensor geometryRaw;
            OwnedTensor textureRaw;
            OwnedTensor meanXNdc;
            OwnedTensor meanYNdc;
            OwnedTensor meanInverseZNdc;
            OwnedTensor scales;
            OwnedTensor quaternions;
            OwnedTensor colors;
            OwnedTensor opacities;
            OwnedTensor globalScale;
            if (!copyOutputTensor(*rawHeadResult, 0, geometryRaw, "fine split tile_00 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, textureRaw, "fine split tile_00 texture_raw") ||
                !copyOutputTensor(*initResult, 1, meanXNdc, "fine split tile_00 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, meanYNdc, "fine split tile_00 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, meanInverseZNdc, "fine split tile_00 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scales, "fine split tile_00 scales") ||
                !copyOutputTensor(*initResult, 5, quaternions, "fine split tile_00 quaternions") ||
                !copyOutputTensor(*initResult, 6, colors, "fine split tile_00 colors") ||
                !copyOutputTensor(*initResult, 7, opacities, "fine split tile_00 opacities") ||
                !copyOutputTensor(*initResult, 8, globalScale, "fine split tile_00 global_scale")) {
                return false;
            }
            if (!validateOwnedTensorShape(geometryRaw, {1, 6, 192, 192}, "fine split tile_00 geometry_raw") ||
                !validateOwnedTensorShape(textureRaw, {1, 22, 192, 192}, "fine split tile_00 texture_raw")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 raw_heads produced unexpected shapes on tile %d", t);
                return false;
            }

            auto geometryRawTensor = from_blob(geometryRaw.data.data(), geometryRaw.sizes);
            auto textureRawTensor = from_blob(textureRaw.data.data(), textureRaw.sizes);
            auto meanXNdcTensor = from_blob(meanXNdc.data.data(), meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(meanYNdc.data.data(), meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(meanInverseZNdc.data.data(), meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scales.data.data(), scales.sizes);
            auto quaternionsTensor = from_blob(quaternions.data.data(), quaternions.sizes);
            auto colorsTensor = from_blob(colors.data.data(), colors.sizes);
            auto opacitiesTensor = from_blob(opacities.data.data(), opacities.sizes);
            auto globalScaleTensor = from_blob(globalScale.data.data(), globalScale.sizes);

            std::vector<EValue> composeInputs;
            composeInputs.reserve(10);
            composeInputs.emplace_back(*geometryRawTensor);
            composeInputs.emplace_back(*textureRawTensor);
            composeInputs.emplace_back(*meanXNdcTensor);
            composeInputs.emplace_back(*meanYNdcTensor);
            composeInputs.emplace_back(*meanInverseZNdcTensor);
            composeInputs.emplace_back(*scalesTensor);
            composeInputs.emplace_back(*quaternionsTensor);
            composeInputs.emplace_back(*colorsTensor);
            composeInputs.emplace_back(*opacitiesTensor);
            composeInputs.emplace_back(*globalScaleTensor);

            LOGD("%s compose begin", tileLabel.c_str());
            auto composeResult = runForwardWithHeartbeat(
                    [&]() { return splitComposeModule->forward(composeInputs); },
                    tileLabel + " compose");
            if (!composeResult.ok() || composeResult->empty() || !(*composeResult)[0].isTensor()) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 compose failed on tile %d", t);
                return false;
            }
            const auto& outTensor = (*composeResult)[0].toTensor();
            const float* packedPtr = outTensor.const_data_ptr<float>();
            if (!packedPtr) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 compose returned null data on tile %d", t);
                return false;
            }
            numFloats = static_cast<int>(outTensor.numel());
            numGaussians = numFloats / PARAMS_PER_GAUSSIAN;
            packedTile.resize(static_cast<size_t>(numFloats));
            std::memcpy(packedTile.data(), packedPtr, static_cast<size_t>(numFloats) * sizeof(float));
            outData = packedTile.data();
        } else if (useSplitTile00) {
            const std::string tileLabel =
                    "runPart4bTiledFullPipeline: tile " + std::to_string(t + 1) + "/" +
                    std::to_string(NUM_TILES) + " split tile_00";
            std::vector<EValue> stageAInputs;
            stageAInputs.reserve(7);
            stageAInputs.emplace_back(*imgTensor);
            stageAInputs.emplace_back(*lat0Tensor);
            stageAInputs.emplace_back(*lat1Tensor);
            stageAInputs.emplace_back(*x0Tensor);
            stageAInputs.emplace_back(*x1Tensor);
            stageAInputs.emplace_back(*x2Tensor);
            stageAInputs.emplace_back(*xlTensor);

            LOGD("%s stage_a begin", tileLabel.c_str());
            auto stageAResult = runForwardWithHeartbeat(
                    [&]() { return splitStageAModule->forward(stageAInputs); },
                    tileLabel + " stage_a");
            if (!stageAResult.ok() || stageAResult->size() < 7) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 stage_a failed on tile %d", t);
                return false;
            }

            OwnedTensor disparity;
            if (!copyOutputTensor(*stageAResult, 0, disparity, "split tile_00 disparity")) {
                return false;
            }
            auto disparityTensor = from_blob(disparity.data.data(), disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgTensor);
            initInputs.emplace_back(*disparityTensor);

            LOGD("%s init_base begin", tileLabel.c_str());
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    tileLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 init_base failed on tile %d", t);
                return false;
            }

            OwnedTensor featureInput;
            OwnedTensor latent0Up;
            OwnedTensor latent1Up;
            OwnedTensor x0Up;
            OwnedTensor x1Up;
            OwnedTensor xFused;
            OwnedTensor decoderFeatures;
            if (!copyOutputTensor(*initResult, 0, featureInput, "split tile_00 feature_input") ||
                !copyOutputTensor(*stageAResult, 1, latent0Up, "split tile_00 latent0_up") ||
                !copyOutputTensor(*stageAResult, 2, latent1Up, "split tile_00 latent1_up") ||
                !copyOutputTensor(*stageAResult, 3, x0Up, "split tile_00 x0_up") ||
                !copyOutputTensor(*stageAResult, 4, x1Up, "split tile_00 x1_up") ||
                !copyOutputTensor(*stageAResult, 5, xFused, "split tile_00 x_fused") ||
                !copyOutputTensor(*stageAResult, 6, decoderFeatures, "split tile_00 decoder_features")) {
                return false;
            }

            auto featureInputTensor = from_blob(featureInput.data.data(), featureInput.sizes);
            auto latent0UpTensor = from_blob(latent0Up.data.data(), latent0Up.sizes);
            auto latent1UpTensor = from_blob(latent1Up.data.data(), latent1Up.sizes);
            auto x0UpTensor = from_blob(x0Up.data.data(), x0Up.sizes);
            auto x1UpTensor = from_blob(x1Up.data.data(), x1Up.sizes);
            auto xFusedTensor = from_blob(xFused.data.data(), xFused.sizes);
            auto decoderFeaturesTensor = from_blob(decoderFeatures.data.data(), decoderFeatures.sizes);

            std::vector<EValue> rawHeadInputs;
            rawHeadInputs.reserve(7);
            rawHeadInputs.emplace_back(*featureInputTensor);
            rawHeadInputs.emplace_back(*latent0UpTensor);
            rawHeadInputs.emplace_back(*latent1UpTensor);
            rawHeadInputs.emplace_back(*x0UpTensor);
            rawHeadInputs.emplace_back(*x1UpTensor);
            rawHeadInputs.emplace_back(*xFusedTensor);
            rawHeadInputs.emplace_back(*decoderFeaturesTensor);

            LOGD("%s raw_heads begin", tileLabel.c_str());
            auto rawHeadResult = runForwardWithHeartbeat(
                    [&]() { return splitRawHeadsModule->forward(rawHeadInputs); },
                    tileLabel + " raw_heads");
            if (!rawHeadResult.ok() || rawHeadResult->size() < 2) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 raw_heads failed on tile %d", t);
                return false;
            }

            OwnedTensor geometryRaw;
            OwnedTensor textureRaw;
            OwnedTensor meanXNdc;
            OwnedTensor meanYNdc;
            OwnedTensor meanInverseZNdc;
            OwnedTensor scales;
            OwnedTensor quaternions;
            OwnedTensor colors;
            OwnedTensor opacities;
            OwnedTensor globalScale;
            if (!copyOutputTensor(*rawHeadResult, 0, geometryRaw, "split tile_00 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, textureRaw, "split tile_00 texture_raw") ||
                !copyOutputTensor(*initResult, 1, meanXNdc, "split tile_00 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, meanYNdc, "split tile_00 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, meanInverseZNdc, "split tile_00 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scales, "split tile_00 scales") ||
                !copyOutputTensor(*initResult, 5, quaternions, "split tile_00 quaternions") ||
                !copyOutputTensor(*initResult, 6, colors, "split tile_00 colors") ||
                !copyOutputTensor(*initResult, 7, opacities, "split tile_00 opacities") ||
                !copyOutputTensor(*initResult, 8, globalScale, "split tile_00 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(geometryRaw.data.data(), geometryRaw.sizes);
            auto textureRawTensor = from_blob(textureRaw.data.data(), textureRaw.sizes);
            auto meanXNdcTensor = from_blob(meanXNdc.data.data(), meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(meanYNdc.data.data(), meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(meanInverseZNdc.data.data(), meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scales.data.data(), scales.sizes);
            auto quaternionsTensor = from_blob(quaternions.data.data(), quaternions.sizes);
            auto colorsTensor = from_blob(colors.data.data(), colors.sizes);
            auto opacitiesTensor = from_blob(opacities.data.data(), opacities.sizes);
            auto globalScaleTensor = from_blob(globalScale.data.data(), globalScale.sizes);

            std::vector<EValue> composeInputs;
            composeInputs.reserve(10);
            composeInputs.emplace_back(*geometryRawTensor);
            composeInputs.emplace_back(*textureRawTensor);
            composeInputs.emplace_back(*meanXNdcTensor);
            composeInputs.emplace_back(*meanYNdcTensor);
            composeInputs.emplace_back(*meanInverseZNdcTensor);
            composeInputs.emplace_back(*scalesTensor);
            composeInputs.emplace_back(*quaternionsTensor);
            composeInputs.emplace_back(*colorsTensor);
            composeInputs.emplace_back(*opacitiesTensor);
            composeInputs.emplace_back(*globalScaleTensor);

            LOGD("%s compose begin", tileLabel.c_str());
            auto composeResult = runForwardWithHeartbeat(
                    [&]() { return splitComposeModule->forward(composeInputs); },
                    tileLabel + " compose");
            if (!composeResult.ok() || composeResult->empty() || !(*composeResult)[0].isTensor()) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 compose failed on tile %d", t);
                return false;
            }
            const auto& outTensor = (*composeResult)[0].toTensor();
            const float* packedPtr = outTensor.const_data_ptr<float>();
            if (!packedPtr) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 compose returned null data on tile %d", t);
                return false;
            }
            numFloats = static_cast<int>(outTensor.numel());
            numGaussians = numFloats / PARAMS_PER_GAUSSIAN;
            packedTile.resize(static_cast<size_t>(numFloats));
            std::memcpy(packedTile.data(), packedPtr, static_cast<size_t>(numFloats) * sizeof(float));
            outData = packedTile.data();
        } else {
            std::vector<EValue> inputs;
            inputs.reserve(7);
            inputs.emplace_back(*imgTensor);
            inputs.emplace_back(*lat0Tensor);
            inputs.emplace_back(*lat1Tensor);
            inputs.emplace_back(*x0Tensor);
            inputs.emplace_back(*x1Tensor);
            inputs.emplace_back(*x2Tensor);
            inputs.emplace_back(*xlTensor);

            const std::string tileLabel =
                    "runPart4bTiledFullPipeline: tile " + std::to_string(t + 1) + "/" +
                    std::to_string(NUM_TILES) + " legacy";
            auto result = runForwardWithHeartbeat(
                    [&]() { return module->forward(inputs); },
                    tileLabel + " forward");
            if (!result.ok() || result->empty() || !(*result)[0].isTensor()) {
                LOGE("runPart4bTiledFullPipeline: tile %d forward failed", t);
                return false;
            }

            const auto& outTensor = (*result)[0].toTensor();
            outData = outTensor.const_data_ptr<float>();
            numFloats = static_cast<int>(outTensor.numel());
            numGaussians = numFloats / PARAMS_PER_GAUSSIAN;
        }

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
    logProcessMemory("Part4b tiled seq: completed");
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

bool warmupCachedPart12Modules() {
    if (!g_moduleCache.mod1 || !g_moduleCache.mod2) return false;

    const size_t patchSz = 3 * PATCH_SIZE * PATCH_SIZE;
    const size_t tokensSz = TOKENS_577 * FEATURE_DIM;
    std::vector<float> zeroPatch(patchSz, 0.0f);
    std::vector<float> zeroTokens(tokensSz, 0.0f);

    long long tPart1 = nowMs();
    {
        auto patchTensor = from_blob(zeroPatch.data(), {1, 3, PATCH_SIZE, PATCH_SIZE});
        std::vector<EValue> in1 = {*patchTensor};
        auto out1 = g_moduleCache.mod1->forward(in1);
        if (!out1.ok() || out1->size() < 2) {
            LOGW("ModuleCache warmup: Part1 forward failed err=%d", out1.ok() ? 0 : static_cast<int>(out1.error()));
            return false;
        }
    }
    LOGD("ModuleCache warmup: Part1 forward finished in %lldms", nowMs() - tPart1);

    long long tPart2 = nowMs();
    {
        auto tokenTensor = from_blob(zeroTokens.data(), {1, TOKENS_577, FEATURE_DIM});
        std::vector<EValue> in2 = {*tokenTensor};
        auto out2 = g_moduleCache.mod2->forward(in2);
        if (!out2.ok() || out2->empty()) {
            LOGW("ModuleCache warmup: Part2 forward failed err=%d", out2.ok() ? 0 : static_cast<int>(out2.error()));
            return false;
        }
    }
    LOGD("ModuleCache warmup: Part2 forward finished in %lldms", nowMs() - tPart2);
    return true;
}

// ── JNI: preload Part1+Part2 (call from Kotlin on init for warm start) ───────
extern "C" {

JNIEXPORT void JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_nativeSetSharpExecVerboseLogging(
        JNIEnv*, jclass, jboolean enabled) {
    g_sharp_exec_verbose.store(enabled == JNI_TRUE ? 1 : 0, std::memory_order_relaxed);
}

JNIEXPORT jboolean JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_preloadCppModules(
        JNIEnv* env, jobject, jstring modelDirPath, jboolean useVulkanForPart12) {
    ensureRuntimeInit();
    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string dir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);
    if (dir.empty()) return JNI_FALSE;
    if (!g_moduleCache.ensureLoaded(dir, (useVulkanForPart12 == JNI_TRUE))) {
        return JNI_FALSE;
    }
    if (useVulkanForPart12 == JNI_TRUE) {
        LOGD("ModuleCache preload: Vulkan load-only (skip warmup forward to avoid startup mem-pressure/device-lost)");
        return JNI_TRUE;
    }
    if (!warmupCachedPart12Modules()) {
        LOGW("ModuleCache preload: load succeeded but warmup failed");
    }
    return JNI_TRUE;
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
        jobject progressReporter,
        jstring etdumpOutputPath) {
    if (useVulkan == JNI_TRUE) {
        return runSharpFullPipeline_Vulkan(
                env, thiz, modelDirPath, imageNCHW, maxGaussians, preferSinglePart4b, part12OnCpu,
                part12ForceSinglePatch, part12_25Only,
                part1MaxPatches1x, part1MaxPatches05x, part12Chunk1x, part12Chunk05x,
                part12YieldMsBetweenChunks, swapTileNdcXY, progressReporter, etdumpOutputPath);
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
