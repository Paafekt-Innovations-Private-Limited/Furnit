/**
 * Shared declarations for SHARP ExecuTorch full pipeline (CPU + Vulkan translation units).
 *
 * CPU split `.pte` basenames are fixed strings in this tree (pathJoin + literal). Kotlin uses the same
 * names in `SharpExecuTorchSplitModelNames`; keep them aligned when renaming exports (see
 * `android/docs/SHARP_CPU_V2_MODEL_SET.md` for the six-file v2 set with `part4b_tile_b4` only).
 */
#pragma once

#include <jni.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <vector>
#include <chrono>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <unistd.h>
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

/** Set from Kotlin (DebugLogger: debuggable + debug_mode) via JNI before native SHARP calls. */
bool sharpExecNativeVerboseLogsEnabled();

#define LOGD(...)                                                                                  \
    do {                                                                                           \
        if (sharpExecNativeVerboseLogsEnabled())                                                   \
            __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__);                              \
    } while (0)
#define LOGI(...)                                                                                  \
    do {                                                                                           \
        if (sharpExecNativeVerboseLogsEnabled())                                                   \
            __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__);                               \
    } while (0)
#define LOGW(...)                                                                                  \
    do {                                                                                           \
        if (sharpExecNativeVerboseLogsEnabled())                                                   \
            __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__);                               \
    } while (0)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

const char* executorchErrorStr(int err);

using ETModule = executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

// ── Constants ────────────────────────────────────────────────────────────────
inline constexpr int IMAGE_SIZE_DEFAULT = 1536;
inline constexpr int IMAGE_SIZE_1280 = 1280;
inline constexpr int PATCH_SIZE = 384;
inline constexpr int FEATURE_DIM = 1024;
inline constexpr int SPATIAL_SIZE = 24;
inline constexpr int M_1X = 96;
inline constexpr int M_05X = 48;
inline constexpr int GRID_1X = 5;
inline constexpr int GRID_05X = 3;
inline constexpr int PADDING_1X = 3;
inline constexpr int PADDING_05X = 6;
inline constexpr int PARAMS_PER_GAUSSIAN = 14;
inline constexpr int TOKENS_577 = 577;
inline constexpr int PATCH_BATCH = 4;
inline constexpr int PATCH_BATCH_2 = 2;
inline constexpr int IMAGE_SIZE = IMAGE_SIZE_DEFAULT;
inline constexpr int HALF_SIZE = IMAGE_SIZE / 2;
inline constexpr int STRIDE_1X = (IMAGE_SIZE - PATCH_SIZE) / 4;
inline constexpr int STRIDE_05X = (HALF_SIZE - PATCH_SIZE) / 2;
inline constexpr int SPATIAL_HW = SPATIAL_SIZE * SPATIAL_SIZE;

inline constexpr int ROW_OFFS_1X[5] = {0, 21, 39, 57, 75};
inline constexpr int COL_OFFS_1X[5] = {0, 21, 39, 57, 75};
inline constexpr int ROW_OFFS_05X[3] = {0, 18, 30};
inline constexpr int COL_OFFS_05X[3] = {0, 18, 30};

// ── Pre-allocated workspace ─────────────────────────────────────────────────
struct Workspace {
    float* latent0 = nullptr;
    float* latent1 = nullptr;
    float* x0Feat = nullptr;
    float* x1Feat = nullptr;
    float* x2Feat = nullptr;
    float* tempSpatial = nullptr;
    float* halfImg = nullptr;
    float* quarterImg = nullptr;
    float* patchBuf = nullptr;
    float* patchBuf4 = nullptr;
    float* tokensCopy = nullptr;
    float* tokensCopy4 = nullptr;
    float* tileImgCrop = nullptr;
    float* tileLat0Crop = nullptr;
    float* tileLat1Crop = nullptr;
    float* tileX0Crop = nullptr;
    float* tileX1Crop = nullptr;
    float* tileX2Crop = nullptr;
    float* tileXlCrop = nullptr;
    bool allocated = false;
    /** True when tile crop buffers are live (Part4b tiled path); separate from encoder scratch. */
    bool tileAllocated = false;

    bool allocate();
    /** Allocate pooled Part4b tile crops; safe to call if already allocated. */
    bool allocateTileScratch();
    /** Free tile crop buffers only (not encoder scratch). */
    void releaseTileScratch();
    void zero();
    void releaseEncoderScratch();
    void release();
};

// ── Singleton module cache ───────────────────────────────────────────────────
struct ModuleCache {
    std::unique_ptr<ETModule> mod1, mod2;
    std::unique_ptr<ETModule> mod1_b4, mod2_b4;
    std::unique_ptr<ETModule> mod1_b2, mod2_b2;
    std::string modelDir;
    bool lastUseVulkan = false;
    std::mutex mu;

    bool ensureLoaded(const std::string& dir, bool useVulkan);
    void release();
};

extern ModuleCache g_moduleCache;
extern Workspace g_workspace;
extern bool g_runtime_initialized;

std::string pathJoin(const std::string& dir, const std::string& name);

void downsample2x(const float* __restrict src, int H, int W, int C, float* __restrict dst);
void downsample4x(const float* __restrict src, int H, int W, int C, float* __restrict dst);
void cropNCHW(const float* __restrict src, int srcH, int srcW, int C,
              int startY, int startX, int cropH, int cropW, float* __restrict dst);
void reshapeToSpatial(const float* __restrict tokens, size_t tokenLen, float* __restrict out);
void mergeCrop(float* __restrict out, int outW, const float* __restrict patch,
               int gI, int gJ, int gS, int pad, const int* rowOff, const int* colOff);

/**
 * Loaded Part4b tile_00 split modules (fine-split, split, or legacy single) for reuse / async preload.
 * Mirrors discovery order in runPart4bTiledFullPipeline.
 */
struct Part4bTile00ModuleBundle {
    std::string modelDir;
    bool useFineSplitTile00 = false;
    bool useSplitTile00 = false;
    std::unique_ptr<ETModule> splitStagePreModule;
    std::unique_ptr<ETModule> splitDecoderHeadModule;
    std::unique_ptr<ETModule> splitStageAModule;
    std::unique_ptr<ETModule> splitInitBaseModule;
    std::unique_ptr<ETModule> splitRawHeadsModule;
    std::unique_ptr<ETModule> splitComposeModule;
    std::unique_ptr<ETModule> legacyTileModule;
    std::string legacyModelPath;
};

/** Load tile_00 Part4b modules from disk (same logic as tiled full pipeline). Returns false if no path found. */
bool loadPart4bTile00ModuleBundle(const std::string& modelDir, Part4bTile00ModuleBundle& out);

/** Start async mmap+load of Part4b tile_00 modules while Part4a runs (joined before Part4b). */
void part4bTile00PreloadStart(const std::string& modelDir);
/** Block until preload thread finishes (no-op if not started). */
void part4bTile00PreloadJoin();

bool runPart4bBatchedTiledPipeline(const std::string& modelDir,
                                   const float* __restrict imageData,
                                   const float* __restrict latent0,
                                   const float* __restrict latent1,
                                   const float* __restrict x0Feat,
                                   const float* __restrict x1Feat,
                                   const float* __restrict x2Feat,
                                   const float* __restrict combinedTokens, size_t tokensNumel,
                                   bool swapTileNdcXY, std::vector<float>& outGaussians,
                                   JNIEnv* progressEnv = nullptr, jobject progressReporter = nullptr,
                                   jmethodID reportProgressMethodId = nullptr);

bool runPart4bTiledFullPipeline(const std::string& modelDir,
                                const float* __restrict imageData,
                                const float* __restrict latent0,
                                const float* __restrict latent1,
                                const float* __restrict x0Feat,
                                const float* __restrict x1Feat,
                                const float* __restrict x2Feat,
                                const float* __restrict combinedTokens, size_t tokensNumel,
                                bool swapTileNdcXY, std::vector<float>& outGaussians,
                                JNIEnv* progressEnv = nullptr, jobject progressReporter = nullptr,
                                jmethodID reportProgressMethodId = nullptr);

void pruneGaussiansByOpacity(std::vector<float>& gaussians, int maxGaussians);
int pruneGaussiansFromPtr(const float* src, int totalGaussians, int maxGaussians,
                          std::vector<float>& prunedOut);

long long nowMs();
/** /proc/meminfo MemAvailable in KiB, or -1 if unknown. Used for hybrid Part1+2 interleave gate. */
long sharpDeviceMemAvailableKb();
void ensureRuntimeInit();
void configureCpuInferenceThreadsOnce();
std::string processMemorySummary();
void logProcessMemory(const char* stage);

void cpuForwardHeartbeatLoop(std::atomic<bool>* keepGoing, const char* stageTag, int intervalSec);
void reportProgress(JNIEnv* env, jobject reporter, jmethodID methodId, float progress, const char* message);
void clearLastMonodepthCapture();

bool moduleCacheLoadPart12Cpu(ModuleCache& cache, const std::string& dir);
bool moduleCacheLoadPart12Vulkan(ModuleCache& cache, const std::string& dir);

jfloatArray runSharpFullPipeline_Cpu(JNIEnv* env, jobject thiz, jstring modelDirPath,
                                     jfloatArray imageNCHW, jint maxGaussians, jboolean preferSinglePart4b,
                                     jboolean part12ForceSinglePatch,
                                     jboolean part12_25Only,
                                     jint part1MaxPatches1x, jint part1MaxPatches05x,
                                     jint part12Chunk1x, jint part12Chunk05x,
                                     jint part12YieldMsBetweenChunks, jboolean swapTileNdcXY,
                                     jobject progressReporter);

jfloatArray runSharpFullPipeline_Vulkan(JNIEnv* env, jobject thiz, jstring modelDirPath,
                                        jfloatArray imageNCHW, jint maxGaussians, jboolean preferSinglePart4b,
                                        jboolean part12OnCpu,
                                        jboolean part12ForceSinglePatch,
                                        jboolean part12_25Only,
                                        jint part1MaxPatches1x, jint part1MaxPatches05x,
                                        jint part12Chunk1x, jint part12Chunk05x,
                                        jint part12YieldMsBetweenChunks, jboolean swapTileNdcXY,
                                        jobject progressReporter,
                                        jstring etdumpOutputPath,
                                        jboolean hybridInterleavePart12,
                                        jlong hybridInterleaveMinAvailMemBytes);
