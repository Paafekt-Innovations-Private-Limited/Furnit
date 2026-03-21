/**
 * ExecuTorch C++ JNI for SHARP 16-tile Part4b pipeline.
 *
 * Runs the entire tile loop in native code: crop, forward, NDC correction,
 * and concatenation all stay on the native heap. Only the final packed
 * Gaussian array crosses back to Java (one copy instead of 32).
 */

#include <jni.h>
#include <atomic>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <thread>
#include <chrono>
#include <android/log.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/platform/runtime.h>

#define TAG "ExecTorchTilesJNI"

static std::atomic<int> g_sharp_tiles_verbose{0};

#define LOGD(...)                                                                                  \
    do {                                                                                           \
        if (g_sharp_tiles_verbose.load(std::memory_order_relaxed) != 0)                           \
            __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__);                              \
    } while (0)
#define LOGE(...)                                                                                  \
    do {                                                                                           \
        if (g_sharp_tiles_verbose.load(std::memory_order_relaxed) != 0)                           \
            __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__);                              \
    } while (0)

using ETModule = executorch::extension::Module;
using executorch::extension::from_blob;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

static constexpr int GRID = 4;
static constexpr int NUM_TILES = GRID * GRID;
static constexpr int GAUSSIANS_PER_TILE = 73728;
static constexpr int PARAMS_PER_GAUSSIAN = 14;
static constexpr int FLOATS_PER_TILE = GAUSSIANS_PER_TILE * PARAMS_PER_GAUSSIAN;

static bool g_runtime_initialized = false;

static void ensureRuntimeInit() {
    if (!g_runtime_initialized) {
        executorch::runtime::runtime_init();
        g_runtime_initialized = true;
    }
}

/** Crop tile (tileRow, tileCol) from a flat NCHW buffer. Output has size C*(H/GRID)*(W/GRID). */
static void cropTileNCHW(
        const float* src, int C, int H, int W,
        int tileRow, int tileCol,
        float* dst) {
    const int tileH = H / GRID;
    const int tileW = W / GRID;
    const int srcHW = H * W;
    const int dstHW = tileH * tileW;
    for (int c = 0; c < C; ++c) {
        const float* srcPlane = src + c * srcHW + tileRow * tileH * W + tileCol * tileW;
        float* dstPlane = dst + c * dstHW;
        for (int y = 0; y < tileH; ++y) {
            std::memcpy(dstPlane + y * tileW, srcPlane + y * W, tileW * sizeof(float));
        }
    }
}

/** Apply NDC correction so tile-local NDC maps to full-image NDC. swapXY: use tileRow for X and tileCol for Y (if export uses transposed tile layout). */
static void correctNDC(float* data, int numGaussians, int tileRow, int tileCol, bool swapXY) {
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
}

/**
 * Run a single tile forward: create input tensors from pre-cropped data, call module.forward(),
 * copy output and apply NDC correction into the destination buffer.
 *
 * Returns 0 on success, -1 on failure.
 */
static int runOneTile(
        ETModule& module,
        float* imgCrop,  int imgTileH,  int imgTileW,
        float* lat0Crop, float* lat1Crop, float* x0Crop,
        float* x1Crop,   float* x2Crop,  float* xlCrop,
        float* outputDst,
        int tileRow, int tileCol, int tileIdx, bool swapNdcXY) {

    auto imgTensor  = from_blob(imgCrop,  {1, 3, imgTileH, imgTileW});
    auto lat0Tensor = from_blob(lat0Crop, {1, 1024, 24, 24});
    auto lat1Tensor = from_blob(lat1Crop, {1, 1024, 24, 24});
    auto x0Tensor   = from_blob(x0Crop,   {1, 1024, 24, 24});
    auto x1Tensor   = from_blob(x1Crop,   {1, 1024, 12, 12});
    auto x2Tensor   = from_blob(x2Crop,   {1, 1024, 6, 6});
    auto xlTensor   = from_blob(xlCrop,   {1, 1024, 6, 6});

    std::vector<EValue> inputs;
    inputs.reserve(7);
    inputs.emplace_back(*imgTensor);
    inputs.emplace_back(*lat0Tensor);
    inputs.emplace_back(*lat1Tensor);
    inputs.emplace_back(*x0Tensor);
    inputs.emplace_back(*x1Tensor);
    inputs.emplace_back(*x2Tensor);
    inputs.emplace_back(*xlTensor);

    auto result = module.forward(inputs);
    if (!result.ok()) {
        LOGE("Tile %d forward failed: error %d", tileIdx, static_cast<int>(result.error()));
        return -1;
    }

    auto& outputs = *result;
    if (outputs.empty() || !outputs[0].isTensor()) {
        LOGE("Tile %d: no tensor output", tileIdx);
        return -1;
    }

    const auto& outTensor = outputs[0].toTensor();
    const float* outData = outTensor.const_data_ptr<float>();
    const int numFloats = static_cast<int>(outTensor.numel());
    const int numGaussians = numFloats / PARAMS_PER_GAUSSIAN;

    std::memcpy(outputDst, outData, numFloats * sizeof(float));
    correctNDC(outputDst, numGaussians, tileRow, tileCol, swapNdcXY);

    return 0;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_nativeSetSharpTilesVerboseLogging(
        JNIEnv*, jclass, jboolean enabled) {
    g_sharp_tiles_verbose.store(enabled == JNI_TRUE ? 1 : 0, std::memory_order_relaxed);
}

JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_runTiledPart4bNative(
        JNIEnv* env, jobject /* this */,
        jstring jModelPath,
        jfloatArray jImageNCHW,
        jfloatArray jLatent0,
        jfloatArray jLatent1,
        jfloatArray jX0Feat,
        jfloatArray jX1Feat,
        jfloatArray jX2Feat,
        jfloatArray jXLowres,
        jint numThreads,
        jboolean parallelTiles,
        jboolean swapNdcXY) {

    ensureRuntimeInit();

    const char* modelPathRaw = env->GetStringUTFChars(jModelPath, nullptr);
    std::string modelPath(modelPathRaw);
    env->ReleaseStringUTFChars(jModelPath, modelPathRaw);

    const bool swapXY = (swapNdcXY == JNI_TRUE);
    LOGD("Starting native tiled Part4b: model=%s threads=%d parallel=%d swapNdcXY=%d",
         modelPath.c_str(), numThreads, parallelTiles, swapXY);

    const int imgH = 1536, imgW = 1536, imgC = 3;
    const int imgTileH = imgH / GRID, imgTileW = imgW / GRID;

    // Pin input arrays from Java heap (no copy; GetFloatArrayElements may pin or copy)
    jfloat* imageData  = env->GetFloatArrayElements(jImageNCHW, nullptr);
    jfloat* latent0    = env->GetFloatArrayElements(jLatent0, nullptr);
    jfloat* latent1    = env->GetFloatArrayElements(jLatent1, nullptr);
    jfloat* x0Feat     = env->GetFloatArrayElements(jX0Feat, nullptr);
    jfloat* x1Feat     = env->GetFloatArrayElements(jX1Feat, nullptr);
    jfloat* x2Feat     = env->GetFloatArrayElements(jX2Feat, nullptr);
    jfloat* xLowres    = env->GetFloatArrayElements(jXLowres, nullptr);

    // Pre-allocate native output buffer (stays on native heap; no Java heap pressure)
    const int totalFloats = NUM_TILES * FLOATS_PER_TILE;
    std::vector<float> gaussianOutput(totalFloats);

    // Pre-compute all 16 tile crops on native heap
    const int imgCropSize  = imgC * imgTileH * imgTileW;
    const int lat96CropSz  = 1024 * 24 * 24;
    const int x1CropSz     = 1024 * 12 * 12;
    const int x2CropSz     = 1024 * 6 * 6;

    struct TileCrops {
        std::vector<float> img;
        std::vector<float> lat0, lat1, x0;
        std::vector<float> x1, x2, xl;
    };
    std::vector<TileCrops> crops(NUM_TILES);

    auto tCropStart = std::chrono::steady_clock::now();
    for (int t = 0; t < NUM_TILES; ++t) {
        int tileRow = t / GRID, tileCol = t % GRID;
        crops[t].img.resize(imgCropSize);
        crops[t].lat0.resize(lat96CropSz);
        crops[t].lat1.resize(lat96CropSz);
        crops[t].x0.resize(lat96CropSz);
        crops[t].x1.resize(x1CropSz);
        crops[t].x2.resize(x2CropSz);
        crops[t].xl.resize(x2CropSz);

        cropTileNCHW(imageData, imgC, imgH, imgW, tileRow, tileCol, crops[t].img.data());
        cropTileNCHW(latent0, 1024, 96, 96, tileRow, tileCol, crops[t].lat0.data());
        cropTileNCHW(latent1, 1024, 96, 96, tileRow, tileCol, crops[t].lat1.data());
        cropTileNCHW(x0Feat, 1024, 96, 96, tileRow, tileCol, crops[t].x0.data());
        cropTileNCHW(x1Feat, 1024, 48, 48, tileRow, tileCol, crops[t].x1.data());
        cropTileNCHW(x2Feat, 1024, 24, 24, tileRow, tileCol, crops[t].x2.data());
        cropTileNCHW(xLowres, 1024, 24, 24, tileRow, tileCol, crops[t].xl.data());
    }
    auto tCropEnd = std::chrono::steady_clock::now();
    auto cropMs = std::chrono::duration_cast<std::chrono::milliseconds>(tCropEnd - tCropStart).count();
    LOGD("Pre-computed 16 tile crops in %lldms", (long long)cropMs);

    env->ReleaseFloatArrayElements(jImageNCHW, imageData, JNI_ABORT);
    env->ReleaseFloatArrayElements(jLatent0, latent0, JNI_ABORT);
    env->ReleaseFloatArrayElements(jLatent1, latent1, JNI_ABORT);
    env->ReleaseFloatArrayElements(jX0Feat, x0Feat, JNI_ABORT);
    env->ReleaseFloatArrayElements(jX1Feat, x1Feat, JNI_ABORT);
    env->ReleaseFloatArrayElements(jX2Feat, x2Feat, JNI_ABORT);
    env->ReleaseFloatArrayElements(jXLowres, xLowres, JNI_ABORT);

    bool success = true;
    auto tInfStart = std::chrono::steady_clock::now();

    const bool useParallel = false;

    if (parallelTiles && useParallel) {
        auto modA = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
        auto modB = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
        auto errA = modA->load();
        auto errB = modB->load();
        if (errA != Error::Ok || errB != Error::Ok) {
            LOGE("Failed to load parallel modules: A=%d B=%d", (int)errA, (int)errB);
            success = false;
        } else {
            LOGD("Loaded 2 tile modules for parallel inference");
            for (int pairStart = 0; pairStart < NUM_TILES && success; pairStart += 2) {
                int idxA = pairStart;
                int idxB = (pairStart + 1 < NUM_TILES) ? pairStart + 1 : -1;

                int resultA = -1, resultB = -1;
                std::thread threadA([&]() {
                    resultA = runOneTile(*modA,
                        crops[idxA].img.data(), imgTileH, imgTileW,
                        crops[idxA].lat0.data(), crops[idxA].lat1.data(), crops[idxA].x0.data(),
                        crops[idxA].x1.data(), crops[idxA].x2.data(), crops[idxA].xl.data(),
                        gaussianOutput.data() + idxA * FLOATS_PER_TILE,
                        idxA / GRID, idxA % GRID, idxA, swapXY);
                });

                if (idxB >= 0) {
                    resultB = runOneTile(*modB,
                        crops[idxB].img.data(), imgTileH, imgTileW,
                        crops[idxB].lat0.data(), crops[idxB].lat1.data(), crops[idxB].x0.data(),
                        crops[idxB].x1.data(), crops[idxB].x2.data(), crops[idxB].xl.data(),
                        gaussianOutput.data() + idxB * FLOATS_PER_TILE,
                        idxB / GRID, idxB % GRID, idxB, swapXY);
                }

                threadA.join();

                if (resultA != 0 || (idxB >= 0 && resultB != 0)) {
                    LOGE("Parallel tiles %d+%d failed: A=%d B=%d", idxA, idxB, resultA, resultB);
                    success = false;
                }
            }
        }
    } else {
        if (parallelTiles) {
            LOGD("Parallel tiles requested; running sequential in native (parallel mode can OOM or crash on this runtime)");
        }
        auto module = std::make_unique<ETModule>(modelPath, ETModule::LoadMode::Mmap);
        auto err = module->load();
        if (err != Error::Ok) {
            LOGE("Failed to load tile module: %d", (int)err);
            success = false;
        } else {
            auto tLoad = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - tInfStart).count();
            LOGD("Loaded tile module in %lldms (reused for all 16 tiles)", (long long)tLoad);

            for (int t = 0; t < NUM_TILES && success; ++t) {
                auto tTile = std::chrono::steady_clock::now();
                int ret = runOneTile(*module,
                    crops[t].img.data(), imgTileH, imgTileW,
                    crops[t].lat0.data(), crops[t].lat1.data(), crops[t].x0.data(),
                    crops[t].x1.data(), crops[t].x2.data(), crops[t].xl.data(),
                    gaussianOutput.data() + t * FLOATS_PER_TILE,
                    t / GRID, t % GRID, t, swapXY);
                if (ret != 0) {
                    LOGE("Tile %d failed", t);
                    success = false;
                } else {
                    auto tileMs = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - tTile).count();
                    LOGD("Tile %d: %d Gaussians, %lldms", t, GAUSSIANS_PER_TILE, (long long)tileMs);
                }
            }
        }
    }

    auto tInfEnd = std::chrono::steady_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(tInfEnd - tInfStart).count();

    if (!success) {
        LOGE("Native tiled Part4b failed after %lldms", (long long)totalMs);
        return nullptr;
    }

    LOGD("[TIMING] Native Part4b tiled total: %lldms, %d Gaussians (parallel=%d, threads=%d)",
         (long long)totalMs, NUM_TILES * GAUSSIANS_PER_TILE, parallelTiles, numThreads);

    jfloatArray jResult = env->NewFloatArray(totalFloats);
    if (jResult == nullptr) {
        LOGE("Failed to allocate Java result array (%d floats)", totalFloats);
        return nullptr;
    }
    env->SetFloatArrayRegion(jResult, 0, totalFloats, gaussianOutput.data());

    return jResult;
}
