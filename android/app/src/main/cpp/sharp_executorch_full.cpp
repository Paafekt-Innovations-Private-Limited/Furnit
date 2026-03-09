/**
 * ExecuTorch C++ full INT8 pipeline: Part1, Part2, Part3, Part4a, Part4b (single).
 *
 * Runs the entire SHARP pipeline in native code. No tiles for Part4b.
 * Matches Kotlin ExecutorchInt8Sharp: 35 patches (25@1x + 9@0.5x + 1@0.25x),
 * Part3 full image, Part4a two chunks, Part4b single forward.
 */

#include <jni.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <android/log.h>

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
static constexpr int TOKENS_576 = 576;

static bool g_runtime_initialized = false;

static void ensureRuntimeInit() {
    if (!g_runtime_initialized) {
        executorch::runtime::runtime_init();
        g_runtime_initialized = true;
    }
}

/** Crop from NCHW image at (startY, startX) size (cropH, cropW). dst = C * cropH * cropW. */
static void cropNCHW(const float* src, int srcH, int srcW, int C,
                     int startY, int startX, int cropH, int cropW,
                     float* dst) {
    const int srcHW = srcH * srcW;
    const int dstHW = cropH * cropW;
    for (int c = 0; c < C; ++c) {
        const float* plane = src + c * srcHW;
        float* outPlane = dst + c * dstHW;
        for (int y = 0; y < cropH; ++y) {
            std::memcpy(outPlane + y * cropW, plane + (startY + y) * srcW + startX, cropW * sizeof(float));
        }
    }
}

/** Downsample NCHW by factor 2 (2x2 box average). src [C,H,W], dst [C, H/2, W/2]. */
static void downsample2x(const float* src, int H, int W, int C, float* dst) {
    const int outH = H / 2;
    const int outW = W / 2;
    for (int c = 0; c < C; ++c) {
        const float* sp = src + c * H * W;
        float* dp = dst + c * outH * outW;
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float v = sp[(2*y  )*W + (2*x)] + sp[(2*y  )*W + (2*x+1)] +
                         sp[(2*y+1)*W + (2*x)] + sp[(2*y+1)*W + (2*x+1)];
                dp[y * outW + x] = v * 0.25f;
            }
        }
    }
}

/** Downsample NCHW by factor 4 (4x4 box average). src [C,H,W], dst [C, H/4, W/4]. */
static void downsample4x(const float* src, int H, int W, int C, float* dst) {
    const int outH = H / 4;
    const int outW = W / 4;
    for (int c = 0; c < C; ++c) {
        const float* sp = src + c * H * W;
        float* dp = dst + c * outH * outW;
        for (int y = 0; y < outH; ++y) {
            for (int x = 0; x < outW; ++x) {
                float v = 0;
                for (int dy = 0; dy < 4; ++dy)
                    for (int dx = 0; dx < 4; ++dx)
                        v += sp[(4*y+dy)*W + (4*x+dx)];
                dp[y * outW + x] = v / 16.f;
            }
        }
    }
}

/** buildColumnOffsets(gS, pad): colOff[i] = 0, 21, 42, ... (24 - pad*2 for interior). */
static void buildColumnOffsets(int gS, int pad, int* colOff) {
    int x = 0;
    for (int i = 0; i < gS; ++i) {
        colOff[i] = x;
        int cellW = SPATIAL_SIZE;
        if (i != 0) cellW -= pad;
        if (i != gS - 1) cellW -= pad;
        x += cellW;
    }
}

/** reshapeToSpatial: tokens [577*1024 or 576*1024] -> out [1024, 24, 24] NCHW. Skip CLS if 577. */
static void reshapeToSpatial(const float* tokens, size_t tokenLen, float* out) {
    const int C = FEATURE_DIM;
    const int H = SPATIAL_SIZE;
    const int W = SPATIAL_SIZE;
    const int spatialCount = H * W;
    const bool hasCls = (tokenLen >= (size_t)((spatialCount + 1) * C));
    const int tokenOffset = hasCls ? 1 : 0;
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            int tokenIdx = (h * W + w) + tokenOffset;
            int outBase = h * W + w;
            for (int c = 0; c < C; ++c) {
                out[c * spatialCount + outBase] = tokens[tokenIdx * C + c];
            }
        }
    }
}

/** mergeCrop: copy cropped patch (24x24) into out at (rowOff[gJ], colOff[gI]). */
static void mergeCrop(float* out, int outW, const float* patch,
                      int gI, int gJ, int gS, int pad,
                      const int* rowOff, const int* colOff) {
    int cT = (gJ != 0) ? pad : 0;
    int cB = (gJ != gS - 1) ? pad : 0;
    int cL = (gI != 0) ? pad : 0;
    int cR = (gI != gS - 1) ? pad : 0;
    int cH = SPATIAL_SIZE - cT - cB;
    int cW = SPATIAL_SIZE - cL - cR;
    const int outHW = outW * outW;
    for (int c = 0; c < FEATURE_DIM; ++c) {
        const float* sB = patch + c * 576;
        float* dB = out + c * outHW;
        int rowStart = rowOff[gJ];
        int colStart = colOff[gI];
        for (int dy = 0; dy < cH; ++dy) {
            for (int dx = 0; dx < cW; ++dx) {
                dB[(rowStart + dy) * outW + (colStart + dx)] = sB[(cT + dy) * 24 + (cL + dx)];
            }
        }
    }
}

/** Precompute row offsets for 5x5 grid with pad 3: 0, 21, 39, 57, 78. */
static void buildRowOffsets1x(int* rowOff) {
    int cH0 = 24 - 0 - 3;
    int cHmid = 24 - 3 - 3;
    int cHlast = 24 - 3 - 0;
    rowOff[0] = 0;
    rowOff[1] = rowOff[0] + cH0;
    rowOff[2] = rowOff[1] + cHmid;
    rowOff[3] = rowOff[2] + cHmid;
    rowOff[4] = rowOff[3] + cHmid;
}

static void buildRowOffsets05x(int* rowOff) {
    int cH0 = 24 - 0 - 6;
    int cHmid = 24 - 6 - 6;
    int cHlast = 24 - 6 - 0;
    rowOff[0] = 0;
    rowOff[1] = rowOff[0] + cH0;
    rowOff[2] = rowOff[1] + cHmid;
}

static std::string pathJoin(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/') return dir + name;
    return dir + "/" + name;
}

extern "C" {

JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_runFullPipelineInt8Native(
        JNIEnv* env,
        jobject /* thiz */,
        jstring modelDirPath,
        jfloatArray imageNCHW) {

    if (modelDirPath == nullptr || imageNCHW == nullptr) {
        LOGE("runFullPipelineInt8: null modelDir or image");
        return nullptr;
    }

    ensureRuntimeInit();

    const char* dirC = env->GetStringUTFChars(modelDirPath, nullptr);
    std::string modelDir(dirC ? dirC : "");
    env->ReleaseStringUTFChars(modelDirPath, dirC);

    if (modelDir.empty()) {
        LOGE("runFullPipelineInt8: empty model dir");
        return nullptr;
    }
    LOGD("runFullPipelineInt8: modelDir=%s", modelDir.c_str());

    jsize imageLen = env->GetArrayLength(imageNCHW);
    if (imageLen != 3 * IMAGE_SIZE * IMAGE_SIZE) {
        LOGE("runFullPipelineInt8: bad image length %d", (int)imageLen);
        return nullptr;
    }

    jfloat* imageData = env->GetFloatArrayElements(imageNCHW, nullptr);
    if (imageData == nullptr) return nullptr;

    const int stride1x = (IMAGE_SIZE - PATCH_SIZE) / 4;
    const int halfSize = IMAGE_SIZE / 2;
    const int stride05x = (halfSize - PATCH_SIZE) / 2;

    std::vector<float> halfImg(3 * halfSize * halfSize);
    downsample2x(imageData, IMAGE_SIZE, IMAGE_SIZE, 3, halfImg.data());

    std::vector<float> quarterImg(3 * PATCH_SIZE * PATCH_SIZE);
    downsample4x(imageData, IMAGE_SIZE, IMAGE_SIZE, 3, quarterImg.data());

    std::vector<float> latent0(FEATURE_DIM * M_1X * M_1X, 0.f);
    std::vector<float> latent1(FEATURE_DIM * M_1X * M_1X, 0.f);
    std::vector<float> x0Feat(FEATURE_DIM * M_1X * M_1X, 0.f);
    std::vector<float> x1Feat(FEATURE_DIM * M_05X * M_05X, 0.f);
    std::vector<float> x2Feat(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE, 0.f);
    std::vector<float> tempSpatial(FEATURE_DIM * SPATIAL_SIZE * SPATIAL_SIZE);

    int colOffs1x[GRID_1X], rowOffs1x[GRID_1X];
    int colOffs05x[GRID_05X], rowOffs05x[GRID_05X];
    buildColumnOffsets(GRID_1X, PADDING_1X, colOffs1x);
    buildColumnOffsets(GRID_05X, PADDING_05X, colOffs05x);
    buildRowOffsets1x(rowOffs1x);
    buildRowOffsets05x(rowOffs05x);

    std::string path1 = pathJoin(modelDir, "sharp_split_part1_int8.pte");
    std::string path2 = pathJoin(modelDir, "sharp_split_part2_int8.pte");
    std::string path3 = pathJoin(modelDir, "sharp_split_part3_int8.pte");
    std::string path4a512 = pathJoin(modelDir, "sharp_split_part4a_chunk_512.pte");
    std::string path4a65 = pathJoin(modelDir, "sharp_split_part4a_chunk_65.pte");
    std::string path4b = pathJoin(modelDir, "sharp_split_part4b.pte");

    std::ifstream f1(path1);
    if (!f1.good()) {
        LOGE("Part1 model not found: %s", path1.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    f1.close();

    LOGD("Loading Part1+Part2...");
    auto mod1 = std::make_unique<ETModule>(path1, ETModule::LoadMode::Mmap);
    if (mod1->load() != Error::Ok) {
        LOGE("Failed to load Part1: %s", path1.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto mod2 = std::make_unique<ETModule>(path2, ETModule::LoadMode::Mmap);
    if (mod2->load() != Error::Ok) {
        LOGE("Failed to load Part2: %s", path2.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    LOGD("Part1+Part2 loaded. Starting 1x patches (5x5)...");

    std::vector<float> patchBuf(3 * PATCH_SIZE * PATCH_SIZE);
    std::vector<float> tokensCopy(TOKENS_577 * FEATURE_DIM);

    for (int i = 0; i < GRID_1X; ++i) {
        for (int j = 0; j < GRID_1X; ++j) {
            cropNCHW(imageData, IMAGE_SIZE, IMAGE_SIZE, 3,
                     i * stride1x, j * stride1x, PATCH_SIZE, PATCH_SIZE,
                     patchBuf.data());
            auto patchTensor = from_blob(patchBuf.data(), {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*patchTensor};
            auto out1 = mod1->forward(in1);
            if (!out1.ok() || out1->size() < 2) {
                LOGE("Part1 forward failed at 1x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const auto& tokensTensor = (*out1)[0].toTensor();
            const auto& spatialTensor = (*out1)[1].toTensor();
            const float* tokensPtr = tokensTensor.const_data_ptr<float>();
            const float* spatialPtr = spatialTensor.const_data_ptr<float>();
            size_t tokensLen = tokensTensor.numel();
            size_t spatialLen = spatialTensor.numel();

            reshapeToSpatial(spatialPtr, spatialLen, tempSpatial.data());
            mergeCrop(latent0.data(), M_1X, tempSpatial.data(), j, i, GRID_1X, PADDING_1X, rowOffs1x, colOffs1x);

            reshapeToSpatial(tokensPtr, tokensLen, tempSpatial.data());
            mergeCrop(latent1.data(), M_1X, tempSpatial.data(), j, i, GRID_1X, PADDING_1X, rowOffs1x, colOffs1x);

            std::memcpy(tokensCopy.data(), tokensPtr, tokensLen * sizeof(float));
            auto tokensInput = from_blob(tokensCopy.data(), {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokensInput};
            auto out2 = mod2->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 forward failed at 1x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            size_t featLen = (*out2)[0].toTensor().numel();
            reshapeToSpatial(featPtr, featLen, tempSpatial.data());
            mergeCrop(x0Feat.data(), M_1X, tempSpatial.data(), j, i, GRID_1X, PADDING_1X, rowOffs1x, colOffs1x);
        }
    }

    LOGD("1x patches done. Starting 0.5x patches (3x3)...");
    for (int i = 0; i < GRID_05X; ++i) {
        for (int j = 0; j < GRID_05X; ++j) {
            cropNCHW(halfImg.data(), halfSize, halfSize, 3,
                     i * stride05x, j * stride05x, PATCH_SIZE, PATCH_SIZE,
                     patchBuf.data());
            auto patchTensor = from_blob(patchBuf.data(), {1, 3, PATCH_SIZE, PATCH_SIZE});
            std::vector<EValue> in1 = {*patchTensor};
            auto out1 = mod1->forward(in1);
            if (!out1.ok() || out1->size() < 1) {
                LOGE("Part1 forward failed at 0.5x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* tokensPtr = (*out1)[0].toTensor().const_data_ptr<float>();
            std::memcpy(tokensCopy.data(), tokensPtr, (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float));
            auto tokensInput = from_blob(tokensCopy.data(), {1, TOKENS_577, FEATURE_DIM});
            std::vector<EValue> in2 = {*tokensInput};
            auto out2 = mod2->forward(in2);
            if (!out2.ok() || out2->empty()) {
                LOGE("Part2 forward failed at 0.5x (%d,%d)", i, j);
                env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
                return nullptr;
            }
            const float* featPtr = (*out2)[0].toTensor().const_data_ptr<float>();
            size_t featLen = (*out2)[0].toTensor().numel();
            reshapeToSpatial(featPtr, featLen, tempSpatial.data());
            mergeCrop(x1Feat.data(), M_05X, tempSpatial.data(), j, i, GRID_05X, PADDING_05X, rowOffs05x, colOffs05x);
        }
    }

    LOGD("0.5x patches done. Running 0.25x patch...");
    auto quarterTensor = from_blob(quarterImg.data(), {1, 3, PATCH_SIZE, PATCH_SIZE});
    std::vector<EValue> inQ = {*quarterTensor};
    auto outQ = mod1->forward(inQ);
    if (!outQ.ok() || outQ->size() < 1) {
        LOGE("Part1 forward failed for 0.25x");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* qTokensPtr = (*outQ)[0].toTensor().const_data_ptr<float>();
    std::memcpy(tokensCopy.data(), qTokensPtr, (size_t)TOKENS_577 * FEATURE_DIM * sizeof(float));
    auto qTokensInput = from_blob(tokensCopy.data(), {1, TOKENS_577, FEATURE_DIM});
    std::vector<EValue> inQ2 = {*qTokensInput};
    auto outQ2 = mod2->forward(inQ2);
    if (!outQ2.ok() || outQ2->empty()) {
        LOGE("Part2 forward failed for 0.25x");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* qFeatPtr = (*outQ2)[0].toTensor().const_data_ptr<float>();
    size_t qFeatLen = (*outQ2)[0].toTensor().numel();
    reshapeToSpatial(qFeatPtr, qFeatLen, x2Feat.data());

    LOGD("All 35 patches done. Releasing Part1+Part2, loading Part3...");
    mod1.reset();
    mod2.reset();

    auto mod3 = std::make_unique<ETModule>(path3, ETModule::LoadMode::Mmap);
    if (mod3->load() != Error::Ok) {
        LOGE("Failed to load Part3: %s", path3.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto fullImgTensor = from_blob(imageData, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
    std::vector<EValue> in3 = {*fullImgTensor};
    auto out3 = mod3->forward(in3);
    if (!out3.ok() || out3->empty()) {
        LOGE("Part3 forward failed");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* imgTokensPtr = (*out3)[0].toTensor().const_data_ptr<float>();
    size_t imgTokensNumel = (*out3)[0].toTensor().numel();
    LOGD("Part3 done: imgTokens numel=%zu", imgTokensNumel);

    std::vector<float> combinedTokens(imgTokensNumel);
    std::memcpy(combinedTokens.data(), imgTokensPtr, imgTokensNumel * sizeof(float));
    mod3.reset();

    LOGD("Loading Part4a (chunk 512)...");
    auto mod4a512 = std::make_unique<ETModule>(path4a512, ETModule::LoadMode::Mmap);
    if (mod4a512->load() != Error::Ok) {
        LOGE("Failed to load Part4a chunk 512: %s", path4a512.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto chunk512Tensor = from_blob(combinedTokens.data(), {1, 512, FEATURE_DIM});
    std::vector<EValue> in4a512 = {*chunk512Tensor};
    auto out4a512 = mod4a512->forward(in4a512);
    if (!out4a512.ok() || out4a512->empty()) {
        LOGE("Part4a chunk 512 forward failed");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out512Ptr = (*out4a512)[0].toTensor().const_data_ptr<float>();
    size_t out512Len = (*out4a512)[0].toTensor().numel();
    std::memcpy(combinedTokens.data(), out512Ptr, out512Len * sizeof(float));
    mod4a512.reset();

    LOGD("Part4a chunk 512 done (out512Len=%zu). Loading Part4a chunk 65...", out512Len);
    auto mod4a65 = std::make_unique<ETModule>(path4a65, ETModule::LoadMode::Mmap);
    if (mod4a65->load() != Error::Ok) {
        LOGE("Failed to load Part4a chunk 65: %s", path4a65.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    auto chunk65Tensor = from_blob(combinedTokens.data() + 512 * FEATURE_DIM, {1, 65, FEATURE_DIM});
    std::vector<EValue> in4a65 = {*chunk65Tensor};
    auto out4a65 = mod4a65->forward(in4a65);
    if (!out4a65.ok() || out4a65->empty()) {
        LOGE("Part4a chunk 65 forward failed");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }
    const float* out65Ptr = (*out4a65)[0].toTensor().const_data_ptr<float>();
    size_t out65Len = (*out4a65)[0].toTensor().numel();
    std::memcpy(combinedTokens.data() + 512 * FEATURE_DIM, out65Ptr, out65Len * sizeof(float));
    mod4a65.reset();

    LOGD("Part4a done. Loading Part4b (single)...");
    auto mod4b = std::make_unique<ETModule>(path4b, ETModule::LoadMode::Mmap);
    if (mod4b->load() != Error::Ok) {
        LOGE("Failed to load Part4b: %s", path4b.c_str());
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    auto tokens4b = from_blob(combinedTokens.data(), {1, TOKENS_577, FEATURE_DIM});
    auto img4b = from_blob(imageData, {1, 3, IMAGE_SIZE, IMAGE_SIZE});
    auto lat0_4b = from_blob(latent0.data(), {1, FEATURE_DIM, M_1X, M_1X});
    auto lat1_4b = from_blob(latent1.data(), {1, FEATURE_DIM, M_1X, M_1X});
    auto x0_4b = from_blob(x0Feat.data(), {1, FEATURE_DIM, M_1X, M_1X});
    auto x1_4b = from_blob(x1Feat.data(), {1, FEATURE_DIM, M_05X, M_05X});
    auto x2_4b = from_blob(x2Feat.data(), {1, FEATURE_DIM, SPATIAL_SIZE, SPATIAL_SIZE});

    std::vector<EValue> in4b;
    in4b.reserve(7);
    in4b.push_back(*tokens4b);
    in4b.push_back(*img4b);
    in4b.push_back(*lat0_4b);
    in4b.push_back(*lat1_4b);
    in4b.push_back(*x0_4b);
    in4b.push_back(*x1_4b);
    in4b.push_back(*x2_4b);

    auto out4b = mod4b->forward(in4b);

    if (!out4b.ok() || out4b->empty()) {
        LOGE("Part4b forward failed");
        env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);
        return nullptr;
    }

    const auto& gaussianTensor = (*out4b)[0].toTensor();
    const float* gaussianPtr = gaussianTensor.const_data_ptr<float>();
    const int numFloats = static_cast<int>(gaussianTensor.numel());

    env->ReleaseFloatArrayElements(imageNCHW, imageData, JNI_ABORT);

    if (numFloats <= 0 || (numFloats % PARAMS_PER_GAUSSIAN) != 0) {
        LOGE("Part4b output size invalid: %d", numFloats);
        return nullptr;
    }

    LOGD("[TIMING] C++ full pipeline done: %d Gaussians", numFloats / PARAMS_PER_GAUSSIAN);

    jfloatArray jResult = env->NewFloatArray(numFloats);
    if (jResult == nullptr) {
        LOGE("Failed to allocate result array");
        return nullptr;
    }
    env->SetFloatArrayRegion(jResult, 0, numFloats, gaussianPtr);

    return jResult;
}

} // extern "C"
