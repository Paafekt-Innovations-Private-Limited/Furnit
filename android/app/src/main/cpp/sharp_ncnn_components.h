/**
 * SHARP NCNN Component-Based Implementation (Memory Optimized)
 *
 * This header provides component-based SHARP inference using multiple NCNN models.
 * It processes patches serially to avoid the batch/channel dimension issue in NCNN.
 *
 * Memory Optimizations:
 * - Free patches immediately after processing
 * - Use reduced resolution feature maps (384x384 instead of 768x768)
 * - Process and free intermediate results aggressively
 * - Skip decoder and generate Gaussians directly from merged features
 */

#ifndef SHARP_NCNN_COMPONENTS_H
#define SHARP_NCNN_COMPONENTS_H

#include <vector>
#include <string>
#include <cmath>
#include <chrono>
#include <future>
#include <thread>
#include <atomic>
#include <cstdlib>
#include <android/log.h>
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"
#include "sharp_custom_layers.h"

#ifndef SHARP_LOG_TAG
#define SHARP_LOG_TAG "SharpNCNN"
#endif

#ifndef LOGD
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, SHARP_LOG_TAG, __VA_ARGS__)
#endif
#ifndef LOGI
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, SHARP_LOG_TAG, __VA_ARGS__)
#endif
#ifndef LOGE
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, SHARP_LOG_TAG, __VA_ARGS__)
#endif

namespace sharp_components {

// Patch configuration
static const int IMAGE_SIZE = 1536;
static const int PATCH_SIZE = 384;
static const int PATCH_EMBED_DIM = 1024;
static const int PATCH_TOKENS = 576;  // 24 * 24
static const int VIT_TOKENS = 577;    // 576 + 1 CLS token

// Patch counts per scale
static const int PATCHES_1X = 25;  // 5x5
static const int PATCHES_05X = 9;  // 3x3
static const int PATCHES_025X = 1; // 1x1
static const int TOTAL_PATCHES = 35;

// Memory-optimized output resolution (reduced from 768 to 384)
static const int OUTPUT_SIZE = 384;

/**
 * Component-based SHARP model with memory optimization
 */
class SharpComponents {
public:
    SharpComponents() = default;
    ~SharpComponents() { release(); }

    bool load(const std::string& modelDir, bool useVulkanCompute, int requestedThreads) {
        LOGI("Loading SHARP components from: %s", modelDir.c_str());
        useVulkanCompute_ = useVulkanCompute;
        modelThreads_ = resolveThreadCount(requestedThreads);
        patchWorkerCount_ = resolvePatchWorkerCount(modelThreads_);
        LOGI("Component runtime config: threads=%d, patchWorkers=%d, vulkan=%s",
             modelThreads_, patchWorkerCount_, useVulkanCompute_ ? "on" : "off");

        if (!loadEmbeddings(modelDir)) {
            LOGE("Failed to load embeddings");
            return false;
        }

        if (!loadModel(patchEmbedNet_, modelDir, "sharp_single_patch_embed")) {
            LOGE("Failed to load patch_embed model");
            return false;
        }

        if (!loadModel(patchEncoderNet_, modelDir, "sharp_single_patch_encoder")) {
            LOGE("Failed to load patch_encoder model");
            return false;
        }

        if (!loadModel(imageEncoderNet_, modelDir, "sharp_image_encoder")) {
            LOGE("Failed to load image_encoder model");
            return false;
        }

        // Load GaussianHead (lightweight alternative to decoder)
        if (!loadModel(gaussianHeadNet_, modelDir, "gaussian_head")) {
            LOGI("GaussianHead not found, will use placeholder generation");
        } else {
            hasGaussianHead_ = true;
            LOGI("GaussianHead loaded (3.5MB lightweight head)");
        }

        LOGI("All SHARP components loaded successfully");
        return true;
    }

    /**
     * Memory-optimized inference
     */
    int infer(const ncnn::Mat& image,
              std::vector<float>& positions,
              std::vector<float>& scales,
              std::vector<float>& rotations,
              std::vector<float>& colors,
              std::vector<float>& opacities) {
        LOGI("Infer runtime config: vulkan=%s threads=%d patchWorkers=%d",
             useVulkanCompute_ ? "on" : "off", modelThreads_, patchWorkerCount_);

        // Process patches one at a time to minimize peak memory
        // Store only the spatial features we need, not full ViT outputs

        std::vector<ncnn::Mat> spatialFeatures1x;
        std::vector<ncnn::Mat> spatialFeatures05x;
        ncnn::Mat spatialFeatures025x;

        spatialFeatures1x.reserve(PATCHES_1X);
        spatialFeatures05x.reserve(PATCHES_05X);

        // Create downsampled images once
        ncnn::Mat image05;
        ncnn::resize_bilinear(image, image05, IMAGE_SIZE / 2, IMAGE_SIZE / 2);

        ncnn::Mat image025;
        ncnn::resize_bilinear(image, image025, PATCH_SIZE, PATCH_SIZE);

        // Process 1.0x scale patches (5x5 grid) with bounded parallelism
        spatialFeatures1x.resize(PATCHES_1X);
        int stride1x = (IMAGE_SIZE - PATCH_SIZE) / 4;  // 288
        if (!processPatchGrid(image, 5, 5, stride1x, 0, PATCHES_1X, spatialFeatures1x)) {
            LOGE("Failed to process 1.0x patch grid");
            return -1;
        }

        // Process 0.5x scale patches (3x3 grid) with bounded parallelism
        spatialFeatures05x.resize(PATCHES_05X);
        int stride05 = (IMAGE_SIZE / 2 - PATCH_SIZE) / 2;  // 192
        if (!processPatchGrid(image05, 3, 3, stride05, PATCHES_1X, TOTAL_PATCHES, spatialFeatures05x)) {
            LOGE("Failed to process 0.5x patch grid");
            return -1;
        }
        // Free 0.5x image
        image05.release();

        // Process 0.25x scale patch
        {
            auto tStart = std::chrono::steady_clock::now();
            PatchStageTiming timing;
            ncnn::Mat features;
            if (!processPatch(image025, features, timing)) {
                LOGE("Failed to process 0.25x patch");
                return -1;
            }
            auto reshapeStart = std::chrono::steady_clock::now();
            spatialFeatures025x = reshapeViTOutput(features);
            auto reshapeEnd = std::chrono::steady_clock::now();
            timing.reshapeMs = std::chrono::duration_cast<std::chrono::milliseconds>(reshapeEnd - reshapeStart).count();
            timing.totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(reshapeEnd - tStart).count();
            LOGI("Patch 35/35 total=%ldms crop=%ld embed=%ld clspos=%ld enc=%ld reshape=%ld",
                 timing.totalMs, timing.cropMs, timing.embedMs, timing.clsPosMs, timing.encoderMs, timing.reshapeMs);
        }

        // Run image encoder on 0.25x image
        auto imageEncoderStart = std::chrono::steady_clock::now();
        ncnn::Mat imageFeatures;
        if (!processImageEncoder(image025, imageFeatures)) {
            LOGE("Failed to run image encoder");
            return -1;
        }
        auto imageEncoderEnd = std::chrono::steady_clock::now();
        LOGI("ImageEncoder: %lldms",
             (long long)std::chrono::duration_cast<std::chrono::milliseconds>(imageEncoderEnd - imageEncoderStart).count());
        image025.release();

        // Merge patches into spatial feature maps with reduced memory

        // Merge 1.0x scale: 5x5 grid -> [96, 96, 1024]
        auto merge1xStart = std::chrono::steady_clock::now();
        ncnn::Mat merged1x = mergePatchGrid(spatialFeatures1x, 5, 3);
        auto merge1xEnd = std::chrono::steady_clock::now();
        LOGI("Merge1x: %lldms",
             (long long)std::chrono::duration_cast<std::chrono::milliseconds>(merge1xEnd - merge1xStart).count());
        // Free 1x spatial features immediately
        spatialFeatures1x.clear();
        spatialFeatures1x.shrink_to_fit();

        // Merge 0.5x scale: 3x3 grid -> [48, 48, 1024]
        auto merge05xStart = std::chrono::steady_clock::now();
        ncnn::Mat merged05x = mergePatchGrid(spatialFeatures05x, 3, 6);
        auto merge05xEnd = std::chrono::steady_clock::now();
        LOGI("Merge05x: %lldms",
             (long long)std::chrono::duration_cast<std::chrono::milliseconds>(merge05xEnd - merge05xStart).count());
        // Free 0.5x spatial features
        spatialFeatures05x.clear();
        spatialFeatures05x.shrink_to_fit();

        // Free image features (not needed for simplified approach)
        imageFeatures.release();

        // Free unused features
        merged05x.release();
        spatialFeatures025x.release();

        // Use GaussianHead if available (lightweight 3.5MB model)
        if (hasGaussianHead_) {
            ncnn::Mat headOutput;
            {
                ncnn::Extractor ex = gaussianHeadNet_.create_extractor();
                ex.set_light_mode(true);
                ex.input("in0", merged1x);

                int ret = ex.extract("out0", headOutput);
                if (ret != 0) {
                    LOGE("GaussianHead failed with code %d, falling back to placeholder", ret);
                    hasGaussianHead_ = false;  // Don't try again
                }
            }

            merged1x.release();

            if (headOutput.empty()) {
                LOGE("GaussianHead produced empty output");
                return -1;
            }

            // Generate Gaussians from GaussianHead output
            // Output is [14, 384, 384]:
            //   channels 0-2: xyz positions (raw, apply tanh*2)
            //   channel 3: opacity (raw, apply sigmoid)
            //   channels 4-6: scales (raw, apply softplus+offset)
            //   channels 7-10: rotation quaternion (raw, normalize)
            //   channels 11-13: RGB color (raw, apply sigmoid)

            int outH = headOutput.h;
            int outW = headOutput.w;
            int numGaussians = outH * outW;

            positions.resize(numGaussians * 3);
            scales.resize(numGaussians * 3);
            rotations.resize(numGaussians * 4);
            colors.resize(numGaussians * 3);
            opacities.resize(numGaussians);

            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    int idx = y * outW + x;

                    // Position: trained model outputs final values directly (L1 loss)
                    // No activation needed - values are in [-2, 2] range from training
                    positions[idx * 3 + 0] = headOutput.channel(0)[idx];
                    positions[idx * 3 + 1] = headOutput.channel(1)[idx];
                    positions[idx * 3 + 2] = headOutput.channel(2)[idx];

                    // Opacity: trained with BCE with logits, so apply sigmoid
                    float rawOpacity = headOutput.channel(3)[idx];
                    opacities[idx] = 1.0f / (1.0f + std::exp(-rawOpacity));

                    // Scale: trained model outputs final scale values directly (L1 loss)
                    // These are already log-space values from SHARP
                    // Clamp to reasonable range to avoid rendering issues
                    scales[idx * 3 + 0] = std::max(0.001f, headOutput.channel(4)[idx]);
                    scales[idx * 3 + 1] = std::max(0.001f, headOutput.channel(5)[idx]);
                    scales[idx * 3 + 2] = std::max(0.001f, headOutput.channel(6)[idx]);

                    // Rotation quaternion: normalize
                    float qw = headOutput.channel(7)[idx];
                    float qx = headOutput.channel(8)[idx];
                    float qy = headOutput.channel(9)[idx];
                    float qz = headOutput.channel(10)[idx];
                    float qnorm = std::sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
                    if (qnorm > 1e-6f) {
                        rotations[idx * 4 + 0] = qw / qnorm;
                        rotations[idx * 4 + 1] = qx / qnorm;
                        rotations[idx * 4 + 2] = qy / qnorm;
                        rotations[idx * 4 + 3] = qz / qnorm;
                    } else {
                        rotations[idx * 4 + 0] = 1.0f;
                        rotations[idx * 4 + 1] = 0.0f;
                        rotations[idx * 4 + 2] = 0.0f;
                        rotations[idx * 4 + 3] = 0.0f;
                    }

                    // Color: model outputs BGR; swap to RGB to match ONNX
                    colors[idx * 3 + 0] = std::max(0.0f, std::min(1.0f, headOutput.channel(13)[idx]));
                    colors[idx * 3 + 1] = std::max(0.0f, std::min(1.0f, headOutput.channel(12)[idx]));
                    colors[idx * 3 + 2] = std::max(0.0f, std::min(1.0f, headOutput.channel(11)[idx]));
                }
            }

            headOutput.release();
            return numGaussians;
        }

        // Fallback: generate placeholder Gaussians from merged features
        int numGaussians = generateGaussiansFromFeatures(
            merged1x, ncnn::Mat(), ncnn::Mat(),
            positions, scales, rotations, colors, opacities);

        merged1x.release();

        return numGaussians;
    }

    void release() {
        patchEmbedNet_.clear();
        patchEncoderNet_.clear();
        imageEncoderNet_.clear();
        gaussianHeadNet_.clear();

        if (clsToken_) { delete[] clsToken_; clsToken_ = nullptr; }
        if (posEmbed_) { delete[] posEmbed_; posEmbed_ = nullptr; }
    }

private:
    struct PatchStageTiming {
        long cropMs = 0;
        long embedMs = 0;
        long clsPosMs = 0;
        long encoderMs = 0;
        long reshapeMs = 0;
        long totalMs = 0;
    };

    int resolveThreadCount(int requestedThreads) const {
        const char* overrideThreads = std::getenv("SHARP_NCNN_THREADS");
        if (overrideThreads != nullptr) {
            int forcedThreads = std::atoi(overrideThreads);
            if (forcedThreads > 0) {
                return std::max(1, std::min(8, forcedThreads));
            }
        }
        if (requestedThreads > 0) {
            return std::max(1, std::min(8, requestedThreads));
        }
        int bigCoreCount = ncnn::get_big_cpu_count();
        if (bigCoreCount <= 0) {
            bigCoreCount = ncnn::get_cpu_count();
        }
        return std::max(1, std::min(4, bigCoreCount));
    }

    int resolvePatchWorkerCount(int modelThreads) const {
        const char* overrideWorkers = std::getenv("SHARP_NCNN_PATCH_WORKERS");
        if (overrideWorkers != nullptr) {
            int forcedWorkers = std::atoi(overrideWorkers);
            if (forcedWorkers > 0) {
                return std::max(1, std::min(4, forcedWorkers));
            }
        }
        return std::max(1, std::min(2, modelThreads));
    }

    bool processPatchGrid(
            const ncnn::Mat& sourceImage,
            int rows,
            int columns,
            int stride,
            int globalPatchOffset,
            int totalPatchCount,
            std::vector<ncnn::Mat>& outputSpatialFeatures) {
        const int patchCount = rows * columns;
        std::atomic<int> nextPatchIndex(0);
        std::atomic<bool> encounteredFailure(false);
        const int workerCount = std::max(1, std::min(patchWorkerCount_, patchCount));
        std::vector<std::future<void>> workers;
        workers.reserve(workerCount);

        for (int workerIndex = 0; workerIndex < workerCount; workerIndex++) {
            workers.push_back(std::async(std::launch::async, [&, workerIndex]() {
                while (!encounteredFailure.load(std::memory_order_relaxed)) {
                    int localPatchIndex = nextPatchIndex.fetch_add(1);
                    if (localPatchIndex >= patchCount) {
                        return;
                    }

                    int row = localPatchIndex / columns;
                    int col = localPatchIndex % columns;
                    int patchX = col * stride;
                    int patchY = row * stride;

                    auto patchStart = std::chrono::steady_clock::now();
                    auto cropStart = patchStart;
                    ncnn::Mat patch = cropPatch(sourceImage, patchX, patchY, PATCH_SIZE, PATCH_SIZE);
                    auto cropEnd = std::chrono::steady_clock::now();

                    PatchStageTiming timing;
                    timing.cropMs = std::chrono::duration_cast<std::chrono::milliseconds>(cropEnd - cropStart).count();

                    ncnn::Mat encodedPatch;
                    if (!processPatch(patch, encodedPatch, timing)) {
                        encounteredFailure.store(true, std::memory_order_relaxed);
                        return;
                    }

                    auto reshapeStart = std::chrono::steady_clock::now();
                    outputSpatialFeatures[localPatchIndex] = reshapeViTOutput(encodedPatch);
                    auto patchEnd = std::chrono::steady_clock::now();
                    timing.reshapeMs = std::chrono::duration_cast<std::chrono::milliseconds>(patchEnd - reshapeStart).count();
                    timing.totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(patchEnd - patchStart).count();

                    int globalPatchIndex = globalPatchOffset + localPatchIndex + 1;
                    LOGI("Patch %d/%d total=%ldms crop=%ld embed=%ld clspos=%ld enc=%ld reshape=%ld",
                         globalPatchIndex, totalPatchCount, timing.totalMs, timing.cropMs, timing.embedMs,
                         timing.clsPosMs, timing.encoderMs, timing.reshapeMs);
                }
            }));
        }

        for (std::future<void>& worker : workers) {
            worker.get();
        }
        return !encounteredFailure.load(std::memory_order_relaxed);
    }

    ncnn::Net patchEmbedNet_;
    ncnn::Net patchEncoderNet_;
    ncnn::Net imageEncoderNet_;
    ncnn::Net gaussianHeadNet_;

    float* clsToken_ = nullptr;
    float* posEmbed_ = nullptr;
    bool hasGaussianHead_ = false;
    bool useVulkanCompute_ = false;
    int modelThreads_ = 1;
    int patchWorkerCount_ = 1;

    bool loadModel(ncnn::Net& net, const std::string& dir, const std::string& name) {
        std::string paramPath = dir + "/" + name + ".ncnn.param";
        std::string binPath = dir + "/" + name + ".ncnn.bin";

        net.opt.num_threads = modelThreads_;
        net.opt.use_vulkan_compute = useVulkanCompute_;
        net.opt.lightmode = true;
        net.opt.use_fp16_packed = false;
        net.opt.use_fp16_storage = false;
        net.opt.use_fp16_arithmetic = false;
        net.opt.use_packing_layout = false;
        net.opt.use_winograd_convolution = true;
        net.opt.use_sgemm_convolution = true;
        net.opt.use_int8_inference = false;
        net.opt.use_bf16_storage = false;

        // Keep built-in SDPA (do not override) for better NCNN backend usage.
        sharp_layers::register_custom_layers_without_sdpa(net);

        int ret = net.load_param(paramPath.c_str());
        if (ret != 0) {
            LOGE("Failed to load param: %s", paramPath.c_str());
            return false;
        }

        ret = net.load_model(binPath.c_str());
        if (ret != 0) {
            LOGE("Failed to load model: %s", binPath.c_str());
            return false;
        }

        LOGD("Loaded model: %s", name.c_str());
        return true;
    }

    bool loadEmbeddings(const std::string& dir) {
        std::string clsPath = dir + "/patch_cls_token.bin";
        FILE* f = fopen(clsPath.c_str(), "rb");
        if (!f) {
            LOGE("Failed to open CLS token file: %s", clsPath.c_str());
            return false;
        }
        clsToken_ = new float[PATCH_EMBED_DIM];
        fread(clsToken_, sizeof(float), PATCH_EMBED_DIM, f);
        fclose(f);
        LOGD("Loaded CLS token");

        std::string posPath = dir + "/patch_pos_embed.bin";
        f = fopen(posPath.c_str(), "rb");
        if (!f) {
            LOGE("Failed to open pos embed file: %s", posPath.c_str());
            return false;
        }
        posEmbed_ = new float[VIT_TOKENS * PATCH_EMBED_DIM];
        fread(posEmbed_, sizeof(float), VIT_TOKENS * PATCH_EMBED_DIM, f);
        fclose(f);
        LOGD("Loaded positional embeddings");

        return true;
    }

    ncnn::Mat cropPatch(const ncnn::Mat& src, int x, int y, int w, int h) {
        ncnn::Mat dst(w, h, src.c);

        for (int c = 0; c < src.c; c++) {
            const float* srcChannel = src.channel(c);
            float* dstChannel = dst.channel(c);

            for (int dy = 0; dy < h; dy++) {
                for (int dx = 0; dx < w; dx++) {
                    dstChannel[dy * w + dx] = srcChannel[(y + dy) * src.w + (x + dx)];
                }
            }
        }

        return dst;
    }

    bool processPatch(const ncnn::Mat& patch, ncnn::Mat& output, PatchStageTiming& timing) {
        // Step 1: Run patch embed -> [576, 1024]
        ncnn::Mat embedded;
        {
            auto embedStart = std::chrono::steady_clock::now();
            ncnn::Extractor ex = patchEmbedNet_.create_extractor();
            ex.set_light_mode(true);
            ex.input("in0", patch);
            if (ex.extract("out0", embedded) != 0) {
                LOGE("patch_embed failed");
                return false;
            }
            auto embedEnd = std::chrono::steady_clock::now();
            timing.embedMs = std::chrono::duration_cast<std::chrono::milliseconds>(embedEnd - embedStart).count();
        }

        // Step 2: Add CLS token -> [577, 1024]
        auto clsPosStart = std::chrono::steady_clock::now();
        ncnn::Mat withCls(PATCH_EMBED_DIM, VIT_TOKENS);

        float* dstCls = withCls.row(0);
        memcpy(dstCls, clsToken_, PATCH_EMBED_DIM * sizeof(float));

        for (int i = 0; i < PATCH_TOKENS; i++) {
            const float* srcRow = embedded.row(i);
            float* dstRow = withCls.row(i + 1);
            memcpy(dstRow, srcRow, PATCH_EMBED_DIM * sizeof(float));
        }

        // Free embedded immediately
        embedded.release();

        // Step 3: Add positional embeddings
        for (int i = 0; i < VIT_TOKENS; i++) {
            float* row = withCls.row(i);
            const float* posRow = posEmbed_ + i * PATCH_EMBED_DIM;
            for (int j = 0; j < PATCH_EMBED_DIM; j++) {
                row[j] += posRow[j];
            }
        }
        auto clsPosEnd = std::chrono::steady_clock::now();
        timing.clsPosMs = std::chrono::duration_cast<std::chrono::milliseconds>(clsPosEnd - clsPosStart).count();

        // Step 4: Run patch encoder -> [577, 1024]
        {
            auto encoderStart = std::chrono::steady_clock::now();
            ncnn::Extractor ex = patchEncoderNet_.create_extractor();
            ex.set_light_mode(true);
            ex.input("in0", withCls);
            if (ex.extract("out0", output) != 0) {
                LOGE("patch_encoder failed");
                return false;
            }
            auto encoderEnd = std::chrono::steady_clock::now();
            timing.encoderMs = std::chrono::duration_cast<std::chrono::milliseconds>(encoderEnd - encoderStart).count();
        }

        return true;
    }

    bool processImageEncoder(const ncnn::Mat& lowResImage, ncnn::Mat& output) {
        ncnn::Extractor ex = imageEncoderNet_.create_extractor();
        ex.set_light_mode(true);
        ex.input("in0", lowResImage);
        return ex.extract("out0", output) == 0;
    }

    ncnn::Mat reshapeViTOutput(const ncnn::Mat& vitOutput) {
        int spatialSize = 24;
        ncnn::Mat spatial(spatialSize, spatialSize, PATCH_EMBED_DIM);

        for (int y = 0; y < spatialSize; y++) {
            for (int x = 0; x < spatialSize; x++) {
                int tokenIdx = y * spatialSize + x + 1;
                const float* srcRow = vitOutput.row(tokenIdx);

                for (int c = 0; c < PATCH_EMBED_DIM; c++) {
                    float* dstChannel = spatial.channel(c);
                    dstChannel[y * spatialSize + x] = srcRow[c];
                }
            }
        }

        return spatial;
    }

    ncnn::Mat mergePatchGrid(const std::vector<ncnn::Mat>& patches,
                             int gridSize, int padding) {
        if (patches.empty()) return ncnn::Mat();

        int patchH = patches[0].h;
        int patchW = patches[0].w;
        int channels = patches[0].c;

        int patchContrib = patchH - 2 * padding;
        int outSize = patchH + (gridSize - 1) * patchContrib;

        ncnn::Mat output(outSize, outSize, channels);
        output.fill(0.0f);

        int idx = 0;
        int outY = 0;

        for (int j = 0; j < gridSize; j++) {
            int outX = 0;
            int srcY0 = (j == 0) ? 0 : padding;
            int srcY1 = (j == gridSize - 1) ? patchH : (patchH - padding);
            int copyH = srcY1 - srcY0;

            for (int i = 0; i < gridSize; i++) {
                const ncnn::Mat& patch = patches[idx++];

                int srcX0 = (i == 0) ? 0 : padding;
                int srcX1 = (i == gridSize - 1) ? patchW : (patchW - padding);
                int copyW = srcX1 - srcX0;

                for (int c = 0; c < channels; c++) {
                    const float* srcChannel = patch.channel(c);
                    float* dstChannel = output.channel(c);

                    for (int dy = 0; dy < copyH; dy++) {
                        for (int dx = 0; dx < copyW; dx++) {
                            int srcIdx = (srcY0 + dy) * patchW + (srcX0 + dx);
                            int dstIdx = (outY + dy) * outSize + (outX + dx);
                            dstChannel[dstIdx] = srcChannel[srcIdx];
                        }
                    }
                }

                outX += copyW;
            }
            outY += copyH;
        }

        return output;
    }

    /**
     * Generate Gaussians directly from merged features without running decoder
     * This saves significant memory by avoiding large intermediate feature maps
     */
    int generateGaussiansFromFeatures(
            const ncnn::Mat& merged1x,    // [1024, 96, 96]
            const ncnn::Mat& merged05x,   // [1024, 48, 48]
            const ncnn::Mat& merged025x,  // [1024, 24, 24]
            std::vector<float>& positions,
            std::vector<float>& scales,
            std::vector<float>& rotations,
            std::vector<float>& colors,
            std::vector<float>& opacities) {

        // Use reduced output size for memory efficiency
        int outH = OUTPUT_SIZE;
        int outW = OUTPUT_SIZE;
        int numGaussians = outH * outW;

        positions.resize(numGaussians * 3);
        scales.resize(numGaussians * 3);
        rotations.resize(numGaussians * 4);
        colors.resize(numGaussians * 3);
        opacities.resize(numGaussians);

        // Upsample merged1x to output size for feature extraction
        ncnn::Mat upsampled;
        ncnn::resize_bilinear(merged1x, upsampled, outW, outH);

        // Generate Gaussians from upsampled features
        // Create a visible 3D point cloud from encoder features

        int idx = 0;
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                // Base position: spread across 3D space [-2, 2] range for visibility
                float baseX = ((float)x / outW - 0.5f) * 4.0f;  // -2 to 2
                float baseY = ((float)y / outH - 0.5f) * 4.0f;  // -2 to 2

                // Use features to create depth variation
                float depthFeature = 0.0f;
                if (upsampled.c >= 3) {
                    // Average multiple channels for more stable depth
                    depthFeature = (upsampled.channel(0)[y * outW + x] +
                                   upsampled.channel(1)[y * outW + x] +
                                   upsampled.channel(2)[y * outW + x]) / 3.0f;
                }
                // Map depth feature to Z range [-1, 1] for 3D effect
                float baseZ = depthFeature * 0.5f;  // Reasonable depth range

                positions[idx * 3 + 0] = baseX;
                positions[idx * 3 + 1] = baseY;
                positions[idx * 3 + 2] = baseZ;

                // Larger scales for visibility (0.02 to 0.05)
                float scaleVal = 0.03f;
                if (upsampled.c >= 6) {
                    float featScale = std::abs(upsampled.channel(3)[y * outW + x]);
                    scaleVal = 0.02f + featScale * 0.03f;
                    scaleVal = std::max(0.02f, std::min(0.08f, scaleVal));
                }
                scales[idx * 3 + 0] = scaleVal;
                scales[idx * 3 + 1] = scaleVal;
                scales[idx * 3 + 2] = scaleVal * 0.5f;  // Slightly flat splats

                // Identity rotation
                rotations[idx * 4 + 0] = 1.0f;
                rotations[idx * 4 + 1] = 0.0f;
                rotations[idx * 4 + 2] = 0.0f;
                rotations[idx * 4 + 3] = 0.0f;

                // Color from features - more vibrant
                if (upsampled.c >= 9) {
                    // Use sigmoid-like mapping for better color distribution
                    float r = upsampled.channel(6)[y * outW + x];
                    float g = upsampled.channel(7)[y * outW + x];
                    float b = upsampled.channel(8)[y * outW + x];
                    // Normalize with sigmoid: 1/(1+exp(-x))
                    colors[idx * 3 + 0] = 1.0f / (1.0f + std::exp(-r * 2.0f));
                    colors[idx * 3 + 1] = 1.0f / (1.0f + std::exp(-g * 2.0f));
                    colors[idx * 3 + 2] = 1.0f / (1.0f + std::exp(-b * 2.0f));
                } else {
                    // Gradient based on position for visibility
                    colors[idx * 3 + 0] = (float)x / outW;
                    colors[idx * 3 + 1] = (float)y / outH;
                    colors[idx * 3 + 2] = 0.5f;
                }

                // High opacity for visibility
                opacities[idx] = 0.9f;

                idx++;
            }
        }

        upsampled.release();

        return numGaussians;
    }
};

} // namespace sharp_components

#endif // SHARP_NCNN_COMPONENTS_H
