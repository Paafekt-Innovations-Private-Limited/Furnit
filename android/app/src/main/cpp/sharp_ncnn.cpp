/**
 * SHARP NCNN JNI Wrapper
 *
 * This file provides JNI bindings for running SHARP 3D Gaussian Splat model
 * using NCNN on Android. SHARP generates 3D Gaussian splats from single images
 * for room reconstruction.
 *
 * Model Architecture:
 * - Input: 1536×1536 RGB image (normalized [0,1])
 * - Outputs: 5 tensors → ~1.1M Gaussians
 *   - positions (N × 3)
 *   - scales (N × 3)
 *   - rotations (N × 4)
 *   - colors (N × 3)
 *   - opacity (N)
 *
 * Setup:
 * 1. Convert SHARP PyTorch model to NCNN using pnnx
 * 2. Place sharp.ncnn.param and sharp.ncnn.bin in assets/
 */

#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// NCNN headers
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"

#define LOG_TAG "SharpNCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// SHARP model configuration
static const int INPUT_SIZE = 1536;
static const int PARAMS_PER_GAUSSIAN = 14;  // pos(3) + scale(3) + rot(4) + opacity(1) + color(3)

// Global NCNN network
static ncnn::Net* g_net = nullptr;

extern "C" {

/**
 * Initialize NCNN network from asset files
 */
JNIEXPORT jlong JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeInit(
    JNIEnv* env,
    jobject thiz,
    jobject assetManager,
    jstring paramPath,
    jstring binPath,
    jboolean useGpu,
    jint numThreads
) {
    if (g_net != nullptr) {
        delete g_net;
        g_net = nullptr;
    }

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("Failed to get AssetManager");
        return 0;
    }

    const char* paramPathStr = env->GetStringUTFChars(paramPath, nullptr);
    const char* binPathStr = env->GetStringUTFChars(binPath, nullptr);

    LOGI("Loading SHARP NCNN model: %s, %s", paramPathStr, binPathStr);

    g_net = new ncnn::Net();

    // Configure network options (CPU-only, optimized for mobile)
    g_net->opt.use_vulkan_compute = false;  // CPU-only build
    g_net->opt.num_threads = numThreads > 0 ? numThreads : 4;
    g_net->opt.use_fp16_packed = true;
    g_net->opt.use_fp16_storage = true;
    g_net->opt.use_fp16_arithmetic = true;
    g_net->opt.use_packing_layout = true;
    g_net->opt.lightmode = true;  // Reduce memory usage

    // Load param from assets
    AAsset* paramAsset = AAssetManager_open(mgr, paramPathStr, AASSET_MODE_BUFFER);
    if (!paramAsset) {
        LOGE("Failed to open param asset: %s", paramPathStr);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    size_t paramSize = AAsset_getLength(paramAsset);
    const void* paramData = AAsset_getBuffer(paramAsset);
    LOGI("Param size: %zu bytes", paramSize);

    int ret = g_net->load_param_mem(static_cast<const char*>(paramData));
    AAsset_close(paramAsset);

    if (ret != 0) {
        LOGE("Failed to load param: %d", ret);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    // Load bin from assets (large file, use streaming mode)
    AAsset* binAsset = AAssetManager_open(mgr, binPathStr, AASSET_MODE_STREAMING);
    if (!binAsset) {
        LOGE("Failed to open bin asset: %s", binPathStr);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    size_t binSize = AAsset_getLength(binAsset);
    LOGI("Bin size: %zu bytes (%.1f MB)", binSize, binSize / 1024.0 / 1024.0);

    // Allocate buffer and read entire file
    unsigned char* binBuffer = (unsigned char*)malloc(binSize);
    if (!binBuffer) {
        LOGE("Failed to allocate %zu bytes for model", binSize);
        AAsset_close(binAsset);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    // Read in chunks
    size_t totalRead = 0;
    while (totalRead < binSize) {
        int bytesRead = AAsset_read(binAsset, binBuffer + totalRead, binSize - totalRead);
        if (bytesRead <= 0) {
            LOGE("Failed to read bin asset at offset %zu", totalRead);
            free(binBuffer);
            AAsset_close(binAsset);
            env->ReleaseStringUTFChars(paramPath, paramPathStr);
            env->ReleaseStringUTFChars(binPath, binPathStr);
            delete g_net;
            g_net = nullptr;
            return 0;
        }
        totalRead += bytesRead;
    }
    AAsset_close(binAsset);
    LOGI("Read %zu bytes from bin asset", totalRead);

    ret = g_net->load_model(binBuffer);
    free(binBuffer);

    if (ret != 0) {
        LOGE("Failed to load model: %d", ret);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    env->ReleaseStringUTFChars(paramPath, paramPathStr);
    env->ReleaseStringUTFChars(binPath, binPathStr);

    LOGI("SHARP NCNN model loaded successfully, input size: %d", INPUT_SIZE);

    return reinterpret_cast<jlong>(g_net);
}

/**
 * Run SHARP inference on a bitmap
 * Returns float array with interleaved Gaussian parameters:
 * [x, y, z, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z, opacity, r, g, b] × N
 */
JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeInfer(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jobject bitmap
) {
    ncnn::Net* net = reinterpret_cast<ncnn::Net*>(handle);
    if (net == nullptr) {
        LOGE("Invalid network handle");
        return nullptr;
    }

    // Get bitmap info
    AndroidBitmapInfo bitmapInfo;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get bitmap info");
        return nullptr;
    }

    if (bitmapInfo.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
        LOGE("Unsupported bitmap format: %d", bitmapInfo.format);
        return nullptr;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock bitmap pixels");
        return nullptr;
    }

    int srcWidth = bitmapInfo.width;
    int srcHeight = bitmapInfo.height;

    LOGD("Input bitmap: %dx%d, target: %dx%d", srcWidth, srcHeight, INPUT_SIZE, INPUT_SIZE);

    // Convert RGBA to NCNN Mat and resize to 1536×1536
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        static_cast<const unsigned char*>(pixels),
        ncnn::Mat::PIXEL_RGBA2RGB,
        srcWidth, srcHeight,
        INPUT_SIZE, INPUT_SIZE
    );

    AndroidBitmap_unlockPixels(env, bitmap);

    // Normalize to [0, 1]
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // Run inference
    ncnn::Extractor ex = net->create_extractor();

    // Input layer name (from SHARP model conversion)
    // The exact name depends on the pnnx conversion output
    ex.input("in0", in);

    // Extract outputs
    // SHARP model outputs 5 tensors:
    // - positions: N × 3
    // - scales: N × 3
    // - rotations: N × 4
    // - colors: N × 3
    // - opacity: N

    ncnn::Mat positionsOut, scalesOut, rotationsOut, colorsOut, opacityOut;

    // Output layer names (these should match the pnnx-converted model)
    // Common naming conventions after pnnx conversion
    int ret = ex.extract("positions", positionsOut);
    if (ret != 0) {
        // Try alternative naming
        ret = ex.extract("out0", positionsOut);
    }
    if (ret != 0) {
        LOGE("Failed to extract positions output");
        return nullptr;
    }

    ret = ex.extract("scales", scalesOut);
    if (ret != 0) {
        ret = ex.extract("out1", scalesOut);
    }
    if (ret != 0) {
        LOGE("Failed to extract scales output");
        return nullptr;
    }

    ret = ex.extract("rotations", rotationsOut);
    if (ret != 0) {
        ret = ex.extract("out2", rotationsOut);
    }
    if (ret != 0) {
        LOGE("Failed to extract rotations output");
        return nullptr;
    }

    ret = ex.extract("colors", colorsOut);
    if (ret != 0) {
        ret = ex.extract("out3", colorsOut);
    }
    if (ret != 0) {
        LOGE("Failed to extract colors output");
        return nullptr;
    }

    ret = ex.extract("opacity", opacityOut);
    if (ret != 0) {
        ret = ex.extract("out4", opacityOut);
    }
    if (ret != 0) {
        LOGE("Failed to extract opacity output");
        return nullptr;
    }

    LOGD("Positions output: c=%d h=%d w=%d", positionsOut.c, positionsOut.h, positionsOut.w);
    LOGD("Scales output: c=%d h=%d w=%d", scalesOut.c, scalesOut.h, scalesOut.w);
    LOGD("Rotations output: c=%d h=%d w=%d", rotationsOut.c, rotationsOut.h, rotationsOut.w);
    LOGD("Colors output: c=%d h=%d w=%d", colorsOut.c, colorsOut.h, colorsOut.w);
    LOGD("Opacity output: c=%d h=%d w=%d", opacityOut.c, opacityOut.h, opacityOut.w);

    // Determine number of Gaussians from positions output
    // Positions should be [N, 3] or [1, N, 3]
    int numGaussians = 0;

    // Handle different possible output shapes
    if (positionsOut.dims == 2) {
        // Shape [N, 3]
        numGaussians = positionsOut.h;
    } else if (positionsOut.dims == 3) {
        // Shape [1, N, 3] or [N, 3, 1]
        numGaussians = positionsOut.c > positionsOut.w ? positionsOut.h : positionsOut.c;
    } else if (positionsOut.dims == 1) {
        // Flattened [N*3]
        numGaussians = positionsOut.w / 3;
    } else {
        LOGE("Unexpected positions output dims: %d", positionsOut.dims);
        return nullptr;
    }

    LOGI("Processing %d Gaussians", numGaussians);

    if (numGaussians <= 0) {
        LOGE("Invalid Gaussian count: %d", numGaussians);
        return nullptr;
    }

    // Allocate result array
    int totalSize = numGaussians * PARAMS_PER_GAUSSIAN;
    std::vector<float> result(totalSize);

    // Interleave Gaussian parameters
    // Format: [x, y, z, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z, opacity, r, g, b]

    for (int i = 0; i < numGaussians; i++) {
        int offset = i * PARAMS_PER_GAUSSIAN;

        // Positions (3 values)
        if (positionsOut.dims == 2) {
            result[offset + 0] = positionsOut.row(i)[0];
            result[offset + 1] = positionsOut.row(i)[1];
            result[offset + 2] = positionsOut.row(i)[2];
        } else {
            // Handle flattened or different layout
            result[offset + 0] = ((float*)positionsOut.data)[i * 3 + 0];
            result[offset + 1] = ((float*)positionsOut.data)[i * 3 + 1];
            result[offset + 2] = ((float*)positionsOut.data)[i * 3 + 2];
        }

        // Scales (3 values)
        if (scalesOut.dims == 2) {
            result[offset + 3] = scalesOut.row(i)[0];
            result[offset + 4] = scalesOut.row(i)[1];
            result[offset + 5] = scalesOut.row(i)[2];
        } else {
            result[offset + 3] = ((float*)scalesOut.data)[i * 3 + 0];
            result[offset + 4] = ((float*)scalesOut.data)[i * 3 + 1];
            result[offset + 5] = ((float*)scalesOut.data)[i * 3 + 2];
        }

        // Rotations (4 values - quaternion)
        if (rotationsOut.dims == 2) {
            result[offset + 6] = rotationsOut.row(i)[0];
            result[offset + 7] = rotationsOut.row(i)[1];
            result[offset + 8] = rotationsOut.row(i)[2];
            result[offset + 9] = rotationsOut.row(i)[3];
        } else {
            result[offset + 6] = ((float*)rotationsOut.data)[i * 4 + 0];
            result[offset + 7] = ((float*)rotationsOut.data)[i * 4 + 1];
            result[offset + 8] = ((float*)rotationsOut.data)[i * 4 + 2];
            result[offset + 9] = ((float*)rotationsOut.data)[i * 4 + 3];
        }

        // Opacity (1 value)
        if (opacityOut.dims == 2) {
            result[offset + 10] = opacityOut.row(i)[0];
        } else if (opacityOut.dims == 1) {
            result[offset + 10] = ((float*)opacityOut.data)[i];
        } else {
            result[offset + 10] = ((float*)opacityOut.data)[i];
        }

        // Colors (3 values)
        if (colorsOut.dims == 2) {
            result[offset + 11] = colorsOut.row(i)[0];
            result[offset + 12] = colorsOut.row(i)[1];
            result[offset + 13] = colorsOut.row(i)[2];
        } else {
            result[offset + 11] = ((float*)colorsOut.data)[i * 3 + 0];
            result[offset + 12] = ((float*)colorsOut.data)[i * 3 + 1];
            result[offset + 13] = ((float*)colorsOut.data)[i * 3 + 2];
        }
    }

    LOGI("Generated %d Gaussians (%d floats)", numGaussians, totalSize);

    // Return as Java float array
    jfloatArray jResult = env->NewFloatArray(totalSize);
    if (jResult == nullptr) {
        LOGE("Failed to allocate Java float array for %d elements", totalSize);
        return nullptr;
    }

    env->SetFloatArrayRegion(jResult, 0, totalSize, result.data());

    return jResult;
}

/**
 * Release NCNN network resources
 */
JNIEXPORT void JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeRelease(
    JNIEnv* env,
    jobject thiz,
    jlong handle
) {
    ncnn::Net* net = reinterpret_cast<ncnn::Net*>(handle);
    if (net != nullptr) {
        delete net;
        LOGI("SHARP NCNN network released");
    }
    // Also clean up global pointer if it matches
    if (g_net == net) {
        g_net = nullptr;
    }
}

} // extern "C"
