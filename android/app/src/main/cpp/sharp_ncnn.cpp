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
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <omp.h>

// NCNN headers
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"

// Custom layers for SHARP model
#include "sharp_custom_layers.h"

// Component-based SHARP implementation
#include "sharp_ncnn_components.h"

#define LOG_TAG "SharpNCNN"
// Use ERROR level for all logs to ensure they appear
#define LOGD(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// SHARP model configuration
static const int INPUT_SIZE = 1536;
static const int PARAMS_PER_GAUSSIAN = 14;  // pos(3) + scale(3) + rot(4) + opacity(1) + color(3)

// Global NCNN network
static ncnn::Net* g_net = nullptr;
// Global component-based model
static sharp_components::SharpComponents* g_components = nullptr;
static bool g_vulkan_instance_created = false;

static bool ensureVulkanInstance() {
#if NCNN_VULKAN
    if (!g_vulkan_instance_created) {
        ncnn::create_gpu_instance();
        g_vulkan_instance_created = true;
    }
    return ncnn::get_gpu_count() > 0;
#else
    return false;
#endif
}

static void maybeDestroyVulkanInstance() {
#if NCNN_VULKAN
    if (g_vulkan_instance_created) {
        ncnn::destroy_gpu_instance();
        g_vulkan_instance_created = false;
    }
#endif
}

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

    int threads = (numThreads > 0 && numThreads <= 8) ? numThreads : 4;
    bool wantGpu = useGpu == JNI_TRUE;
    bool hasVulkanGpu = wantGpu && ensureVulkanInstance();

    ncnn::Option opt;
    opt.use_vulkan_compute = hasVulkanGpu;
    opt.num_threads = threads;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;
    opt.lightmode = true;
    opt.use_local_pool_allocator = false;
    opt.use_winograd_convolution = false;
    opt.use_sgemm_convolution = true;
    opt.use_int8_inference = false;
    opt.use_bf16_storage = false;

    g_net = new ncnn::Net();

    g_net->opt = opt;

    sharp_layers::register_custom_layers(*g_net);
    LOGI("Registered custom SHARP layers");
    LOGI("NCNN configured: threads=%d, SGEMM on, vulkan=%s", threads, hasVulkanGpu ? "on" : "off");

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
 * Get available memory on device (returns MB)
 */
static long getAvailableMemoryMB() {
    FILE* meminfo = fopen("/proc/meminfo", "r");
    if (!meminfo) return -1;

    char line[256];
    long availableMB = -1;

    while (fgets(line, sizeof(line), meminfo)) {
        if (strncmp(line, "MemAvailable:", 13) == 0) {
            long availableKB;
            if (sscanf(line + 13, "%ld", &availableKB) == 1) {
                availableMB = availableKB / 1024;
            }
            break;
        }
    }
    fclose(meminfo);
    return availableMB;
}

/**
 * Initialize NCNN network from file system paths
 */
JNIEXPORT jlong JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeInitFromPath(
    JNIEnv* env,
    jobject thiz,
    jstring paramPath,
    jstring binPath,
    jboolean useGpu,
    jint numThreads
) {
    if (g_net != nullptr) {
        delete g_net;
        g_net = nullptr;
    }

    const char* paramPathStr = env->GetStringUTFChars(paramPath, nullptr);
    const char* binPathStr = env->GetStringUTFChars(binPath, nullptr);

    // Check available memory before loading
    long availMB = getAvailableMemoryMB();
    LOGI("Available memory: %ld MB", availMB);

    if (availMB > 0 && availMB < 2000) {
        LOGE("Insufficient memory for SHARP model (need ~2GB, have %ld MB)", availMB);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        return 0;
    }

    LOGI("=== SHARP NCNN INIT START ===");
    LOGI("Loading SHARP NCNN model from files: %s, %s", paramPathStr, binPathStr);

    int threads = (numThreads > 0 && numThreads <= 8) ? numThreads : 4;
    char ompThreads[16];
    snprintf(ompThreads, sizeof(ompThreads), "%d", threads);
    setenv("OMP_NUM_THREADS", ompThreads, 1);
    omp_set_num_threads(threads);
    omp_set_dynamic(0);

    bool wantGpu = useGpu == JNI_TRUE;
    bool hasVulkanGpu = wantGpu && ensureVulkanInstance();

    ncnn::Option opt;
    opt.use_vulkan_compute = hasVulkanGpu;
    opt.num_threads = threads;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_packing_layout = false;
    opt.lightmode = true;
    opt.use_local_pool_allocator = false;
    opt.use_winograd_convolution = false;  // conv_106 crash
    opt.use_sgemm_convolution = true;      // GEMM for conv - major speedup
    opt.use_int8_inference = false;
    opt.use_bf16_storage = false;

    LOGI("NCNN options: %s, %d threads, SGEMM on", hasVulkanGpu ? "vulkan" : "cpu", threads);

    g_net = new ncnn::Net();

    // Assign options IMMEDIATELY after creation, BEFORE any other operations
    g_net->opt = opt;
    LOGI("ncnn::Net created with safe options assigned");

    // Register custom layers for SHARP model
    sharp_layers::register_custom_layers(*g_net);
    LOGI("Registered custom SHARP layers");

    // Load param file - options are already set
    int ret = g_net->load_param(paramPathStr);
    if (ret != 0) {
        LOGE("Failed to load param file: %s (error: %d)", paramPathStr, ret);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }
    LOGI("Loaded param file successfully");

    // Check memory after loading param
    availMB = getAvailableMemoryMB();
    LOGI("Available memory after param load: %ld MB", availMB);

    // Load bin file (large - 2.4GB)
    LOGI("Loading model weights (2.4GB)...");
    ret = g_net->load_model(binPathStr);
    if (ret != 0) {
        LOGE("Failed to load model file: %s (error: %d)", binPathStr, ret);
        env->ReleaseStringUTFChars(paramPath, paramPathStr);
        env->ReleaseStringUTFChars(binPath, binPathStr);
        delete g_net;
        g_net = nullptr;
        return 0;
    }

    // Check memory after loading model
    availMB = getAvailableMemoryMB();
    LOGI("Available memory after model load: %ld MB", availMB);

    env->ReleaseStringUTFChars(paramPath, paramPathStr);
    env->ReleaseStringUTFChars(binPath, binPathStr);

    LOGI("SHARP NCNN model loaded from files successfully, input size: %d", INPUT_SIZE);

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
    ex.set_light_mode(true);  // Reduce memory usage

    // Input layer name (from SHARP model conversion)
    // The exact name depends on the pnnx conversion output
    int input_ret = ex.input("in0", in);
    if (input_ret != 0) {
        LOGE("Failed to set input: %d", input_ret);
        return nullptr;
    }

    // Extract outputs
    // SHARP model outputs 5 tensors (from pnnx conversion):
    // - out0: positions (1179648, 3) - N gaussians with x,y,z
    // - out1: scales (1179648, 3) - N gaussians with sx,sy,sz
    // - out2: rotations (1179648, 4) - N gaussians with quaternion w,x,y,z
    // - out3: colors (1179648, 3) - N gaussians with r,g,b
    // - out4: opacity (1179648,) - N gaussians with opacity

    ncnn::Mat positionsOut, scalesOut, rotationsOut, colorsOut, opacityOut;
    int ret;

    ret = ex.extract("out0", positionsOut);

    if (ret != 0) {
        LOGE("Failed to extract out0 (positions): %d", ret);
        return nullptr;
    }
    ret = ex.extract("out1", scalesOut);
    if (ret != 0) {
        LOGE("Failed to extract out1 (scales): %d", ret);
        return nullptr;
    }
    ret = ex.extract("out2", rotationsOut);
    if (ret != 0) {
        LOGE("Failed to extract out2 (rotations): %d", ret);
        return nullptr;
    }
    ret = ex.extract("out3", colorsOut);
    if (ret != 0) {
        LOGE("Failed to extract out3 (colors): %d", ret);
        return nullptr;
    }
    ret = ex.extract("out4", opacityOut);
    if (ret != 0) {
        LOGE("Failed to extract out4 (opacity): %d", ret);
        return nullptr;
    }

    // Determine number of Gaussians from positions output
    // pnnx output format: (h=N, w=3) for positions, meaning N gaussians with 3 components each
    int numGaussians = 0;

    if (positionsOut.dims == 2 && positionsOut.w == 3) {
        // Shape (h=N, w=3) - standard format from pnnx
        numGaussians = positionsOut.h;
    } else if (positionsOut.dims == 1) {
        // Flattened (w=N*3)
        numGaussians = positionsOut.w / 3;
    } else if (positionsOut.dims == 2 && positionsOut.h == 3) {
        // Transposed shape (h=3, w=N)
        numGaussians = positionsOut.w;
    } else {
        numGaussians = positionsOut.total() / 3;
    }

    if (numGaussians <= 0) {
        LOGE("Invalid Gaussian count: %d", numGaussians);
        return nullptr;
    }

    // Allocate result array
    int totalSize = numGaussians * PARAMS_PER_GAUSSIAN;
    std::vector<float> result(totalSize);

    // Interleave Gaussian parameters
    // Format: [x, y, z, scale_x, scale_y, scale_z, rot_w, rot_x, rot_y, rot_z, opacity, r, g, b]

    // Get raw data pointers
    const float* posData = (const float*)positionsOut.data;
    const float* scaleData = (const float*)scalesOut.data;
    const float* rotData = (const float*)rotationsOut.data;
    const float* colorData = (const float*)colorsOut.data;
    const float* opacityData = (const float*)opacityOut.data;

    // Determine data layout: row-major (N,C) or column-major (C,N)
    bool posRowMajor = (positionsOut.dims == 2 && positionsOut.w == 3);
    bool scaleRowMajor = (scalesOut.dims == 2 && scalesOut.w == 3);
    bool rotRowMajor = (rotationsOut.dims == 2 && rotationsOut.w == 4);
    bool colorRowMajor = (colorsOut.dims == 2 && colorsOut.w == 3);

    for (int i = 0; i < numGaussians; i++) {
        int offset = i * PARAMS_PER_GAUSSIAN;

        // Positions (3 values)
        if (posRowMajor) {
            // Row-major: data[i*3 + component]
            result[offset + 0] = posData[i * 3 + 0];
            result[offset + 1] = posData[i * 3 + 1];
            result[offset + 2] = posData[i * 3 + 2];
        } else {
            // Column-major: data[component * N + i]
            result[offset + 0] = posData[0 * numGaussians + i];
            result[offset + 1] = posData[1 * numGaussians + i];
            result[offset + 2] = posData[2 * numGaussians + i];
        }

        // Scales (3 values)
        if (scaleRowMajor) {
            result[offset + 3] = scaleData[i * 3 + 0];
            result[offset + 4] = scaleData[i * 3 + 1];
            result[offset + 5] = scaleData[i * 3 + 2];
        } else {
            result[offset + 3] = scaleData[0 * numGaussians + i];
            result[offset + 4] = scaleData[1 * numGaussians + i];
            result[offset + 5] = scaleData[2 * numGaussians + i];
        }

        // Rotations (4 values - quaternion)
        if (rotRowMajor) {
            result[offset + 6] = rotData[i * 4 + 0];
            result[offset + 7] = rotData[i * 4 + 1];
            result[offset + 8] = rotData[i * 4 + 2];
            result[offset + 9] = rotData[i * 4 + 3];
        } else {
            result[offset + 6] = rotData[0 * numGaussians + i];
            result[offset + 7] = rotData[1 * numGaussians + i];
            result[offset + 8] = rotData[2 * numGaussians + i];
            result[offset + 9] = rotData[3 * numGaussians + i];
        }

        // Opacity (1 value) - always linear
        result[offset + 10] = opacityData[i];

        // Colors (3 values): PNNX/NCNN outputs BGR; swap to RGB to match ONNX
        if (colorRowMajor) {
            result[offset + 11] = colorData[i * 3 + 2];
            result[offset + 12] = colorData[i * 3 + 1];
            result[offset + 13] = colorData[i * 3 + 0];
        } else {
            result[offset + 11] = colorData[2 * numGaussians + i];
            result[offset + 12] = colorData[1 * numGaussians + i];
            result[offset + 13] = colorData[0 * numGaussians + i];
        }
    }

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
    if (g_net == nullptr && g_components == nullptr) {
        maybeDestroyVulkanInstance();
    }
}

// ============================================================================
// Component-Based SHARP Implementation
// ============================================================================

/**
 * Initialize component-based SHARP model
 * This loads multiple NCNN models for patch-by-patch processing
 */
JNIEXPORT jlong JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeInitComponents(
    JNIEnv* env,
    jobject thiz,
    jstring modelDir,
    jboolean useGpu,
    jint numThreads
) {
    const char* modelDirStr = env->GetStringUTFChars(modelDir, nullptr);

    LOGI("=== SHARP NCNN COMPONENTS INIT ===");
    LOGI("Model directory: %s", modelDirStr);

    if (g_components != nullptr) {
        delete g_components;
        g_components = nullptr;
    }

    bool wantGpu = useGpu == JNI_TRUE;
    bool hasVulkanGpu = wantGpu && ensureVulkanInstance();
    int threads = (numThreads > 0 && numThreads <= 8) ? numThreads : 0;
    const char* overrideThreads = std::getenv("SHARP_NCNN_THREADS");
    const char* overrideWorkers = std::getenv("SHARP_NCNN_PATCH_WORKERS");

    g_components = new sharp_components::SharpComponents();

    if (!g_components->load(modelDirStr, hasVulkanGpu, threads)) {
        LOGE("Failed to load SHARP components");
        delete g_components;
        g_components = nullptr;
        env->ReleaseStringUTFChars(modelDir, modelDirStr);
        return 0;
    }

    env->ReleaseStringUTFChars(modelDir, modelDirStr);

    LOGI("SHARP components initialized successfully (vulkan=%s, requestedThreads=%d, envThreads=%s, envWorkers=%s)",
         hasVulkanGpu ? "on" : "off", threads,
         overrideThreads ? overrideThreads : "unset",
         overrideWorkers ? overrideWorkers : "unset");
    return reinterpret_cast<jlong>(g_components);
}

/**
 * Run component-based SHARP inference
 * This processes patches one at a time to avoid NCNN batch issues
 */
JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeInferComponents(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jobject bitmap
) {
    sharp_components::SharpComponents* components =
        reinterpret_cast<sharp_components::SharpComponents*>(handle);

    if (components == nullptr) {
        LOGE("Invalid components handle");
        return nullptr;
    }

    // Get bitmap info
    AndroidBitmapInfo bitmapInfo;
    if (AndroidBitmap_getInfo(env, bitmap, &bitmapInfo) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to get bitmap info");
        return nullptr;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) != ANDROID_BITMAP_RESULT_SUCCESS) {
        LOGE("Failed to lock bitmap pixels");
        return nullptr;
    }

    LOGI("Input bitmap: %dx%d", bitmapInfo.width, bitmapInfo.height);

    // Convert to NCNN Mat and resize
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(
        static_cast<const unsigned char*>(pixels),
        ncnn::Mat::PIXEL_RGBA2RGB,
        bitmapInfo.width, bitmapInfo.height,
        INPUT_SIZE, INPUT_SIZE
    );

    AndroidBitmap_unlockPixels(env, bitmap);

    // Normalize to [0, 1]
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    input.substract_mean_normalize(mean_vals, norm_vals);

    LOGI("Running component-based inference...");

    // Run inference
    std::vector<float> positions, scales, rotations, colors, opacities;
    int numGaussians = components->infer(input, positions, scales, rotations, colors, opacities);

    if (numGaussians <= 0) {
        LOGE("Component-based inference failed");
        return nullptr;
    }

    LOGI("Generated %d Gaussians", numGaussians);

    // Interleave into output format
    int totalSize = numGaussians * PARAMS_PER_GAUSSIAN;
    std::vector<float> result(totalSize);

    for (int i = 0; i < numGaussians; i++) {
        int offset = i * PARAMS_PER_GAUSSIAN;

        // Positions
        result[offset + 0] = positions[i * 3 + 0];
        result[offset + 1] = positions[i * 3 + 1];
        result[offset + 2] = positions[i * 3 + 2];

        // Scales
        result[offset + 3] = scales[i * 3 + 0];
        result[offset + 4] = scales[i * 3 + 1];
        result[offset + 5] = scales[i * 3 + 2];

        // Rotations
        result[offset + 6] = rotations[i * 4 + 0];
        result[offset + 7] = rotations[i * 4 + 1];
        result[offset + 8] = rotations[i * 4 + 2];
        result[offset + 9] = rotations[i * 4 + 3];

        // Opacity
        result[offset + 10] = opacities[i];

        // Colors
        result[offset + 11] = colors[i * 3 + 0];
        result[offset + 12] = colors[i * 3 + 1];
        result[offset + 13] = colors[i * 3 + 2];
    }

    jfloatArray jResult = env->NewFloatArray(totalSize);
    if (jResult == nullptr) {
        LOGE("Failed to allocate result array");
        return nullptr;
    }

    env->SetFloatArrayRegion(jResult, 0, totalSize, result.data());

    return jResult;
}

/**
 * Release component-based model
 */
JNIEXPORT void JNICALL
Java_com_furnit_android_services_NcnnSharp_nativeReleaseComponents(
    JNIEnv* env,
    jobject thiz,
    jlong handle
) {
    sharp_components::SharpComponents* components =
        reinterpret_cast<sharp_components::SharpComponents*>(handle);

    if (components != nullptr) {
        delete components;
        LOGI("SHARP components released");
    }

    if (g_components == components) {
        g_components = nullptr;
    }
    if (g_net == nullptr && g_components == nullptr) {
        maybeDestroyVulkanInstance();
    }
}

} // extern "C"
