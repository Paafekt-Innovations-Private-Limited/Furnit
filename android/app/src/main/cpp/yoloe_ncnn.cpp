/**
 * YOLOE NCNN JNI Wrapper
 *
 * This file provides JNI bindings for running YOLOE segmentation model
 * using NCNN on Android. NCNN is highly optimized for mobile ARM CPUs
 * and supports Vulkan GPU acceleration.
 *
 * Setup:
 * 1. Download NCNN Android SDK from: https://github.com/Tencent/ncnn/releases
 * 2. Extract and copy libs to jniLibs/arm64-v8a/ and jniLibs/armeabi-v7a/
 * 3. Copy include/ folder to cpp/include/
 * 4. Export your YOLOE model: model.export(format="ncnn")
 * 5. Place model.ncnn.param and model.ncnn.bin in assets/
 */

#include <jni.h>
#include <string>
#include <vector>
#include <algorithm>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>

// NCNN headers
#include "ncnn/net.h"
#include "ncnn/mat.h"
#include "ncnn/cpu.h"

#define LOG_TAG "YoloeNCNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Detection structure
struct Detection {
    float x, y, w, h;
    float confidence;
    int classId;
    std::vector<float> maskCoeffs;
};

// Global NCNN network
static ncnn::Net* g_net = nullptr;
static bool g_useGpu = false;
static int g_inputSize = 640;  // YOLOE-PF uses 640x640 input
static ncnn::Mat g_protoOut;   // Store prototype output for mask generation

// Sigmoid function
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// IoU calculation
float calculateIoU(const Detection& a, const Detection& b) {
    float x1Min = a.x - a.w / 2;
    float y1Min = a.y - a.h / 2;
    float x1Max = a.x + a.w / 2;
    float y1Max = a.y + a.h / 2;

    float x2Min = b.x - b.w / 2;
    float y2Min = b.y - b.h / 2;
    float x2Max = b.x + b.w / 2;
    float y2Max = b.y + b.h / 2;

    float interXMin = std::max(x1Min, x2Min);
    float interYMin = std::max(y1Min, y2Min);
    float interXMax = std::min(x1Max, x2Max);
    float interYMax = std::min(y1Max, y2Max);

    float interW = std::max(0.0f, interXMax - interXMin);
    float interH = std::max(0.0f, interYMax - interYMin);
    float interArea = interW * interH;

    float area1 = a.w * a.h;
    float area2 = b.w * b.h;
    float unionArea = area1 + area2 - interArea;

    return unionArea > 0 ? interArea / unionArea : 0.0f;
}

// NMS
std::vector<Detection> nms(std::vector<Detection>& detections, float iouThreshold) {
    std::sort(detections.begin(), detections.end(),
        [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });

    std::vector<Detection> result;
    std::vector<bool> suppressed(detections.size(), false);

    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j]) continue;
            if (calculateIoU(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

extern "C" {

/**
 * Initialize NCNN network from asset files
 */
JNIEXPORT jlong JNICALL
Java_com_furnit_android_services_NcnnYoloe_nativeInit(
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

    g_useGpu = useGpu;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    if (!mgr) {
        LOGE("Failed to get AssetManager");
        return 0;
    }

    const char* paramPathStr = env->GetStringUTFChars(paramPath, nullptr);
    const char* binPathStr = env->GetStringUTFChars(binPath, nullptr);

    LOGI("Loading NCNN model: %s, %s", paramPathStr, binPathStr);

    g_net = new ncnn::Net();

    // Configure network options (CPU-only, no Vulkan)
    g_net->opt.use_vulkan_compute = false;  // CPU-only build
    g_net->opt.num_threads = numThreads > 0 ? numThreads : 4;
    g_net->opt.use_fp16_packed = true;
    g_net->opt.use_fp16_storage = true;
    g_net->opt.use_fp16_arithmetic = true;
    g_net->opt.use_packing_layout = true;
    g_net->opt.lightmode = true;  // Reduce memory usage

    // Load param from assets (small file, use buffer mode)
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

    // Load bin from assets (large file, use streaming mode and manual buffer)
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

    LOGI("NCNN model loaded successfully, input size: %d, GPU: %s",
         g_inputSize, g_useGpu ? "enabled" : "disabled");

    // Return handle (pointer cast to long)
    return reinterpret_cast<jlong>(g_net);
}

/**
 * Check if GPU (Vulkan) is available
 * Note: This CPU-only build always returns false
 */
JNIEXPORT jboolean JNICALL
Java_com_furnit_android_services_NcnnYoloe_nativeHasGpu(
    JNIEnv* env,
    jobject thiz
) {
    // CPU-only build - GPU not available
    return JNI_FALSE;
}

/**
 * Run YOLOE segmentation inference on a bitmap
 * Returns float array with detection results
 */
JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_NcnnYoloe_nativeDetect(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jobject bitmap,
    jfloat confThreshold,
    jfloat iouThreshold
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

    LOGD("Input bitmap: %dx%d, target: %dx%d", srcWidth, srcHeight, g_inputSize, g_inputSize);

    // Convert RGBA to NCNN Mat and resize
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(
        static_cast<const unsigned char*>(pixels),
        ncnn::Mat::PIXEL_RGBA2RGB,
        srcWidth, srcHeight,
        g_inputSize, g_inputSize
    );

    AndroidBitmap_unlockPixels(env, bitmap);

    // Normalize: /255.0
    const float mean_vals[3] = {0.0f, 0.0f, 0.0f};
    const float norm_vals[3] = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    // Run inference
    // Note: num_threads and vulkan_compute are set in net->opt during initialization
    ncnn::Extractor ex = net->create_extractor();

    // Input layer name from param file
    ex.input("in0", in);

    // Get detection output (typically "output0")
    ncnn::Mat detOut;
    ex.extract("output0", detOut);

    // Get prototype output for masks (typically "output1")
    ncnn::Mat protoOut;
    ex.extract("output1", protoOut);

    LOGD("Detection output: c=%d h=%d w=%d", detOut.c, detOut.h, detOut.w);
    LOGD("Proto output: c=%d h=%d w=%d", protoOut.c, protoOut.h, protoOut.w);

    // Store prototype output for mask generation
    g_protoOut = protoOut.clone();

    // Parse detections
    // YOLOE-PF output format: [num_features, num_anchors]
    // Features: bbox(4) + class_scores(80) + mask_coeffs(32) = 116
    std::vector<Detection> detections;

    int numAnchors = detOut.w;
    int numFeatures = detOut.h;
    int numMaskCoeffs = 32;
    int numClasses = numFeatures - 4 - numMaskCoeffs;  // 116 - 4 - 32 = 80

    LOGD("Parsing: features=%d, anchors=%d, classes=%d", numFeatures, numAnchors, numClasses);

    if (numClasses <= 0) {
        LOGE("Invalid feature count: %d", numFeatures);
        return nullptr;
    }

    float scaleX = static_cast<float>(srcWidth) / g_inputSize;
    float scaleY = static_cast<float>(srcHeight) / g_inputSize;

    for (int a = 0; a < numAnchors; a++) {
        // Extract bbox (already decoded by model)
        float x = detOut.row(0)[a];
        float y = detOut.row(1)[a];
        float w = detOut.row(2)[a];
        float h = detOut.row(3)[a];

        // Find max class score (indices 4 to 4+numClasses)
        float conf = 0.0f;
        int classId = 0;

        for (int c = 0; c < numClasses; c++) {
            float score = detOut.row(4 + c)[a];
            if (score > conf) {
                conf = score;
                classId = c;
            }
        }

        if (conf > confThreshold && w > 0 && h > 0) {
            Detection det;
            det.x = x * scaleX;
            det.y = y * scaleY;
            det.w = w * scaleX;
            det.h = h * scaleY;
            det.confidence = conf;
            det.classId = classId;

            // Extract mask coefficients (last 32 features)
            int coeffStart = 4 + numClasses;  // 84
            det.maskCoeffs.resize(numMaskCoeffs);
            for (int c = 0; c < numMaskCoeffs; c++) {
                det.maskCoeffs[c] = detOut.row(coeffStart + c)[a];
            }

            detections.push_back(det);
        }
    }

    LOGD("Found %zu detections above threshold %.2f", detections.size(), confThreshold);

    // Apply NMS
    detections = nms(detections, iouThreshold);

    LOGD("After NMS: %zu detections", detections.size());

    // Pack results: [numDets, det0_data..., det1_data..., ...]
    // Each detection: x, y, w, h, conf, classId, maskCoeffs[32]
    int detSize = 6 + numMaskCoeffs;
    int totalSize = 1 + detections.size() * detSize;

    std::vector<float> result(totalSize);
    result[0] = static_cast<float>(detections.size());

    for (size_t i = 0; i < detections.size(); i++) {
        int offset = 1 + i * detSize;
        result[offset + 0] = detections[i].x;
        result[offset + 1] = detections[i].y;
        result[offset + 2] = detections[i].w;
        result[offset + 3] = detections[i].h;
        result[offset + 4] = detections[i].confidence;
        result[offset + 5] = static_cast<float>(detections[i].classId);
        for (int c = 0; c < numMaskCoeffs; c++) {
            result[offset + 6 + c] = detections[i].maskCoeffs[c];
        }
    }

    // Return as Java float array
    jfloatArray jResult = env->NewFloatArray(totalSize);
    env->SetFloatArrayRegion(jResult, 0, totalSize, result.data());

    return jResult;
}

/**
 * Generate segmentation mask from mask coefficients and prototypes
 */
JNIEXPORT jintArray JNICALL
Java_com_furnit_android_services_NcnnYoloe_nativeGenerateMask(
    JNIEnv* env,
    jobject thiz,
    jlong handle,
    jfloatArray coeffs,
    jint numDetections,
    jint width,
    jint height,
    jfloat maskThreshold
) {
    if (numDetections <= 0 || g_protoOut.empty()) {
        LOGD("No detections or empty proto");
        return nullptr;
    }

    jfloat* coeffsData = env->GetFloatArrayElements(coeffs, nullptr);
    if (coeffsData == nullptr) return nullptr;

    // Proto output shape: [32, protoH, protoW]
    int numProtos = g_protoOut.c;
    int protoH = g_protoOut.h;
    int protoW = g_protoOut.w;

    LOGD("Generating mask: proto=%dx%dx%d, dets=%d, output=%dx%d",
         numProtos, protoH, protoW, numDetections, width, height);

    // Create mask at proto resolution
    std::vector<float> maskProto(protoH * protoW, 0.0f);

    // Compute combined mask from all detections
    for (int d = 0; d < numDetections; d++) {
        float* detCoeffs = coeffsData + d * 32;

        // For each pixel in proto
        for (int py = 0; py < protoH; py++) {
            for (int px = 0; px < protoW; px++) {
                float sum = 0.0f;
                for (int c = 0; c < 32; c++) {
                    sum += detCoeffs[c] * g_protoOut.channel(c).row(py)[px];
                }
                float sigmoidVal = sigmoid(sum);
                int idx = py * protoW + px;
                if (sigmoidVal > maskProto[idx]) {
                    maskProto[idx] = sigmoidVal;
                }
            }
        }
    }

    env->ReleaseFloatArrayElements(coeffs, coeffsData, 0);

    // Create output mask at original resolution
    int numPixels = width * height;
    std::vector<int> maskPixels(numPixels, 0);

    float scaleX = static_cast<float>(protoW) / width;
    float scaleY = static_cast<float>(protoH) / height;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int px = static_cast<int>(x * scaleX);
            int py = static_cast<int>(y * scaleY);
            px = std::min(px, protoW - 1);
            py = std::min(py, protoH - 1);

            float v = maskProto[py * protoW + px];
            if (v > maskThreshold) {
                maskPixels[y * width + x] = 0xCC00FF00;  // Semi-transparent green
            }
        }
    }

    jintArray jMask = env->NewIntArray(numPixels);
    env->SetIntArrayRegion(jMask, 0, numPixels, maskPixels.data());

    return jMask;
}

/**
 * Release NCNN network resources
 */
JNIEXPORT void JNICALL
Java_com_furnit_android_services_NcnnYoloe_nativeRelease(
    JNIEnv* env,
    jobject thiz,
    jlong handle
) {
    ncnn::Net* net = reinterpret_cast<ncnn::Net*>(handle);
    if (net != nullptr) {
        delete net;
        LOGI("NCNN network released");
    }
    // Also clean up global pointer if it matches
    if (g_net == net) {
        g_net = nullptr;
    }
}

} // extern "C"
