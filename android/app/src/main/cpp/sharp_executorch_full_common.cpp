#include "sharp_executorch_full_internal.h"

#include <array>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>

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
    if (ws.tileImgCrop) total += (size_t)3 * (IMAGE_SIZE / 4) * (IMAGE_SIZE / 4) * sizeof(float);
    if (ws.tileLat0Crop) total += (size_t)1024 * 24 * 24 * sizeof(float);
    if (ws.tileLat1Crop) total += (size_t)1024 * 24 * 24 * sizeof(float);
    if (ws.tileX0Crop) total += (size_t)1024 * 24 * 24 * sizeof(float);
    if (ws.tileX1Crop) total += (size_t)1024 * 12 * 12 * sizeof(float);
    if (ws.tileX2Crop) total += (size_t)1024 * 6 * 6 * sizeof(float);
    if (ws.tileXlCrop) total += (size_t)1024 * 6 * 6 * sizeof(float);
    return total;
}

bool fileExists(const std::string& path) {
    std::ifstream f(path);
    return f.good();
}

struct OwnedTensor {
    std::vector<float> data;
    std::vector<executorch::aten::SizesType> sizes;

    void ensureCapacity(size_t numel) {
        if (data.capacity() < numel) {
            data.reserve(numel);
        }
        data.resize(numel);
    }
};

struct Part4bIntermediateScratch {
    OwnedTensor latent0Up;
    OwnedTensor latent1Up;
    OwnedTensor x0Up;
    OwnedTensor x1Up;
    OwnedTensor xFused;
    OwnedTensor disparity;
    OwnedTensor decoderFeatures;
    OwnedTensor featureInput;
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
};

constexpr int kPlyBatchSize = 512;
constexpr int kPlyFloatsPerVertex = 62;
constexpr int kPlyZeroShFloatCount = 45;
constexpr int kPlyLogitLutSize = 1024;
constexpr int kPlySrgbLutSize = 4096;
constexpr int kPlyLnLutSize = 2048;
constexpr float kPlyShC0 = 0.28209479177387814f;
constexpr float kPlyLinearToSrgbThreshold = 0.0031308f;
constexpr float kPlyLnLutMin = 0.001f;
constexpr float kPlyLnLutMax = 5.0f;
constexpr float kPlyScaleBias = 1.3f;

struct PlyExportStats {
    float minX = std::numeric_limits<float>::max();
    float maxX = -std::numeric_limits<float>::max();
    float minY = std::numeric_limits<float>::max();
    float maxY = -std::numeric_limits<float>::max();
    float minZ = std::numeric_limits<float>::max();
    float maxZ = -std::numeric_limits<float>::max();
    float maxAbsX = 0.0f;
    float maxAbsY = 0.0f;
    float maxAbsZ = 0.0f;
};

const std::array<float, kPlyLogitLutSize>& plyLogitLut() {
    static const std::array<float, kPlyLogitLutSize> lut = []() {
        std::array<float, kPlyLogitLutSize> values{};
        for (int i = 0; i < kPlyLogitLutSize; ++i) {
            const float p = std::clamp(
                    static_cast<float>(i) / static_cast<float>(kPlyLogitLutSize - 1),
                    1e-4f,
                    1.0f - 1e-4f);
            values[static_cast<size_t>(i)] = std::log(p / (1.0f - p));
        }
        return values;
    }();
    return lut;
}

const std::array<float, kPlySrgbLutSize>& plySrgbLut() {
    static const std::array<float, kPlySrgbLutSize> lut = []() {
        std::array<float, kPlySrgbLutSize> values{};
        for (int i = 0; i < kPlySrgbLutSize; ++i) {
            const float v = static_cast<float>(i) / static_cast<float>(kPlySrgbLutSize - 1);
            if (v <= kPlyLinearToSrgbThreshold) {
                values[static_cast<size_t>(i)] = v * 12.92f;
            } else {
                values[static_cast<size_t>(i)] =
                        std::clamp(1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f, 0.0f, 1.0f);
            }
        }
        return values;
    }();
    return lut;
}

const std::array<float, kPlyLnLutSize>& plyLnLut() {
    static const std::array<float, kPlyLnLutSize> lut = []() {
        std::array<float, kPlyLnLutSize> values{};
        for (int i = 0; i < kPlyLnLutSize; ++i) {
            const float t = static_cast<float>(i) / static_cast<float>(kPlyLnLutSize - 1);
            values[static_cast<size_t>(i)] =
                    std::log(kPlyLnLutMin + (kPlyLnLutMax - kPlyLnLutMin) * t);
        }
        return values;
    }();
    return lut;
}

inline float plyLnLutValue(float x) {
    if (x <= kPlyLnLutMin) return plyLnLut()[0];
    if (x >= kPlyLnLutMax) return plyLnLut()[kPlyLnLutSize - 1];
    const float scale = static_cast<float>(kPlyLnLutSize - 1) / (kPlyLnLutMax - kPlyLnLutMin);
    const int index = static_cast<int>((x - kPlyLnLutMin) * scale);
    return plyLnLut()[static_cast<size_t>(std::clamp(index, 0, kPlyLnLutSize - 1))];
}

inline float plyLinearToSrgb(float linear) {
    const float clamped = std::clamp(linear, 0.0f, 1.0f);
    const int index = std::clamp(
            static_cast<int>(clamped * static_cast<float>(kPlySrgbLutSize - 1)),
            0,
            kPlySrgbLutSize - 1);
    return plySrgbLut()[static_cast<size_t>(index)];
}

inline float plyOpacityLogit(float opacity) {
    const int index = std::clamp(
            static_cast<int>(opacity * static_cast<float>(kPlyLogitLutSize - 1)),
            0,
            kPlyLogitLutSize - 1);
    return plyLogitLut()[static_cast<size_t>(index)];
}

bool writePlyBinaryNative(const std::string& outputPath,
                          const float* params,
                          int count,
                          float aspectCorrX,
                          float aspectCorrY,
                          float metricScale,
                          PlyExportStats* statsOut) {
    if (!params || count <= 0 || !statsOut) {
        return false;
    }

    std::ofstream out(outputPath, std::ios::binary);
    if (!out.good()) {
        LOGE("writePlyBinaryNative: failed to open %s", outputPath.c_str());
        return false;
    }

    const std::string header =
            "ply\nformat binary_little_endian 1.0\nelement vertex " + std::to_string(count) +
            "\nproperty float x\nproperty float y\nproperty float z\nproperty float nx\nproperty float ny\nproperty float nz\n" +
            "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n" +
            "property float f_rest_0\nproperty float f_rest_1\nproperty float f_rest_2\nproperty float f_rest_3\nproperty float f_rest_4\nproperty float f_rest_5\nproperty float f_rest_6\nproperty float f_rest_7\nproperty float f_rest_8\nproperty float f_rest_9\nproperty float f_rest_10\nproperty float f_rest_11\nproperty float f_rest_12\nproperty float f_rest_13\nproperty float f_rest_14\nproperty float f_rest_15\nproperty float f_rest_16\nproperty float f_rest_17\nproperty float f_rest_18\nproperty float f_rest_19\nproperty float f_rest_20\nproperty float f_rest_21\nproperty float f_rest_22\nproperty float f_rest_23\nproperty float f_rest_24\nproperty float f_rest_25\nproperty float f_rest_26\nproperty float f_rest_27\nproperty float f_rest_28\nproperty float f_rest_29\nproperty float f_rest_30\nproperty float f_rest_31\nproperty float f_rest_32\nproperty float f_rest_33\nproperty float f_rest_34\nproperty float f_rest_35\nproperty float f_rest_36\nproperty float f_rest_37\nproperty float f_rest_38\nproperty float f_rest_39\nproperty float f_rest_40\nproperty float f_rest_41\nproperty float f_rest_42\nproperty float f_rest_43\nproperty float f_rest_44\n" +
            "property float opacity\nproperty float scale_0\nproperty float scale_1\nproperty float scale_2\nproperty float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\nend_header\n";
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (!out.good()) {
        LOGE("writePlyBinaryNative: failed to write header for %s", outputPath.c_str());
        return false;
    }

    std::array<float, kPlyBatchSize * kPlyFloatsPerVertex> batch{};
    int batchCount = 0;
    float* dst = batch.data();
    const float xScale = aspectCorrX * metricScale;
    const float yScale = -aspectCorrY * metricScale;
    const float zScale = -metricScale;
    const float scaleScale = kPlyScaleBias * metricScale;
#if HAS_NEON
    const float32x4_t xyzMul = {xScale, yScale, zScale, 1.0f};
    const float32x4_t scaleMul = {scaleScale, scaleScale, scaleScale, 1.0f};
#endif

    PlyExportStats stats;
    for (int i = 0; i < count; ++i) {
        const float* src = params + static_cast<size_t>(i) * PARAMS_PER_GAUSSIAN;
        float x;
        float y;
        float z;
        float opacityRaw;
        float scale0;
        float scale1;
        float scale2;
        float rw;
#if HAS_NEON
        const float32x4_t xyzOpacity = vld1q_f32(src);
        const float32x4_t xyzScaled = vmulq_f32(xyzOpacity, xyzMul);
        x = vgetq_lane_f32(xyzScaled, 0);
        y = vgetq_lane_f32(xyzScaled, 1);
        z = vgetq_lane_f32(xyzScaled, 2);
        opacityRaw = vgetq_lane_f32(xyzOpacity, 3);

        const float32x4_t scalesRw = vld1q_f32(src + 4);
        const float32x4_t scalesScaled = vmulq_f32(scalesRw, scaleMul);
        scale0 = vgetq_lane_f32(scalesScaled, 0);
        scale1 = vgetq_lane_f32(scalesScaled, 1);
        scale2 = vgetq_lane_f32(scalesScaled, 2);
        rw = vgetq_lane_f32(scalesRw, 3);
#else
        x = src[0] * xScale;
        y = src[1] * yScale;
        z = src[2] * zScale;
        opacityRaw = src[3];
        scale0 = src[4] * scaleScale;
        scale1 = src[5] * scaleScale;
        scale2 = src[6] * scaleScale;
        rw = src[7];
#endif
        const float rx = src[8];
        const float ry = src[9];
        const float rz = src[10];
        const float b = plyLinearToSrgb(src[11]);
        const float g = plyLinearToSrgb(src[12]);
        const float r = plyLinearToSrgb(src[13]);

        stats.minX = std::min(stats.minX, x);
        stats.maxX = std::max(stats.maxX, x);
        stats.minY = std::min(stats.minY, y);
        stats.maxY = std::max(stats.maxY, y);
        stats.minZ = std::min(stats.minZ, z);
        stats.maxZ = std::max(stats.maxZ, z);
        stats.maxAbsX = std::max(stats.maxAbsX, std::fabs(x));
        stats.maxAbsY = std::max(stats.maxAbsY, std::fabs(y));
        stats.maxAbsZ = std::max(stats.maxAbsZ, std::fabs(z));

        dst[0] = x;
        dst[1] = y;
        dst[2] = z;
        dst[3] = 0.0f;
        dst[4] = 0.0f;
        dst[5] = 0.0f;
        dst[6] = (r - 0.5f) / kPlyShC0;
        dst[7] = (g - 0.5f) / kPlyShC0;
        dst[8] = (b - 0.5f) / kPlyShC0;
        std::memset(dst + 9, 0, static_cast<size_t>(kPlyZeroShFloatCount) * sizeof(float));
        dst[54] = plyOpacityLogit(opacityRaw);
        dst[55] = plyLnLutValue(std::max(scale0, 0.001f));
        dst[56] = plyLnLutValue(std::max(scale1, 0.001f));
        dst[57] = plyLnLutValue(std::max(scale2, 0.001f));

        const float normSq = rw * rw + rx * rx + ry * ry + rz * rz;
        const float invNorm = (normSq > 1e-16f) ? (1.0f / std::sqrt(normSq)) : 1.0f;
        dst[58] = rw * invNorm;
        dst[59] = rx * invNorm;
        dst[60] = ry * invNorm;
        dst[61] = rz * invNorm;

        dst += kPlyFloatsPerVertex;
        batchCount++;
        if (batchCount == kPlyBatchSize || i == count - 1) {
            out.write(
                    reinterpret_cast<const char*>(batch.data()),
                    static_cast<std::streamsize>(batchCount * kPlyFloatsPerVertex * sizeof(float)));
            if (!out.good()) {
                LOGE("writePlyBinaryNative: failed while writing vertex payload for %s", outputPath.c_str());
                return false;
            }
            batchCount = 0;
            dst = batch.data();
        }
    }

    out.flush();
    if (!out.good()) {
        LOGE("writePlyBinaryNative: flush failed for %s", outputPath.c_str());
        return false;
    }
    *statsOut = stats;
    return true;
}

std::mutex g_lastMonodepthMutex;
std::vector<float> g_lastMonodepth;
int g_lastMonodepthWidth = 0;
int g_lastMonodepthHeight = 0;
int g_lastMonodepthChannels = 0;

void clearLastMonodepthCaptureLocked() {
    g_lastMonodepth.clear();
    g_lastMonodepthWidth = 0;
    g_lastMonodepthHeight = 0;
    g_lastMonodepthChannels = 0;
}

void ensureLastMonodepthCaptureLocked(int fullWidth, int fullHeight, int channels) {
    if (fullWidth <= 0 || fullHeight <= 0 || channels <= 0) {
        clearLastMonodepthCaptureLocked();
        return;
    }
    const size_t required = static_cast<size_t>(fullWidth) * static_cast<size_t>(fullHeight) * static_cast<size_t>(channels);
    if (g_lastMonodepthWidth == fullWidth &&
        g_lastMonodepthHeight == fullHeight &&
        g_lastMonodepthChannels == channels &&
        g_lastMonodepth.size() == required) {
        return;
    }
    g_lastMonodepth.assign(required, 0.0f);
    g_lastMonodepthWidth = fullWidth;
    g_lastMonodepthHeight = fullHeight;
    g_lastMonodepthChannels = channels;
}

void stitchMonodepthTile(const OwnedTensor& monodepth,
                         int batchIndex,
                         int tileRow,
                         int tileCol,
                         int grid,
                         const char* label) {
    if (monodepth.sizes.size() != 4) {
        LOGW("%s: monodepth rank %zu unsupported for capture", label, monodepth.sizes.size());
        return;
    }
    const int batch = static_cast<int>(monodepth.sizes[0]);
    const int channels = static_cast<int>(monodepth.sizes[1]);
    const int tileHeight = static_cast<int>(monodepth.sizes[2]);
    const int tileWidth = static_cast<int>(monodepth.sizes[3]);
    if (batch <= 0 || channels <= 0 || tileHeight <= 0 || tileWidth <= 0) {
        LOGW("%s: monodepth capture got invalid shape [%d,%d,%d,%d]",
             label, batch, channels, tileHeight, tileWidth);
        return;
    }
    if (batchIndex < 0 || batchIndex >= batch) {
        LOGW("%s: monodepth capture batch index %d out of range for batch %d", label, batchIndex, batch);
        return;
    }
    if (tileRow < 0 || tileCol < 0 || tileRow >= grid || tileCol >= grid) {
        LOGW("%s: monodepth capture tile (%d,%d) out of grid %d", label, tileRow, tileCol, grid);
        return;
    }

    const int fullWidth = tileWidth * grid;
    const int fullHeight = tileHeight * grid;
    const int tilePixels = tileWidth * tileHeight;
    const int batchStride = channels * tilePixels;
    const int srcBase = batchIndex * batchStride;

    std::lock_guard<std::mutex> lock(g_lastMonodepthMutex);
    ensureLastMonodepthCaptureLocked(fullWidth, fullHeight, channels);
    if (g_lastMonodepth.empty()) {
        return;
    }
    for (int channel = 0; channel < channels; ++channel) {
        const int srcChannelBase = srcBase + channel * tilePixels;
        const int dstChannelBase = channel * fullWidth * fullHeight;
        for (int y = 0; y < tileHeight; ++y) {
            const int dstY = tileRow * tileHeight + y;
            float* dst = g_lastMonodepth.data() + dstChannelBase + dstY * fullWidth + tileCol * tileWidth;
            const float* src = monodepth.data.data() + srcChannelBase + y * tileWidth;
            std::memcpy(dst, src, static_cast<size_t>(tileWidth) * sizeof(float));
        }
    }
}

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
    dst.ensureCapacity(static_cast<size_t>(numel));
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

void clearLastMonodepthCapture() {
    std::lock_guard<std::mutex> lock(g_lastMonodepthMutex);
    clearLastMonodepthCaptureLocked();
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

bool Workspace::allocateTileScratch() {
    if (tileAllocated && tileImgCrop && tileLat0Crop && tileLat1Crop && tileX0Crop && tileX1Crop &&
        tileX2Crop && tileXlCrop) {
        return true;
    }
    releaseTileScratch();
    const size_t tileImgSz = 3 * (IMAGE_SIZE / 4) * (IMAGE_SIZE / 4);
    const size_t tileLat96Sz = (size_t)1024 * 24 * 24;
    const size_t tileX1Sz = (size_t)1024 * 12 * 12;
    const size_t tileX2Sz = (size_t)1024 * 6 * 6;
    tileImgCrop = alignedAlloc(tileImgSz);
    tileLat0Crop = alignedAlloc(tileLat96Sz);
    tileLat1Crop = alignedAlloc(tileLat96Sz);
    tileX0Crop = alignedAlloc(tileLat96Sz);
    tileX1Crop = alignedAlloc(tileX1Sz);
    tileX2Crop = alignedAlloc(tileX2Sz);
    tileXlCrop = alignedAlloc(tileX2Sz);
    tileAllocated = tileImgCrop && tileLat0Crop && tileLat1Crop && tileX0Crop && tileX1Crop && tileX2Crop &&
                    tileXlCrop;
    if (!tileAllocated) {
        releaseTileScratch();
        LOGE("Workspace::allocateTileScratch: allocation failed");
        return false;
    }
    return true;
}

void Workspace::releaseTileScratch() {
    alignedFree(tileImgCrop);
    tileImgCrop = nullptr;
    alignedFree(tileLat0Crop);
    tileLat0Crop = nullptr;
    alignedFree(tileLat1Crop);
    tileLat1Crop = nullptr;
    alignedFree(tileX0Crop);
    tileX0Crop = nullptr;
    alignedFree(tileX1Crop);
    tileX1Crop = nullptr;
    alignedFree(tileX2Crop);
    tileX2Crop = nullptr;
    alignedFree(tileXlCrop);
    tileXlCrop = nullptr;
    tileAllocated = false;
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
        alignedFree(tileImgCrop); tileImgCrop = nullptr;
        alignedFree(tileLat0Crop); tileLat0Crop = nullptr;
        alignedFree(tileLat1Crop); tileLat1Crop = nullptr;
        alignedFree(tileX0Crop); tileX0Crop = nullptr;
        alignedFree(tileX1Crop); tileX1Crop = nullptr;
        alignedFree(tileX2Crop); tileX2Crop = nullptr;
        alignedFree(tileXlCrop); tileXlCrop = nullptr;
    }

void Workspace::release() {
        alignedFree(latent0);   alignedFree(latent1);   alignedFree(x0Feat);
        alignedFree(x1Feat);    alignedFree(x2Feat);    alignedFree(tempSpatial);
        alignedFree(halfImg);   alignedFree(quarterImg); alignedFree(patchBuf);
        alignedFree(patchBuf4); alignedFree(tokensCopy); alignedFree(tokensCopy4);
        alignedFree(tileImgCrop); alignedFree(tileLat0Crop); alignedFree(tileLat1Crop);
        alignedFree(tileX0Crop); alignedFree(tileX1Crop); alignedFree(tileX2Crop); alignedFree(tileXlCrop);
        latent0 = latent1 = x0Feat = x1Feat = x2Feat = tempSpatial = nullptr;
        halfImg = quarterImg = patchBuf = patchBuf4 = tokensCopy = tokensCopy4 = nullptr;
        tileImgCrop = tileLat0Crop = tileLat1Crop = tileX0Crop = tileX1Crop = tileX2Crop = tileXlCrop = nullptr;
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

long sharpDeviceMemAvailableKb() {
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

namespace {
std::mutex g_part4bTile00PreloadMu;
std::unique_ptr<Part4bTile00ModuleBundle> g_part4bTile00PreloadReady;
}  // namespace

bool loadPart4bTile00ModuleBundle(const std::string& modelDir, Part4bTile00ModuleBundle& out) {
    out = Part4bTile00ModuleBundle{};
    out.modelDir = modelDir;

    const auto pathOk = [](const std::string& p) {
        std::ifstream f(p);
        return f.good();
    };

    const std::string fineSplitStagePre = pathJoin(modelDir, "sharp_split_part4b_tile_00_stage_pre_vulkan.pte");
    const std::string fineSplitDecoderHead = pathJoin(modelDir, "sharp_split_part4b_tile_00_decoder_head.pte");
    const std::string splitStageA = pathJoin(modelDir, "sharp_split_part4b_tile_00_stage_a_vulkan.pte");
    const std::string splitInitBase = pathJoin(modelDir, "sharp_split_part4b_tile_00_init_base.pte");
    const std::string splitRawHeads = pathJoin(modelDir, "sharp_split_part4b_tile_00_raw_heads_vulkan.pte");
    const std::string splitCompose = pathJoin(modelDir, "sharp_split_part4b_tile_00_compose.pte");
    const bool hasFineSplitTile00 =
            pathOk(fineSplitStagePre) &&
            pathOk(fineSplitDecoderHead) &&
            pathOk(splitInitBase) &&
            pathOk(splitRawHeads) &&
            pathOk(splitCompose);
    const bool hasSplitTile00 =
            pathOk(splitStageA) &&
            pathOk(splitInitBase) &&
            pathOk(splitRawHeads) &&
            pathOk(splitCompose);
    const std::string tileFullFp32 = "sharp_split_part4b_tile_full.pte";
    const std::string tile00Fp32 = "sharp_split_part4b_tile_00.pte";
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
        out.splitStagePreModule = std::make_unique<ETModule>(fineSplitStagePre, ETModule::LoadMode::Mmap);
        out.splitDecoderHeadModule = std::make_unique<ETModule>(fineSplitDecoderHead, ETModule::LoadMode::Mmap);
        out.splitInitBaseModule = std::make_unique<ETModule>(splitInitBase, ETModule::LoadMode::Mmap);
        out.splitRawHeadsModule = std::make_unique<ETModule>(splitRawHeads, ETModule::LoadMode::Mmap);
        out.splitComposeModule = std::make_unique<ETModule>(splitCompose, ETModule::LoadMode::Mmap);
        const bool loaded =
                out.splitStagePreModule->load() == Error::Ok &&
                out.splitDecoderHeadModule->load() == Error::Ok &&
                out.splitInitBaseModule->load() == Error::Ok &&
                out.splitRawHeadsModule->load() == Error::Ok &&
                out.splitComposeModule->load() == Error::Ok;
        if (loaded) {
            out.useFineSplitTile00 = true;
            return true;
        }
        out.splitStagePreModule.reset();
        out.splitDecoderHeadModule.reset();
        out.splitInitBaseModule.reset();
        out.splitRawHeadsModule.reset();
        out.splitComposeModule.reset();
    }
    if (hasSplitTile00) {
        out.splitStageAModule = std::make_unique<ETModule>(splitStageA, ETModule::LoadMode::Mmap);
        out.splitInitBaseModule = std::make_unique<ETModule>(splitInitBase, ETModule::LoadMode::Mmap);
        out.splitRawHeadsModule = std::make_unique<ETModule>(splitRawHeads, ETModule::LoadMode::Mmap);
        out.splitComposeModule = std::make_unique<ETModule>(splitCompose, ETModule::LoadMode::Mmap);
        const bool loaded =
                out.splitStageAModule->load() == Error::Ok &&
                out.splitInitBaseModule->load() == Error::Ok &&
                out.splitRawHeadsModule->load() == Error::Ok &&
                out.splitComposeModule->load() == Error::Ok;
        if (loaded) {
            out.useSplitTile00 = true;
            return true;
        }
        out.splitStageAModule.reset();
        out.splitInitBaseModule.reset();
        out.splitRawHeadsModule.reset();
        out.splitComposeModule.reset();
    }
    out.legacyModelPath = chooseLegacyTileModel();
    if (out.legacyModelPath.empty()) {
        return false;
    }
    out.legacyTileModule = std::make_unique<ETModule>(out.legacyModelPath, ETModule::LoadMode::Mmap);
    if (out.legacyTileModule->load() != Error::Ok) {
        out.legacyTileModule.reset();
        out.legacyModelPath.clear();
        return false;
    }
    return true;
}

void part4bTile00PreloadStart(const std::string& /*modelDir*/) {
    // Disabled: ExecuTorch runtime is not thread-safe — async Module::load() races CPU/GPU forward() on the
    // inference thread. Part4b tile_00 loads only on the main pipeline path (see runPart4bTiledFullPipeline).
    (void)0;
}

void part4bTile00PreloadJoin() {
    // No-op (preload disabled).
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
        std::vector<float>& outGaussians,
        JNIEnv* progressEnv,
        jobject progressReporter,
        jmethodID reportProgressMethodId) {

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
    Part4bIntermediateScratch scratch;

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

            if (!copyOutputTensor(*stagePreResult, 0, scratch.latent0Up, "fine split tile_b2 latent0_up") ||
                !copyOutputTensor(*stagePreResult, 1, scratch.latent1Up, "fine split tile_b2 latent1_up") ||
                !copyOutputTensor(*stagePreResult, 2, scratch.x0Up, "fine split tile_b2 x0_up") ||
                !copyOutputTensor(*stagePreResult, 3, scratch.x1Up, "fine split tile_b2 x1_up") ||
                !copyOutputTensor(*stagePreResult, 4, scratch.xFused, "fine split tile_b2 x_fused")) {
                return false;
            }

            auto latent0UpTensor = from_blob(scratch.latent0Up.data.data(), scratch.latent0Up.sizes);
            auto latent1UpTensor = from_blob(scratch.latent1Up.data.data(), scratch.latent1Up.sizes);
            auto x0UpTensor = from_blob(scratch.x0Up.data.data(), scratch.x0Up.sizes);
            auto x1UpTensor = from_blob(scratch.x1Up.data.data(), scratch.x1Up.sizes);
            auto xFusedTensor = from_blob(scratch.xFused.data.data(), scratch.xFused.sizes);

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

            if (!copyOutputTensor(*decoderHeadResult, 0, scratch.disparity, "fine split tile_b2 disparity") ||
                !copyOutputTensor(*decoderHeadResult, 1, scratch.decoderFeatures, "fine split tile_b2 decoder_features")) {
                return false;
            }

            // Convert raw disparity → monodepth (= disparity_factor / disparity).
            // The exported Part4bTileInitBasePortable passes its input directly to
            // init_model which expects monodepth (metric depth), not raw disparity.
            // disparity_factor is 1.0 (baked in export), so monodepth = 1/disparity.
            for (size_t k = 0; k < scratch.disparity.data.size(); ++k) {
                float v = scratch.disparity.data[k];
                v = std::clamp(v, 1e-4f, 1e4f);
                scratch.disparity.data[k] = 1.0f / v;
            }
            for (int i = 0; i < batch; ++i) {
                const int t = b * batch + i;
                stitchMonodepthTile(scratch.disparity, i, t / GRID, t % GRID, GRID, "fine split tile_b2 disparity");
            }
            auto disparityTensor = from_blob(scratch.disparity.data.data(), scratch.disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgT);
            initInputs.emplace_back(*disparityTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d fine split tile_b2 init_base begin (disparity→monodepth applied)",
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

            if (!copyOutputTensor(*initResult, 0, scratch.featureInput, "fine split tile_b2 feature_input")) {
                return false;
            }

            auto featureInputTensor = from_blob(scratch.featureInput.data.data(), scratch.featureInput.sizes);
            auto decoderFeaturesTensor = from_blob(scratch.decoderFeatures.data.data(), scratch.decoderFeatures.sizes);

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

            if (!copyOutputTensor(*rawHeadResult, 0, scratch.geometryRaw, "fine split tile_b2 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, scratch.textureRaw, "fine split tile_b2 texture_raw") ||
                !copyOutputTensor(*initResult, 1, scratch.meanXNdc, "fine split tile_b2 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, scratch.meanYNdc, "fine split tile_b2 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, scratch.meanInverseZNdc, "fine split tile_b2 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scratch.scales, "fine split tile_b2 scales") ||
                !copyOutputTensor(*initResult, 5, scratch.quaternions, "fine split tile_b2 quaternions") ||
                !copyOutputTensor(*initResult, 6, scratch.colors, "fine split tile_b2 colors") ||
                !copyOutputTensor(*initResult, 7, scratch.opacities, "fine split tile_b2 opacities") ||
                !copyOutputTensor(*initResult, 8, scratch.globalScale, "fine split tile_b2 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(scratch.geometryRaw.data.data(), scratch.geometryRaw.sizes);
            auto textureRawTensor = from_blob(scratch.textureRaw.data.data(), scratch.textureRaw.sizes);
            auto meanXNdcTensor = from_blob(scratch.meanXNdc.data.data(), scratch.meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(scratch.meanYNdc.data.data(), scratch.meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(scratch.meanInverseZNdc.data.data(), scratch.meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scratch.scales.data.data(), scratch.scales.sizes);
            auto quaternionsTensor = from_blob(scratch.quaternions.data.data(), scratch.quaternions.sizes);
            auto colorsTensor = from_blob(scratch.colors.data.data(), scratch.colors.sizes);
            auto opacitiesTensor = from_blob(scratch.opacities.data.data(), scratch.opacities.sizes);
            auto globalScaleTensor = from_blob(scratch.globalScale.data.data(), scratch.globalScale.sizes);

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

            if (!copyOutputTensor(*stageAResult, 0, scratch.disparity, "split tile_b2 disparity")) {
                return false;
            }
            for (size_t k = 0; k < scratch.disparity.data.size(); ++k) {
                float v = std::clamp(scratch.disparity.data[k], 1e-4f, 1e4f);
                scratch.disparity.data[k] = 1.0f / v;
            }
            for (int i = 0; i < batch; ++i) {
                const int t = b * batch + i;
                stitchMonodepthTile(scratch.disparity, i, t / GRID, t % GRID, GRID, "split tile_b2 disparity");
            }
            auto disparityTensor = from_blob(scratch.disparity.data.data(), scratch.disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgT);
            initInputs.emplace_back(*disparityTensor);

            LOGD("runPart4bBatchedTiledPipeline: batch %d/%d split tile_b2 init_base begin (disparity→monodepth applied)",
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

            if (!copyOutputTensor(*initResult, 0, scratch.featureInput, "split tile_b2 feature_input") ||
                !copyOutputTensor(*stageAResult, 1, scratch.latent0Up, "split tile_b2 latent0_up") ||
                !copyOutputTensor(*stageAResult, 2, scratch.latent1Up, "split tile_b2 latent1_up") ||
                !copyOutputTensor(*stageAResult, 3, scratch.x0Up, "split tile_b2 x0_up") ||
                !copyOutputTensor(*stageAResult, 4, scratch.x1Up, "split tile_b2 x1_up") ||
                !copyOutputTensor(*stageAResult, 5, scratch.xFused, "split tile_b2 x_fused") ||
                !copyOutputTensor(*stageAResult, 6, scratch.decoderFeatures, "split tile_b2 decoder_features")) {
                return false;
            }

            auto featureInputTensor = from_blob(scratch.featureInput.data.data(), scratch.featureInput.sizes);
            auto latent0UpTensor = from_blob(scratch.latent0Up.data.data(), scratch.latent0Up.sizes);
            auto latent1UpTensor = from_blob(scratch.latent1Up.data.data(), scratch.latent1Up.sizes);
            auto x0UpTensor = from_blob(scratch.x0Up.data.data(), scratch.x0Up.sizes);
            auto x1UpTensor = from_blob(scratch.x1Up.data.data(), scratch.x1Up.sizes);
            auto xFusedTensor = from_blob(scratch.xFused.data.data(), scratch.xFused.sizes);
            auto decoderFeaturesTensor = from_blob(scratch.decoderFeatures.data.data(), scratch.decoderFeatures.sizes);

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

            if (!copyOutputTensor(*rawHeadResult, 0, scratch.geometryRaw, "split tile_b2 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, scratch.textureRaw, "split tile_b2 texture_raw") ||
                !copyOutputTensor(*initResult, 1, scratch.meanXNdc, "split tile_b2 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, scratch.meanYNdc, "split tile_b2 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, scratch.meanInverseZNdc, "split tile_b2 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scratch.scales, "split tile_b2 scales") ||
                !copyOutputTensor(*initResult, 5, scratch.quaternions, "split tile_b2 quaternions") ||
                !copyOutputTensor(*initResult, 6, scratch.colors, "split tile_b2 colors") ||
                !copyOutputTensor(*initResult, 7, scratch.opacities, "split tile_b2 opacities") ||
                !copyOutputTensor(*initResult, 8, scratch.globalScale, "split tile_b2 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(scratch.geometryRaw.data.data(), scratch.geometryRaw.sizes);
            auto textureRawTensor = from_blob(scratch.textureRaw.data.data(), scratch.textureRaw.sizes);
            auto meanXNdcTensor = from_blob(scratch.meanXNdc.data.data(), scratch.meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(scratch.meanYNdc.data.data(), scratch.meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(scratch.meanInverseZNdc.data.data(), scratch.meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scratch.scales.data.data(), scratch.scales.sizes);
            auto quaternionsTensor = from_blob(scratch.quaternions.data.data(), scratch.quaternions.sizes);
            auto colorsTensor = from_blob(scratch.colors.data.data(), scratch.colors.sizes);
            auto opacitiesTensor = from_blob(scratch.opacities.data.data(), scratch.opacities.sizes);
            auto globalScaleTensor = from_blob(scratch.globalScale.data.data(), scratch.globalScale.sizes);

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
            // Per-tile progress (0.68→0.90) avoids a large jump when numBatches is small (e.g. 2 batches of 8 tiles).
            if (progressEnv && progressReporter && reportProgressMethodId) {
                const int tilesDone = t + 1;
                const float p = 0.68f + 0.22f * (float)tilesDone / (float)NUM_TILES;
                char buf[96];
                snprintf(buf, sizeof(buf), "Part 4b: tile %d/%d…", tilesDone, NUM_TILES);
                reportProgress(progressEnv, progressReporter, reportProgressMethodId, p, buf);
            }
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
        std::vector<float>& outGaussians,
        JNIEnv* progressEnv,
        jobject progressReporter,
        jmethodID reportProgressMethodId) {

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
    bool consumedPart4bPreload = false;
    {
        std::lock_guard<std::mutex> lock(g_part4bTile00PreloadMu);
        if (g_part4bTile00PreloadReady && g_part4bTile00PreloadReady->modelDir == modelDir) {
            splitStagePreModule = std::move(g_part4bTile00PreloadReady->splitStagePreModule);
            splitDecoderHeadModule = std::move(g_part4bTile00PreloadReady->splitDecoderHeadModule);
            splitStageAModule = std::move(g_part4bTile00PreloadReady->splitStageAModule);
            splitInitBaseModule = std::move(g_part4bTile00PreloadReady->splitInitBaseModule);
            splitRawHeadsModule = std::move(g_part4bTile00PreloadReady->splitRawHeadsModule);
            splitComposeModule = std::move(g_part4bTile00PreloadReady->splitComposeModule);
            module = std::move(g_part4bTile00PreloadReady->legacyTileModule);
            modelPath = std::move(g_part4bTile00PreloadReady->legacyModelPath);
            useFineSplitTile00 = g_part4bTile00PreloadReady->useFineSplitTile00;
            useSplitTile00 = g_part4bTile00PreloadReady->useSplitTile00;
            g_part4bTile00PreloadReady.reset();
            consumedPart4bPreload = true;
            LOGD("runPart4bTiledFullPipeline: consumed async Part4b tile_00 preload");
        }
    }
    if (!consumedPart4bPreload) {
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
    }
    logProcessMemory("Part4b tiled seq: after module load");

    if (!g_workspace.allocateTileScratch()) {
        LOGE("runPart4bTiledFullPipeline: tile scratch allocation failed");
        return false;
    }
    struct TileScratchGuard {
        ~TileScratchGuard() { g_workspace.releaseTileScratch(); }
    } tileScratchGuard;

    // Per-tile scratch buffers (pooled in workspace; released when this function returns)
    float* imgCrop = g_workspace.tileImgCrop;
    float* lat0Crop = g_workspace.tileLat0Crop;
    float* lat1Crop = g_workspace.tileLat1Crop;
    float* x0Crop = g_workspace.tileX0Crop;
    float* x1Crop = g_workspace.tileX1Crop;
    float* x2Crop = g_workspace.tileX2Crop;
    float* xlCrop = g_workspace.tileXlCrop;
    Part4bIntermediateScratch scratch;

    outGaussians.clear();
    int floatsPerTile = 0;

    auto tStart = std::chrono::steady_clock::now();
    for (int t = 0; t < NUM_TILES; ++t) {
        const int tileRow = t / GRID;
        const int tileCol = t % GRID;
        auto tileStart = std::chrono::steady_clock::now();

        cropTileNCHWLocal(imageData, imgC, imgH, imgW, tileRow, tileCol, imgCrop);
        cropTileNCHWLocal(latent0, 1024, lat96H, lat96W, tileRow, tileCol, lat0Crop);
        cropTileNCHWLocal(latent1, 1024, lat96H, lat96W, tileRow, tileCol, lat1Crop);
        cropTileNCHWLocal(x0Feat,  1024, lat96H, lat96W, tileRow, tileCol, x0Crop);
        cropTileNCHWLocal(x1Feat,  1024, x1H,    x1W,    tileRow, tileCol, x1Crop);
        cropTileNCHWLocal(x2Feat,  1024, x2H,    x2W,    tileRow, tileCol, x2Crop);
        cropTileNCHWLocal(xLowres.data(), 1024, xLowH, xLowW, tileRow, tileCol, xlCrop);

        auto imgTensor  = from_blob(imgCrop,  {1, 3, imgTileH, imgTileW});
        auto lat0Tensor = from_blob(lat0Crop, {1, 1024, 24, 24});
        auto lat1Tensor = from_blob(lat1Crop, {1, 1024, 24, 24});
        auto x0Tensor   = from_blob(x0Crop,   {1, 1024, 24, 24});
        auto x1Tensor   = from_blob(x1Crop,   {1, 1024, 12, 12});
        auto x2Tensor   = from_blob(x2Crop,   {1, 1024, 6, 6});
        auto xlTensor   = from_blob(xlCrop,   {1, 1024, 6, 6});

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

            if (!copyOutputTensor(*stagePreResult, 0, scratch.latent0Up, "fine split tile_00 latent0_up") ||
                !copyOutputTensor(*stagePreResult, 1, scratch.latent1Up, "fine split tile_00 latent1_up") ||
                !copyOutputTensor(*stagePreResult, 2, scratch.x0Up, "fine split tile_00 x0_up") ||
                !copyOutputTensor(*stagePreResult, 3, scratch.x1Up, "fine split tile_00 x1_up") ||
                !copyOutputTensor(*stagePreResult, 4, scratch.xFused, "fine split tile_00 x_fused")) {
                return false;
            }
            if (!validateOwnedTensorShape(scratch.latent0Up, {1, 256, 192, 192}, "fine split tile_00 latent0_up") ||
                !validateOwnedTensorShape(scratch.latent1Up, {1, 256, 96, 96}, "fine split tile_00 latent1_up") ||
                !validateOwnedTensorShape(scratch.x0Up, {1, 512, 48, 48}, "fine split tile_00 x0_up") ||
                !validateOwnedTensorShape(scratch.x1Up, {1, 1024, 24, 24}, "fine split tile_00 x1_up") ||
                !validateOwnedTensorShape(scratch.xFused, {1, 1024, 12, 12}, "fine split tile_00 x_fused")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 stage_pre produced unexpected shapes on tile %d", t);
                return false;
            }

            auto latent0UpTensor = from_blob(scratch.latent0Up.data.data(), scratch.latent0Up.sizes);
            auto latent1UpTensor = from_blob(scratch.latent1Up.data.data(), scratch.latent1Up.sizes);
            auto x0UpTensor = from_blob(scratch.x0Up.data.data(), scratch.x0Up.sizes);
            auto x1UpTensor = from_blob(scratch.x1Up.data.data(), scratch.x1Up.sizes);
            auto xFusedTensor = from_blob(scratch.xFused.data.data(), scratch.xFused.sizes);

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

            if (!copyOutputTensor(*decoderHeadResult, 0, scratch.disparity, "fine split tile_00 disparity") ||
                !copyOutputTensor(*decoderHeadResult, 1, scratch.decoderFeatures, "fine split tile_00 decoder_features")) {
                return false;
            }
            if (!validateOwnedTensorShape(scratch.disparity, {1, 2, 384, 384}, "fine split tile_00 disparity") ||
                !validateOwnedTensorShape(scratch.decoderFeatures, {1, 256, 192, 192}, "fine split tile_00 decoder_features")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 decoder_head produced unexpected shapes on tile %d", t);
                return false;
            }

            for (size_t k = 0; k < scratch.disparity.data.size(); ++k) {
                float v = std::clamp(scratch.disparity.data[k], 1e-4f, 1e4f);
                scratch.disparity.data[k] = 1.0f / v;
            }
            stitchMonodepthTile(scratch.disparity, 0, tileRow, tileCol, GRID, "fine split tile_00 disparity");
            auto disparityTensor = from_blob(scratch.disparity.data.data(), scratch.disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgTensor);
            initInputs.emplace_back(*disparityTensor);

            LOGD("%s init_base begin (disparity→monodepth applied)", tileLabel.c_str());
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    tileLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 init_base failed on tile %d", t);
                return false;
            }

            if (!copyOutputTensor(*initResult, 0, scratch.featureInput, "fine split tile_00 feature_input")) {
                return false;
            }
            if (!validateOwnedTensorShape(scratch.featureInput, {1, 5, 384, 384}, "fine split tile_00 feature_input")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 init_base produced unexpected feature_input shape on tile %d", t);
                return false;
            }

            auto featureInputTensor = from_blob(scratch.featureInput.data.data(), scratch.featureInput.sizes);
            auto decoderFeaturesTensor = from_blob(scratch.decoderFeatures.data.data(), scratch.decoderFeatures.sizes);

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

            if (!copyOutputTensor(*rawHeadResult, 0, scratch.geometryRaw, "fine split tile_00 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, scratch.textureRaw, "fine split tile_00 texture_raw") ||
                !copyOutputTensor(*initResult, 1, scratch.meanXNdc, "fine split tile_00 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, scratch.meanYNdc, "fine split tile_00 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, scratch.meanInverseZNdc, "fine split tile_00 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scratch.scales, "fine split tile_00 scales") ||
                !copyOutputTensor(*initResult, 5, scratch.quaternions, "fine split tile_00 quaternions") ||
                !copyOutputTensor(*initResult, 6, scratch.colors, "fine split tile_00 colors") ||
                !copyOutputTensor(*initResult, 7, scratch.opacities, "fine split tile_00 opacities") ||
                !copyOutputTensor(*initResult, 8, scratch.globalScale, "fine split tile_00 global_scale")) {
                return false;
            }
            if (!validateOwnedTensorShape(scratch.geometryRaw, {1, 6, 192, 192}, "fine split tile_00 geometry_raw") ||
                !validateOwnedTensorShape(scratch.textureRaw, {1, 22, 192, 192}, "fine split tile_00 texture_raw")) {
                LOGE("runPart4bTiledFullPipeline: fine split tile_00 raw_heads produced unexpected shapes on tile %d", t);
                return false;
            }

            auto geometryRawTensor = from_blob(scratch.geometryRaw.data.data(), scratch.geometryRaw.sizes);
            auto textureRawTensor = from_blob(scratch.textureRaw.data.data(), scratch.textureRaw.sizes);
            auto meanXNdcTensor = from_blob(scratch.meanXNdc.data.data(), scratch.meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(scratch.meanYNdc.data.data(), scratch.meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(scratch.meanInverseZNdc.data.data(), scratch.meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scratch.scales.data.data(), scratch.scales.sizes);
            auto quaternionsTensor = from_blob(scratch.quaternions.data.data(), scratch.quaternions.sizes);
            auto colorsTensor = from_blob(scratch.colors.data.data(), scratch.colors.sizes);
            auto opacitiesTensor = from_blob(scratch.opacities.data.data(), scratch.opacities.sizes);
            auto globalScaleTensor = from_blob(scratch.globalScale.data.data(), scratch.globalScale.sizes);

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

            if (!copyOutputTensor(*stageAResult, 0, scratch.disparity, "split tile_00 disparity")) {
                return false;
            }
            for (size_t k = 0; k < scratch.disparity.data.size(); ++k) {
                float v = std::clamp(scratch.disparity.data[k], 1e-4f, 1e4f);
                scratch.disparity.data[k] = 1.0f / v;
            }
            stitchMonodepthTile(scratch.disparity, 0, tileRow, tileCol, GRID, "split tile_00 disparity");
            auto disparityTensor = from_blob(scratch.disparity.data.data(), scratch.disparity.sizes);
            std::vector<EValue> initInputs;
            initInputs.reserve(2);
            initInputs.emplace_back(*imgTensor);
            initInputs.emplace_back(*disparityTensor);

            LOGD("%s init_base begin (disparity→monodepth applied)", tileLabel.c_str());
            auto initResult = runForwardWithHeartbeat(
                    [&]() { return splitInitBaseModule->forward(initInputs); },
                    tileLabel + " init_base");
            if (!initResult.ok() || initResult->size() < 9) {
                LOGE("runPart4bTiledFullPipeline: split tile_00 init_base failed on tile %d", t);
                return false;
            }

            if (!copyOutputTensor(*initResult, 0, scratch.featureInput, "split tile_00 feature_input") ||
                !copyOutputTensor(*stageAResult, 1, scratch.latent0Up, "split tile_00 latent0_up") ||
                !copyOutputTensor(*stageAResult, 2, scratch.latent1Up, "split tile_00 latent1_up") ||
                !copyOutputTensor(*stageAResult, 3, scratch.x0Up, "split tile_00 x0_up") ||
                !copyOutputTensor(*stageAResult, 4, scratch.x1Up, "split tile_00 x1_up") ||
                !copyOutputTensor(*stageAResult, 5, scratch.xFused, "split tile_00 x_fused") ||
                !copyOutputTensor(*stageAResult, 6, scratch.decoderFeatures, "split tile_00 decoder_features")) {
                return false;
            }

            auto featureInputTensor = from_blob(scratch.featureInput.data.data(), scratch.featureInput.sizes);
            auto latent0UpTensor = from_blob(scratch.latent0Up.data.data(), scratch.latent0Up.sizes);
            auto latent1UpTensor = from_blob(scratch.latent1Up.data.data(), scratch.latent1Up.sizes);
            auto x0UpTensor = from_blob(scratch.x0Up.data.data(), scratch.x0Up.sizes);
            auto x1UpTensor = from_blob(scratch.x1Up.data.data(), scratch.x1Up.sizes);
            auto xFusedTensor = from_blob(scratch.xFused.data.data(), scratch.xFused.sizes);
            auto decoderFeaturesTensor = from_blob(scratch.decoderFeatures.data.data(), scratch.decoderFeatures.sizes);

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

            if (!copyOutputTensor(*rawHeadResult, 0, scratch.geometryRaw, "split tile_00 geometry_raw") ||
                !copyOutputTensor(*rawHeadResult, 1, scratch.textureRaw, "split tile_00 texture_raw") ||
                !copyOutputTensor(*initResult, 1, scratch.meanXNdc, "split tile_00 mean_x_ndc") ||
                !copyOutputTensor(*initResult, 2, scratch.meanYNdc, "split tile_00 mean_y_ndc") ||
                !copyOutputTensor(*initResult, 3, scratch.meanInverseZNdc, "split tile_00 mean_inverse_z_ndc") ||
                !copyOutputTensor(*initResult, 4, scratch.scales, "split tile_00 scales") ||
                !copyOutputTensor(*initResult, 5, scratch.quaternions, "split tile_00 quaternions") ||
                !copyOutputTensor(*initResult, 6, scratch.colors, "split tile_00 colors") ||
                !copyOutputTensor(*initResult, 7, scratch.opacities, "split tile_00 opacities") ||
                !copyOutputTensor(*initResult, 8, scratch.globalScale, "split tile_00 global_scale")) {
                return false;
            }

            auto geometryRawTensor = from_blob(scratch.geometryRaw.data.data(), scratch.geometryRaw.sizes);
            auto textureRawTensor = from_blob(scratch.textureRaw.data.data(), scratch.textureRaw.sizes);
            auto meanXNdcTensor = from_blob(scratch.meanXNdc.data.data(), scratch.meanXNdc.sizes);
            auto meanYNdcTensor = from_blob(scratch.meanYNdc.data.data(), scratch.meanYNdc.sizes);
            auto meanInverseZNdcTensor = from_blob(scratch.meanInverseZNdc.data.data(), scratch.meanInverseZNdc.sizes);
            auto scalesTensor = from_blob(scratch.scales.data.data(), scratch.scales.sizes);
            auto quaternionsTensor = from_blob(scratch.quaternions.data.data(), scratch.quaternions.sizes);
            auto colorsTensor = from_blob(scratch.colors.data.data(), scratch.colors.sizes);
            auto opacitiesTensor = from_blob(scratch.opacities.data.data(), scratch.opacities.sizes);
            auto globalScaleTensor = from_blob(scratch.globalScale.data.data(), scratch.globalScale.sizes);

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
        if (progressEnv && progressReporter && reportProgressMethodId) {
            // Align with batched path: span 0.68→0.90 across tiles (Kotlin PLY phase starts at 0.91).
            const float p = 0.68f + 0.22f * (float)(t + 1) / (float)NUM_TILES;
            char buf[96];
            snprintf(buf, sizeof(buf), "Part 4b: tile %d/%d…", t + 1, NUM_TILES);
            reportProgress(progressEnv, progressReporter, reportProgressMethodId, p, buf);
        }
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
    std::sort(indices.begin(), indices.begin() + maxGaussians);

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
    std::sort(indices.begin(), indices.begin() + maxGaussians);

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
    clearLastMonodepthCapture();
    LOGD("Released module cache + workspace");
}

JNIEXPORT jintArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_getLastMonodepthInfoNative(
        JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(g_lastMonodepthMutex);
    if (g_lastMonodepth.empty() || g_lastMonodepthWidth <= 0 || g_lastMonodepthHeight <= 0 || g_lastMonodepthChannels <= 0) {
        return nullptr;
    }
    jint values[3] = {
            static_cast<jint>(g_lastMonodepthWidth),
            static_cast<jint>(g_lastMonodepthHeight),
            static_cast<jint>(g_lastMonodepthChannels),
    };
    jintArray out = env->NewIntArray(3);
    if (!out) {
        return nullptr;
    }
    env->SetIntArrayRegion(out, 0, 3, values);
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_getLastMonodepthBufferNative(
        JNIEnv* env, jobject) {
    std::lock_guard<std::mutex> lock(g_lastMonodepthMutex);
    if (g_lastMonodepth.empty()) {
        return nullptr;
    }
    jfloatArray out = env->NewFloatArray(static_cast<jsize>(g_lastMonodepth.size()));
    if (!out) {
        return nullptr;
    }
    env->SetFloatArrayRegion(out, 0, static_cast<jsize>(g_lastMonodepth.size()), g_lastMonodepth.data());
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_sampleMonodepthAtPointsNative(
        JNIEnv* env, jobject, jintArray pixelXs, jintArray pixelYs, jint channel) {
    if (!pixelXs || !pixelYs) {
        return nullptr;
    }
    const jsize count = env->GetArrayLength(pixelXs);
    if (count != env->GetArrayLength(pixelYs)) {
        LOGE("sampleMonodepthAtPointsNative: xs/ys length mismatch");
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_lastMonodepthMutex);
    if (g_lastMonodepth.empty() || g_lastMonodepthWidth <= 0 || g_lastMonodepthHeight <= 0 || g_lastMonodepthChannels <= 0) {
        return nullptr;
    }

    jint* xs = env->GetIntArrayElements(pixelXs, nullptr);
    jint* ys = env->GetIntArrayElements(pixelYs, nullptr);
    if (!xs || !ys) {
        if (xs) env->ReleaseIntArrayElements(pixelXs, xs, JNI_ABORT);
        if (ys) env->ReleaseIntArrayElements(pixelYs, ys, JNI_ABORT);
        return nullptr;
    }

    const int planeSize = g_lastMonodepthWidth * g_lastMonodepthHeight;
    const int channelIndex = std::clamp(static_cast<int>(channel), 0, g_lastMonodepthChannels - 1);
    const int channelBase = channelIndex * planeSize;
    std::vector<float> results(static_cast<size_t>(count), 0.0f);
    for (jsize i = 0; i < count; ++i) {
        const int x = std::clamp(static_cast<int>(xs[i]), 0, g_lastMonodepthWidth - 1);
        const int y = std::clamp(static_cast<int>(ys[i]), 0, g_lastMonodepthHeight - 1);
        results[static_cast<size_t>(i)] = g_lastMonodepth[static_cast<size_t>(channelBase + y * g_lastMonodepthWidth + x)];
    }

    env->ReleaseIntArrayElements(pixelXs, xs, JNI_ABORT);
    env->ReleaseIntArrayElements(pixelYs, ys, JNI_ABORT);

    jfloatArray out = env->NewFloatArray(count);
    if (!out) {
        return nullptr;
    }
    env->SetFloatArrayRegion(out, 0, count, results.data());
    return out;
}

JNIEXPORT jfloatArray JNICALL
Java_com_furnit_android_services_ExecutorchInt8Sharp_writePlyNative(
        JNIEnv* env,
        jobject,
        jstring outputPath,
        jfloatArray params,
        jfloat aspectCorrX,
        jfloat aspectCorrY,
        jfloat metricScale) {
    if (!outputPath || !params) {
        return nullptr;
    }
    const jsize numFloats = env->GetArrayLength(params);
    if (numFloats <= 0 || (numFloats % PARAMS_PER_GAUSSIAN) != 0) {
        LOGE("writePlyNative: invalid param length %d", static_cast<int>(numFloats));
        return nullptr;
    }

    const char* outputPathChars = env->GetStringUTFChars(outputPath, nullptr);
    if (!outputPathChars) {
        return nullptr;
    }
    jboolean isCopy = JNI_FALSE;
    jfloat* paramsData = env->GetFloatArrayElements(params, &isCopy);
    if (!paramsData) {
        env->ReleaseStringUTFChars(outputPath, outputPathChars);
        return nullptr;
    }

    PlyExportStats stats;
    const bool ok = writePlyBinaryNative(
            outputPathChars,
            paramsData,
            static_cast<int>(numFloats / PARAMS_PER_GAUSSIAN),
            aspectCorrX,
            aspectCorrY,
            metricScale,
            &stats);

    env->ReleaseFloatArrayElements(params, paramsData, JNI_ABORT);
    env->ReleaseStringUTFChars(outputPath, outputPathChars);
    if (!ok) {
        return nullptr;
    }

    const jfloat values[9] = {
            stats.minX, stats.maxX,
            stats.minY, stats.maxY,
            stats.minZ, stats.maxZ,
            stats.maxAbsX, stats.maxAbsY, stats.maxAbsZ,
    };
    jfloatArray out = env->NewFloatArray(9);
    if (!out) {
        return nullptr;
    }
    env->SetFloatArrayRegion(out, 0, 9, values);
    return out;
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
        jstring etdumpOutputPath,
        jboolean hybridInterleavePart12,
        jlong hybridInterleaveMinAvailMemBytes) {
    if (useVulkan == JNI_TRUE) {
        return runSharpFullPipeline_Vulkan(
                env, thiz, modelDirPath, imageNCHW, maxGaussians, preferSinglePart4b, part12OnCpu,
                part12ForceSinglePatch, part12_25Only,
                part1MaxPatches1x, part1MaxPatches05x, part12Chunk1x, part12Chunk05x,
                part12YieldMsBetweenChunks, swapTileNdcXY, progressReporter, etdumpOutputPath,
                hybridInterleavePart12,
                hybridInterleaveMinAvailMemBytes);
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
