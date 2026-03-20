/**
 * Tiny Vulkan test: query device limits and print whether 1536x1536 is supported.
 * Used to verify device capability before running ExecuTorch Vulkan SHARP pipeline.
 * Log tag: Vulkan1536Test
 */
#include <android/log.h>
#include <jni.h>
#include <cstring>
#include <vector>
#include <vulkan/vulkan.h>
#include <vulkan/vulkan_android.h>

#define LOG_TAG "Vulkan1536Test"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static const uint32_t TEST_SIZE = 1536u;
static const size_t   TEST_IMAGE_BYTES = (size_t)3 * TEST_SIZE * TEST_SIZE * sizeof(float);  // NCHW float

extern "C" JNIEXPORT void JNICALL
Java_com_furnit_android_utils_Vulkan1536Test_runVulkan1536Test(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;

    VkResult err;
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;

    LOGI("=== Vulkan 1536x1536 support check ===");

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Vulkan1536Test";
    appInfo.applicationVersion = 1;
    appInfo.pEngineName = "none";
    appInfo.engineVersion = 1;
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    err = vkCreateInstance(&instInfo, nullptr, &instance);
    if (err != VK_SUCCESS) {
        LOGE("vkCreateInstance failed: %d (Vulkan 1.1 not available?)", (int)err);
        return;
    }

    uint32_t deviceCount = 0;
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (err != VK_SUCCESS || deviceCount == 0) {
        LOGE("vkEnumeratePhysicalDevices failed or no devices: %d, count=%u", (int)err, deviceCount);
        vkDestroyInstance(instance, nullptr);
        return;
    }

    VkPhysicalDevice* devices = new VkPhysicalDevice[deviceCount];
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    if (err != VK_SUCCESS) {
        LOGE("vkEnumeratePhysicalDevices (get) failed: %d", (int)err);
        delete[] devices;
        vkDestroyInstance(instance, nullptr);
        return;
    }

    physicalDevice = devices[0];
    VkPhysicalDeviceProperties props = {};
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    LOGI("Device: %s (API %u.%u)", props.deviceName, VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));

    VkPhysicalDeviceLimits limits = props.limits;
    uint32_t maxDim2D = limits.maxImageDimension2D;
    VkDeviceSize maxStorageRange = limits.maxStorageBufferRange;

    LOGI("maxImageDimension2D: %u (need >= %u)", maxDim2D, TEST_SIZE);
    LOGI("maxStorageBufferRange: %llu (need >= %zu for 3x1536x1536 float)", (unsigned long long)maxStorageRange, TEST_IMAGE_BYTES);
    LOGI("maxMemoryAllocationCount: %u", limits.maxMemoryAllocationCount);

    int supported = 1;
    if (maxDim2D < TEST_SIZE) {
        LOGI("1536x1536 NOT supported: maxImageDimension2D %u < 1536", maxDim2D);
        supported = 0;
    }
    if (maxStorageRange < TEST_IMAGE_BYTES) {
        LOGI("1536x1536 buffer NOT supported: maxStorageBufferRange %llu < %zu", (unsigned long long)maxStorageRange, TEST_IMAGE_BYTES);
        supported = 0;
    }
    if (supported)
        LOGI("1536x1536 supported: YES (limits OK; runtime ExecuTorch may still fail on this device)");
    else
        LOGI("1536x1536 supported: NO (limits insufficient)");

    delete[] devices;
    vkDestroyInstance(instance, nullptr);
}

// Log tag for full diagnostics (shaders/sync); filter with: adb logcat -s VulkanDiag:D
#define DIAG_TAG "VulkanDiag"

extern "C" JNIEXPORT void JNICALL
Java_com_furnit_android_utils_Vulkan1536Test_runVulkanDiagnostics(JNIEnv* env, jclass clazz) {
    (void)env;
    (void)clazz;

    VkResult err;
    VkInstance instance = VK_NULL_HANDLE;

    __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "========== Vulkan & ExecuTorch diagnostics ==========");

    // Instance extensions (optional; often empty on Android)
    uint32_t instExtCount = 0;
    err = vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, nullptr);
    if (err == VK_SUCCESS && instExtCount > 0) {
        std::vector<VkExtensionProperties> instExts(instExtCount);
        err = vkEnumerateInstanceExtensionProperties(nullptr, &instExtCount, instExts.data());
        if (err == VK_SUCCESS) {
            __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "Instance extensions: %u", instExtCount);
            for (uint32_t i = 0; i < instExtCount; ++i)
                __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "  [%u] %s (spec %u)", i, instExts[i].extensionName, instExts[i].specVersion);
        }
    } else {
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "Instance extensions: 0 or enum failed (%d)", (int)err);
    }

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "VulkanDiag";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instInfo = {};
    instInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instInfo.pApplicationInfo = &appInfo;

    err = vkCreateInstance(&instInfo, nullptr, &instance);
    if (err != VK_SUCCESS) {
        __android_log_print(ANDROID_LOG_ERROR, DIAG_TAG, "vkCreateInstance failed: %d", (int)err);
        return;
    }

    uint32_t deviceCount = 0;
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (err != VK_SUCCESS || deviceCount == 0) {
        __android_log_print(ANDROID_LOG_ERROR, DIAG_TAG, "No physical devices: err=%d count=%u", (int)err, deviceCount);
        vkDestroyInstance(instance, nullptr);
        return;
    }

    VkPhysicalDevice* devices = new VkPhysicalDevice[deviceCount];
    err = vkEnumeratePhysicalDevices(instance, &deviceCount, devices);
    if (err != VK_SUCCESS) {
        delete[] devices;
        vkDestroyInstance(instance, nullptr);
        return;
    }

    for (uint32_t d = 0; d < deviceCount; ++d) {
        VkPhysicalDevice phys = devices[d];
        VkPhysicalDeviceProperties props = {};
        vkGetPhysicalDeviceProperties(phys, &props);

        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "--- Device %u ---", d);
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "deviceName: %s", props.deviceName);
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "driverVersion: %u (0x%x)", props.driverVersion, props.driverVersion);
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "apiVersion: %u.%u.%u", VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion), VK_VERSION_PATCH(props.apiVersion));
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "vendorID: 0x%x deviceID: 0x%x", props.vendorID, props.deviceID);

        uint32_t extCount = 0;
        err = vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, nullptr);
        if (err != VK_SUCCESS) {
            __android_log_print(ANDROID_LOG_ERROR, DIAG_TAG, "device extension enum failed: %d", (int)err);
            continue;
        }

        std::vector<VkExtensionProperties> exts(extCount);
        err = vkEnumerateDeviceExtensionProperties(phys, nullptr, &extCount, exts.data());
        if (err != VK_SUCCESS) continue;

        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "Device extensions: %u", extCount);
        int hasSync2 = 0, hasTimeline = 0;
        for (uint32_t i = 0; i < extCount; ++i) {
            __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "  [%u] %s (spec %u)", i, exts[i].extensionName, exts[i].specVersion);
            if (strcmp(exts[i].extensionName, "VK_KHR_synchronization2") == 0) hasSync2 = 1;
            if (strcmp(exts[i].extensionName, "VK_KHR_timeline_semaphore") == 0) hasTimeline = 1;
        }
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "Sync: VK_KHR_synchronization2=%s VK_KHR_timeline_semaphore=%s",
                            hasSync2 ? "YES" : "NO", hasTimeline ? "YES" : "NO");
        __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "  (Missing sync2 can contribute to fence/device-lost with ExecuTorch Vulkan)");
    }

    __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "Shader registry: inside ExecuTorch lib (not enumerable from app). Build from source with EXECUTORCH_BUILD_VULKAN=ON for full shaders.");
    __android_log_print(ANDROID_LOG_INFO, DIAG_TAG, "========== end diagnostics ==========");

    delete[] devices;
    vkDestroyInstance(instance, nullptr);
}
