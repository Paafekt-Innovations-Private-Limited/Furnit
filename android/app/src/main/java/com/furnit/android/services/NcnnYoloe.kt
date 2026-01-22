package com.furnit.android.services

import android.content.Context
import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.graphics.Color
import android.util.Log

/**
 * NcnnYoloe provides high-performance YOLOE inference using NCNN.
 *
 * This class wraps native C++ code that uses NCNN for GPU-accelerated
 * object detection and instance segmentation on Android devices.
 *
 * Usage:
 * 1. Call `init()` with AssetManager to load the model
 * 2. Call `detect()` with a Bitmap to run inference
 * 3. Call `release()` when done
 *
 * Model files required in assets folder:
 * - yoloe-11l-seg.param (NCNN graph definition)
 * - yoloe-11l-seg.bin (NCNN weights)
 */
class NcnnYoloe {

    companion object {
        private const val TAG = "NcnnYoloe"

        // Try to load the native library
        private var libraryLoaded = false
        private var libraryLoadError: String? = null

        init {
            try {
                // NCNN rebuilt with 16KB page alignment for Android 15+ compatibility
                System.loadLibrary("yoloe_ncnn")
                libraryLoaded = true
                Log.i(TAG, "NCNN native library loaded successfully (16KB aligned)")
            } catch (e: UnsatisfiedLinkError) {
                Log.w(TAG, "Failed to load yoloe_ncnn library: ${e.message}")
                libraryLoaded = false
                libraryLoadError = e.message
            }
        }

        fun isAvailable(): Boolean = libraryLoaded
        fun getLoadError(): String? = libraryLoadError
    }

    private var nativeHandle: Long = 0
    private var isInitialized = false
    private var inputWidth = 1280
    private var inputHeight = 1280

    /**
     * Initialize the NCNN model from assets.
     *
     * @param context Application context for accessing assets
     * @param paramAsset Name of the .param file in assets (default: "yoloe-11l-seg.param")
     * @param binAsset Name of the .bin file in assets (default: "yoloe-11l-seg.bin")
     * @param useGpu Whether to use GPU (Vulkan) acceleration (default: true)
     * @param numThreads Number of CPU threads (default: 4)
     * @return true if initialization succeeded
     */
    fun init(
        context: Context,
        paramAsset: String = "yoloe-11l-seg.param",
        binAsset: String = "yoloe-11l-seg.bin",
        useGpu: Boolean = true,
        numThreads: Int = 4
    ): Boolean {
        if (!libraryLoaded) {
            Log.e(TAG, "Native library not loaded, cannot initialize")
            return false
        }

        if (isInitialized) {
            Log.w(TAG, "Already initialized, call release() first")
            return true
        }

        return try {
            val assetManager = context.assets

            // Check if GPU is available
            val gpuAvailable = nativeHasGpu()
            val actualUseGpu = useGpu && gpuAvailable

            if (useGpu && !gpuAvailable) {
                Log.w(TAG, "GPU requested but not available, using CPU")
            }

            Log.i(TAG, "Initializing NCNN: param=$paramAsset, bin=$binAsset, gpu=$actualUseGpu, threads=$numThreads")

            nativeHandle = nativeInit(
                assetManager,
                paramAsset,
                binAsset,
                actualUseGpu,
                numThreads
            )

            if (nativeHandle != 0L) {
                isInitialized = true
                Log.i(TAG, "NCNN initialization successful, handle=$nativeHandle")
                true
            } else {
                Log.e(TAG, "NCNN initialization failed")
                false
            }
        } catch (e: Exception) {
            Log.e(TAG, "NCNN init exception: ${e.message}", e)
            false
        }
    }

    /**
     * Run object detection on a bitmap.
     *
     * @param bitmap Input image (any size, will be resized internally)
     * @param confThreshold Confidence threshold (0.0 - 1.0, default: 0.25)
     * @param iouThreshold IoU threshold for NMS (0.0 - 1.0, default: 0.45)
     * @return List of Detection objects, or empty list if failed
     */
    fun detect(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f
    ): List<Detection> {
        if (!isInitialized || nativeHandle == 0L) {
            Log.w(TAG, "Not initialized, returning empty detections")
            return emptyList()
        }

        return try {
            // Convert bitmap to ARGB_8888 if needed
            val inputBitmap = if (bitmap.config == Config.ARGB_8888) {
                bitmap
            } else {
                bitmap.copy(Config.ARGB_8888, false)
            }

            val results = nativeDetect(
                nativeHandle,
                inputBitmap,
                confThreshold,
                iouThreshold
            )

            // Parse results from native code
            // Format: [x, y, w, h, confidence, classId, coeff0...coeff31] per detection
            parseDetections(results, bitmap.width, bitmap.height)
        } catch (e: Exception) {
            Log.e(TAG, "Detection failed: ${e.message}", e)
            emptyList()
        }
    }

    /**
     * Generate a segmentation mask from detections.
     *
     * @param bitmap Original input image
     * @param detections List of detections from detect()
     * @param maskThreshold Threshold for mask binarization (default: 0.5)
     * @return Bitmap mask with alpha channel, or null if failed
     */
    fun generateMask(
        bitmap: Bitmap,
        detections: List<Detection>,
        maskThreshold: Float = 0.5f
    ): Bitmap? {
        if (!isInitialized || nativeHandle == 0L) {
            Log.w(TAG, "Not initialized, cannot generate mask")
            return null
        }

        if (detections.isEmpty()) {
            return null
        }

        return try {
            // Prepare coefficients array
            val numDetections = detections.size
            val coeffsArray = FloatArray(numDetections * 32)

            for (i in detections.indices) {
                val det = detections[i]
                for (j in 0 until 32) {
                    coeffsArray[i * 32 + j] = det.maskCoeffs[j]
                }
            }

            // Get mask from native code
            val maskPixels = nativeGenerateMask(
                nativeHandle,
                coeffsArray,
                numDetections,
                bitmap.width,
                bitmap.height,
                maskThreshold
            )

            if (maskPixels == null || maskPixels.isEmpty()) {
                return null
            }

            // Create bitmap from pixels
            val maskBitmap = Bitmap.createBitmap(bitmap.width, bitmap.height, Config.ARGB_8888)
            maskBitmap.setPixels(maskPixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
            maskBitmap
        } catch (e: Exception) {
            Log.e(TAG, "Mask generation failed: ${e.message}", e)
            null
        }
    }

    /**
     * Run detection and generate mask in one call (more efficient).
     *
     * @param bitmap Input image
     * @param confThreshold Confidence threshold
     * @param iouThreshold IoU threshold
     * @param maskThreshold Mask binarization threshold
     * @return DetectionResult containing detections and optional mask
     */
    fun detectWithMask(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maskThreshold: Float = 0.5f
    ): DetectionResult {
        val detections = detect(bitmap, confThreshold, iouThreshold)
        val mask = if (detections.isNotEmpty()) {
            generateMask(bitmap, detections, maskThreshold)
        } else {
            null
        }
        return DetectionResult(detections, mask)
    }

    /**
     * Release native resources.
     */
    fun release() {
        if (nativeHandle != 0L) {
            nativeRelease(nativeHandle)
            nativeHandle = 0
        }
        isInitialized = false
        Log.i(TAG, "NCNN resources released")
    }

    /**
     * Check if GPU (Vulkan) is available.
     */
    fun hasGpu(): Boolean {
        return if (libraryLoaded) nativeHasGpu() else false
    }

    private fun parseDetections(results: FloatArray?, imgWidth: Int, imgHeight: Int): List<Detection> {
        if (results == null || results.isEmpty()) {
            return emptyList()
        }

        val detections = mutableListOf<Detection>()
        val valuesPerDetection = 6 + 32 // x, y, w, h, conf, classId, + 32 mask coeffs
        val numDetections = results.size / valuesPerDetection

        for (i in 0 until numDetections) {
            val offset = i * valuesPerDetection

            // Convert from model coordinates to image coordinates
            val x = results[offset] * imgWidth / inputWidth
            val y = results[offset + 1] * imgHeight / inputHeight
            val w = results[offset + 2] * imgWidth / inputWidth
            val h = results[offset + 3] * imgHeight / inputHeight
            val confidence = results[offset + 4]
            val classId = results[offset + 5].toInt()

            val maskCoeffs = FloatArray(32)
            for (j in 0 until 32) {
                maskCoeffs[j] = results[offset + 6 + j]
            }

            detections.add(Detection(
                x = x,
                y = y,
                width = w,
                height = h,
                confidence = confidence,
                classId = classId,
                label = getClassName(classId),
                maskCoeffs = maskCoeffs
            ))
        }

        Log.d(TAG, "Parsed ${detections.size} detections")
        return detections
    }

    private fun getClassName(classId: Int): String {
        return if (classId >= 0 && classId < COCO_CLASSES.size) {
            COCO_CLASSES[classId]
        } else {
            "object"
        }
    }

    // Native methods
    private external fun nativeInit(
        assetManager: AssetManager,
        paramAsset: String,
        binAsset: String,
        useGpu: Boolean,
        numThreads: Int
    ): Long

    private external fun nativeDetect(
        handle: Long,
        bitmap: Bitmap,
        confThreshold: Float,
        iouThreshold: Float
    ): FloatArray?

    private external fun nativeGenerateMask(
        handle: Long,
        coeffs: FloatArray,
        numDetections: Int,
        width: Int,
        height: Int,
        maskThreshold: Float
    ): IntArray?

    private external fun nativeRelease(handle: Long)

    private external fun nativeHasGpu(): Boolean

    /**
     * Data class for a single detection.
     */
    data class Detection(
        val x: Float,           // Center X
        val y: Float,           // Center Y
        val width: Float,       // Bounding box width
        val height: Float,      // Bounding box height
        val confidence: Float,  // Detection confidence
        val classId: Int,       // Class ID
        val label: String,      // Class name
        val maskCoeffs: FloatArray  // 32 mask coefficients for segmentation
    ) {
        // Bounding box in corner format (for drawing)
        val left: Float get() = x - width / 2
        val top: Float get() = y - height / 2
        val right: Float get() = x + width / 2
        val bottom: Float get() = y + height / 2

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false
            other as Detection
            return x == other.x && y == other.y && width == other.width &&
                   height == other.height && classId == other.classId
        }

        override fun hashCode(): Int {
            return arrayOf(x, y, width, height, classId).contentHashCode()
        }
    }

    /**
     * Result container for detection with optional mask.
     */
    data class DetectionResult(
        val detections: List<Detection>,
        val mask: Bitmap?
    )
}

// COCO class names for labeling
private val COCO_CLASSES = arrayOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
)
