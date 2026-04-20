package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Handler
import android.os.Looper
import com.furnit.android.utils.LogUtil
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtException
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import com.furnit.android.DetectionResult
import com.furnit.android.ar.ArSupportChecker
import java.io.File
import java.io.IOException
import org.json.JSONObject
import kotlin.math.ceil
import kotlin.math.exp
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

// Result containing mask and detections
data class SegmentationResult(
    val mask: Bitmap?,
    val detections: List<DetectionResult>,
    val inputSize: Int = 640,
    val primaryDetection: DetectionResult? = null,
)

/**
 * FurnitureFitManager handles object detection and segmentation using YOLOE models.
 *
 * Inference backend:
 * 1. ONNX Runtime (`yoloe-11l-seg-pf.onnx` in generated assets)
 *
 * Furniture segmentation is ONNX-only. NCNN and TensorFlow Lite are not used here.
 */
class FurnitureFitManager(private val context: Context) {
    private val mainHandler = Handler(Looper.getMainLooper())
    private val bboxExpandMargin = 0.08f
    private val includeSupportingTableForMonitorScene = true
    private val enableMorphCloseForMask = false
    private val monitorLikeClassIds = setOf(1063, 2675, 4105)
    private val supportingTableClassIds = setOf(1061, 1301, 1325, 1503, 1885, 2324, 2836, 4564)
    private val classNames: Map<Int, String> by lazy(LazyThreadSafetyMode.NONE) { loadClassNames() }
    private val ignoredClassIds: Set<Int> by lazy(LazyThreadSafetyMode.NONE) { loadIgnoredClassIds() }

    companion object {
        private const val TAG = "FurnitureFitManager"
        private const val DEFAULT_ONNX_MODEL_ASSET = "yoloe-11l-seg-pf.onnx"
        private const val DEFAULT_CONFIDENCE_THRESHOLD = 0.10f
        private const val DEFAULT_NMS_IOU_THRESHOLD = 0.50f
        private const val DEFAULT_MAX_DETECTIONS = 1000

        /**
         * When true, shows ⋮ "Calibrate wall" and the brain-session "Tap to calibrate" pill (matches iOS `show_room_furniture_calibrate`).
         * Default false — same as iOS @AppStorage default.
         */
        const val KEY_SHOW_ROOM_FURNITURE_CALIBRATE_UI = "show_room_furniture_calibrate"
        const val KEY_SHOW_FULL_VIDEO_WITH_IDENTIFICATIONS = "show_full_video_with_identifications"

        /**
         * Metric overlay sizing uses ARCore depth/planes when the device supports ARCore; otherwise non-metric fallback.
         */
        fun isArAssistedFurnitureSizingEnabled(context: android.content.Context): Boolean =
            ArSupportChecker.isArCoreSupported(context)

        fun isRoomFurnitureCalibrateUiEnabled(context: Context): Boolean {
            return context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
                .getBoolean(KEY_SHOW_ROOM_FURNITURE_CALIBRATE_UI, false)
        }

        fun isFullVideoWithIdentificationsEnabled(context: Context): Boolean {
            return context.getSharedPreferences("furnit_prefs", Context.MODE_PRIVATE)
                .getBoolean(KEY_SHOW_FULL_VIDEO_WITH_IDENTIFICATIONS, true)
        }
    }
    private val inferenceExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()

    // ONNX Runtime objects
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    /**
     * Initializes the ONNX Runtime segmentation backend.
     */
    fun initializeAuto(): Boolean {
        LogUtil.i(TAG, "Initializing iOS-parity YOLOE 11L ONNX segmentation backend...")

        try {
            initializeOnnx()
            if (ortSession != null) {
                LogUtil.i(TAG, "Using ONNX Runtime backend")
                return true
            }
        } catch (e: Exception) {
            LogUtil.w(TAG, "ONNX initialization failed: ${e.message}")
        }

        LogUtil.e(TAG, "ONNX initialization failed - segmentation disabled")
        return false
    }

    /** Initialize ONNX Runtime session from asset ONNX model. */
    fun initializeOnnx(onnxAssetName: String = DEFAULT_ONNX_MODEL_ASSET) {
        val initStartNanos = System.nanoTime()
        LogUtil.i(TAG, "initializeOnnx called with '$onnxAssetName'")
        try {
            LogUtil.d(TAG, "Copying asset to cache...")
            val file = copyAssetToFile(onnxAssetName)
            LogUtil.d(
                TAG,
                "Asset copied to: ${file.absolutePath}, size: ${file.length()}"
            )

            LogUtil.d(TAG, "Getting ORT environment...")
            ortEnv = OrtEnvironment.getEnvironment()

            LogUtil.d(TAG, "Creating ONNX session...")
            val opts = SessionOptions()
            ortSession = ortEnv!!.createSession(file.absolutePath, opts)

            // Log input/output info
            LogUtil.i(TAG, "ONNX model loaded successfully")
            for ((name, info) in ortSession!!.inputInfo) {
                LogUtil.i(TAG, "ONNX input: $name -> ${info.info}")
            }
            for ((name, info) in ortSession!!.outputInfo) {
                LogUtil.i(TAG, "ONNX output: $name -> ${info.info}")
            }
            LogUtil.i(
                TAG,
                "Loaded ONNX model '$onnxAssetName' into ONNX Runtime in ${elapsedMillis(initStartNanos)}ms"
            )
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to load onnx: ${e.message}", e)
            ortSession = null
            ortEnv = null
        }
    }

    fun segmentImageAsync(frame: Bitmap?, callback: (Bitmap?) -> Unit) {
        // Wrapper that discards detection info
        segmentWithDetectionsAsync(frame) { result ->
            callback(result?.mask)
        }
    }

    fun segmentWithDetectionsAsync(
        frame: Bitmap?,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(frame, includeMask = true, selectedClassIds = emptySet(), pinnedDetections = null, callback = callback)
    }

    fun detectWithDetectionsAsync(
        frame: Bitmap?,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(frame, includeMask = false, selectedClassIds = emptySet(), pinnedDetections = null, callback = callback)
    }

    fun segmentSelectedClassesAsync(
        frame: Bitmap?,
        selectedClassIds: Set<Int>,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(frame, includeMask = true, selectedClassIds = selectedClassIds, pinnedDetections = null, callback = callback)
    }

    /**
     * Segment only the object instances matching [pinnedDetections] (same class + IoU to live box), not every box of that class.
     */
    fun segmentSelectedInstancesAsync(
        frame: Bitmap?,
        pinnedDetections: List<DetectionResult>,
        callback: (SegmentationResult?) -> Unit,
    ) {
        analyzeFrameAsync(
            frame,
            includeMask = true,
            selectedClassIds = emptySet(),
            pinnedDetections = pinnedDetections,
            callback = callback,
        )
    }

    private fun analyzeFrameAsync(
        frame: Bitmap?,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>? = null,
        callback: (SegmentationResult?) -> Unit,
    ) {
        if (frame == null) {
            mainHandler.postDelayed({ callback(null) }, 200)
            return
        }

        inferenceExecutor.execute {
            try {
                if (ortSession != null) {
                    runOnnxInferenceWithDetections(frame, includeMask, selectedClassIds, pinnedDetections, callback)
                    return@execute
                }

                mainHandler.post { callback(null) }
            } catch (e: Exception) {
                LogUtil.e("FurnitureFitManager", "inference error", e)
                mainHandler.post { callback(null) }
            }
        }
    }

    /**
     * Nearest-neighbor map from prototype mask to full frame, copy camera ARGB only where mask > [threshold].
     * Fills [outPixels] with transparent black, then only scans the primary band [x0,x1)×[y0,y1) and uses
     * horizontal spans that share the same proto column (fewer branches than per-pixel double loop).
     */
    private fun composeNearestProtoMaskCutoutArgb(
        framePixels: IntArray,
        outPixels: IntArray,
        maskProto: FloatArray,
        frameW: Int,
        frameH: Int,
        protoW: Int,
        protoH: Int,
        x0: Int,
        x1: Int,
        y0: Int,
        y1: Int,
        threshold: Float = 0.5f,
    ) {
        outPixels.fill(0)
        if (x0 >= x1 || y0 >= y1 || frameW <= 0 || frameH <= 0) return
        val xStart = x0.coerceIn(0, frameW)
        val xEnd = x1.coerceIn(0, frameW)
        val yStart = y0.coerceIn(0, frameH)
        val yEnd = y1.coerceIn(0, frameH)
        if (xStart >= xEnd || yStart >= yEnd) return

        for (y in yStart until yEnd) {
            val protoY = (y * protoH) / frameH
            val protoRow = protoY * protoW
            val rowBase = y * frameW
            var x = xStart
            while (x < xEnd) {
                val protoX = (x * protoW) / frameW
                val nextX = minOf(
                    xEnd,
                    ((protoX + 1) * frameW + protoW - 1) / protoW,
                )
                if (maskProto[protoRow + protoX] > threshold) {
                    System.arraycopy(framePixels, rowBase + x, outPixels, rowBase + x, nextX - x)
                }
                x = nextX
            }
        }
    }

    private fun runOnnxInference(frame: Bitmap, callback: (Bitmap?) -> Unit) {
        try {
            val session = ortSession ?: run {
                LogUtil.e("FurnitureFitManager", "ortSession is null")
                mainHandler.post { callback(null) }
                return
            }
            val env = ortEnv ?: run {
                LogUtil.e("FurnitureFitManager", "ortEnv is null")
                mainHandler.post { callback(null) }
                return
            }

            // Use first input info to determine shape
            val firstInput = session.inputInfo.entries.firstOrNull()
            if (firstInput == null) {
                LogUtil.w("FurnitureFitManager", "ONNX session has no inputs")
                mainHandler.post { callback(null) }
                return
            }

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info

            // --- FIX #1 (important): use model-declared input H/W instead of hardcoding 640 ---
            // Many YOLOE exports are 768x768 (your anchors like 48384 and proto like 384 strongly suggest 768).
            var inputH = 640
            var inputW = 640
            if (tensorInfo is ai.onnxruntime.TensorInfo) {
                val sh = tensorInfo.shape
                // Expected [1,3,H,W] (NCHW). H/W may be -1 if dynamic.
                if (sh.size == 4) {
                    val hCandidate = sh[2].toInt()
                    val wCandidate = sh[3].toInt()
                    if (hCandidate > 0 && wCandidate > 0) {
                        inputH = hCandidate
                        inputW = wCandidate
                    }
                }
            }

            // Resize to model input size
            LogUtil.d(
                "FurnitureFitManager",
                "Resizing frame ${frame.width}x${frame.height} to ${inputW}x${inputH}"
            )
            val resized = Bitmap.createScaledBitmap(frame, inputW, inputH, true).copy(Config.ARGB_8888, false)

            // Prepare float array in NCHW format
            val floatCount = 1 * 3 * inputH * inputW
            val inputFloats = FloatArray(floatCount)
            val intValues = IntArray(resized.width * resized.height)
            resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

            // Fill NCHW layout: channel-first ordering
            val hw = inputH * inputW
            for (y in 0 until inputH) {
                val rowOff = y * inputW
                for (x in 0 until inputW) {
                    val v = intValues[rowOff + x]
                    val r = ((v shr 16) and 0xFF) / 255.0f
                    val g = ((v shr 8) and 0xFF) / 255.0f
                    val b = (v and 0xFF) / 255.0f
                    val pixelIdx = rowOff + x
                    inputFloats[0 * hw + pixelIdx] = r
                    inputFloats[1 * hw + pixelIdx] = g
                    inputFloats[2 * hw + pixelIdx] = b
                }
            }

            val shapeLong = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
            LogUtil.d("FurnitureFitManager", "Creating input tensor with shape ${shapeLong.toList()}")
            LogUtil.d(
                "FurnitureFitManager",
                "Input sample - R[0]=${inputFloats[0]}, G[0]=${inputFloats[hw]}, B[0]=${inputFloats[2 * hw]}"
            )
            LogUtil.d("FurnitureFitManager", "Input range - min=${inputFloats.minOrNull()}, max=${inputFloats.maxOrNull()}")

            val tensor = OnnxTensor.createTensor(
                env,
                java.nio.FloatBuffer.wrap(inputFloats),
                shapeLong
            )

            var maskResult: Bitmap? = null

            LogUtil.d("FurnitureFitManager", "Running ONNX inference...")
            session.run(mapOf(inputName to tensor)).use { results ->
                LogUtil.d("FurnitureFitManager", "Inference complete, processing outputs...")

                val outInfos = ortSession!!.outputInfo.entries.toList()

                // Find detection output (3D) and prototype output (4D)
                var detIndex = -1
                var protoIndex = -1
                var detShape: LongArray? = null
                var protoShape: LongArray? = null

                for (i in outInfos.indices) {
                    val info = outInfos[i].value.info
                    if (info is ai.onnxruntime.TensorInfo) {
                        val sh = info.shape
                        LogUtil.d("FurnitureFitManager", "Output $i shape: ${sh.toList()}")
                        if (sh.size == 3 && detIndex == -1) {
                            detIndex = i
                            detShape = sh
                        }
                        if (sh.size == 4 && protoIndex == -1) {
                            protoIndex = i
                            protoShape = sh
                        }
                    }
                }

                if (detIndex == -1 || protoIndex == -1 || detShape == null || protoShape == null) {
                    LogUtil.w("FurnitureFitManager", "Could not find detection/prototype outputs")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d("FurnitureFitManager", "Detection output[$detIndex] shape: ${detShape.toList()}")
                LogUtil.d("FurnitureFitManager", "Proto output[$protoIndex] shape: ${protoShape.toList()}")

                val detResult = results.get(detIndex)
                val protoResult = results.get(protoIndex)

                val detValue = detResult?.value
                val protoValue = protoResult?.value

                val numFeatures = detShape[1].toInt()
                val numAnchors = detShape[2].toInt()

                val numMaskCoeffs = 32

                // Your comment says prompt-free (embeddings + mask coeffs), but below you treat as standard YOLO format.
                // We'll keep your logic as-is, but the critical fix is: input size must match model export (768 vs 640).
                val numClasses = numFeatures - 4 - numMaskCoeffs
                val classStartIdx = 4
                val maskCoeffStartIdx = 4 + numClasses

                val numProtos = protoShape[1].toInt()
                val protoH = protoShape[2].toInt()
                val protoW = protoShape[3].toInt()

                LogUtil.d(
                    "FurnitureFitManager",
                    "Features=$numFeatures Anchors=$numAnchors Classes=$numClasses MaskCoeffs=$numMaskCoeffs Protos=$numProtos ProtoSize=${protoW}x${protoH}"
                )

                if (numFeatures < (4 + numMaskCoeffs + 1) || numAnchors <= 0 || numProtos <= 0) {
                    LogUtil.e("FurnitureFitManager", "Invalid tensor dimensions")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d("FurnitureFitManager", "DetValue type: ${detValue?.javaClass}")

                // Extract detection tensor (3D preferred)
                var det3d: Array<Array<FloatArray>>? = null
                var detFlat: FloatArray? = null

                when (detValue) {
                    is Array<*> -> {
                        try {
                            @Suppress("UNCHECKED_CAST")
                            det3d = detValue as Array<Array<FloatArray>>
                            LogUtil.d(
                                "FurnitureFitManager",
                                "Det as 3D array: [${det3d.size}][${det3d[0].size}][${det3d[0][0].size}]"
                            )
                        } catch (e: Exception) {
                            LogUtil.w(
                                "FurnitureFitManager",
                                "Failed to cast as 3D array, trying flatten: ${e.message}"
                            )
                            detFlat = extractFloatArray(detValue)
                        }
                    }
                    is FloatArray -> {
                        detFlat = detValue
                        LogUtil.d("FurnitureFitManager", "Det as flat FloatArray: ${detFlat.size}")
                    }
                    is java.nio.FloatBuffer -> {
                        detFlat = FloatArray(detValue.remaining())
                        detValue.get(detFlat)
                        LogUtil.d("FurnitureFitManager", "Det as FloatBuffer: ${detFlat.size}")
                    }
                    else -> {
                        LogUtil.w("FurnitureFitManager", "Unknown det type: ${detValue?.javaClass}")
                        detFlat = extractFloatArray(detValue)
                    }
                }

                if (det3d == null && (detFlat == null || detFlat.isEmpty())) {
                    LogUtil.e("FurnitureFitManager", "Could not extract detection tensor")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                // Extract prototype tensor
                LogUtil.d("FurnitureFitManager", "Extracting proto array...")
                val proto = extractFloatArray(protoValue)
                LogUtil.d("FurnitureFitManager", "Proto extracted: ${proto.size} floats")

                if (proto.isEmpty()) {
                    LogUtil.w("FurnitureFitManager", "Empty proto output")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                LogUtil.d(
                    "FurnitureFitManager",
                    "Proto[0]=${proto[0]}, Proto[1]=${proto[1]}, Proto[160]=${proto.getOrNull(160)}, Proto[25600]=${proto.getOrNull(25600)}"
                )

                val stride = numAnchors

                val confThreshold = 0.25f
                val iouThreshold = 0.5f
                val maxDetections = 100

                val detections = mutableListOf<Detection>()

                val det3dArr = det3d
                val getDetValue: (Int, Int) -> Float = if (det3dArr != null) {
                    { feature, anchor -> det3dArr[0][feature][anchor] }
                } else {
                    { feature, anchor -> detFlat!![feature * stride + anchor] }
                }

                // Debug: find global max class score
                var globalMaxScore = Float.MIN_VALUE
                var globalMaxAnchor = -1
                var globalMaxClass = -1
                for (anchor in 0 until numAnchors step 10) {
                    for (c in 0 until numClasses) {
                        val score = getDetValue(classStartIdx + c, anchor)
                        if (score > globalMaxScore) {
                            globalMaxScore = score
                            globalMaxAnchor = anchor
                            globalMaxClass = c
                        }
                    }
                }
                LogUtil.d(
                    "FurnitureFitManager",
                    "Global max class score: $globalMaxScore at anchor $globalMaxAnchor, class $globalMaxClass"
                )

                val dbgAnchor = 100
                val dbgX = getDetValue(0, dbgAnchor)
                val dbgY = getDetValue(1, dbgAnchor)
                val dbgW = getDetValue(2, dbgAnchor)
                val dbgH = getDetValue(3, dbgAnchor)
                val dbgC0 = getDetValue(4, dbgAnchor)
                val dbgC1 = getDetValue(5, dbgAnchor)
                LogUtil.d(
                    "FurnitureFitManager",
                    "Anchor[$dbgAnchor]: bbox=($dbgX,$dbgY,$dbgW,$dbgH), class0=$dbgC0, class1=$dbgC1"
                )

                LogUtil.d(
                    "FurnitureFitManager",
                    "Scanning $numAnchors anchors with conf threshold $confThreshold..."
                )
                val startTime = System.currentTimeMillis()

                for (anchor in 0 until numAnchors) {
                    var maxClassScore = Float.MIN_VALUE
                    var bestClassId = -1
                    for (c in 0 until numClasses) {
                        val score = getDetValue(classStartIdx + c, anchor)
                        if (score > maxClassScore) {
                            maxClassScore = score
                            bestClassId = c
                        }
                    }

                    if (maxClassScore > confThreshold) {
                        val x = getDetValue(0, anchor)
                        val y = getDetValue(1, anchor)
                        val bw = getDetValue(2, anchor)
                        val bh = getDetValue(3, anchor)

                        if (x.isFinite() && y.isFinite() && bw.isFinite() && bh.isFinite() && bw > 0 && bh > 0) {
                            val coeffs = FloatArray(numProtos)
                            for (c in 0 until numProtos) {
                                coeffs[c] = getDetValue(maskCoeffStartIdx + c, anchor)
                            }

                            detections.add(
                                Detection(
                                    anchorIdx = anchor,
                                    x = x, y = y, w = bw, h = bh,
                                    confidence = maxClassScore,
                                    classId = bestClassId,
                                    coeffs = coeffs
                                )
                            )
                        }
                    }
                }

                val scanTime = System.currentTimeMillis() - startTime
                LogUtil.d(
                    "FurnitureFitManager",
                    "Found ${detections.size} detections above conf $confThreshold in ${scanTime}ms"
                )

                if (detections.isEmpty()) {
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                val topDets = detections.sortedByDescending { it.confidence }.take(5)
                LogUtil.d("FurnitureFitManager", "=== TOP DETECTIONS ===")
                for ((idx, det) in topDets.withIndex()) {
                    val label = labelForClassId(det.classId)
                    LogUtil.d("FurnitureFitManager", "  [$idx] $label: conf=${String.format("%.3f", det.confidence)}")
                }
                LogUtil.d("FurnitureFitManager", "======================")

                val sortedDets = detections.sortedByDescending { it.confidence }.take(maxDetections)

                val keepDets = mutableListOf<Detection>()
                val suppressed = BooleanArray(sortedDets.size)

                for (i in sortedDets.indices) {
                    if (suppressed[i]) continue
                    keepDets.add(sortedDets[i])

                    for (j in i + 1 until sortedDets.size) {
                        if (suppressed[j]) continue
                        val iou = calculateIoU(sortedDets[i], sortedDets[j])
                        if (iou > iouThreshold) suppressed[j] = true
                    }
                }

                LogUtil.d("FurnitureFitManager", "After NMS: ${keepDets.size} detections kept")

                if (keepDets.isNotEmpty()) {
                    val firstDet = keepDets[0]
                    LogUtil.d(
                        "FurnitureFitManager",
                        "First det coeffs[0..3]: ${firstDet.coeffs[0]}, ${firstDet.coeffs[1]}, ${firstDet.coeffs[2]}, ${firstDet.coeffs[3]}"
                    )
                }

                // --- FIX #2: protoScale must use actual model inputW/inputH (768 vs 640) ---
                val protoScaleX = inputW.toFloat() / protoW.toFloat()
                val protoScaleY = inputH.toFloat() / protoH.toFloat()

                val maskProto = FloatArray(protoH * protoW)

                // Generate combined mask from all detections
                // NOTE: Your bbox values might already be in input pixel coords OR model head coords.
                // The biggest real-world cause of "green zigzag" here was resizing to 640 when the model is 768.
                for (detection in keepDets) {
                    // Convert bbox from input coords to proto coords
                    val bboxLeft = ((detection.x - detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                    val bboxTop = ((detection.y - detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)
                    val bboxRight = ((detection.x + detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                    val bboxBottom = ((detection.y + detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)

                    // Only compute mask within bbox region
                    for (py in bboxTop..bboxBottom) {
                        val rowBase = py * protoW
                        for (px in bboxLeft..bboxRight) {
                            var sum = 0f
                            val p = rowBase + px
                            val hwProto = protoH * protoW
                            var c = 0
                            while (c < numProtos) {
                                val protoIdx = c * hwProto + p
                                sum += detection.coeffs[c] * proto[protoIdx]
                                c++
                            }
                            val sigmoidVal = 1f / (1f + exp(-sum))
                            if (sigmoidVal > maskProto[p]) {
                                maskProto[p] = sigmoidVal
                            }
                        }
                    }
                }

                // Debug mask values
                val maskMin = maskProto.minOrNull() ?: 0f
                val maskMax = maskProto.maxOrNull() ?: 0f
                val maskAbove05 = maskProto.count { it > 0.5f }
                LogUtil.d("FurnitureFitManager", "Mask stats: min=$maskMin, max=$maskMax, pixels>0.5=$maskAbove05")

                // Create mask bitmap from computed maskProto values
                val maskBmp = Bitmap.createBitmap(protoW, protoH, Config.ARGB_8888)
                val pixels = IntArray(protoW * protoH)
                for (i in pixels.indices) {
                    val v = maskProto[i]
                    // Semi-transparent green where mask > 0.5
                    val alpha = if (v > 0.5f) 0xCC else 0x00
                    pixels[i] = (alpha shl 24) or 0x00FF00
                }
                maskBmp.setPixels(pixels, 0, protoW, 0, 0, protoW, protoH)

                // Scale mask to original frame size
                val outMask = Bitmap.createScaledBitmap(maskBmp, frame.width, frame.height, true)
                maskResult = outMask
                LogUtil.d("FurnitureFitManager", "Mask generated: ${outMask.width}x${outMask.height}")
            }

            tensor.close()
            val finalMask = maskResult
            mainHandler.post { callback(finalMask) }
        } catch (e: OrtException) {
            LogUtil.e("FurnitureFitManager", "ONNX inference failed", e)
            mainHandler.post { callback(null) }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "ONNX inference exception", e)
            e.printStackTrace()
            mainHandler.post { callback(null) }
        }
    }

    private fun runOnnxInferenceWithDetections(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>?,
        callback: (SegmentationResult?) -> Unit,
    ) {
        try {
            val base = runOnnxSegmentationOnce(frame, includeMask, selectedClassIds, pinnedDetections) ?: run {
                mainHandler.post { callback(null) }
                return
            }
            mainHandler.post { callback(base) }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "ONNX inference with detections failed", e)
            mainHandler.post { callback(null) }
        }
    }

    /** Single ONNX forward + mask; no main-thread hop. */
    private fun runOnnxSegmentationOnce(
        frame: Bitmap,
        includeMask: Boolean,
        selectedClassIds: Set<Int>,
        pinnedDetections: List<DetectionResult>? = null,
    ): SegmentationResult? {
        var tensor: OnnxTensor? = null
        return try {
            val totalStartNanos = System.nanoTime()
            val session = ortSession ?: return null
            val env = ortEnv ?: return null

            val firstInput = session.inputInfo.entries.firstOrNull() ?: return null

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info

            var inputH = 640
            var inputW = 640
            if (tensorInfo is ai.onnxruntime.TensorInfo) {
                val sh = tensorInfo.shape
                if (sh.size == 4) {
                    if (sh[2] > 0) inputH = sh[2].toInt()
                    if (sh[3] > 0) inputW = sh[3].toInt()
                }
            }
            val usesLetterboxPreprocess = inputH >= 1280 || inputW >= 1280
            LogUtil.i(
                TAG,
                "YOLOE 11L ONNX start: ${frame.width}x${frame.height} -> ${if (usesLetterboxPreprocess) "letterbox" else "stretch"} ${inputW}x${inputH}"
            )

            val preprocessStartNanos = System.nanoTime()
            val preparedBitmap = preprocessFrameForModel(frame, inputW, inputH, usesLetterboxPreprocess)
            val intValues = IntArray(inputW * inputH)
            preparedBitmap.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)
            val inputFloats = FloatArray(3 * inputH * inputW)
            val hw = inputH * inputW
            for (y in 0 until inputH) {
                val rowOff = y * inputW
                for (x in 0 until inputW) {
                    val v = intValues[rowOff + x]
                    val pixelIdx = rowOff + x
                    inputFloats[0 * hw + pixelIdx] = ((v shr 16) and 0xFF) / 255.0f
                    inputFloats[1 * hw + pixelIdx] = ((v shr 8) and 0xFF) / 255.0f
                    inputFloats[2 * hw + pixelIdx] = (v and 0xFF) / 255.0f
                }
            }
            LogUtil.i(TAG, "YOLOE 11L preprocess: ${elapsedMillis(preprocessStartNanos)}ms")

            val shapeLong = longArrayOf(1, 3, inputH.toLong(), inputW.toLong())
            tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inputFloats), shapeLong)

            val inferenceStartNanos = System.nanoTime()
            session.run(mapOf(inputName to tensor)).use { results ->
                LogUtil.i(TAG, "YOLOE 11L inference: ${elapsedMillis(inferenceStartNanos)}ms")

                val outputSelectStartNanos = System.nanoTime()
                val outputs = selectDetectionProtoOutputs(session, results) ?: run {
                    LogUtil.e(TAG, "Could not find detection/prototype outputs in YOLOE 11L session")
                    return null
                }
                LogUtil.i(
                    TAG,
                    "YOLOE 11L outputs: det=${outputs.detectionName}${outputs.detectionShape.contentToString()} proto=${outputs.protoName}${outputs.protoShape.contentToString()} in ${elapsedMillis(outputSelectStartNanos)}ms"
                )

                val detValue = outputs.detectionValue
                val protoValue = outputs.protoValue
                val numMaskCoeffs = outputs.protoShape[1].toInt()

                val parseStartNanos = System.nanoTime()
                val proto = extractFloatArray(protoValue)
                val protoH = outputs.protoShape[2].toInt()
                val protoW = outputs.protoShape[3].toInt()
                val detections = parseDetectionsForCurrentModel(
                    outputs = outputs,
                    detValue = detValue,
                    confidenceThreshold = DEFAULT_CONFIDENCE_THRESHOLD,
                )

                if (detections.isEmpty()) {
                    LogUtil.i(TAG, "YOLOE 11L detections: 0 candidates in ${elapsedMillis(parseStartNanos)}ms")
                    return SegmentationResult(null, emptyList(), inputW, null)
                }

                val sortedDets = detections.sortedByDescending { it.confidence }.take(300)
                val keepDets = mutableListOf<Detection>()
                val suppressed = BooleanArray(sortedDets.size)
                for (i in sortedDets.indices) {
                    if (suppressed[i]) continue
                    keepDets.add(sortedDets[i])
                    for (j in i + 1 until sortedDets.size) {
                        if (suppressed[j]) continue
                        if (sortedDets[i].classId != sortedDets[j].classId) continue
                        val iou = calculateIoU(sortedDets[i], sortedDets[j])
                        if (iou > DEFAULT_NMS_IOU_THRESHOLD) suppressed[j] = true
                    }
                }
                LogUtil.i(
                    TAG,
                    "YOLOE 11L detections: raw=${detections.size} kept=${keepDets.size} parse+nms=${elapsedMillis(parseStartNanos)}ms"
                )

                val pinList = pinnedDetections.orEmpty()
                val restrictToSelection =
                    selectedClassIds.isNotEmpty() || pinList.isNotEmpty()
                val primaryCandidates = when {
                    pinList.isNotEmpty() -> {
                        val iouPinThreshold = 0.45f
                        keepDets.filter { det ->
                            pinList.any { pin ->
                                det.classId == pin.classId && calculateIoU(det, pin) >= iouPinThreshold
                            }
                        }
                    }
                    selectedClassIds.isEmpty() -> keepDets
                    else -> keepDets.filter { it.classId in selectedClassIds }
                }
                val primaryDet = pickPrimaryOnnxDetection(
                    detections = primaryCandidates,
                    frameWidth = inputW.toFloat(),
                    frameHeight = inputH.toFloat(),
                )
                val maskSourceDetections = if (!restrictToSelection) {
                    if (primaryDet != null) {
                        collectMaskDetections(primaryDet, keepDets)
                    } else {
                        emptyList()
                    }
                } else {
                    primaryCandidates
                }
                val maskDetectionsForBuild = if (restrictToSelection) {
                    maskSourceDetections.map { detection ->
                        expandedPrimaryForMaskBuild(
                            primaryDetection = detection,
                            frameWidth = inputW.toFloat(),
                            frameHeight = inputH.toFloat(),
                        )
                    }
                } else if (primaryDet != null) {
                    val expandedPrimary = expandedPrimaryForMaskBuild(
                        primaryDetection = primaryDet,
                        frameWidth = inputW.toFloat(),
                        frameHeight = inputH.toFloat(),
                    )
                    buildList {
                        add(expandedPrimary)
                        for (detection in maskSourceDetections) {
                            if (calculateIoUForMaskSelection(detection, primaryDet) < 0.999f) {
                                add(detection)
                            }
                        }
                    }
                } else {
                    emptyList()
                }

                val orderedDisplayDetections = if (primaryDet != null) {
                    buildList {
                        add(primaryDet)
                        for (detection in keepDets) {
                            if (detection.anchorIdx != primaryDet.anchorIdx) {
                                add(detection)
                            }
                        }
                    }
                } else {
                    keepDets
                }
                val detectionResults = orderedDisplayDetections
                    .take(DEFAULT_MAX_DETECTIONS)
                    .map { detection ->
                        DetectionResult(
                            x = detection.x,
                            y = detection.y,
                            w = detection.w,
                            h = detection.h,
                            confidence = detection.confidence,
                            label = labelForClassId(detection.classId),
                            classId = detection.classId,
                        )
                    }
                if (primaryDet != null) {
                    val topLabels = keepDets
                        .take(3)
                        .joinToString(", ") { "${labelForClassId(it.classId)}:${String.format("%.2f", it.confidence)}" }
                    LogUtil.d(
                        TAG,
                        "Primary=${labelForClassId(primaryDet.classId)} conf=${String.format("%.2f", primaryDet.confidence)} " +
                            "maskBuildDets=${maskDetectionsForBuild.size} keepDets=${keepDets.size} top=[$topLabels]",
                    )
                }

                if (!includeMask) {
                    LogUtil.i(TAG, "YOLOE 11L total (detections only): ${elapsedMillis(totalStartNanos)}ms")
                    return SegmentationResult(
                        mask = null,
                        detections = detectionResults,
                        inputSize = inputW,
                        primaryDetection = detectionResults.firstOrNull(),
                    )
                }

                var maskResult: Bitmap? = null
                if (primaryDet != null && proto.isNotEmpty()) {
                    val maskBuildStartNanos = System.nanoTime()
                    val protoScaleX = inputW.toFloat() / protoW.toFloat()
                    val protoScaleY = inputH.toFloat() / protoH.toFloat()
                    val maskProto = FloatArray(protoH * protoW)

                    for (detection in maskDetectionsForBuild) {
                        val bboxLeft = ((detection.x - detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                        val bboxTop = ((detection.y - detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)
                        val bboxRight = ((detection.x + detection.w / 2f) / protoScaleX).toInt().coerceIn(0, protoW - 1)
                        val bboxBottom = ((detection.y + detection.h / 2f) / protoScaleY).toInt().coerceIn(0, protoH - 1)

                        for (py in bboxTop..bboxBottom) {
                            val rowBase = py * protoW
                            for (px in bboxLeft..bboxRight) {
                                var sum = 0f
                                val p = rowBase + px
                                val hwProto = protoH * protoW
                                var coeffIndex = 0
                                while (coeffIndex < numMaskCoeffs) {
                                    val protoIdx = coeffIndex * hwProto + p
                                    sum += detection.coeffs[coeffIndex] * proto[protoIdx]
                                    coeffIndex++
                                }
                                val sigmoidVal = 1f / (1f + exp(-sum))
                                if (sigmoidVal > maskProto[p]) {
                                    maskProto[p] = sigmoidVal
                                }
                            }
                        }
                    }

                    if (enableMorphCloseForMask) {
                        applyMorphClose3x3ToFloatMask(
                            mask = maskProto,
                            width = protoW,
                            height = protoH,
                            threshold = 0.5f,
                        )
                    }
                    val clipCandidates = if (!restrictToSelection) maskDetectionsForBuild else primaryCandidates
                    val clipLeftModel = clipCandidates.minOfOrNull { it.x - it.w / 2f } ?: (primaryDet.x - primaryDet.w / 2f)
                    val clipTopModel = clipCandidates.minOfOrNull { it.y - it.h / 2f } ?: (primaryDet.y - primaryDet.h / 2f)
                    val clipRightModel = clipCandidates.maxOfOrNull { it.x + it.w / 2f } ?: (primaryDet.x + primaryDet.w / 2f)
                    val clipBottomModel = clipCandidates.maxOfOrNull { it.y + it.h / 2f } ?: (primaryDet.y + primaryDet.h / 2f)
                    val protoClipLeft = floor((clipLeftModel / protoScaleX).toDouble()).toInt().coerceIn(0, protoW)
                    val protoClipTop = floor((clipTopModel / protoScaleY).toDouble()).toInt().coerceIn(0, protoH)
                    val protoClipRight = ceil((clipRightModel / protoScaleX).toDouble()).toInt().coerceIn(0, protoW)
                    val protoClipBottom = ceil((clipBottomModel / protoScaleY).toDouble()).toInt().coerceIn(0, protoH)
                    clipProtoMaskOutsideRect(
                        mask = maskProto,
                        protoW = protoW,
                        protoH = protoH,
                        clipX0 = protoClipLeft,
                        clipY0 = protoClipTop,
                        clipX1 = protoClipRight,
                        clipY1 = protoClipBottom,
                    )

                    val frameW = frame.width
                    val frameH = frame.height
                    val sxf = frameW.toFloat() / inputW.toFloat()
                    val syf = frameH.toFloat() / inputH.toFloat()
                    val tightFx0 = clipLeftModel * sxf
                    val tightFx1 = clipRightModel * sxf
                    val tightFy0 = clipTopModel * syf
                    val tightFy1 = clipBottomModel * syf
                    val bandMarginW = max(1f, tightFx1 - tightFx0) * bboxExpandMargin
                    val bandMarginH = max(1f, tightFy1 - tightFy0) * bboxExpandMargin
                    val bandX0 = floor((tightFx0 - bandMarginW).toDouble()).toInt().coerceIn(0, frameW)
                    val bandX1 = ceil((tightFx1 + bandMarginW).toDouble()).toInt().coerceIn(0, frameW)
                    val bandY0 = floor((tightFy0 - bandMarginH).toDouble()).toInt().coerceIn(0, frameH)
                    val bandY1 = ceil((tightFy1 + bandMarginH).toDouble()).toInt().coerceIn(0, frameH)

                    val framePixels = IntArray(frameW * frameH)
                    frame.getPixels(framePixels, 0, frameW, 0, 0, frameW, frameH)

                    val outPixels = IntArray(frameW * frameH)
                    composeNearestProtoMaskCutoutArgb(
                        framePixels = framePixels,
                        outPixels = outPixels,
                        maskProto = maskProto,
                        frameW = frameW,
                        frameH = frameH,
                        protoW = protoW,
                        protoH = protoH,
                        x0 = bandX0,
                        x1 = bandX1,
                        y0 = bandY0,
                        y1 = bandY1,
                    )

                    val maskBmp = Bitmap.createBitmap(frameW, frameH, Config.ARGB_8888)
                    maskBmp.setPixels(outPixels, 0, frameW, 0, 0, frameW, frameH)
                    maskResult = maskBmp
                    LogUtil.i(TAG, "YOLOE 11L mask build: ${elapsedMillis(maskBuildStartNanos)}ms")
                }

                LogUtil.i(TAG, "YOLOE 11L total: ${elapsedMillis(totalStartNanos)}ms")
                SegmentationResult(maskResult, detectionResults, inputW, detectionResults.firstOrNull())
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "ONNX segmentation once failed", e)
            null
        } finally {
            tensor?.close()
        }
    }

    private data class DetectionProtoOutputs(
        val detectionName: String,
        val protoName: String,
        val detectionValue: Any?,
        val protoValue: Any?,
        val detectionShape: LongArray,
        val protoShape: LongArray,
    )

    private fun elapsedMillis(startNanos: Long): Long =
        (System.nanoTime() - startNanos) / 1_000_000L

    private fun preprocessFrameForModel(
        frame: Bitmap,
        inputW: Int,
        inputH: Int,
        usesLetterbox: Boolean,
    ): Bitmap {
        if (!usesLetterbox) {
            return Bitmap.createScaledBitmap(frame, inputW, inputH, true)
                .copy(Config.ARGB_8888, false)
        }

        val scale = min(inputW.toFloat() / frame.width.toFloat(), inputH.toFloat() / frame.height.toFloat())
        val scaledW = max(1, (frame.width * scale).toInt())
        val scaledH = max(1, (frame.height * scale).toInt())
        val scaledBitmap = Bitmap.createScaledBitmap(frame, scaledW, scaledH, true)
        val output = Bitmap.createBitmap(inputW, inputH, Config.ARGB_8888)
        val canvas = Canvas(output)
        canvas.drawColor(Color.rgb(114, 114, 114))
        val left = (inputW - scaledW) * 0.5f
        val top = (inputH - scaledH) * 0.5f
        canvas.drawBitmap(scaledBitmap, left, top, Paint(Paint.FILTER_BITMAP_FLAG))
        if (scaledBitmap !== frame && !scaledBitmap.isRecycled) {
            scaledBitmap.recycle()
        }
        return output
    }

    private fun selectDetectionProtoOutputs(
        session: OrtSession,
        results: OrtSession.Result,
    ): DetectionProtoOutputs? {
        val outputEntries = session.outputInfo.entries.toList()
        val knownPairs = listOf(
            "detections" to "protos",
            "output0" to "output1",
            "output" to "proto",
        )
        for ((detName, protoName) in knownPairs) {
            val detIndex = outputEntries.indexOfFirst { it.key == detName }
            val protoIndex = outputEntries.indexOfFirst { it.key == protoName }
            if (detIndex == -1 || protoIndex == -1) continue
            val detectionValue = results.get(detIndex)?.value
            val protoValue = results.get(protoIndex)?.value
            val detectionInfo = session.outputInfo[detName]?.info as? ai.onnxruntime.TensorInfo
            val protoInfo = session.outputInfo[protoName]?.info as? ai.onnxruntime.TensorInfo
            if (detectionValue != null && protoValue != null && detectionInfo != null && protoInfo != null) {
                return DetectionProtoOutputs(
                    detectionName = detName,
                    protoName = protoName,
                    detectionValue = detectionValue,
                    protoValue = protoValue,
                    detectionShape = detectionInfo.shape,
                    protoShape = protoInfo.shape,
                )
            }
        }

        var detectionCandidate: Pair<String, ai.onnxruntime.TensorInfo>? = null
        var protoCandidate: Pair<String, ai.onnxruntime.TensorInfo>? = null
        for ((name, nodeInfo) in session.outputInfo) {
            val tensorInfo = nodeInfo.info as? ai.onnxruntime.TensorInfo ?: continue
            val shape = tensorInfo.shape
            if (shape.size == 4 && shape.getOrNull(1) == 32L && protoCandidate == null) {
                protoCandidate = name to tensorInfo
            } else if (shape.size == 3 && detectionCandidate == null) {
                detectionCandidate = name to tensorInfo
            }
        }
        val det = detectionCandidate ?: return null
        val proto = protoCandidate ?: return null
        val detIndex = outputEntries.indexOfFirst { it.key == det.first }
        val protoIndex = outputEntries.indexOfFirst { it.key == proto.first }
        if (detIndex == -1 || protoIndex == -1) return null
        return DetectionProtoOutputs(
            detectionName = det.first,
            protoName = proto.first,
            detectionValue = results.get(detIndex)?.value,
            protoValue = results.get(protoIndex)?.value,
            detectionShape = det.second.shape,
            protoShape = proto.second.shape,
        )
    }

    private data class PrimaryCandidateScore(
        val score: Float,
        val isInteriorCandidate: Boolean,
    )

    private fun primaryDetectionScore(
        centerX: Float,
        centerY: Float,
        width: Float,
        height: Float,
        confidence: Float,
        frameWidth: Float,
        frameHeight: Float,
    ): PrimaryCandidateScore {
        if (!centerX.isFinite() || !centerY.isFinite() || !width.isFinite() || !height.isFinite() || !confidence.isFinite()) {
            return PrimaryCandidateScore(-1f, false)
        }
        if (frameWidth <= 1f || frameHeight <= 1f || width <= 0f || height <= 0f) {
            return PrimaryCandidateScore(-1f, false)
        }

        val frameArea = frameWidth * frameHeight
        val areaNormalized = (width * height) / frameArea
        val minimumConfidence = 0.15f
        val minimumAreaNormalized = 0.02f
        if (confidence < minimumConfidence || areaNormalized < minimumAreaNormalized) {
            return PrimaryCandidateScore(-1f, false)
        }

        val frameCenterX = frameWidth * 0.5f
        val frameCenterY = frameHeight * 0.5f
        val deltaX = (centerX - frameCenterX) / frameCenterX.coerceAtLeast(1f)
        val deltaY = (centerY - frameCenterY) / frameCenterY.coerceAtLeast(1f)
        val centerDistance = min(1f, sqrt(deltaX * deltaX + deltaY * deltaY))
        val centerScore = 1f - centerDistance

        val boxLeft = centerX - width * 0.5f
        val boxTop = centerY - height * 0.5f
        val boxRight = centerX + width * 0.5f
        val boxBottom = centerY + height * 0.5f
        val edgeMarginX = max(frameWidth * 0.04f, 1f)
        val edgeMarginY = max(frameHeight * 0.04f, 1f)
        val leftClearance = (boxLeft / edgeMarginX).coerceIn(0f, 1f)
        val topClearance = (boxTop / edgeMarginY).coerceIn(0f, 1f)
        val rightClearance = ((frameWidth - boxRight) / edgeMarginX).coerceIn(0f, 1f)
        val bottomClearance = ((frameHeight - boxBottom) / edgeMarginY).coerceIn(0f, 1f)
        val edgeClearanceScore = max(0.1f, min(min(leftClearance, topClearance), min(rightClearance, bottomClearance)))
        val isInteriorCandidate =
            leftClearance >= 1f &&
                topClearance >= 1f &&
                rightClearance >= 1f &&
                bottomClearance >= 1f

        val confidenceTerm = confidence.pow(1.0f)
        val areaTerm = areaNormalized.pow(0.8f)
        val centerTerm = max(0f, centerScore).pow(1.0f)
        val edgeTerm = edgeClearanceScore.pow(1.0f)
        return PrimaryCandidateScore(
            score = confidenceTerm * areaTerm * centerTerm * edgeTerm,
            isInteriorCandidate = isInteriorCandidate,
        )
    }

    private fun pickPrimaryOnnxDetection(
        detections: List<Detection>,
        frameWidth: Float,
        frameHeight: Float,
    ): Detection? {
        if (detections.isEmpty()) return null

        var bestDetection: Detection? = null
        var bestScore = -1f
        var bestEdgeFallback: Detection? = null
        var bestEdgeFallbackScore = -1f
        for (detection in detections) {
            val candidateScore = primaryDetectionScore(
                centerX = detection.x,
                centerY = detection.y,
                width = detection.w,
                height = detection.h,
                confidence = detection.confidence,
                frameWidth = frameWidth,
                frameHeight = frameHeight,
            )
            if (candidateScore.isInteriorCandidate && candidateScore.score > bestScore) {
                bestScore = candidateScore.score
                bestDetection = detection
            } else if (!candidateScore.isInteriorCandidate && candidateScore.score > bestEdgeFallbackScore) {
                bestEdgeFallbackScore = candidateScore.score
                bestEdgeFallback = detection
            }
        }
        return bestDetection ?: bestEdgeFallback ?: detections.maxByOrNull { it.confidence }
    }

    private fun pickSupportingTableForMonitorScene(
        primaryDetection: Detection,
        detections: List<Detection>,
    ): Detection? {
        if (!includeSupportingTableForMonitorScene) return null
        if (!monitorLikeClassIds.contains(primaryDetection.classId)) return null

        val primaryLeft = primaryDetection.x - primaryDetection.w / 2f
        val primaryRight = primaryDetection.x + primaryDetection.w / 2f
        val primaryBottom = primaryDetection.y + primaryDetection.h / 2f
        val primaryArea = max(1e-3f, primaryDetection.w * primaryDetection.h)

        var bestDetection: Detection? = null
        var bestScore = -1f

        for (detection in detections) {
            if (detection === primaryDetection) continue
            if (!supportingTableClassIds.contains(detection.classId)) continue

            val candidateLeft = detection.x - detection.w / 2f
            val candidateRight = detection.x + detection.w / 2f
            val candidateTop = detection.y - detection.h / 2f
            val overlapWidth = max(0f, min(primaryRight, candidateRight) - max(primaryLeft, candidateLeft))
            val horizontalOverlapRatio = overlapWidth / max(1e-3f, min(primaryDetection.w, detection.w))
            if (horizontalOverlapRatio < 0.35f) continue

            if (detection.y <= primaryDetection.y) continue

            val verticalGap = candidateTop - primaryBottom
            if (verticalGap < -primaryDetection.h * 0.20f || verticalGap > primaryDetection.h * 0.60f) continue

            val widthRatio = detection.w / max(1e-3f, primaryDetection.w)
            if (widthRatio < 0.75f || widthRatio > 5.0f) continue

            val areaRatio = (detection.w * detection.h) / primaryArea
            if (areaRatio < 0.50f || areaRatio > 12.0f) continue

            val closenessTerm = 1f - min(1f, kotlin.math.abs(verticalGap) / max(primaryDetection.h * 0.60f, 1e-3f))
            val score = detection.confidence * horizontalOverlapRatio * max(0.1f, closenessTerm)

            if (score > bestScore) {
                bestScore = score
                bestDetection = detection
            }
        }

        if (bestDetection != null) {
            LogUtil.d(
                "FurnitureFitManager",
                "Support table picked for monitor scene: class=${bestDetection.classId} conf=${bestDetection.confidence}",
            )
        }

        return bestDetection
    }

    private fun calculateIoUForMaskSelection(first: Detection, second: Detection): Float {
        val firstX1 = first.x - first.w / 2f
        val firstY1 = first.y - first.h / 2f
        val firstX2 = first.x + first.w / 2f
        val firstY2 = first.y + first.h / 2f

        val secondX1 = second.x - second.w / 2f
        val secondY1 = second.y - second.h / 2f
        val secondX2 = second.x + second.w / 2f
        val secondY2 = second.y + second.h / 2f

        val interX1 = max(firstX1, secondX1)
        val interY1 = max(firstY1, secondY1)
        val interX2 = min(firstX2, secondX2)
        val interY2 = min(firstY2, secondY2)
        val interW = max(0f, interX2 - interX1)
        val interH = max(0f, interY2 - interY1)
        val interArea = interW * interH
        val unionArea = first.w * first.h + second.w * second.h - interArea
        return if (unionArea > 0f) interArea / unionArea else 0f
    }

    private fun collectMaskDetections(
        primaryDetection: Detection,
        detections: List<Detection>,
    ): List<Detection> {
        val supportingTableDetection = pickSupportingTableForMonitorScene(primaryDetection, detections)
        val primaryLeft = primaryDetection.x - primaryDetection.w / 2f
        val primaryTop = primaryDetection.y - primaryDetection.h / 2f
        val primaryRight = primaryDetection.x + primaryDetection.w / 2f
        val primaryBottom = primaryDetection.y + primaryDetection.h / 2f
        val encompassTolerance = 2f
        // Fusion-only confidence floor (redundant with parse threshold); kept for reference.
        // val minimumCandidateConfidence = 0.1f
        val bboxDuplicateThreshold = 0.7f

        val bboxKept = mutableListOf<Detection>()
        for (detection in detections) {
            if (detection == primaryDetection) continue

            val candidateLeft = detection.x - detection.w / 2f
            val candidateTop = detection.y - detection.h / 2f
            val candidateRight = detection.x + detection.w / 2f
            val candidateBottom = detection.y + detection.h / 2f

            val encompassesPrimary =
                candidateLeft <= primaryLeft + encompassTolerance &&
                    candidateTop <= primaryTop + encompassTolerance &&
                    candidateRight >= primaryRight - encompassTolerance &&
                    candidateBottom >= primaryBottom - encompassTolerance
            if (encompassesPrimary) continue

            val intersectsPrimary =
                !(candidateRight < primaryLeft || candidateLeft > primaryRight || candidateBottom < primaryTop || candidateTop > primaryBottom)
            if (!intersectsPrimary) continue

            val tooLarge =
                detection.w > primaryDetection.w * 1.5f &&
                    detection.h > primaryDetection.h * 1.5f
            if (tooLarge) continue

            if (calculateIoUForMaskSelection(detection, primaryDetection) > bboxDuplicateThreshold) continue

            var shouldSkip = false
            var replaceIndex = -1
            for ((index, keptDetection) in bboxKept.withIndex()) {
                val iou = calculateIoUForMaskSelection(detection, keptDetection)
                if (iou > bboxDuplicateThreshold) {
                    if (detection.confidence > keptDetection.confidence) {
                        replaceIndex = index
                    } else {
                        shouldSkip = true
                    }
                    break
                }
            }
            if (shouldSkip) continue
            if (replaceIndex >= 0) {
                bboxKept[replaceIndex] = detection
            } else {
                bboxKept += detection
            }
        }

        val maskDetections = mutableListOf(primaryDetection)
        maskDetections += bboxKept
        if (supportingTableDetection != null && !maskDetections.contains(supportingTableDetection)) {
            maskDetections += supportingTableDetection
        }
        return maskDetections
    }

    private fun expandedPrimaryForMaskBuild(
        primaryDetection: Detection,
        frameWidth: Float,
        frameHeight: Float,
    ): Detection {
        val maxHalfW = min(primaryDetection.x, frameWidth - primaryDetection.x)
        val maxHalfH = min(primaryDetection.y, frameHeight - primaryDetection.y)
        val capW = 2f * max(maxHalfW, 1f)
        val capH = 2f * max(maxHalfH, 1f)
        val expandedW = min(primaryDetection.w * (1f + 2f * bboxExpandMargin), capW)
        val expandedH = min(primaryDetection.h * (1f + 2f * bboxExpandMargin), capH)
        return primaryDetection.copy(
            w = expandedW,
            h = expandedH,
        )
    }

    private fun clipProtoMaskOutsideRect(
        mask: FloatArray,
        protoW: Int,
        protoH: Int,
        clipX0: Int,
        clipY0: Int,
        clipX1: Int,
        clipY1: Int,
    ) {
        if (protoW <= 0 || protoH <= 0 || mask.size != protoW * protoH) return
        val x0 = clipX0.coerceIn(0, protoW)
        val y0 = clipY0.coerceIn(0, protoH)
        val x1 = clipX1.coerceIn(0, protoW)
        val y1 = clipY1.coerceIn(0, protoH)
        if (x0 >= x1 || y0 >= y1) {
            mask.fill(0f)
            return
        }

        for (y in 0 until protoH) {
            val rowBase = y * protoW
            if (y < y0 || y >= y1) {
                for (x in 0 until protoW) {
                    mask[rowBase + x] = 0f
                }
                continue
            }
            for (x in 0 until x0) {
                mask[rowBase + x] = 0f
            }
            for (x in x1 until protoW) {
                mask[rowBase + x] = 0f
            }
        }
    }

    private fun extractFloatArray(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> flattenArrayToFloat(value)
            is java.nio.FloatBuffer -> {
                val arr = FloatArray(value.remaining())
                value.get(arr)
                arr
            }
            else -> {
                LogUtil.w("FurnitureFitManager", "Unknown output type: ${value?.javaClass}")
                FloatArray(0)
            }
        }
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        val x1Min = det1.x - det1.w / 2
        val y1Min = det1.y - det1.h / 2
        val x1Max = det1.x + det1.w / 2
        val y1Max = det1.y + det1.h / 2

        val x2Min = det2.x - det2.w / 2
        val y2Min = det2.y - det2.h / 2
        val x2Max = det2.x + det2.w / 2
        val y2Max = det2.y + det2.h / 2

        val interXMin = maxOf(x1Min, x2Min)
        val interYMin = maxOf(y1Min, y2Min)
        val interXMax = minOf(x1Max, x2Max)
        val interYMax = minOf(y1Max, y2Max)

        val interWidth = maxOf(0f, interXMax - interXMin)
        val interHeight = maxOf(0f, interYMax - interYMin)
        val interArea = interWidth * interHeight

        val area1 = det1.w * det1.h
        val area2 = det2.w * det2.h
        val unionArea = area1 + area2 - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    private fun calculateIoU(det: Detection, pin: DetectionResult): Float {
        val x1Min = det.x - det.w / 2
        val y1Min = det.y - det.h / 2
        val x1Max = det.x + det.w / 2
        val y1Max = det.y + det.h / 2

        val x2Min = pin.x - pin.w / 2
        val y2Min = pin.y - pin.h / 2
        val x2Max = pin.x + pin.w / 2
        val y2Max = pin.y + pin.h / 2

        val interXMin = maxOf(x1Min, x2Min)
        val interYMin = maxOf(y1Min, y2Min)
        val interXMax = minOf(x1Max, x2Max)
        val interYMax = minOf(y1Max, y2Max)

        val interWidth = maxOf(0f, interXMax - interXMin)
        val interHeight = maxOf(0f, interYMax - interYMin)
        val interArea = interWidth * interHeight

        val area1 = det.w * det.h
        val area2 = pin.w * pin.h
        val unionArea = area1 + area2 - interArea

        return if (unionArea > 0) interArea / unionArea else 0f
    }

    private fun applyMorphClose3x3ToFloatMask(
        mask: FloatArray,
        width: Int,
        height: Int,
        threshold: Float,
    ) {
        if (width <= 0 || height <= 0 || mask.size != width * height) return

        val binaryMask = BooleanArray(mask.size) { idx -> mask[idx] > threshold }
        val dilatedMask = dilate3x3(binaryMask, width, height)
        val closedMask = erode3x3(dilatedMask, width, height)

        for (index in mask.indices) {
            mask[index] = if (closedMask[index]) 1f else 0f
        }
    }

    private fun applyMorphClose3x3ToBitmapMask(frame: Bitmap, mask: Bitmap): Bitmap {
        val width = mask.width
        val height = mask.height
        if (width <= 0 || height <= 0) return mask

        val framePixels = IntArray(width * height)
        val maskPixels = IntArray(width * height)
        frame.getPixels(framePixels, 0, width, 0, 0, width, height)
        mask.getPixels(maskPixels, 0, width, 0, 0, width, height)

        val binaryMask = BooleanArray(maskPixels.size) { idx ->
            ((maskPixels[idx] ushr 24) and 0xFF) > 0
        }
        val dilatedMask = dilate3x3(binaryMask, width, height)
        val closedMask = erode3x3(dilatedMask, width, height)

        val outputPixels = IntArray(width * height)
        for (index in outputPixels.indices) {
            outputPixels[index] = if (closedMask[index]) {
                framePixels[index]
            } else {
                0x00000000
            }
        }

        return Bitmap.createBitmap(outputPixels, width, height, Config.ARGB_8888)
    }

    private fun dilate3x3(mask: BooleanArray, width: Int, height: Int): BooleanArray {
        val outputMask = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var isForeground = false
                val yStart = maxOf(0, y - 1)
                val yEnd = minOf(height - 1, y + 1)
                val xStart = maxOf(0, x - 1)
                val xEnd = minOf(width - 1, x + 1)
                for (kernelY in yStart..yEnd) {
                    val rowOffset = kernelY * width
                    for (kernelX in xStart..xEnd) {
                        if (mask[rowOffset + kernelX]) {
                            isForeground = true
                            break
                        }
                    }
                    if (isForeground) break
                }
                outputMask[y * width + x] = isForeground
            }
        }
        return outputMask
    }

    private fun erode3x3(mask: BooleanArray, width: Int, height: Int): BooleanArray {
        val outputMask = BooleanArray(mask.size)
        for (y in 0 until height) {
            for (x in 0 until width) {
                var isForeground = true
                val yStart = maxOf(0, y - 1)
                val yEnd = minOf(height - 1, y + 1)
                val xStart = maxOf(0, x - 1)
                val xEnd = minOf(width - 1, x + 1)
                for (kernelY in yStart..yEnd) {
                    val rowOffset = kernelY * width
                    for (kernelX in xStart..xEnd) {
                        if (!mask[rowOffset + kernelX]) {
                            isForeground = false
                            break
                        }
                    }
                    if (!isForeground) break
                }
                outputMask[y * width + x] = isForeground
            }
        }
        return outputMask
    }

    // Inner class for detection data
    private data class Detection(
        val anchorIdx: Int,
        val x: Float, val y: Float, val w: Float, val h: Float,
        val confidence: Float,
        val classId: Int,
        val coeffs: FloatArray
    )

    private fun parseDetectionsForCurrentModel(
        outputs: DetectionProtoOutputs,
        detValue: Any?,
        confidenceThreshold: Float,
    ): List<Detection> {
        val dim1 = outputs.detectionShape[1].toInt()
        val dim2 = outputs.detectionShape[2].toInt()
        val numMaskCoeffs = outputs.protoShape[1].toInt()
        val detections = mutableListOf<Detection>()

        val detFlat = extractFloatArray(detValue)
        if (detFlat.isEmpty()) {
            LogUtil.w(TAG, "Detection tensor was empty after extraction")
            return emptyList()
        }

        val isEndToEndFormat = dim2 < 100
        if (isEndToEndFormat) {
            val numDetections = dim1
            val featuresPerDetection = dim2
            LogUtil.i(
                TAG,
                "YOLOE parser layout: end-to-end [1,$numDetections,$featuresPerDetection] with $numMaskCoeffs mask coeffs"
            )

            for (detIndex in 0 until numDetections) {
                val base = detIndex * featuresPerDetection
                if (base + 6 + numMaskCoeffs > detFlat.size) break

                val x1 = detFlat[base + 0]
                val y1 = detFlat[base + 1]
                val x2 = detFlat[base + 2]
                val y2 = detFlat[base + 3]
                val confidence = detFlat[base + 4]
                val classIdxFloat = detFlat[base + 5]
                if (!x1.isFinite() || !y1.isFinite() || !x2.isFinite() || !y2.isFinite() ||
                    !confidence.isFinite() || !classIdxFloat.isFinite()
                ) {
                    continue
                }

                val width = x2 - x1
                val height = y2 - y1
                if (confidence < confidenceThreshold || confidence > 1f || width <= 0f || height <= 0f) {
                    continue
                }

                val classId = classIdxFloat.toInt()
                if (classId < 0 || classId in ignoredClassIds) continue

                val coeffs = FloatArray(numMaskCoeffs)
                var validCoefficients = true
                for (coeffIndex in 0 until numMaskCoeffs) {
                    val value = detFlat[base + 6 + coeffIndex]
                    if (!value.isFinite()) {
                        validCoefficients = false
                        break
                    }
                    coeffs[coeffIndex] = value
                }
                if (!validCoefficients) continue

                detections.add(
                    Detection(
                        anchorIdx = detIndex,
                        x = (x1 + x2) * 0.5f,
                        y = (y1 + y2) * 0.5f,
                        w = width,
                        h = height,
                        confidence = confidence,
                        classId = classId,
                        coeffs = coeffs,
                    )
                )
            }
            return detections
        }

        val det3d = detValue as? Array<Array<FloatArray>>
        val numFeatures = dim1
        val numAnchors = dim2
        val numClasses = numFeatures - 4 - numMaskCoeffs
        if (numFeatures < 4 + numMaskCoeffs + 1 || numAnchors <= 0 || numClasses <= 0) {
            LogUtil.e(
                TAG,
                "Invalid one-to-many detection layout: features=$numFeatures anchors=$numAnchors maskCoeffs=$numMaskCoeffs classes=$numClasses"
            )
            return emptyList()
        }

        LogUtil.i(
            TAG,
            "YOLOE parser layout: one-to-many [1,$numFeatures,$numAnchors] with $numClasses classes and $numMaskCoeffs mask coeffs"
        )
        val getDetValue: (Int, Int) -> Float = if (det3d != null) {
            { feature, anchor -> det3d[0][feature][anchor] }
        } else {
            { feature, anchor -> detFlat[feature * numAnchors + anchor] }
        }
        val classStartIdx = 4
        val maskCoeffStartIdx = 4 + numClasses

        for (anchor in 0 until numAnchors) {
            var maxScore = Float.MIN_VALUE
            var bestClass = -1
            for (classIndex in 0 until numClasses) {
                val score = getDetValue(classStartIdx + classIndex, anchor)
                if (score > maxScore) {
                    maxScore = score
                    bestClass = classIndex
                }
            }
            if (maxScore < confidenceThreshold || bestClass in ignoredClassIds) continue

            val x = getDetValue(0, anchor)
            val y = getDetValue(1, anchor)
            val width = getDetValue(2, anchor)
            val height = getDetValue(3, anchor)
            if (!x.isFinite() || !y.isFinite() || !width.isFinite() || !height.isFinite() || width <= 0f || height <= 0f) {
                continue
            }

            val coeffs = FloatArray(numMaskCoeffs)
            var validCoefficients = true
            for (coeffIndex in 0 until numMaskCoeffs) {
                val value = getDetValue(maskCoeffStartIdx + coeffIndex, anchor)
                if (!value.isFinite()) {
                    validCoefficients = false
                    break
                }
                coeffs[coeffIndex] = value
            }
            if (!validCoefficients) continue

            detections.add(
                Detection(
                    anchorIdx = anchor,
                    x = x,
                    y = y,
                    w = width,
                    h = height,
                    confidence = maxScore,
                    classId = bestClass,
                    coeffs = coeffs,
                )
            )
        }
        return detections
    }

    fun close() {
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
        inferenceExecutor.shutdown()
    }

    private fun flattenArrayToFloat(arr: Array<*>): FloatArray {
        val list = ArrayList<Float>()
        fun rec(a: Any?) {
            when (a) {
                is Float -> list.add(a)
                is Double -> list.add(a.toFloat())
                is Int -> list.add(a.toFloat())
                is FloatArray -> for (v in a) list.add(v)
                is DoubleArray -> for (v in a) list.add(v.toFloat())
                is Array<*> -> for (e in a) rec(e)
                else -> {}
            }
        }
        rec(arr)
        return list.toFloatArray()
    }

    @Throws(IOException::class)
    private fun copyAssetToFile(assetName: String): File {
        val outFile = File(context.cacheDir, assetName)
        context.assets.open(assetName).use { input ->
            java.io.FileOutputStream(outFile).use { output ->
                input.copyTo(output)
            }
        }
        return outFile
    }

    private fun loadClassNames(): Map<Int, String> {
        return try {
            context.assets.open("classes.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                buildMap {
                    json.keys().forEach { key ->
                        val id = key.toIntOrNull() ?: return@forEach
                        val label = json.optString(key).trim()
                        if (label.isNotEmpty()) put(id, label)
                    }
                }
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "loadClassNames failed: ${e.message}")
            emptyMap()
        }
    }

    private fun loadIgnoredClassIds(): Set<Int> {
        return try {
            context.assets.open("blacklist.json").bufferedReader().use { reader ->
                val json = JSONObject(reader.readText())
                buildSet {
                    json.keys().forEach { key ->
                        key.toIntOrNull()?.let { add(it) }
                    }
                }
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "loadIgnoredClassIds failed: ${e.message}")
            emptySet()
        }
    }

    private fun labelForClassId(classId: Int): String {
        return classNames[classId]?.takeIf { it.isNotBlank() } ?: "class_$classId"
    }
}
