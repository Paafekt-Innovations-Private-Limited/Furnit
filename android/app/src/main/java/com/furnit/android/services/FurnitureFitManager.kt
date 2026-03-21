package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.os.Handler
import android.os.Looper
import com.furnit.android.utils.LogUtil
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtException
import com.furnit.android.DetectionResult
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.exp

// Result containing mask and detections
data class SegmentationResult(
    val mask: Bitmap?,
    val detections: List<DetectionResult>,
    val inputSize: Int = 640
)

/**
 * FurnitureFitManager handles object detection and segmentation using YOLOE models.
 *
 * Inference backends (in order of preference):
 * 1. NCNN - Fastest, GPU-accelerated with Vulkan (recommended)
 * 2. ONNX Runtime - Good performance, cross-platform
 * 3. TensorFlow Lite - Fallback option
 *
 * For best performance, use NCNN with exported .param/.bin model files.
 * Place model files in `app/src/main/assets/`.
 */
class FurnitureFitManager(private val context: Context) {
    private val mainHandler = Handler(Looper.getMainLooper())

    companion object {
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
        fun getClassName(classId: Int): String = COCO_CLASSES.getOrElse(classId) { "unknown" }
    }
    private val inferenceExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private var interpreter: Interpreter? = null
    private var inputShape: IntArray? = null
    private var inputDataType: DataType? = null

    // ONNX Runtime objects
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

    // NCNN inference engine (preferred)
    private var ncnnYoloe: NcnnYoloe? = null
    private var useNcnn = false

    fun initialize(tfliteAssetName: String = "yoloe_11l.tflite") {
        try {
            val model = loadModelFile(tfliteAssetName)
            val opts = Interpreter.Options().apply { setNumThreads(4) }
            interpreter = Interpreter(model, opts)

            // Inspect input tensor
            val idx = 0
            val t = interpreter!!.getInputTensor(idx)
            inputShape = t.shape()
            inputDataType = t.dataType()
            LogUtil.i(
                "FurnitureFitManager",
                "Loaded TFLite model '$tfliteAssetName' inputShape=${inputShape?.joinToString()} dataType=$inputDataType"
            )
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "Failed to load tflite: ${e.message}")
            interpreter = null
        }
    }

    /**
     * Initialize NCNN backend for high-performance inference.
     * This is the recommended backend for best performance on Android.
     *
     * @param paramAsset NCNN .param file in assets (default: "yoloe-11l-seg.param")
     * @param binAsset NCNN .bin file in assets (default: "yoloe-11l-seg.bin")
     * @param useGpu Whether to use GPU (Vulkan) acceleration (default: true)
     * @return true if NCNN initialization succeeded
     */
    fun initializeNcnn(
        paramAsset: String = "yoloe-11l-seg.param",
        binAsset: String = "yoloe-11l-seg.bin",
        useGpu: Boolean = true
    ): Boolean {
        LogUtil.i(
            "FurnitureFitManager",
            "initializeNcnn called with param='$paramAsset', bin='$binAsset', gpu=$useGpu"
        )

        if (!NcnnYoloe.isAvailable()) {
            val error = NcnnYoloe.getLoadError() ?: "unknown reason"
            LogUtil.w("FurnitureFitManager", "NCNN native library not available: $error")
            return false
        }

        try {
            ncnnYoloe = NcnnYoloe()
            val success = ncnnYoloe!!.init(context, paramAsset, binAsset, useGpu)

            if (success) {
                useNcnn = true
                LogUtil.i(
                    "FurnitureFitManager",
                    "NCNN initialization successful (GPU: ${ncnnYoloe!!.hasGpu()})"
                )
                return true
            } else {
                LogUtil.e("FurnitureFitManager", "NCNN initialization failed")
                ncnnYoloe = null
                return false
            }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "NCNN init exception: ${e.message}", e)
            ncnnYoloe = null
            return false
        }
    }

    /**
     * Auto-initialize with the best available backend (ONNX Runtime, then TFLite).
     */
    fun initializeAuto(): Boolean {
        LogUtil.i("FurnitureFitManager", "Auto-initializing with best available backend...")

        // ONNX Runtime
        try {
            initializeOnnx()
            if (ortSession != null) {
                LogUtil.i("FurnitureFitManager", "Using ONNX Runtime backend")
                return true
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "ONNX initialization failed: ${e.message}")
        }

        // Fall back to TFLite
        try {
            initialize()
            if (interpreter != null) {
                LogUtil.i("FurnitureFitManager", "Using TFLite backend")
                return true
            }
        } catch (e: Exception) {
            LogUtil.w("FurnitureFitManager", "TFLite initialization failed: ${e.message}")
        }

        LogUtil.e("FurnitureFitManager", "No inference backend available - segmentation disabled")
        return false
    }

    /** Initialize ONNX Runtime session from asset ONNX model. */
    fun initializeOnnx(onnxAssetName: String = "yoloe-11l-seg-pf.onnx") {
        LogUtil.i("FurnitureFitManager", "initializeOnnx called with '$onnxAssetName'")
        try {
            LogUtil.d("FurnitureFitManager", "Copying asset to cache...")
            val file = copyAssetToFile(onnxAssetName)
            LogUtil.d(
                "FurnitureFitManager",
                "Asset copied to: ${file.absolutePath}, size: ${file.length()}"
            )

            LogUtil.d("FurnitureFitManager", "Getting ORT environment...")
            ortEnv = OrtEnvironment.getEnvironment()

            LogUtil.d("FurnitureFitManager", "Creating ONNX session...")
            val opts = SessionOptions()
            ortSession = ortEnv!!.createSession(file.absolutePath, opts)

            // Log input/output info
            LogUtil.i("FurnitureFitManager", "ONNX model loaded successfully")
            for ((name, info) in ortSession!!.inputInfo) {
                LogUtil.i("FurnitureFitManager", "ONNX input: $name -> ${info.info}")
            }
            for ((name, info) in ortSession!!.outputInfo) {
                LogUtil.i("FurnitureFitManager", "ONNX output: $name -> ${info.info}")
            }
            LogUtil.i("FurnitureFitManager", "Loaded ONNX model '$onnxAssetName' into ONNX Runtime")
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "Failed to load onnx: ${e.message}", e)
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

    fun segmentWithDetectionsAsync(frame: Bitmap?, callback: (SegmentationResult?) -> Unit) {
        if (frame == null) {
            mainHandler.postDelayed({ callback(null) }, 200)
            return
        }

        inferenceExecutor.execute {
            try {
                // Prefer NCNN if available (best performance)
                if (useNcnn && ncnnYoloe != null) {
                    runNcnnInferenceWithDetections(frame, callback)
                    return@execute
                }

                // Prefer ONNX Runtime if available
                if (ortSession != null) {
                    runOnnxInferenceWithDetections(frame, callback)
                    return@execute
                }

                // Fallback to TFLite if initialized
                if (interpreter == null) {
                    mainHandler.post { callback(null) }
                    return@execute
                }

                val inShape = inputShape ?: throw IllegalArgumentException("Missing input shape")
                // Expecting input shape like [1, H, W, C] or [1, C, H, W]
                val h: Int
                val w: Int
                val c: Int
                if (inShape.size == 4) {
                    // assume NHWC
                    h = inShape[1]
                    w = inShape[2]
                    c = inShape[3]
                } else if (inShape.size == 3) {
                    h = inShape[1]
                    w = inShape[2]
                    c = 3
                } else {
                    throw IllegalArgumentException("Unsupported input shape: ${inShape.joinToString()}")
                }

                // Resize frame to model input size
                val resized = Bitmap.createScaledBitmap(frame, w, h, true).copy(Config.ARGB_8888, false)

                // Prepare input ByteBuffer
                val bb = convertBitmapToByteBuffer(resized, c, inputDataType ?: DataType.FLOAT32)

                // Prepare output buffers by inspecting model outputs (best-effort)
                val outputMap = HashMap<Int, Any>()
                val outputCount = interpreter!!.outputTensorCount
                for (i in 0 until outputCount) {
                    val outT = interpreter!!.getOutputTensor(i)
                    val shape = outT.shape()
                    val dt = outT.dataType()
                    // allocate a FloatArray for common float outputs
                    if (dt == DataType.FLOAT32) {
                        var size = 1
                        for (d in shape) size *= d
                        outputMap[i] = FloatArray(size)
                    } else {
                        // default fallback: ByteBuffer
                        var size = 1
                        for (d in shape) size *= d
                        outputMap[i] =
                            ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
                    }
                }

                // Run inference
                interpreter!!.runForMultipleInputsOutputs(arrayOf(bb), outputMap)

                mainHandler.post { callback(null) }
            } catch (e: Exception) {
                LogUtil.e("FurnitureFitManager", "inference error", e)
                mainHandler.post { callback(null) }
            }
        }
    }

    /**
     * Run inference using NCNN backend (fastest).
     */
    private fun runNcnnInference(frame: Bitmap, callback: (Bitmap?) -> Unit) {
        try {
            val ncnn = ncnnYoloe ?: run {
                LogUtil.e("FurnitureFitManager", "NCNN not initialized")
                mainHandler.post { callback(null) }
                return
            }

            val startTime = System.currentTimeMillis()

            // Run detection with mask generation
            val result = ncnn.detectWithMask(
                bitmap = frame,
                confThreshold = 0.25f,
                iouThreshold = 0.45f,
                maskThreshold = 0.5f
            )

            val inferenceTime = System.currentTimeMillis() - startTime
            LogUtil.d(
                "FurnitureFitManager",
                "NCNN inference: ${result.detections.size} detections in ${inferenceTime}ms"
            )

            // Log top detections
            for ((idx, det) in result.detections.take(5).withIndex()) {
                LogUtil.d(
                    "FurnitureFitManager",
                    "  [$idx] ${det.label} (${det.classId}): conf=${det.confidence}, bbox=(${det.x},${det.y},${det.width},${det.height})"
                )
            }

            // Return the mask bitmap
            if (result.mask != null) {
                mainHandler.post { callback(result.mask) }
            } else if (result.detections.isEmpty()) {
                // No detections, return null mask
                mainHandler.post { callback(null) }
            } else {
                // Detections but no mask - create a simple bounding box visualization
                val maskBmp = createBboxMask(frame, result.detections)
                mainHandler.post { callback(maskBmp) }
            }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "NCNN inference error: ${e.message}", e)
            mainHandler.post { callback(null) }
        }
    }

    /**
     * Create a simple mask from bounding boxes (fallback when segmentation masks unavailable).
     */
    private fun createBboxMask(frame: Bitmap, detections: List<NcnnYoloe.Detection>): Bitmap {
        val mask = Bitmap.createBitmap(frame.width, frame.height, Config.ARGB_8888)
        val pixels = IntArray(frame.width * frame.height)

        // Fill with transparent
        for (i in pixels.indices) {
            pixels[i] = 0x00000000
        }

        // Draw filled rectangles for each detection
        for (det in detections) {
            val left = maxOf(0, det.left.toInt())
            val top = maxOf(0, det.top.toInt())
            val right = minOf(frame.width - 1, det.right.toInt())
            val bottom = minOf(frame.height - 1, det.bottom.toInt())

            // Fill with semi-transparent green
            val color = 0xCC00FF00.toInt()
            for (y in top..bottom) {
                for (x in left..right) {
                    pixels[y * frame.width + x] = color
                }
            }
        }

        mask.setPixels(pixels, 0, frame.width, 0, 0, frame.width, frame.height)
        return mask
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

                val getDetValue: (Int, Int) -> Float = if (det3d != null) {
                    { feature, anchor -> det3d!![0][feature][anchor] }
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
                    val label = getClassName(det.classId)
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

    // Version that returns detections along with mask
    private fun runOnnxInferenceWithDetections(frame: Bitmap, callback: (SegmentationResult?) -> Unit) {
        try {
            val session = ortSession ?: run {
                mainHandler.post { callback(null) }
                return
            }
            val env = ortEnv ?: run {
                mainHandler.post { callback(null) }
                return
            }

            val firstInput = session.inputInfo.entries.firstOrNull()
            if (firstInput == null) {
                mainHandler.post { callback(null) }
                return
            }

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info

            var inputH = 640
            var inputW = 640
            if (tensorInfo is ai.onnxruntime.TensorInfo) {
                val sh = tensorInfo.shape
                if (sh.size == 4) {
                    inputH = sh[2].toInt()
                    inputW = sh[3].toInt()
                }
            }

            LogUtil.d("FurnitureFitManager", "Resizing frame ${frame.width}x${frame.height} to ${inputW}x${inputH}")
            val resized = Bitmap.createScaledBitmap(frame, inputW, inputH, true)
                .copy(Config.ARGB_8888, false)

            val floatCount = 1 * 3 * inputH * inputW
            val inputFloats = FloatArray(floatCount)
            val intValues = IntArray(inputW * inputH)
            resized.getPixels(intValues, 0, inputW, 0, 0, inputW, inputH)

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
            val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inputFloats), shapeLong)

            val results = session.run(mapOf(inputName to tensor))

            val detResult = results.get(0)
            val protoResult = results.get(1)

            val detValue = detResult?.value
            val protoValue = protoResult?.value

            @Suppress("UNCHECKED_CAST")
            val det3d = detValue as? Array<Array<FloatArray>>
            if (det3d == null) {
                mainHandler.post { callback(null) }
                tensor.close()
                return
            }

            val numFeatures = det3d[0].size
            val numAnchors = det3d[0][0].size
            val numClasses = 80
            val numMaskCoeffs = 32
            val classStartIdx = 4
            val maskCoeffStartIdx = 4 + numClasses

            val proto = extractFloatArray(protoValue)
            val numProtos = 32
            val protoH = 160
            val protoW = 160

            val confThreshold = 0.25f
            val iouThreshold = 0.45f
            val maxDetections = 100

            val detections = mutableListOf<Detection>()
            for (anchor in 0 until numAnchors) {
                var maxScore = Float.MIN_VALUE
                var bestClass = -1
                for (c in 0 until numClasses) {
                    val score = det3d[0][classStartIdx + c][anchor]
                    if (score > maxScore) {
                        maxScore = score
                        bestClass = c
                    }
                }
                if (maxScore > confThreshold) {
                    val x = det3d[0][0][anchor]
                    val y = det3d[0][1][anchor]
                    val bw = det3d[0][2][anchor]
                    val bh = det3d[0][3][anchor]
                    val coeffs = FloatArray(numMaskCoeffs)
                    for (c in 0 until numMaskCoeffs) {
                        coeffs[c] = det3d[0][maskCoeffStartIdx + c][anchor]
                    }
                    detections.add(Detection(anchor, x, y, bw, bh, maxScore, bestClass, coeffs))
                }
            }

            if (detections.isEmpty()) {
                mainHandler.post { callback(SegmentationResult(null, emptyList(), inputW)) }
                tensor.close()
                return
            }

            // NMS
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

            // Only keep the highest confidence detection (primary furniture)
            val primaryDet = keepDets.firstOrNull()

            // Convert to DetectionResult for overlay - only the primary one
            val detectionResults = if (primaryDet != null) {
                listOf(DetectionResult(
                    x = primaryDet.x,
                    y = primaryDet.y,
                    w = primaryDet.w,
                    h = primaryDet.h,
                    confidence = primaryDet.confidence,
                    label = getClassName(primaryDet.classId)
                ))
            } else {
                emptyList()
            }

            // Generate mask for primary detection only
            var maskResult: Bitmap? = null
            if (primaryDet != null && proto.isNotEmpty()) {
                val protoScaleX = inputW.toFloat() / protoW.toFloat()
                val protoScaleY = inputH.toFloat() / protoH.toFloat()
                val maskProto = FloatArray(protoH * protoW)

                // Only process the primary (highest confidence) detection
                val detection = primaryDet
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

                // Scale mask to frame size and apply to original image (remove background)
                val frameW = frame.width
                val frameH = frame.height
                val scaleX = frameW.toFloat() / protoW
                val scaleY = frameH.toFloat() / protoH

                // Get original frame pixels
                val framePixels = IntArray(frameW * frameH)
                frame.getPixels(framePixels, 0, frameW, 0, 0, frameW, frameH)

                // Create output with transparent background where mask <= 0.5
                val outPixels = IntArray(frameW * frameH)
                for (y in 0 until frameH) {
                    val protoY = (y / scaleY).toInt().coerceIn(0, protoH - 1)
                    for (x in 0 until frameW) {
                        val protoX = (x / scaleX).toInt().coerceIn(0, protoW - 1)
                        val maskVal = maskProto[protoY * protoW + protoX]
                        val frameIdx = y * frameW + x
                        if (maskVal > 0.5f) {
                            // Keep original pixel
                            outPixels[frameIdx] = framePixels[frameIdx]
                        } else {
                            // Transparent background
                            outPixels[frameIdx] = 0x00000000
                        }
                    }
                }

                val maskBmp = Bitmap.createBitmap(frameW, frameH, Config.ARGB_8888)
                maskBmp.setPixels(outPixels, 0, frameW, 0, 0, frameW, frameH)
                maskResult = maskBmp
            }

            tensor.close()
            val result = SegmentationResult(maskResult, detectionResults, inputW)
            mainHandler.post { callback(result) }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "ONNX inference with detections failed", e)
            mainHandler.post { callback(null) }
        }
    }

    // NCNN inference with detections - only returns highest confidence detection
    private fun runNcnnInferenceWithDetections(frame: Bitmap, callback: (SegmentationResult?) -> Unit) {
        try {
            val ncnn = ncnnYoloe ?: run {
                LogUtil.e("FurnitureFitManager", "NCNN not initialized")
                mainHandler.post { callback(null) }
                return
            }

            val startTime = System.currentTimeMillis()

            // Get all detections first
            val detections = ncnn.detect(
                bitmap = frame,
                confThreshold = 0.25f,
                iouThreshold = 0.45f
            )

            if (detections.isEmpty()) {
                mainHandler.post { callback(SegmentationResult(null, emptyList(), 640)) }
                return
            }

            // Sort by confidence and take only the highest
            val primaryDet = detections.maxByOrNull { it.confidence }

            if (primaryDet == null) {
                mainHandler.post { callback(SegmentationResult(null, emptyList(), 640)) }
                return
            }

            // Generate mask for only the primary (highest confidence) detection
            val mask = ncnn.generateMask(
                bitmap = frame,
                detections = listOf(primaryDet),  // Only one detection
                maskThreshold = 0.5f
            )

            val inferenceTime = System.currentTimeMillis() - startTime
            LogUtil.d("FurnitureFitManager", "NCNN inference: primary detection ${primaryDet.label} (${String.format("%.2f", primaryDet.confidence)}) in ${inferenceTime}ms")

            // Convert to DetectionResult
            val detectionResult = DetectionResult(
                x = primaryDet.x,
                y = primaryDet.y,
                w = primaryDet.width,
                h = primaryDet.height,
                confidence = primaryDet.confidence,
                label = primaryDet.label
            )

            val result = SegmentationResult(mask, listOf(detectionResult), 640)
            mainHandler.post { callback(result) }
        } catch (e: Exception) {
            LogUtil.e("FurnitureFitManager", "NCNN inference error: ${e.message}", e)
            mainHandler.post { callback(null) }
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

    // Inner class for detection data
    private data class Detection(
        val anchorIdx: Int,
        val x: Float, val y: Float, val w: Float, val h: Float,
        val confidence: Float,
        val classId: Int,
        val coeffs: FloatArray
    )

    fun close() {
        // Release NCNN resources
        ncnnYoloe?.release()
        ncnnYoloe = null
        useNcnn = false

        interpreter?.close()
        interpreter = null
        ortSession?.close()
        ortSession = null
        ortEnv?.close()
        ortEnv = null
        inferenceExecutor.shutdown()
    }

    @Throws(IOException::class)
    private fun loadModelFile(assetName: String): ByteBuffer {
        val assetFileDescriptor = context.assets.openFd(assetName)
        FileInputStream(assetFileDescriptor.fileDescriptor).use { input ->
            val fileChannel: FileChannel = input.channel
            val startOffset = assetFileDescriptor.startOffset
            val declaredLength = assetFileDescriptor.declaredLength
            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        }
    }

    private fun convertBitmapToByteBuffer(bmp: Bitmap, channels: Int, dtype: DataType): ByteBuffer {
        val bb: ByteBuffer
        if (dtype == DataType.FLOAT32) {
            bb = ByteBuffer.allocateDirect(4 * bmp.width * bmp.height * channels)
                .order(ByteOrder.nativeOrder())
            val intValues = IntArray(bmp.width * bmp.height)
            bmp.getPixels(intValues, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            var px = 0
            for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val v = intValues[px++]
                    bb.putFloat(((v shr 16 and 0xFF) / 255.0f))
                    bb.putFloat(((v shr 8 and 0xFF) / 255.0f))
                    bb.putFloat(((v and 0xFF) / 255.0f))
                }
            }
        } else {
            bb = ByteBuffer.allocateDirect(bmp.width * bmp.height * channels)
                .order(ByteOrder.nativeOrder())
            val intValues = IntArray(bmp.width * bmp.height)
            bmp.getPixels(intValues, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            var px = 0
            for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val v = intValues[px++]
                    bb.put((v shr 16 and 0xFF).toByte())
                    bb.put((v shr 8 and 0xFF).toByte())
                    bb.put((v and 0xFF).toByte())
                }
            }
        }
        bb.rewind()
        return bb
    }

    private fun flattenArrayToFloat(arr: Array<*>): FloatArray {
        val list = ArrayList<Float>()
        fun rec(a: Any?) {
            when (a) {
                is Float -> list.add(a)
                is java.lang.Float -> list.add(a.toFloat())
                is Double -> list.add(a.toFloat())
                is java.lang.Double -> list.add(a.toDouble().toFloat())
                is Int -> list.add(a.toFloat())
                is java.lang.Integer -> list.add(a.toInt().toFloat())
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
}
