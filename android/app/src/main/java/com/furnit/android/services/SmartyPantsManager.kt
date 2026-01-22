package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Bitmap.Config
import android.os.Handler
import android.os.Looper
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.OrtSession.SessionOptions
import ai.onnxruntime.OrtException
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.lang.IllegalArgumentException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * SmartyPantsManager handles object detection and segmentation using YOLOE models.
 *
 * Inference backends (in order of preference):
 * 1. NCNN - Fastest, GPU-accelerated with Vulkan (recommended)
 * 2. ONNX Runtime - Good performance, cross-platform
 * 3. TensorFlow Lite - Fallback option
 *
 * For best performance, use NCNN with exported .param/.bin model files.
 * Place model files in `app/src/main/assets/`.
 */
class SmartyPantsManager(private val context: Context) {
    private val mainHandler = Handler(Looper.getMainLooper())
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
            Log.i("SmartyPantsManager", "Loaded TFLite model '$tfliteAssetName' inputShape=${inputShape?.joinToString()} dataType=$inputDataType")
        } catch (e: Exception) {
            Log.w("SmartyPantsManager", "Failed to load tflite: ${e.message}")
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
        Log.i("SmartyPantsManager", "initializeNcnn called with param='$paramAsset', bin='$binAsset', gpu=$useGpu")

        if (!NcnnYoloe.isAvailable()) {
            val error = NcnnYoloe.getLoadError() ?: "unknown reason"
            Log.w("SmartyPantsManager", "NCNN native library not available: $error")
            return false
        }

        try {
            ncnnYoloe = NcnnYoloe()
            val success = ncnnYoloe!!.init(context, paramAsset, binAsset, useGpu)

            if (success) {
                useNcnn = true
                Log.i("SmartyPantsManager", "NCNN initialization successful (GPU: ${ncnnYoloe!!.hasGpu()})")
                return true
            } else {
                Log.e("SmartyPantsManager", "NCNN initialization failed")
                ncnnYoloe = null
                return false
            }
        } catch (e: Exception) {
            Log.e("SmartyPantsManager", "NCNN init exception: ${e.message}", e)
            ncnnYoloe = null
            return false
        }
    }

    /**
     * Auto-initialize with the best available backend.
     * Tries NCNN first, then ONNX, then TFLite.
     */
    fun initializeAuto(): Boolean {
        Log.i("SmartyPantsManager", "Auto-initializing with best available backend...")

        // Try NCNN first (best performance)
        if (initializeNcnn()) {
            Log.i("SmartyPantsManager", "Using NCNN backend")
            return true
        }

        // Fall back to ONNX Runtime (640x640 model, ~156MB output)
        try {
            initializeOnnx()
            if (ortSession != null) {
                Log.i("SmartyPantsManager", "Using ONNX Runtime backend")
                return true
            }
        } catch (e: Exception) {
            Log.w("SmartyPantsManager", "ONNX initialization failed: ${e.message}")
        }

        // Fall back to TFLite
        try {
            initialize()
            if (interpreter != null) {
                Log.i("SmartyPantsManager", "Using TFLite backend")
                return true
            }
        } catch (e: Exception) {
            Log.w("SmartyPantsManager", "TFLite initialization failed: ${e.message}")
        }

        Log.e("SmartyPantsManager", "No inference backend available - segmentation disabled")
        return false
    }

    /** Initialize ONNX Runtime session from asset ONNX model. */
    fun initializeOnnx(onnxAssetName: String = "yoloe-11l-seg-pf.onnx") {
        Log.i("SmartyPantsManager", "initializeOnnx called with '$onnxAssetName'")
        try {
            Log.d("SmartyPantsManager", "Copying asset to cache...")
            val file = copyAssetToFile(onnxAssetName)
            Log.d("SmartyPantsManager", "Asset copied to: ${file.absolutePath}, size: ${file.length()}")

            Log.d("SmartyPantsManager", "Getting ORT environment...")
            ortEnv = OrtEnvironment.getEnvironment()

            Log.d("SmartyPantsManager", "Creating ONNX session...")
            val opts = SessionOptions()
            ortSession = ortEnv!!.createSession(file.absolutePath, opts)

            // Log input/output info
            Log.i("SmartyPantsManager", "ONNX model loaded successfully")
            for ((name, info) in ortSession!!.inputInfo) {
                Log.i("SmartyPantsManager", "ONNX input: $name -> ${info.info}")
            }
            for ((name, info) in ortSession!!.outputInfo) {
                Log.i("SmartyPantsManager", "ONNX output: $name -> ${info.info}")
            }
            Log.i("SmartyPantsManager", "Loaded ONNX model '$onnxAssetName' into ONNX Runtime")
        } catch (e: Exception) {
            Log.e("SmartyPantsManager", "Failed to load onnx: ${e.message}", e)
            ortSession = null
            ortEnv = null
        }
    }

    fun segmentImageAsync(frame: Bitmap?, callback: (Bitmap?) -> Unit) {
        if (frame == null) {
            mainHandler.postDelayed({ callback(null) }, 200)
            return
        }

        inferenceExecutor.execute {
            try {
                // Prefer NCNN if available (best performance)
                if (useNcnn && ncnnYoloe != null) {
                    runNcnnInference(frame, callback)
                    return@execute
                }

                // Prefer ONNX Runtime if available
                if (ortSession != null) {
                    runOnnxInference(frame, callback)
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
                        outputMap[i] = ByteBuffer.allocateDirect(size * 4).order(ByteOrder.nativeOrder())
                    }
                }

                // Run inference
                interpreter!!.runForMultipleInputsOutputs(arrayOf(bb), outputMap)

                mainHandler.post { callback(null) }
            } catch (e: Exception) {
                Log.e("SmartyPantsManager", "inference error", e)
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
                Log.e("SmartyPantsManager", "NCNN not initialized")
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
            Log.d("SmartyPantsManager", "NCNN inference: ${result.detections.size} detections in ${inferenceTime}ms")

            // Log top detections
            for ((idx, det) in result.detections.take(5).withIndex()) {
                Log.d("SmartyPantsManager", "  [$idx] ${det.label} (${det.classId}): conf=${det.confidence}, bbox=(${det.x},${det.y},${det.width},${det.height})")
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
            Log.e("SmartyPantsManager", "NCNN inference error: ${e.message}", e)
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
                Log.e("SmartyPantsManager", "ortSession is null")
                mainHandler.post { callback(null) }
                return
            }
            val env = ortEnv ?: run {
                Log.e("SmartyPantsManager", "ortEnv is null")
                mainHandler.post { callback(null) }
                return
            }

            // Use first input info to determine shape
            val firstInput = session.inputInfo.entries.firstOrNull()
            if (firstInput == null) {
                Log.w("SmartyPantsManager", "ONNX session has no inputs")
                mainHandler.post { callback(null) }
                return
            }

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info

            val shape = when (tensorInfo) {
                is ai.onnxruntime.TensorInfo -> tensorInfo.shape
                else -> null
            }

            // YOLOe model expects [1, 3, 640, 640] for 640 export
            // Output tensor is ~156MB (vs 900MB for 1536)
            var h = 640
            var w = 640

            Log.d("SmartyPantsManager", "Resizing frame ${frame.width}x${frame.height} to ${w}x${h}")
            val resized = Bitmap.createScaledBitmap(frame, w, h, true).copy(Config.ARGB_8888, false)

            // Prepare float array in NCHW format
            val floatCount = 1 * 3 * h * w
            val inputFloats = FloatArray(floatCount)
            val intValues = IntArray(resized.width * resized.height)
            resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)

            // Fill NCHW layout: channel-first ordering
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val v = intValues[y * w + x]
                    val r = ((v shr 16) and 0xFF) / 255.0f
                    val g = ((v shr 8) and 0xFF) / 255.0f
                    val b = (v and 0xFF) / 255.0f
                    val pixelIdx = y * w + x
                    inputFloats[0 * h * w + pixelIdx] = r  // R channel
                    inputFloats[1 * h * w + pixelIdx] = g  // G channel
                    inputFloats[2 * h * w + pixelIdx] = b  // B channel
                }
            }

            val shapeLong = longArrayOf(1, 3, h.toLong(), w.toLong())
            Log.d("SmartyPantsManager", "Creating input tensor with shape ${shapeLong.toList()}")
            val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inputFloats), shapeLong)
            var maskResult: Bitmap? = null

            Log.d("SmartyPantsManager", "Running ONNX inference...")
            session.run(mapOf(inputName to tensor)).use { results ->
                Log.d("SmartyPantsManager", "Inference complete, processing outputs...")

                // Get output info
                val outInfos = ortSession!!.outputInfo.entries.toList()

                // Find detection output (output0: [1, features, anchors]) and prototype output (output1: [1, 32, H, W])
                var detIndex = -1
                var protoIndex = -1
                var detShape: LongArray? = null
                var protoShape: LongArray? = null

                for (i in outInfos.indices) {
                    val info = outInfos[i].value.info
                    if (info is ai.onnxruntime.TensorInfo) {
                        val sh = info.shape
                        Log.d("SmartyPantsManager", "Output $i shape: ${sh.toList()}")
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
                    Log.w("SmartyPantsManager", "Could not find detection/prototype outputs")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                Log.d("SmartyPantsManager", "Detection output[$detIndex] shape: ${detShape.toList()}")
                Log.d("SmartyPantsManager", "Proto output[$protoIndex] shape: ${protoShape.toList()}")

                // Extract output tensors
                val detResult = results.get(detIndex)
                val protoResult = results.get(protoIndex)

                val detValue = detResult?.value
                val protoValue = protoResult?.value

                // Parse detection output: [1, numFeatures, numAnchors]
                // For YOLOe prompt-free: [1, 4621, 48384] means 4621 features, 48384 anchors
                // Structure: bbox(4) + embeddings(4585) + mask_coeffs(32) = 4621
                val numFeatures = detShape[1].toInt()
                val numAnchors = detShape[2].toInt()
                val numMaskCoeffs = 32
                val numEmbeddings = numFeatures - 4 - numMaskCoeffs  // 4585 for this model
                val embeddingStart = 4
                val maskCoeffStart = 4 + numEmbeddings  // 4589

                // Parse prototype output: [1, 32, protoH, protoW]
                val numProtos = protoShape[1].toInt()  // 32
                val protoH = protoShape[2].toInt()     // 384
                val protoW = protoShape[3].toInt()     // 384

                Log.d("SmartyPantsManager", "Prompt-Free Format: Features=$numFeatures, Anchors=$numAnchors, Embeddings=$numEmbeddings, MaskCoeffs=$numMaskCoeffs, ProtoSize=${protoH}x${protoW}")

                if (numFeatures < 36 || numAnchors <= 0 || numEmbeddings <= 0) {
                    Log.e("SmartyPantsManager", "Invalid tensor dimensions")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                Log.d("SmartyPantsManager", "DetValue type: ${detValue?.javaClass}")

                // Handle detection output - try 3D array first, fall back to flattened
                // For shape [1, numFeatures, numAnchors], ONNX Runtime returns float[1][numFeatures][numAnchors]
                var det3d: Array<Array<FloatArray>>? = null
                var detFlat: FloatArray? = null

                when (detValue) {
                    is Array<*> -> {
                        try {
                            // Try to cast as 3D array
                            @Suppress("UNCHECKED_CAST")
                            det3d = detValue as Array<Array<FloatArray>>
                            Log.d("SmartyPantsManager", "Det as 3D array: [${det3d.size}][${det3d[0].size}][${det3d[0][0].size}]")
                        } catch (e: Exception) {
                            Log.w("SmartyPantsManager", "Failed to cast as 3D array, trying flatten: ${e.message}")
                            detFlat = extractFloatArray(detValue)
                        }
                    }
                    is FloatArray -> {
                        detFlat = detValue
                        Log.d("SmartyPantsManager", "Det as flat FloatArray: ${detFlat.size}")
                    }
                    is java.nio.FloatBuffer -> {
                        detFlat = FloatArray(detValue.remaining())
                        detValue.get(detFlat)
                        Log.d("SmartyPantsManager", "Det as FloatBuffer: ${detFlat.size}")
                    }
                    else -> {
                        Log.w("SmartyPantsManager", "Unknown det type: ${detValue?.javaClass}")
                        detFlat = extractFloatArray(detValue)
                    }
                }

                if (det3d == null && (detFlat == null || detFlat.isEmpty())) {
                    Log.e("SmartyPantsManager", "Could not extract detection tensor")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                // Extract prototype tensor
                Log.d("SmartyPantsManager", "Extracting proto array...")
                val proto = extractFloatArray(protoValue)
                Log.d("SmartyPantsManager", "Proto extracted: ${proto.size} floats")

                if (proto.isEmpty()) {
                    Log.w("SmartyPantsManager", "Empty proto output")
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                val stride = numAnchors

                // Find best detections using embedding norm as confidence
                // For prompt-free model, higher embedding norm indicates object presence
                val normThreshold = 0.3f  // Embedding norm threshold
                val iouThreshold = 0.7f
                val maxDetections = 100

                val detections = mutableListOf<Detection>()

                // OPTIMIZATION: Only sample a subset of embeddings for norm calculation
                // Using first 64 embeddings is enough to detect objects without processing all 4585
                val embeddingSampleSize = minOf(64, numEmbeddings)
                // OPTIMIZATION: Skip anchors - process every Nth anchor to reduce computation
                val anchorStride = maxOf(1, numAnchors / 5000)  // Process ~5000 anchors max

                Log.d("SmartyPantsManager", "Scanning $numAnchors anchors (stride=$anchorStride, ~${numAnchors/anchorStride} samples) with norm threshold $normThreshold...")
                val startTime = System.currentTimeMillis()

                // Define accessor function based on available format
                val getDetValue: (Int, Int) -> Float = if (det3d != null) {
                    { feature, anchor -> det3d[0][feature][anchor] }
                } else {
                    { feature, anchor -> detFlat!![feature * stride + anchor] }
                }

                // Scan anchors with stride - use embedding norm as confidence proxy
                var anchor = 0
                while (anchor < numAnchors) {
                    // Get bbox values
                    val x = getDetValue(0, anchor)
                    val y = getDetValue(1, anchor)
                    val dw = getDetValue(2, anchor)
                    val dh = getDetValue(3, anchor)

                    // Validate bbox - quick rejection
                    if (x.isFinite() && y.isFinite() && dw.isFinite() && dh.isFinite() &&
                        dw > 0 && dh > 0 && dw < 2000 && dh < 2000) {

                        // Calculate embedding L2 norm using sampled embeddings
                        var normSq = 0f
                        for (e in 0 until embeddingSampleSize) {
                            val v = getDetValue(embeddingStart + e, anchor)
                            normSq += v * v
                        }
                        val embeddingNorm = kotlin.math.sqrt(normSq)

                        if (embeddingNorm > normThreshold) {
                            // Extract mask coefficients (last 32 features)
                            val coeffs = FloatArray(numProtos)
                            for (c in 0 until numProtos) {
                                coeffs[c] = getDetValue(maskCoeffStart + c, anchor)
                            }

                            detections.add(Detection(
                                anchorIdx = anchor,
                                x = x, y = y, w = dw, h = dh,
                                confidence = embeddingNorm,
                                classId = 0,  // Unknown class for prompt-free
                                coeffs = coeffs
                            ))

                            // Early exit if we have enough good detections
                            if (detections.size >= maxDetections * 2) break
                        }
                    }
                    anchor += anchorStride
                }

                val scanTime = System.currentTimeMillis() - startTime
                Log.d("SmartyPantsManager", "Found ${detections.size} detections above norm threshold $normThreshold in ${scanTime}ms")

                if (detections.isEmpty()) {
                    // No detections - return empty mask
                    mainHandler.post { callback(null) }
                    tensor.close()
                    return
                }

                // Log top detections
                val topDets = detections.sortedByDescending { it.confidence }.take(5)
                Log.d("SmartyPantsManager", "Top ${topDets.size} detections:")
                for ((idx, det) in topDets.withIndex()) {
                    Log.d("SmartyPantsManager", "  [$idx] class=${det.classId}, conf=${det.confidence}, bbox=(${det.x},${det.y},${det.w},${det.h})")
                }

                // Sort by confidence and take top detections
                val sortedDets = detections.sortedByDescending { it.confidence }.take(maxDetections)
                Log.d("SmartyPantsManager", "Top detection: conf=${sortedDets[0].confidence}, class=${sortedDets[0].classId}, bbox=(${sortedDets[0].x},${sortedDets[0].y},${sortedDets[0].w},${sortedDets[0].h})")

                // Simple NMS
                val keepDets = mutableListOf<Detection>()
                val suppressed = BooleanArray(sortedDets.size)

                for (i in sortedDets.indices) {
                    if (suppressed[i]) continue
                    keepDets.add(sortedDets[i])

                    for (j in i + 1 until sortedDets.size) {
                        if (suppressed[j]) continue
                        val iou = calculateIoU(sortedDets[i], sortedDets[j])
                        if (iou > iouThreshold) {
                            suppressed[j] = true
                        }
                    }
                }

                Log.d("SmartyPantsManager", "After NMS: ${keepDets.size} detections kept")

                // Generate combined mask from all detections
                val maskProto = FloatArray(protoH * protoW)

                for (detection in keepDets) {
                    // Compute mask: sum(coeff[c] * proto[c, y, x]) for each pixel
                    for (py in 0 until protoH) {
                        for (px in 0 until protoW) {
                            var sum = 0f
                            for (c in 0 until numProtos) {
                                // Proto is [1, 32, H, W], flattened as [c * H * W + y * W + x]
                                val protoIdx = c * protoH * protoW + py * protoW + px
                                sum += detection.coeffs[c] * proto[protoIdx]
                            }
                            // Apply sigmoid and take max
                            val sigmoidVal = 1f / (1f + kotlin.math.exp(-sum))
                            if (sigmoidVal > maskProto[py * protoW + px]) {
                                maskProto[py * protoW + px] = sigmoidVal
                            }
                        }
                    }
                }

                // Create mask bitmap
                val maskBmp = Bitmap.createBitmap(protoW, protoH, Config.ARGB_8888)
                for (py in 0 until protoH) {
                    for (px in 0 until protoW) {
                        val v = maskProto[py * protoW + px]
                        val alpha = if (v > 0.5f) 0xCC else 0x00
                        val color = (alpha shl 24) or 0x00FF00  // Green mask
                        maskBmp.setPixel(px, py, color)
                    }
                }

                // Scale mask to original frame size
                val outMask = Bitmap.createScaledBitmap(maskBmp, frame.width, frame.height, true)
                maskResult = outMask
                Log.d("SmartyPantsManager", "Mask generated: ${outMask.width}x${outMask.height}")
            }

            tensor.close()
            val finalMask = maskResult
            mainHandler.post { callback(finalMask) }
        } catch (e: OrtException) {
            Log.e("SmartyPantsManager", "ONNX inference failed", e)
            mainHandler.post { callback(null) }
        } catch (e: Exception) {
            Log.e("SmartyPantsManager", "ONNX inference exception", e)
            e.printStackTrace()
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
                Log.w("SmartyPantsManager", "Unknown output type: ${value?.javaClass}")
                FloatArray(0)
            }
        }
    }

    private fun calculateIoU(det1: Detection, det2: Detection): Float {
        // Convert from center format to corner format
        val x1Min = det1.x - det1.w / 2
        val y1Min = det1.y - det1.h / 2
        val x1Max = det1.x + det1.w / 2
        val y1Max = det1.y + det1.h / 2

        val x2Min = det2.x - det2.w / 2
        val y2Min = det2.y - det2.h / 2
        val x2Max = det2.x + det2.w / 2
        val y2Max = det2.y + det2.h / 2

        // Calculate intersection
        val interXMin = maxOf(x1Min, x2Min)
        val interYMin = maxOf(y1Min, y2Min)
        val interXMax = minOf(x1Max, x2Max)
        val interYMax = minOf(y1Max, y2Max)

        val interWidth = maxOf(0f, interXMax - interXMin)
        val interHeight = maxOf(0f, interYMax - interYMin)
        val interArea = interWidth * interHeight

        // Calculate union
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
            bb = ByteBuffer.allocateDirect(4 * bmp.width * bmp.height * channels).order(ByteOrder.nativeOrder())
            val intValues = IntArray(bmp.width * bmp.height)
            bmp.getPixels(intValues, 0, bmp.width, 0, 0, bmp.width, bmp.height)
            var px = 0
            for (y in 0 until bmp.height) {
                for (x in 0 until bmp.width) {
                    val v = intValues[px++]
                    // Extract RGB and normalize to [0,1]
                    bb.putFloat(((v shr 16 and 0xFF) / 255.0f))
                    bb.putFloat(((v shr 8 and 0xFF) / 255.0f))
                    bb.putFloat(((v and 0xFF) / 255.0f))
                }
            }
        } else {
            // Fallback: pack as bytes (UINT8)
            bb = ByteBuffer.allocateDirect(bmp.width * bmp.height * channels).order(ByteOrder.nativeOrder())
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
