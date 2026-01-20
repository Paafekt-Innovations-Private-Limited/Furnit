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
 * SmartyPantsManager handles loading a TensorFlow Lite model (e.g. a converted `yoloe-11l.tflite`)
 * and running inference on camera frames. This is a best-effort skeleton that inspects the
 * model's input tensor shape and attempts to prepare an input buffer from a provided Bitmap.
 *
 * Place your converted `yoloe_11l.tflite` in `app/src/main/assets/` and call `initialize()`.
 */
class SmartyPantsManager(private val context: Context) {
    private val handler = Handler(Looper.getMainLooper())
    private var interpreter: Interpreter? = null
    private var inputShape: IntArray? = null
    private var inputDataType: DataType? = null
    // ONNX Runtime objects
    private var ortEnv: OrtEnvironment? = null
    private var ortSession: OrtSession? = null

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

    /** Initialize ONNX Runtime session from asset ONNX model. */
    fun initializeOnnx(onnxAssetName: String = "yoloe-11l-seg-pf.onnx") {
        try {
            val file = copyAssetToFile(onnxAssetName)
            ortEnv = OrtEnvironment.getEnvironment()
            val opts = SessionOptions()
            ortSession = ortEnv!!.createSession(file.absolutePath, opts)
            // Log input/output info
            for ((name, info) in ortSession!!.inputInfo) {
                Log.i("SmartyPantsManager", "ONNX input: $name -> ${info.info}")
            }
            for ((name, info) in ortSession!!.outputInfo) {
                Log.i("SmartyPantsManager", "ONNX output: $name -> ${info.info}")
            }
            Log.i("SmartyPantsManager", "Loaded ONNX model '$onnxAssetName' into ONNX Runtime")
        } catch (e: Exception) {
            Log.w("SmartyPantsManager", "Failed to load onnx: ${e.message}")
            ortSession = null
            ortEnv = null
        }
    }

    fun segmentImageAsync(frame: Bitmap?, callback: (Bitmap?) -> Unit) {
        if (frame == null) {
            handler.postDelayed({ callback(null) }, 200)
            return
        }

        handler.post {
            try {
                // Prefer ONNX Runtime if available
                if (ortSession != null) {
                    runOnnxInference(frame, callback)
                    return@post
                }

                // Fallback to TFLite if initialized
                if (interpreter == null) {
                    callback(null)
                    return@post
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

                callback(null)
            } catch (e: Exception) {
                Log.e("SmartyPantsManager", "inference error", e)
                callback(null)
            }
        }
    }

    private fun runOnnxInference(frame: Bitmap, callback: (Bitmap?) -> Unit) {
        try {
            val session = ortSession ?: run { callback(null); return }
            val env = ortEnv ?: run { callback(null); return }

            // Use first input info to determine shape
            val firstInput = session.inputInfo.entries.firstOrNull()
            if (firstInput == null) {
                Log.w("SmartyPantsManager", "ONNX session has no inputs")
                callback(null)
                return
            }

            val inputName = firstInput.key
            val tensorInfo = firstInput.value.info
            val shape = when (tensorInfo) {
                is ai.onnxruntime.TensorInfo -> tensorInfo.shape
                else -> null
            }

            // Heuristic: if shape contains dims > 1, pick H and W; default to 1536
            var h = 1536
            var w = 1536
            if (shape != null) {
                // shape could be [1,3,H,W] or [1,H,W,3]
                if (shape.size >= 4) {
                    val s1 = shape.map { if (it < 0) -1L else it }
                    if (s1[1] == 3L) { // N C H W
                        if (s1[2] > 0) h = s1[2].toInt()
                        if (s1[3] > 0) w = s1[3].toInt()
                    } else if (s1[3] == 3L) { // N H W C
                        if (s1[1] > 0) h = s1[1].toInt()
                        if (s1[2] > 0) w = s1[2].toInt()
                    }
                }
            }

            val resized = Bitmap.createScaledBitmap(frame, w, h, true).copy(Config.ARGB_8888, false)

            // Prepare float array in NCHW (common format for exported models)
            val floatCount = 1 * 3 * h * w
            val inputFloats = FloatArray(floatCount)
            val intValues = IntArray(resized.width * resized.height)
            resized.getPixels(intValues, 0, resized.width, 0, 0, resized.width, resized.height)
            var p = 0
            for (y in 0 until h) {
                for (x in 0 until w) {
                    val v = intValues[y * w + x]
                    val r = ((v shr 16) and 0xFF) / 255.0f
                    val g = ((v shr 8) and 0xFF) / 255.0f
                    val b = (v and 0xFF) / 255.0f
                    // NCHW layout
                    val base = y * w + x
                    inputFloats[0 * (3 * h * w) + 0 * (h * w) + base] = r
                    inputFloats[0 * (3 * h * w) + 1 * (h * w) + base] = g
                    inputFloats[0 * (3 * h * w) + 2 * (h * w) + base] = b
                    p++
                }
            }

            val shapeLong = longArrayOf(1, 3, h.toLong(), w.toLong())
            val tensor = OnnxTensor.createTensor(env, java.nio.FloatBuffer.wrap(inputFloats), shapeLong)

            session.run(mapOf(inputName to tensor)).use { results ->
                // Collect outputs into FloatArray(s)
                val outInfos = ortSession!!.outputInfo.entries.toList()
                val outputs = ArrayList<FloatArray>()
                var idx = 0
                for (out in results) {
                    val v = out.value
                    try {
                        val fa = when (v) {
                            is FloatArray -> v
                            is DoubleArray -> FloatArray(v.size) { i -> v[i].toFloat() }
                            is java.nio.FloatBuffer -> {
                                val a = FloatArray(v.remaining())
                                v.get(a)
                                a
                            }
                            is Array<*> -> flattenArrayToFloat(v)
                            else -> {
                                Log.w("SmartyPantsManager", "Unsupported ONNX output type: ${v::class.java}")
                                FloatArray(0)
                            }
                        }
                        outputs.add(fa)
                    } catch (e: Exception) {
                        Log.w("SmartyPantsManager", "Failed to extract output[$idx]: ${e.message}")
                        outputs.add(FloatArray(0))
                    }
                    idx++
                }

                // Heuristic: find prototype output (rank 4) and detection output (rank 3)
                var protoIndex = -1
                var detIndex = -1
                for (i in outInfos.indices) {
                    val info = outInfos[i].value.info
                    if (info is ai.onnxruntime.TensorInfo) {
                        val shape = info.shape
                        if (shape.size == 4 && protoIndex == -1) protoIndex = i
                        if (shape.size == 3 && detIndex == -1) detIndex = i
                    }
                }

                if (protoIndex == -1 || detIndex == -1) {
                    Log.w("SmartyPantsManager", "Could not identify proto/detection outputs; returning null mask")
                    return@use
                }

                val protoInfo = ortSession!!.outputInfo.entries.elementAt(protoIndex).value.info as ai.onnxruntime.TensorInfo
                val detInfo = ortSession!!.outputInfo.entries.elementAt(detIndex).value.info as ai.onnxruntime.TensorInfo
                val protoShape = protoInfo.shape.map { it.toInt() }
                val detShape = detInfo.shape.map { it.toInt() }

                // outputs[protoIndex] is flattened proto tensor: [1, P, H, W]
                val proto = outputs[protoIndex]
                val det = outputs[detIndex]

                if (proto.isEmpty() || det.isEmpty()) {
                    Log.w("SmartyPantsManager", "Empty outputs from ONNX; returning null mask")
                    return@use
                }

                val P = protoShape[1]
                val HP = protoShape[2]
                val WP = protoShape[3]

                val numPreds = detShape[1]
                val detVecLen = detShape[2]

                // Pick the top-scoring prediction if possible; otherwise pick first.
                var bestIdx = 0
                var bestScore = -Float.MAX_VALUE
                for (iPred in 0 until numPreds) {
                    val base = iPred * detVecLen
                    val scoreCandidates = listOfNotNull(
                        det.getOrNull(base + 4),
                        det.getOrNull(base + 5)
                    )
                    val score = if (scoreCandidates.isNotEmpty()) scoreCandidates.maxOrNull()!! else det.getOrNull(base) ?: 0f
                    if (score > bestScore) { bestScore = score; bestIdx = iPred }
                }

                // Extract prototype coefficients from chosen detection
                val coeffs = FloatArray(P)
                val coeffStart = bestIdx * detVecLen + (detVecLen - P)
                for (j in 0 until P) {
                    coeffs[j] = det.getOrNull(coeffStart + j) ?: 0f
                }

                // Compute mask over prototype spatial size
                val maskProto = FloatArray(HP * WP)
                for (y in 0 until HP) {
                    for (x in 0 until WP) {
                        var s = 0f
                        val pos = y * WP + x
                        for (c in 0 until P) {
                            val protoIdx = c * (HP * WP) + pos
                            s += proto[protoIdx] * coeffs[c]
                        }
                        maskProto[pos] = (1.0f / (1.0f + kotlin.math.exp(-s))) // sigmoid
                    }
                }

                // Resize prototype mask to original frame size
                val maskBmp = Bitmap.createBitmap(WP, HP, Config.ARGB_8888)
                for (y in 0 until HP) {
                    for (x in 0 until WP) {
                        val v = maskProto[y * WP + x]
                        val alpha = if (v > 0.5f) 0xFF else 0x00
                        val color = (alpha shl 24) or (0xFFFFFF)
                        maskBmp.setPixel(x, y, color)
                    }
                }

                // Scale mask to original frame size
                val outMask = Bitmap.createScaledBitmap(maskBmp, frame.width, frame.height, true)
                callback(outMask)
            }

            tensor.close()
        } catch (e: OrtException) {
            Log.e("SmartyPantsManager", "ONNX inference failed", e)
            callback(null)
        }
    }

    fun close() {
        interpreter?.close()
        interpreter = null
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
