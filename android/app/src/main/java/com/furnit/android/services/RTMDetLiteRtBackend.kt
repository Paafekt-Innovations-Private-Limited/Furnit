package com.furnit.android.services

import android.content.Context
import com.furnit.android.utils.LogUtil
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.Tensor
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.gpu.GpuDelegateFactory

internal data class RTMDetLiteRtRun(
    val cls80: FloatArray,
    val cls40: FloatArray,
    val cls20: FloatArray,
    val bbox80: FloatArray,
    val bbox40: FloatArray,
    val bbox20: FloatArray,
    val kernel80: FloatArray,
    val kernel40: FloatArray,
    val kernel20: FloatArray,
    /** NHWC (`pixel * 8 + channel`) — every other output is transposed to NCHW, this one is not. */
    val maskFeat: FloatArray,
    val inputPackingMillis: Long,
    val inferenceMillis: Long,
    val outputCopyMillis: Long,
    val nativeInferenceMillis: Long?,
)

/**
 * Reuse and extraction rules for the on-device model copy, kept free of `Context` so they stay
 * covered by focused JVM tests.
 *
 * The model ships in an install-time asset pack and must be extracted to `cacheDir` before it can
 * be memory-mapped. The original rule reused any cached file with `length() > 0`, which cannot
 * tell a finished 57.9 MB copy from a truncated one, and it wrote straight to the final path. An
 * extraction interrupted by process death or by the platform reclaiming `cacheDir` therefore left
 * a short file that satisfied the check forever: every later launch mapped a corrupt FlatBuffer,
 * failed interpreter creation on both GPU and CPU, and disabled segmentation permanently until
 * app storage was cleared by hand.
 *
 * So the byte count is now recorded in the token and verified on reuse, and extraction lands on a
 * temporary file that is renamed into place only once complete. A partial copy can never be
 * observed as valid, and a bad one re-extracts itself on the next launch.
 */
internal object RTMDetModelCache {
    private const val TOKEN_SEPARATOR = ":"

    fun buildToken(installTime: Long, byteCount: Long): String =
        "$installTime$TOKEN_SEPARATOR$byteCount"

    /** A cached copy is reusable only when this install wrote it and it is still byte-complete. */
    fun isReusable(cachedFile: File, token: String?, installTime: Long): Boolean {
        if (!cachedFile.isFile || token == null) return false
        val cachedInstallTime = token.substringBefore(TOKEN_SEPARATOR).toLongOrNull() ?: return false
        // Tokens written before the byte count was recorded have no separator, so they read as
        // unverifiable and force one clean re-extraction on upgrade.
        val expectedLength = token.substringAfter(TOKEN_SEPARATOR, "").toLongOrNull() ?: return false
        return cachedInstallTime == installTime && cachedFile.length() == expectedLength
    }

    /**
     * Copies through [temporary] and renames on completion, so [destination] never exposes a
     * partial file. Returns the number of bytes written.
     */
    fun extract(destination: File, temporary: File, openSource: () -> InputStream): Long {
        temporary.delete()
        return try {
            val written = openSource().use { input ->
                temporary.outputStream().use(input::copyTo)
            }
            check(written > 0L) { "model asset is empty" }
            destination.delete()
            check(temporary.renameTo(destination)) { "could not move extracted model into place" }
            written
        } finally {
            temporary.delete()
        }
    }
}

/** Pure tensor-layout helpers kept visible to focused JVM parity tests. */
internal object RTMDetLiteRtTensorLayout {
    fun packArgbToNhwcBgr(
        source: IntArray,
        destination: FloatArray,
    ) {
        require(destination.size == source.size * 3)
        var destinationIndex = 0
        for (pixel in source) {
            destination[destinationIndex++] = (pixel and 0xFF).toFloat()
            destination[destinationIndex++] = ((pixel ushr 8) and 0xFF).toFloat()
            destination[destinationIndex++] = ((pixel ushr 16) and 0xFF).toFloat()
        }
    }

    fun nhwcToNchw(
        source: FloatArray,
        destination: FloatArray,
        height: Int,
        width: Int,
        channels: Int,
    ) {
        val spatialSize = height * width
        require(source.size == spatialSize * channels)
        require(destination.size == source.size)
        for (channel in 0 until channels) {
            var sourceIndex = channel
            var destinationIndex = channel * spatialSize
            repeat(spatialSize) {
                destination[destinationIndex++] = source[sourceIndex]
                sourceIndex += channels
            }
        }
    }
}

/**
 * Persistent FP16 LiteRT runner for RTMDet.
 *
 * LiteRT's GPU delegate is thread-affine. This backend owns one dedicated serial thread and
 * creates, warms, invokes, and closes both the delegate and interpreter on that same thread.
 * Callers may therefore preload from an IO thread and later invoke from Furniture Fit's rotating
 * frame lanes without violating the GPU contract.
 */
internal class RTMDetLiteRtBackend private constructor(
    private val executor: ExecutorService,
    private val state: RuntimeState,
) : AutoCloseable {
    val executionProvider: String = state.executionProvider
    val inputWidth: Int = state.inputWidth
    val inputHeight: Int = state.inputHeight

    /**
     * True once [close] has run. This instance is shared between every Furniture Fit surface, so a
     * holder that cached the reference must re-check it rather than assume it is still alive.
     */
    val isClosed: Boolean get() = executor.isShutdown

    fun run(
        argbPixels: IntArray,
        requireKernels: Boolean,
        requireMaskFeat: Boolean,
    ): RTMDetLiteRtRun {
        check(!executor.isShutdown) { "RTMDet LiteRT backend is closed" }
        return executor.submit<RTMDetLiteRtRun> {
            state.run(argbPixels, requireKernels, requireMaskFeat)
        }.get()
    }

    override fun close() {
        if (executor.isShutdown) return
        runCatching {
            executor.submit {
                state.interpreter.close()
                state.gpuDelegate?.close()
            }.get()
        }.onFailure { error ->
            LogUtil.w(TAG, "Could not close RTMDet LiteRT backend cleanly: ${error.message}")
        }
        executor.shutdown()
        runCatching { executor.awaitTermination(5, TimeUnit.SECONDS) }
    }

    private data class OutputBuffer(
        val name: String,
        val shape: IntArray,
        val bytes: ByteBuffer,
        val floats: FloatBuffer,
        val nhwcValues: FloatArray,
        val nchwValues: FloatArray,
    ) {
        val height: Int get() = shape[1]
        val width: Int get() = shape[2]
        val channels: Int get() = shape[3]

        fun prepareForWrite() {
            bytes.clear()
            floats.clear()
        }

        /**
         * LiteRT already writes NHWC, so this is a straight copy with no transpose. The mask head
         * consumes it directly: its eight per-pixel features then share a cache line instead of
         * being gathered across a 102 KB channel stride.
         */
        fun copyToNhwc(): FloatArray {
            floats.position(0)
            floats.get(nhwcValues)
            return nhwcValues
        }

        fun copyToNchw(): FloatArray {
            copyToNhwc()
            RTMDetLiteRtTensorLayout.nhwcToNchw(
                source = nhwcValues,
                destination = nchwValues,
                height = height,
                width = width,
                channels = channels,
            )
            return nchwValues
        }
    }

    private class RuntimeState(
        val interpreter: Interpreter,
        val gpuDelegate: GpuDelegate?,
        val executionProvider: String,
        val inputWidth: Int,
        val inputHeight: Int,
        private val inputBytes: ByteBuffer,
        private val inputFloats: FloatBuffer,
        private val inputValues: FloatArray,
        private val outputsByName: Map<String, OutputBuffer>,
    ) {
        private val signatureInputs = mapOf<String, Any>(INPUT_NAME to inputBytes)
        private val signatureOutputs = outputsByName.mapValues { it.value.bytes as Any }

        fun warmUp(): Long {
            inputValues.fill(0f)
            inputFloats.clear()
            inputFloats.put(inputValues)
            inputBytes.position(0)
            outputsByName.values.forEach(OutputBuffer::prepareForWrite)
            val startNanos = System.nanoTime()
            interpreter.runSignature(signatureInputs, signatureOutputs, SIGNATURE_KEY)
            return elapsedMillis(startNanos)
        }

        fun run(
            argbPixels: IntArray,
            requireKernels: Boolean,
            requireMaskFeat: Boolean,
        ): RTMDetLiteRtRun {
            require(argbPixels.size == inputWidth * inputHeight) {
                "Expected ${inputWidth}x$inputHeight pixels, received ${argbPixels.size}"
            }

            val packingStartNanos = System.nanoTime()
            RTMDetLiteRtTensorLayout.packArgbToNhwcBgr(argbPixels, inputValues)
            inputFloats.clear()
            // One bulk native copy avoids 1.2 million direct-buffer method calls per frame. The
            // values remain byte-for-byte equivalent to Swift's ImageType(BGR) input contract.
            inputFloats.put(inputValues)
            inputBytes.position(0)
            val inputPackingMillis = elapsedMillis(packingStartNanos)

            outputsByName.values.forEach(OutputBuffer::prepareForWrite)
            val inferenceStartNanos = System.nanoTime()
            interpreter.runSignature(signatureInputs, signatureOutputs, SIGNATURE_KEY)
            val inferenceMillis = elapsedMillis(inferenceStartNanos)
            val nativeInferenceMillis = interpreter.lastNativeInferenceDurationNanoseconds
                ?.let { TimeUnit.NANOSECONDS.toMillis(it) }

            val outputCopyStartNanos = System.nanoTime()
            fun output(name: String, required: Boolean = true): FloatArray =
                if (required) checkNotNull(outputsByName[name]) { "Missing LiteRT output '$name'" }.copyToNchw()
                else FloatArray(0)

            val result = RTMDetLiteRtRun(
                cls80 = output("cls_80"),
                cls40 = output("cls_40"),
                cls20 = output("cls_20"),
                bbox80 = output("bbox_80"),
                bbox40 = output("bbox_40"),
                bbox20 = output("bbox_20"),
                kernel80 = output("kernel_80", requireKernels),
                kernel40 = output("kernel_40", requireKernels),
                kernel20 = output("kernel_20", requireKernels),
                // NHWC, unlike every other output: the mask head reads it per pixel.
                maskFeat = if (requireMaskFeat) {
                    checkNotNull(outputsByName["mask_feat"]) { "Missing LiteRT output 'mask_feat'" }
                        .copyToNhwc()
                } else {
                    FloatArray(0)
                },
                inputPackingMillis = inputPackingMillis,
                inferenceMillis = inferenceMillis,
                outputCopyMillis = elapsedMillis(outputCopyStartNanos),
                nativeInferenceMillis = nativeInferenceMillis,
            )
            return result
        }
    }

    companion object {
        private const val TAG = "RTMDetLiteRt"
        private const val SIGNATURE_KEY = "serving_default"
        private const val INPUT_NAME = "input"
        private const val MODEL_TOKEN = "furnit_rtmdet_ins_m_fp16_7edbd669"
        private val REQUIRED_OUTPUT_NAMES = setOf(
            "cls_80", "cls_40", "cls_20",
            "bbox_80", "bbox_40", "bbox_20",
            "kernel_80", "kernel_40", "kernel_20",
            "mask_feat",
        )

        fun create(context: Context, assetName: String): RTMDetLiteRtBackend? {
            val executor = Executors.newSingleThreadExecutor { runnable ->
                Thread(runnable, "RTMDetLiteRt")
            }
            return try {
                val appContext = context.applicationContext
                val state = executor.submit<RuntimeState> {
                    createRuntimeState(appContext, assetName)
                }.get()
                RTMDetLiteRtBackend(executor, state)
            } catch (error: Exception) {
                executor.shutdownNow()
                LogUtil.e(TAG, "LiteRT initialization failed: ${rootMessage(error)}", error)
                null
            }
        }

        private fun createRuntimeState(context: Context, assetName: String): RuntimeState {
            val initStartNanos = System.nanoTime()
            val modelFile = copyAssetToCache(context, assetName)
            val modelBuffer = FileInputStream(modelFile).channel.use { channel ->
                channel.map(java.nio.channels.FileChannel.MapMode.READ_ONLY, 0, channel.size())
            }

            val gpuOptions = try {
                CompatibilityList().use { compatibility ->
                    if (!compatibility.isDelegateSupportedOnThisDevice) {
                        // The allowlist is keyed on SoC/GPU/driver/OS, so a system update can flip
                        // this to false with no code change. Log it: otherwise the app drops to the
                        // XNNPACK CPU path silently and the only symptom is that it feels slow.
                        LogUtil.w(TAG, "GPU delegate unsupported on this device; using LiteRT XNNPACK")
                        return@use null
                    }
                    compatibility.bestOptionsForThisDevice
                        .setPrecisionLossAllowed(true)
                        .setInferencePreference(GpuDelegateFactory.Options.INFERENCE_PREFERENCE_SUSTAINED_SPEED)
                        .setSerializationParams(context.codeCacheDir.absolutePath, MODEL_TOKEN)
                }
            } catch (error: Exception) {
                LogUtil.w(TAG, "Could not query GPU compatibility: ${rootMessage(error)}")
                null
            }

            if (gpuOptions != null) {
                var gpuDelegate: GpuDelegate? = null
                var gpuInterpreter: Interpreter? = null
                try {
                    gpuDelegate = GpuDelegate(gpuOptions)
                    modelBuffer.position(0)
                    gpuInterpreter = Interpreter(
                        modelBuffer,
                        Interpreter.Options()
                            .setNumThreads(1)
                            .addDelegate(gpuDelegate),
                    )
                    return prepareRuntimeState(
                        interpreter = gpuInterpreter,
                        gpuDelegate = gpuDelegate,
                        executionProvider = "LITERT_GPU_FP16",
                        assetName = assetName,
                        initStartNanos = initStartNanos,
                    )
                } catch (gpuError: Exception) {
                    runCatching { gpuInterpreter?.close() }
                    runCatching { gpuDelegate?.close() }
                    LogUtil.w(
                        TAG,
                        "GPU delegate compile/warm-up failed; using LiteRT XNNPACK: ${rootMessage(gpuError)}",
                    )
                }
            }

            modelBuffer.position(0)
            val cpuInterpreter = Interpreter(
                modelBuffer,
                Interpreter.Options()
                    .setUseXNNPACK(true)
                    .setNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(2, 4)),
            )
            return try {
                prepareRuntimeState(
                    interpreter = cpuInterpreter,
                    gpuDelegate = null,
                    executionProvider = "LITERT_CPU_XNNPACK_FP16",
                    assetName = assetName,
                    initStartNanos = initStartNanos,
                )
            } catch (error: Exception) {
                cpuInterpreter.close()
                throw error
            }
        }

        private fun prepareRuntimeState(
            interpreter: Interpreter,
            gpuDelegate: GpuDelegate?,
            executionProvider: String,
            assetName: String,
            initStartNanos: Long,
        ): RuntimeState {
            interpreter.allocateTensors()
            validateSignature(interpreter)
            val inputTensor = interpreter.getInputTensorFromSignature(INPUT_NAME, SIGNATURE_KEY)
            require(inputTensor.dataType() == DataType.FLOAT32) {
                "Expected Float32 LiteRT input, got ${inputTensor.dataType()}"
            }
            val inputShape = inputTensor.shape()
            require(inputShape.size == 4 && inputShape[0] == 1 && inputShape[3] == 3) {
                "Expected NHWC input [1,H,W,3], got ${inputShape.contentToString()}"
            }
            val inputHeight = inputShape[1]
            val inputWidth = inputShape[2]
            val inputBytes = directBuffer(inputTensor.numBytes())
            val outputsByName = REQUIRED_OUTPUT_NAMES.associateWith { name ->
                outputBuffer(name, interpreter.getOutputTensorFromSignature(name, SIGNATURE_KEY))
            }
            val state = RuntimeState(
                interpreter = interpreter,
                gpuDelegate = gpuDelegate,
                executionProvider = executionProvider,
                inputWidth = inputWidth,
                inputHeight = inputHeight,
                inputBytes = inputBytes,
                inputFloats = inputBytes.asFloatBuffer(),
                inputValues = FloatArray(inputTensor.numElements()),
                outputsByName = outputsByName,
            )
            val warmUpMillis = state.warmUp()
            LogUtil.i(
                TAG,
                "Loaded '$assetName' provider=$executionProvider input=${inputWidth}x$inputHeight " +
                    "warmUp=${warmUpMillis}ms total=${elapsedMillis(initStartNanos)}ms " +
                    "thread=${Thread.currentThread().name}",
            )
            return state
        }

        private fun validateSignature(interpreter: Interpreter) {
            require(SIGNATURE_KEY in interpreter.signatureKeys) {
                "Missing LiteRT signature '$SIGNATURE_KEY': ${interpreter.signatureKeys.contentToString()}"
            }
            require(INPUT_NAME in interpreter.getSignatureInputs(SIGNATURE_KEY)) {
                "Missing LiteRT input '$INPUT_NAME'"
            }
            val actualOutputs = interpreter.getSignatureOutputs(SIGNATURE_KEY).toSet()
            require(actualOutputs.containsAll(REQUIRED_OUTPUT_NAMES)) {
                "Missing LiteRT outputs: ${REQUIRED_OUTPUT_NAMES - actualOutputs}"
            }
        }

        private fun outputBuffer(name: String, tensor: Tensor): OutputBuffer {
            require(tensor.dataType() == DataType.FLOAT32) {
                "$name must be Float32, got ${tensor.dataType()}"
            }
            val shape = tensor.shape()
            require(shape.size == 4 && shape[0] == 1) {
                "$name must be rank-4 NHWC, got ${shape.contentToString()}"
            }
            val bytes = directBuffer(tensor.numBytes())
            val elementCount = tensor.numElements()
            return OutputBuffer(
                name = name,
                shape = shape,
                bytes = bytes,
                floats = bytes.asFloatBuffer(),
                nhwcValues = FloatArray(elementCount),
                nchwValues = FloatArray(elementCount),
            )
        }

        private fun directBuffer(byteCount: Int): ByteBuffer =
            ByteBuffer.allocateDirect(byteCount).order(ByteOrder.nativeOrder())

        /// Extracts the model out of its install-time asset pack into `cacheDir`, where it can be
        /// memory-mapped. The reuse and extraction rules live in ``RTMDetModelCache`` so they can
        /// be unit tested without a device.
        private fun copyAssetToCache(context: Context, assetName: String): File {
            val output = File(context.cacheDir, assetName)
            val tokenFile = File(context.cacheDir, "$assetName.install-token")
            val installTime = context.packageManager
                .getPackageInfo(context.packageName, 0)
                .lastUpdateTime
            val currentToken = runCatching { tokenFile.takeIf(File::isFile)?.readText() }.getOrNull()

            if (RTMDetModelCache.isReusable(output, currentToken, installTime)) return output
            if (output.isFile) {
                LogUtil.w(
                    TAG,
                    "Cached '$assetName' is stale or incomplete (${output.length()} bytes); re-extracting",
                )
            }

            tokenFile.delete()
            val copiedBytes = RTMDetModelCache.extract(
                destination = output,
                temporary = File(context.cacheDir, "$assetName.tmp"),
            ) { context.assets.open(assetName) }
            tokenFile.writeText(RTMDetModelCache.buildToken(installTime, copiedBytes))
            LogUtil.i(TAG, "Extracted '$assetName' to cache ($copiedBytes bytes)")
            return output
        }

        private fun elapsedMillis(startNanos: Long): Long =
            TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos)

        private fun rootMessage(error: Throwable): String {
            var root = error
            while (root.cause != null && root.cause !== root) root = root.cause!!
            return root.message ?: root.javaClass.simpleName
        }
    }
}
