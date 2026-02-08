package com.furnit.android.services

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

/**
 * SHARP Gaussian Splatting using ExecuTorch component-mode inference.
 *
 * Uses sliding pyramid patches to avoid OOM:
 * 1. Load sharp_single_patch.pte (~1.1GB) - processes one 384x384 patch at a time
 * 2. Extract 35 patches (25 at 1x, 9 at 0.5x, 1 at 0.25x)
 * 3. Merge encoded features on CPU into [1024, 96, 96]
 * 4. Load sharp_gaussian_head.pte (~7MB) - lightweight decoder
 * 5. Produce [14, 384, 384] Gaussian parameters → 147,456 Gaussians
 */
class ExecutorchSharp private constructor(private val context: Context) {

    companion object {
        private const val TAG = "ExecutorchSharp"
        private const val IMAGE_SIZE = 1536
        private const val PATCH_SIZE = 384
        private const val FEATURE_DIM = 1024
        private const val SPATIAL_SIZE = 24 // 384/16
        private const val GAUSSIAN_CHANNELS = 14
        private const val OUTPUT_SPATIAL = 384

        // Sliding pyramid configuration
        private const val GRID_1X = 5
        private const val GRID_05X = 3
        private const val PATCHES_1X = 25  // 5x5
        private const val PATCHES_05X = 9  // 3x3
        private const val PATCHES_025X = 1
        private const val TOTAL_PATCHES = 35

        // Merge overlap padding (matching NCNN component mode)
        private const val PADDING_1X = 3
        private const val PADDING_05X = 6

        private const val PARAMS_PER_GAUSSIAN = 14
        private const val SH_C0 = 0.28209479177387814f
        private const val BYTES_PER_VERTEX = 62 * 4

        private const val LOGIT_LUT_SIZE = 1024
        private val LOGIT_LUT = FloatArray(LOGIT_LUT_SIZE) { i ->
            val p = (i.toFloat() / (LOGIT_LUT_SIZE - 1)).coerceIn(1e-4f, 1f - 1e-4f)
            ln(p / (1f - p))
        }

        private const val LN_LUT_SIZE = 2048
        private const val LN_LUT_MIN = 0.001f
        private const val LN_LUT_MAX = 5.0f
        private val LN_LUT_SCALE = (LN_LUT_SIZE - 1).toFloat() / (LN_LUT_MAX - LN_LUT_MIN)
        private val LN_LUT = FloatArray(LN_LUT_SIZE) { i ->
            val x = LN_LUT_MIN + (LN_LUT_MAX - LN_LUT_MIN) * i / (LN_LUT_SIZE - 1)
            ln(x)
        }

        private fun lnLut(x: Float): Float {
            if (x <= LN_LUT_MIN) return LN_LUT[0]
            if (x >= LN_LUT_MAX) return LN_LUT[LN_LUT_SIZE - 1]
            return LN_LUT[((x - LN_LUT_MIN) * LN_LUT_SCALE).toInt()]
        }

        // Prefer hybrid INT8+Vulkan/XNNPACK (smallest, fastest), then XNNPACK, then portable
        private val PATCH_ENCODER_FILENAMES = arrayOf(
            "sharp_single_patch_hybrid_standalone.pte",  // INT8 ~275MB, Vulkan+XNNPACK
            "sharp_single_patch_hybrid.pte",             // INT8 ~275MB, with .ptd separation
            "sharp_single_patch_xnnpack.pte",            // FP32 ~1.1GB, XNNPACK only
            "sharp_single_patch.pte",                    // FP32 ~1.1GB, portable fallback
        )
        private const val GAUSSIAN_HEAD_FILENAME = "sharp_gaussian_head.pte"

        private val EXTRA_SEARCH_DIRS = arrayOf(
            "/data/local/tmp/furnit/",
        )

        @Volatile
        private var instance: ExecutorchSharp? = null

        fun getInstance(context: Context): ExecutorchSharp {
            return instance ?: synchronized(this) {
                instance ?: ExecutorchSharp(context.applicationContext).also {
                    instance = it
                    Log.d(TAG, "ExecutorchSharp singleton created")
                }
            }
        }
    }

    private var isInitialized = false

    private val modelsDir: File by lazy {
        context.getExternalFilesDir("models") ?: File(context.filesDir, "models")
    }

    data class StreamingResult(
        val plyFile: File,
        val classicPlyFile: File,
        val gaussianCount: Int,
        val roomWidth: Float,
        val roomHeight: Float,
        val roomDepth: Float
    )

    private fun findFile(filename: String): File? {
        val inModelsDir = File(modelsDir, filename)
        if (inModelsDir.exists() && inModelsDir.length() > 0) return inModelsDir

        for (dir in EXTRA_SEARCH_DIRS) {
            val file = File(dir, filename)
            if (file.exists() && file.length() > 0) return file
        }
        return null
    }

    private fun findPatchEncoder(): File? {
        for (filename in PATCH_ENCODER_FILENAMES) {
            val file = findFile(filename)
            if (file != null) return file
        }
        return null
    }

    fun isModelReady(): Boolean {
        return findPatchEncoder() != null &&
               findFile(GAUSSIAN_HEAD_FILENAME) != null
    }

    fun initialize(): Boolean {
        val patchFile = findPatchEncoder()
        val headFile = findFile(GAUSSIAN_HEAD_FILENAME)

        if (patchFile == null) {
            Log.e(TAG, "Patch encoder not found (tried: ${PATCH_ENCODER_FILENAMES.joinToString()})")
            return false
        }
        if (headFile == null) {
            Log.e(TAG, "Gaussian head not found: $GAUSSIAN_HEAD_FILENAME")
            return false
        }

        Log.d(TAG, "Component models found:")
        Log.d(TAG, "  Patch encoder: ${patchFile.absolutePath} (${patchFile.length() / 1024 / 1024}MB)")
        Log.d(TAG, "  Gaussian head: ${headFile.absolutePath} (${headFile.length() / 1024}KB)")
        isInitialized = true
        return true
    }

    /**
     * Run component-mode SHARP inference.
     */
    suspend fun inferStreaming(
        bitmap: Bitmap,
        progressCallback: ((Float, String) -> Unit)? = null
    ): StreamingResult? = withContext(Dispatchers.IO) {
        if (!isInitialized) {
            Log.e(TAG, "Not initialized")
            return@withContext null
        }

        try {
            val startTime = System.currentTimeMillis()

            // Step 1: Scale input image to 1536x1536
            progressCallback?.invoke(0.02f, "Preprocessing image...")
            val scaledBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
            Log.d(TAG, "Image scaled to ${IMAGE_SIZE}x${IMAGE_SIZE}")

            // Step 2: Load patch encoder
            progressCallback?.invoke(0.05f, "Loading patch encoder...")
            val patchEncoderFile = findPatchEncoder()!!
            val patchEncoder = Module.load(patchEncoderFile.absolutePath)
            Log.d(TAG, "Patch encoder loaded")

            // Step 3: Extract and process 1x scale patches (5x5 grid)
            // Only 1x patches are used by gaussian head - skip 0.5x/0.25x for speed
            val patchFeatures1x = ArrayList<FloatArray>(PATCHES_1X)

            val stride1x = (IMAGE_SIZE - PATCH_SIZE) / (GRID_1X - 1) // 288
            for (i in 0 until GRID_1X) {
                for (j in 0 until GRID_1X) {
                    val patchIdx = i * GRID_1X + j
                    val y = i * stride1x
                    val x = j * stride1x
                    val patchBitmap = Bitmap.createBitmap(scaledBitmap, x, y, PATCH_SIZE, PATCH_SIZE)
                    val features = encodePatch(patchEncoder, patchBitmap)
                    patchBitmap.recycle()
                    patchFeatures1x.add(features)

                    val progress = 0.05f + (patchIdx.toFloat() / PATCHES_1X) * 0.50f
                    progressCallback?.invoke(progress, "Encoding patch ${patchIdx + 1}/$PATCHES_1X...")
                }
            }
            Log.d(TAG, "1x patches encoded: ${patchFeatures1x.size}")
            progressCallback?.invoke(0.55f, "All patches encoded")

            scaledBitmap.recycle()

            // Step 4: Unload patch encoder to free memory
            patchEncoder.destroy()
            Log.d(TAG, "Patch encoder released")
            System.gc()

            // Step 5: Merge features on CPU
            progressCallback?.invoke(0.58f, "Merging features...")
            val merged1x = mergePatchGrid(patchFeatures1x, GRID_1X, PADDING_1X)
            patchFeatures1x.clear()
            Log.d(TAG, "Merged 1x features: ${merged1x.size} floats")

            // Step 6: Load gaussian head
            progressCallback?.invoke(0.62f, "Loading Gaussian head...")
            val headFile = findFile(GAUSSIAN_HEAD_FILENAME)!!
            val gaussianHead = Module.load(headFile.absolutePath)
            Log.d(TAG, "Gaussian head loaded")

            // Step 7: Run gaussian head on merged features
            progressCallback?.invoke(0.65f, "Running Gaussian head...")
            val mergedSize = getMergedSize(GRID_1X, SPATIAL_SIZE, PADDING_1X)
            val mergedTensor = Tensor.fromBlob(
                merged1x,
                longArrayOf(1, FEATURE_DIM.toLong(), mergedSize.toLong(), mergedSize.toLong())
            )

            val headStart = System.currentTimeMillis()
            val headOutputs = gaussianHead.forward(EValue.from(mergedTensor))
            val headTime = System.currentTimeMillis() - headStart
            Log.d(TAG, "Gaussian head completed in ${headTime}ms")

            val headOutput = headOutputs[0].toTensor().getDataAsFloatArray()

            // Step 8: Release gaussian head
            gaussianHead.destroy()
            Log.d(TAG, "Gaussian head released")

            // Step 9: Extract Gaussians from [1, 14, 384, 384] output
            progressCallback?.invoke(0.70f, "Extracting Gaussians...")
            val gaussianCount = OUTPUT_SPATIAL * OUTPUT_SPATIAL // 147,456
            val params = extractGaussians(headOutput, OUTPUT_SPATIAL, OUTPUT_SPATIAL)
            Log.d(TAG, "Extracted $gaussianCount Gaussians")

            // Step 10: Write PLY
            progressCallback?.invoke(0.75f, "Writing PLY ($gaussianCount Gaussians)...")
            val result = writeStreamingPlyPacked(gaussianCount, params, progressCallback)

            val elapsed = System.currentTimeMillis() - startTime
            Log.d(TAG, "ExecuTorch component-mode SHARP completed: $gaussianCount Gaussians in ${elapsed}ms")

            progressCallback?.invoke(1.0f, "Done!")
            return@withContext result

        } catch (e: Exception) {
            Log.e(TAG, "ExecuTorch SHARP component inference failed", e)
            progressCallback?.invoke(0f, "Error: ${e.message}")
            return@withContext null
        }
    }

    /**
     * Encode a single 384x384 patch through the patch encoder.
     * Input: [1, 3, 384, 384] → Output: [1, 1024, 24, 24]
     * Returns flattened [1024, 24, 24] features as FloatArray.
     */
    private fun encodePatch(module: Module, patchBitmap: Bitmap): FloatArray {
        val inputData = preprocessPatch(patchBitmap)
        val inputTensor = Tensor.fromBlob(
            inputData,
            longArrayOf(1, 3, PATCH_SIZE.toLong(), PATCH_SIZE.toLong())
        )
        val outputs = module.forward(EValue.from(inputTensor))
        return outputs[0].toTensor().getDataAsFloatArray()
    }

    /**
     * Preprocess a 384x384 patch to CHW FloatArray normalized [0, 1].
     */
    private fun preprocessPatch(bitmap: Bitmap): FloatArray {
        val width = bitmap.width
        val height = bitmap.height
        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        val floatArray = FloatArray(3 * width * height)
        val channelSize = width * height

        for (i in pixels.indices) {
            floatArray[i] = ((pixels[i] shr 16) and 0xFF) / 255f
            floatArray[channelSize + i] = ((pixels[i] shr 8) and 0xFF) / 255f
            floatArray[2 * channelSize + i] = (pixels[i] and 0xFF) / 255f
        }
        return floatArray
    }

    /**
     * Compute merged grid output size.
     * First patch contributes full spatial size, subsequent patches contribute (spatial - 2*padding).
     */
    private fun getMergedSize(gridSize: Int, patchSpatial: Int, padding: Int): Int {
        val patchContrib = patchSpatial - 2 * padding
        return patchSpatial + (gridSize - 1) * patchContrib
    }

    /**
     * Merge patch features into a single spatial feature map.
     * Each patch is [1024, 24, 24] stored as a flat FloatArray in CHW order.
     * Output: [1024, outH, outW] stored as a flat FloatArray.
     *
     * Matching NCNN mergePatchGrid: uses overlap padding to blend edges.
     */
    private fun mergePatchGrid(
        patches: List<FloatArray>,
        gridSize: Int,
        padding: Int
    ): FloatArray {
        val patchH = SPATIAL_SIZE
        val patchW = SPATIAL_SIZE
        val outSize = getMergedSize(gridSize, patchH, padding)

        // Output: [FEATURE_DIM, outSize, outSize] in CHW format
        val output = FloatArray(FEATURE_DIM * outSize * outSize)

        var idx = 0
        var outY = 0

        for (gridJ in 0 until gridSize) {
            val srcY0 = if (gridJ == 0) 0 else padding
            val srcY1 = if (gridJ == gridSize - 1) patchH else (patchH - padding)
            val copyH = srcY1 - srcY0

            var outX = 0
            for (gridI in 0 until gridSize) {
                val patch = patches[idx++]
                val srcX0 = if (gridI == 0) 0 else padding
                val srcX1 = if (gridI == gridSize - 1) patchW else (patchW - padding)
                val copyW = srcX1 - srcX0

                // Copy region from patch to output for each channel
                for (c in 0 until FEATURE_DIM) {
                    val srcBase = c * patchH * patchW
                    val dstBase = c * outSize * outSize

                    for (dy in 0 until copyH) {
                        for (dx in 0 until copyW) {
                            val srcIdx = srcBase + (srcY0 + dy) * patchW + (srcX0 + dx)
                            val dstIdx = dstBase + (outY + dy) * outSize + (outX + dx)
                            output[dstIdx] = patch[srcIdx]
                        }
                    }
                }
                outX += copyW
            }
            outY += copyH
        }

        Log.d(TAG, "Merged ${patches.size} patches into [${FEATURE_DIM}, $outSize, $outSize]")
        return output
    }

    /**
     * Extract Gaussians from GaussianHead output [1, 14, H, W].
     *
     * Channel layout:
     *   [0-2]: xyz position (raw values)
     *   [3]: opacity (apply sigmoid)
     *   [4-6]: scale (clamp to min)
     *   [7-10]: rotation quaternion (normalize)
     *   [11-13]: RGB color (clamp to [0,1])
     *
     * Output: packed [N, 14] in format: pos(3), scale(3), rot(4), opacity(1), color(3)
     */
    private fun extractGaussians(headOutput: FloatArray, outH: Int, outW: Int): FloatArray {
        val numGaussians = outH * outW
        val channelStride = outH * outW
        val params = FloatArray(numGaussians * PARAMS_PER_GAUSSIAN)

        for (pixIdx in 0 until numGaussians) {
            val offset = pixIdx * PARAMS_PER_GAUSSIAN

            // Position (channels 0-2, raw)
            params[offset + 0] = headOutput[0 * channelStride + pixIdx]
            params[offset + 1] = headOutput[1 * channelStride + pixIdx]
            params[offset + 2] = headOutput[2 * channelStride + pixIdx]

            // Scale (channels 4-6, clamp to min 0.001)
            params[offset + 3] = max(0.001f, headOutput[4 * channelStride + pixIdx])
            params[offset + 4] = max(0.001f, headOutput[5 * channelStride + pixIdx])
            params[offset + 5] = max(0.001f, headOutput[6 * channelStride + pixIdx])

            // Rotation (channels 7-10, normalize quaternion)
            val qw = headOutput[7 * channelStride + pixIdx]
            val qx = headOutput[8 * channelStride + pixIdx]
            val qy = headOutput[9 * channelStride + pixIdx]
            val qz = headOutput[10 * channelStride + pixIdx]
            val qnorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
            val invNorm = if (qnorm > 1e-6f) 1f / qnorm else 1f
            params[offset + 6] = qw * invNorm
            params[offset + 7] = qx * invNorm
            params[offset + 8] = qy * invNorm
            params[offset + 9] = qz * invNorm

            // Opacity (channel 3, apply sigmoid)
            val rawOpacity = headOutput[3 * channelStride + pixIdx]
            params[offset + 10] = 1f / (1f + kotlin.math.exp(-rawOpacity))

            // Color (channels 11-13, clamp to [0,1])
            params[offset + 11] = headOutput[11 * channelStride + pixIdx].coerceIn(0f, 1f)
            params[offset + 12] = headOutput[12 * channelStride + pixIdx].coerceIn(0f, 1f)
            params[offset + 13] = headOutput[13 * channelStride + pixIdx].coerceIn(0f, 1f)
        }

        return params
    }

    /**
     * Write packed [N, 14] Gaussian output to PLY file.
     * Layout per gaussian: pos(3), scale(3), rot(4), opacity(1), color(3)
     */
    private fun writeStreamingPlyPacked(
        gaussianCount: Int,
        params: FloatArray,
        progressCallback: ((Float, String) -> Unit)?
    ): StreamingResult {
        val roomsDir = File(context.filesDir, "sharp_rooms")
        roomsDir.mkdirs()

        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val roomFolder = File(roomsDir, "room_$timestamp")
        roomFolder.mkdirs()

        val plyFile = File(roomFolder, "room.ply")
        val classicPlyFile = File(roomFolder, "room_classic.ply")

        var minX = Float.MAX_VALUE; var maxX = -Float.MAX_VALUE
        var minY = Float.MAX_VALUE; var maxY = -Float.MAX_VALUE
        var minZ = Float.MAX_VALUE; var maxZ = -Float.MAX_VALUE

        val header = buildPlyHeader(gaussianCount)

        FileOutputStream(plyFile).use { fos ->
            val channel = fos.channel

            val headerBytes = header.toByteArray(Charsets.UTF_8)
            val headerBuffer = ByteBuffer.allocateDirect(headerBytes.size)
            headerBuffer.put(headerBytes)
            headerBuffer.flip()
            channel.write(headerBuffer)

            val batchSize = 512
            val batchBuffer = ByteBuffer.allocateDirect(BYTES_PER_VERTEX * batchSize)
            batchBuffer.order(ByteOrder.LITTLE_ENDIAN)
            val scaleBoost = 1.3f
            val minScale = 0.001f
            val lutScale = (LOGIT_LUT_SIZE - 1).toFloat()

            val progressEvery = max(1, gaussianCount / 10)
            var processed = 0

            while (processed < gaussianCount) {
                val currentBatch = minOf(batchSize, gaussianCount - processed)

                for (j in 0 until currentBatch) {
                    val offset = (processed + j) * PARAMS_PER_GAUSSIAN

                    val x = params[offset + 0]
                    val y = -params[offset + 1]
                    val z = -params[offset + 2]

                    if (x < minX) minX = x; if (x > maxX) maxX = x
                    if (y < minY) minY = y; if (y > maxY) maxY = y
                    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z

                    batchBuffer.putFloat(x)
                    batchBuffer.putFloat(y)
                    batchBuffer.putFloat(z)

                    // Normals
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)
                    batchBuffer.putFloat(0f)

                    // Colors (at offset 11-13) -> SH DC
                    val r = params[offset + 11].coerceIn(0f, 1f)
                    val g = params[offset + 12].coerceIn(0f, 1f)
                    val b = params[offset + 13].coerceIn(0f, 1f)
                    batchBuffer.putFloat((r - 0.5f) / SH_C0)
                    batchBuffer.putFloat((g - 0.5f) / SH_C0)
                    batchBuffer.putFloat((b - 0.5f) / SH_C0)

                    // Higher order SH (45 zeros)
                    repeat(45) { batchBuffer.putFloat(0f) }

                    // Opacity (at offset 10) -> logit via LUT
                    val rawOpacity = params[offset + 10].coerceIn(0f, 1f)
                    val lutIndex = (rawOpacity * lutScale).toInt().coerceIn(0, LOGIT_LUT_SIZE - 1)
                    batchBuffer.putFloat(LOGIT_LUT[lutIndex])

                    // Scale (at offset 3-5) -> log via LN LUT
                    batchBuffer.putFloat(lnLut(max(params[offset + 3] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 4] * scaleBoost, minScale)))
                    batchBuffer.putFloat(lnLut(max(params[offset + 5] * scaleBoost, minScale)))

                    // Rotation (at offset 6-9) -> normalize
                    val rw = params[offset + 6]
                    val rx = params[offset + 7]
                    val ry = params[offset + 8]
                    val rz = params[offset + 9]
                    val mag = sqrt(rw * rw + rx * rx + ry * ry + rz * rz)
                    val invMag = if (mag > 1e-8f) 1f / mag else 1f
                    batchBuffer.putFloat(rw * invMag)
                    batchBuffer.putFloat(rx * invMag)
                    batchBuffer.putFloat(ry * invMag)
                    batchBuffer.putFloat(rz * invMag)
                }

                batchBuffer.flip()
                batchBuffer.limit(currentBatch * BYTES_PER_VERTEX)
                while (batchBuffer.hasRemaining()) {
                    channel.write(batchBuffer)
                }
                batchBuffer.clear()

                processed += currentBatch
                if (processed % progressEvery == 0 || processed == gaussianCount) {
                    val progress = 0.75f + (processed.toFloat() / gaussianCount) * 0.20f
                    progressCallback?.invoke(progress, "Writing PLY ($processed/$gaussianCount)...")
                }
            }
        }

        plyFile.copyTo(classicPlyFile, overwrite = true)

        Log.d(TAG, "PLY written: $gaussianCount Gaussians")
        Log.d(TAG, "Room bounds: ${maxX - minX}m x ${maxY - minY}m x ${maxZ - minZ}m")

        return StreamingResult(
            plyFile = plyFile,
            classicPlyFile = classicPlyFile,
            gaussianCount = gaussianCount,
            roomWidth = maxX - minX,
            roomHeight = maxY - minY,
            roomDepth = maxZ - minZ
        )
    }

    private fun buildPlyHeader(gaussianCount: Int): String {
        return buildString {
            append("ply\n")
            append("format binary_little_endian 1.0\n")
            append("element vertex $gaussianCount\n")
            append("property float x\n")
            append("property float y\n")
            append("property float z\n")
            append("property float nx\n")
            append("property float ny\n")
            append("property float nz\n")
            for (i in 0 until 3) append("property float f_dc_$i\n")
            for (i in 0 until 45) append("property float f_rest_$i\n")
            append("property float opacity\n")
            append("property float scale_0\n")
            append("property float scale_1\n")
            append("property float scale_2\n")
            append("property float rot_0\n")
            append("property float rot_1\n")
            append("property float rot_2\n")
            append("property float rot_3\n")
            append("end_header\n")
        }
    }

    fun release() {
        isInitialized = false
        Log.d(TAG, "ExecutorchSharp released")
    }
}
