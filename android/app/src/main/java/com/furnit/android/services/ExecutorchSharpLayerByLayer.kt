package com.furnit.android.services

import android.content.Context
import android.os.Debug
import com.furnit.android.utils.LogUtil
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Memory-optimized SHARP inference using layer-by-layer execution with:
 * NOTE: Requires a .pte that exposes named parameters (layers.*.attn, layers.*.mlp, etc.)
 * via getParameterAsFloatArray. The standard Furnit SHARP export is a single forward graph
 * and may not support this. Use for testing with compatible exports.
 *
 * - ScratchBuffers: pre-allocated fixed set, reused every layer, zero new heap
 * - ActivationPool: reusable buffer pool capped at scratchPoolBytes (256 MB) for attention tiles
 * - ChunkedAttention: O(S·chunk) instead of O(S²)
 * - LayerNorm, SharpMLP with in-place / buffer reuse
 */
data class SharpConfig(
    val hiddenDim: Int = 1024,
    val mlpDim: Int = 4096,
    val numHeads: Int = 16,
    val numLayers: Int = 24,
    val maxSeqLen: Int = 2048,
    val attentionChunkSize: Int = 256,
    val useFp16Activations: Boolean = true,
    val maxVertices: Int = 500_000,
    val scratchPoolBytes: Long = 256L * 1024 * 1024, // 256 MB
    val heapPressureThreshold: Float = 0.75f,  // GC when heap > 75% of max
    val plyOutputDir: String = "sharp_output",
    val webViewerPort: Int = 8080,
) {
    val headDim: Int get() = hiddenDim / numHeads
    val bytesPerElement: Int get() = if (useFp16Activations) 2 else 4
}

/**
 * Pre-allocated scratch buffers - FIXED set reused every layer, zero new heap.
 */
class ScratchBuffers(seqLen: Int, private val cfg: SharpConfig) {
    val hiddenA = FloatArray(seqLen * cfg.hiddenDim) // normed / q
    val hiddenB = FloatArray(seqLen * cfg.hiddenDim) // k / projected
    val hiddenC = FloatArray(seqLen * cfg.hiddenDim) // v / normed2
    val hiddenD = FloatArray(seqLen * cfg.hiddenDim) // attnout / mlpout
    val hiddenE = FloatArray(seqLen * cfg.hiddenDim) // residual accumulator (layer input -> output)
    val mlpBuf = FloatArray(seqLen * cfg.mlpDim)     // 4096-wide MLP intermediate

    fun totalBytes(): Long = (5L * hiddenA.size + 1L * mlpBuf.size) * 4

    fun zero(buf: FloatArray) { buf.fill(0f) }
}

/**
 * Reusable off-heap pool for attention score tiles.
 */
class ActivationPool(private val cfg: SharpConfig) {
    private val pool = ConcurrentLinkedQueue<FloatBuffer>()
    @Volatile
    private var allocated = 0L

    fun acquire(elements: Int): FloatBuffer {
        var buf = pool.poll()
        if (buf != null && buf.capacity() >= elements) {
            buf.clear()
            buf.limit(elements)
            return buf
        }
        val bytes = elements.toLong() * 4
        check(allocated + bytes <= cfg.scratchPoolBytes) {
            "Activation pool exceeded ${cfg.scratchPoolBytes / 1024 / 1024} MB " +
                "(requested ${bytes / 1024} KB, already allocated ${allocated / 1024} KB)"
        }
        allocated += bytes
        buf = ByteBuffer.allocateDirect(elements * 4)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        return buf
    }

    fun release(buf: FloatBuffer) {
        pool.offer(buf)
    }

    fun reset() {
        pool.clear()
        allocated = 0L
    }

    fun usedBytes(): Long = allocated
}

/**
 * Chunked multi-head attention - O(S·chunk) instead of O(S²).
 * Writes into pre-allocated out array.
 */
class ChunkedAttention(private val cfg: SharpConfig, private val pool: ActivationPool) {
    private fun computeScores(
        q: FloatArray, k: FloatArray, scores: FloatBuffer,
        qStart: Int, qLen: Int, kvLen: Int, headOff: Int, scale: Float
    ) {
        scores.clear()
        for (i in 0 until qLen) {
            val qRow = qStart + i
            for (j in 0 until kvLen) {
                var dot = 0f
                for (d in 0 until cfg.headDim) {
                    dot += q[qRow * cfg.hiddenDim + headOff + d] * k[j * cfg.hiddenDim + headOff + d]
                }
                scores.put(i * kvLen + j, dot * scale)
            }
        }
    }

    private fun softmaxRows(scores: FloatBuffer, rows: Int, cols: Int) {
        for (i in 0 until rows) {
            var maxVal = Float.NEGATIVE_INFINITY
            for (j in 0 until cols) maxVal = max(maxVal, scores.get(i * cols + j))
            var sum = 0f
            for (j in 0 until cols) {
                val v = exp((scores.get(i * cols + j) - maxVal).toDouble()).toFloat()
                scores.put(i * cols + j, v)
                sum += v
            }
            if (sum > 0f) {
                val inv = 1f / sum
                for (j in 0 until cols) {
                    scores.put(i * cols + j, scores.get(i * cols + j) * inv)
                }
            }
        }
    }

    private fun accumulateValues(
        scores: FloatBuffer, v: FloatArray, out: FloatArray,
        qStart: Int, qLen: Int, kvLen: Int, headOff: Int
    ) {
        for (i in 0 until qLen) {
            val outRow = qStart + i
            val scoreRowOff = i * kvLen
            for (d in 0 until cfg.headDim) {
                var acc = 0f
                for (j in 0 until kvLen) {
                    acc += scores.get(scoreRowOff + j) * v[j * cfg.hiddenDim + headOff + d]
                }
                out[outRow * cfg.hiddenDim + headOff + d] = acc
            }
        }
    }

    fun forward(
        query: FloatArray,
        key: FloatArray,
        value: FloatArray,
        out: FloatArray,
        seqLen: Int
    ) {
        val scale = 1.0f / sqrt(cfg.headDim.toFloat())
        val chunkSize = cfg.attentionChunkSize
        for (h in 0 until cfg.numHeads) {
            val headOff = h * cfg.headDim
            val scoresBuf = pool.acquire(chunkSize * seqLen)
            for (qStart in 0 until seqLen step chunkSize) {
                val qLen = min(chunkSize, seqLen - qStart)
                computeScores(query, key, scoresBuf, qStart, qLen, seqLen, headOff, scale)
                softmaxRows(scoresBuf, qLen, seqLen)
                accumulateValues(scoresBuf, value, out, qStart, qLen, seqLen, headOff)
            }
            pool.release(scoresBuf)
        }
    }
}

/**
 * MLP with buffer reuse - avoids double-allocating the 4096-wide intermediate.
 */
class SharpMLP(private val cfg: SharpConfig) {
    private fun gelu(x: Float): Float {
        val cdf = 0.5f * (1f + tanh(sqrt(2f / Math.PI.toFloat()) * (x + 0.044715f * x * x * x)))
        return x * cdf
    }

    private fun tanh(x: Float): Float {
        val e2x = exp((2.0 * x).coerceIn(-20.0, 20.0)).toFloat()
        return (e2x - 1f) / (e2x + 1f)
    }

    fun forward(
        input: FloatArray,
        weightsUp: FloatArray,
        weightsDown: FloatArray,
        biasUp: FloatArray?,
        biasDown: FloatArray?,
        seqLen: Int,
        mlpScratch: FloatArray,
        output: FloatArray
    ) {
        for (s in 0 until seqLen) {
            val inBase = s * cfg.hiddenDim
            val midBase = s * cfg.mlpDim
            for (m in 0 until cfg.mlpDim) {
                var acc = biasUp?.get(m) ?: 0f
                for (h in 0 until cfg.hiddenDim) {
                    acc += input[inBase + h] * weightsUp[h * cfg.mlpDim + m]
                }
                mlpScratch[midBase + m] = gelu(acc)
            }
        }
        for (s in 0 until seqLen) {
            val midBase = s * cfg.mlpDim
            val outBase = s * cfg.hiddenDim
            for (h in 0 until cfg.hiddenDim) {
                var acc = biasDown?.get(h) ?: 0f
                for (m in 0 until cfg.mlpDim) {
                    acc += mlpScratch[midBase + m] * weightsDown[m * cfg.hiddenDim + h]
                }
                output[outBase + h] = acc
            }
        }
    }
}

/**
 * LayerNorm with in-place residual add.
 */
object LayerNorm {
    fun forwardWithResidual(
        input: FloatArray,
        residual: FloatArray?,
        gamma: FloatArray,
        beta: FloatArray,
        dim: Int,
        seqLen: Int,
        output: FloatArray,
        eps: Float = 1e-5f
    ) {
        for (s in 0 until seqLen) {
            val off = s * dim
            var mean = 0f
            for (d in 0 until dim) {
                mean += input[off + d] + (residual?.get(off + d) ?: 0f)
            }
            mean /= dim
            var variance = 0f
            for (d in 0 until dim) {
                val centered = input[off + d] + (residual?.get(off + d) ?: 0f) - mean
                variance += centered * centered
            }
            val invStd = 1f / sqrt(variance / dim + eps)
            for (d in 0 until dim) {
                val x = input[off + d] + (residual?.get(off + d) ?: 0f)
                output[off + d] = gamma[d] * ((x - mean) * invStd) + beta[d]
            }
        }
    }
}

/**
 * Main inference pipeline - loads .pte, runs layer-by-layer with buffer reuse.
 */
class SharpInferencePipeline(
    private val context: Context,
    private val cfg: SharpConfig = SharpConfig()
) {
    companion object {
        private const val TAG = "SharpPipeline"
    }

    private var module: Module? = null
    private val pool = ActivationPool(cfg)
    private val attention = ChunkedAttention(cfg, pool)
    private val mlp = SharpMLP(cfg)

    fun loadModel(ptePath: String) {
        val modelFile = if (File(ptePath).exists()) File(ptePath) else assetToFile(ptePath)
        // LOAD_MODE_MMAP: OS pages in only current layer's weights; cold pages evicted.
        module = Module.load(modelFile.absolutePath, Module.LOAD_MODE_MMAP)
        LogUtil.i(TAG, "Loaded SHARP model (mmap) from $ptePath (pool capacity ${cfg.scratchPoolBytes / 1024 / 1024}MB)")
    }

    private fun assetToFile(assetName: String): File {
        val outFile = File(context.cacheDir, assetName)
        if (!outFile.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(outFile).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return outFile
    }

    fun runInference(inputImage: FloatArray, seqLen: Int): SharpOutput {
        val mod = module ?: throw IllegalStateException("Call loadModel first")
        pool.reset()
        val scratch = ScratchBuffers(seqLen, cfg)
        LogUtil.i(TAG, "Scratch allocated: ${scratch.totalBytes() / 1024 / 1024} MB (heap free: ${heapFreeMB()} MB)")
        val t0 = System.nanoTime()

        val inputTensor = Tensor.fromBlob(inputImage, longArrayOf(1, seqLen.toLong(), cfg.hiddenDim.toLong()))
        val encoderOut = mod.forward(EValue.from(inputTensor))
        @Suppress("UNCHECKED_CAST")
        var hidden = encoderOut[0].toTensor().getDataAsFloatArray()

        // Copy encoder output into hiddenE (ping-pong buffer), allow encoder tensor to be freed
        System.arraycopy(hidden, 0, scratch.hiddenE, 0, hidden.size)
        @Suppress("UNUSED_VALUE")
        hidden = scratch.hiddenE

        for (layer in 0 until cfg.numLayers) {
            runTransformerLayer(mod, scratch, seqLen, layer)
            // gcIfPressured every 4 layers to avoid GC thrash (not every layer)
            if ((layer + 1) % 4 == 0) gcIfPressured()
            if ((layer + 1) % 6 == 0) logMemory(layer)
            LogUtil.d(TAG, "Layer $layer/${cfg.numLayers} - heap free: ${heapFreeMB()} MB")
        }

        val numVerts = min(seqLen, cfg.maxVertices)
        val vertices = FloatArray(numVerts * 3)
        val normals = FloatArray(numVerts * 3)
        val colors = FloatArray(numVerts * 3)
        decodeGeometry(scratch.hiddenE, numVerts, vertices, normals, colors)

        val elapsed = (System.nanoTime() - t0) / 1_000_000
        LogUtil.i(TAG, "Inference done in ${elapsed}ms - $numVerts vertices")

        return SharpOutput(vertices, normals, colors, numVerts)
    }

    /**
     * Single transformer layer - ALL writes go into pre-allocated scratch.
     * Buffer roles: hiddenE = layer input; hiddenA = normed->Q; hiddenB = k/proj; hiddenC = v;
     * hiddenD = attnout/mlpout; mlpBuf = MLP intermediate; hiddenE = final output.
     */
    private fun runTransformerLayer(mod: Module, s: ScratchBuffers, seqLen: Int, layerIdx: Int) {
        val lp = "layers.$layerIdx"

        // Pre-attention LayerNorm: hiddenE -> hiddenA
        var gamma = loadParam(mod, "$lp.ln1.weight")
        var beta = loadParam(mod, "$lp.ln1.bias")
        LayerNorm.forwardWithResidual(s.hiddenE, null, gamma, beta, cfg.hiddenDim, seqLen, s.hiddenA)

        // Q,K,V projections from normed(hiddenA) into separate targets
        var w = loadParam(mod, "$lp.attn.q_proj.weight")
        linearProjectInto(s.hiddenA, w, seqLen, cfg.hiddenDim, cfg.hiddenDim, s.hiddenB)
        w = loadParam(mod, "$lp.attn.k_proj.weight")
        linearProjectInto(s.hiddenA, w, seqLen, cfg.hiddenDim, cfg.hiddenDim, s.hiddenC)
        w = loadParam(mod, "$lp.attn.v_proj.weight")
        linearProjectInto(s.hiddenA, w, seqLen, cfg.hiddenDim, cfg.hiddenDim, s.hiddenD)

        // Chunked attention: Q(hiddenB), K(hiddenC), V(hiddenD) -> hiddenA
        attention.forward(s.hiddenB, s.hiddenC, s.hiddenD, s.hiddenA, seqLen)

        // Output projection: hiddenA -> hiddenB
        w = loadParam(mod, "$lp.attn.out_proj.weight")
        linearProjectInto(s.hiddenA, w, seqLen, cfg.hiddenDim, cfg.hiddenDim, s.hiddenB)

        // Residual: hiddenE += hiddenB (post-attention)
        addResidualInPlace(s.hiddenE, s.hiddenB)

        // Post-attention LayerNorm -> hiddenA
        gamma = loadParam(mod, "$lp.ln2.weight")
        beta = loadParam(mod, "$lp.ln2.bias")
        LayerNorm.forwardWithResidual(s.hiddenE, null, gamma, beta, cfg.hiddenDim, seqLen, s.hiddenA)

        // MLP: hiddenA -> mlpBuf -> hiddenB
        val wUp = loadParam(mod, "$lp.mlp.up.weight")
        val wDown = loadParam(mod, "$lp.mlp.down.weight")
        val bUp = loadParamOrNull(mod, "$lp.mlp.up.bias")
        val bDown = loadParamOrNull(mod, "$lp.mlp.down.bias")
        mlp.forward(s.hiddenA, wUp, wDown, bUp, bDown, seqLen, s.mlpBuf, s.hiddenB)

        // Residual: hiddenE += hiddenB (post-MLP)
        addResidualInPlace(s.hiddenE, s.hiddenB)
    }

    private fun linearProjectInto(
        input: FloatArray,
        weight: FloatArray,
        seqLen: Int,
        inDim: Int,
        outDim: Int,
        output: FloatArray
    ) {
        for (s in 0 until seqLen) {
            val inBase = s * inDim
            val outBase = s * outDim
            for (o in 0 until outDim) {
                var acc = 0f
                for (i in 0 until inDim) {
                    acc += input[inBase + i] * weight[i * outDim + o]
                }
                output[outBase + o] = acc
            }
        }
    }

    private fun addResidualInPlace(accumulator: FloatArray, delta: FloatArray) {
        for (i in accumulator.indices) {
            accumulator[i] += delta[i]
        }
    }

    private fun loadParam(mod: Module, name: String): FloatArray {
        return try {
            val method = mod.javaClass.getMethod("getParameterAsFloatArray", String::class.java)
            method.invoke(mod, name) as FloatArray
        } catch (e: Exception) {
            throw IllegalStateException("Failed to load param: $name", e)
        }
    }

    private fun loadParamOrNull(mod: Module, name: String): FloatArray? {
        return try {
            loadParam(mod, name)
        } catch (_: Exception) {
            null
        }
    }

    private fun gcIfPressured() {
        val rt = Runtime.getRuntime()
        val used = rt.totalMemory() - rt.freeMemory()
        val max = rt.maxMemory()
        if (used.toFloat() / max > cfg.heapPressureThreshold) {
            LogUtil.w(TAG, "Heap at ${used / 1024 / 1024}/${max / 1024 / 1024} MB - forcing GC")
            System.gc()
        }
    }

    private fun heapFreeMB(): Long {
        val rt = Runtime.getRuntime()
        return (rt.maxMemory() - (rt.totalMemory() - rt.freeMemory())) / 1024 / 1024
    }

    /** Log heap, native heap, PSS every 6 layers to trace RSS. */
    private fun logMemory(layer: Int) {
        val rt = Runtime.getRuntime()
        val used = rt.totalMemory() - rt.freeMemory()
        val max = rt.maxMemory()
        val nativeHeap = try { Debug.getNativeHeapAllocatedSize() } catch (_: Exception) { -1L }
        val pss = try {
            android.os.Debug.getPss()  // Process statm PSS in bytes
        } catch (_: Exception) { -1L }
        LogUtil.i(TAG, "Memory layer=$layer heap=${used / 1024 / 1024}/${max / 1024 / 1024}MB " +
            "native=${nativeHeap / 1024 / 1024}MB PSS=${pss / 1024}MB")
    }

    private fun decodeGeometry(
        hidden: FloatArray,
        numVerts: Int,
        vertices: FloatArray,
        normals: FloatArray,
        colors: FloatArray
    ) {
        for (i in 0 until numVerts) {
            val base = i * cfg.hiddenDim
            vertices[i * 3 + 0] = hidden[base + 0]
            vertices[i * 3 + 1] = hidden[base + 1]
            vertices[i * 3 + 2] = hidden[base + 2]
            val nx = hidden[base + 3]
            val ny = hidden[base + 4]
            val nz = hidden[base + 5]
            val len = sqrt(nx * nx + ny * ny + nz * nz).coerceAtLeast(1e-8f)
            normals[i * 3 + 0] = nx / len
            normals[i * 3 + 1] = ny / len
            normals[i * 3 + 2] = nz / len
            colors[i * 3 + 0] = sigmoid(hidden[base + 6])
            colors[i * 3 + 1] = sigmoid(hidden[base + 7])
            colors[i * 3 + 2] = sigmoid(hidden[base + 8])
        }
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + exp(-x.toDouble()).toFloat())

    fun release() {
        module?.destroy()
        module = null
        pool.reset()
    }
}

/** Model output container. */
data class SharpOutput(
    val vertices: FloatArray,
    val normals: FloatArray,
    val colors: FloatArray,
    val numVertices: Int
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as SharpOutput
        return vertices.contentEquals(other.vertices) &&
            normals.contentEquals(other.normals) &&
            colors.contentEquals(other.colors) &&
            numVertices == other.numVertices
    }

    override fun hashCode(): Int {
        var result = vertices.contentHashCode()
        result = 31 * result + normals.contentHashCode()
        result = 31 * result + colors.contentHashCode()
        result = 31 * result + numVertices
        return result
    }
}

/**
 * PLY file generator - binary little-endian for speed, ASCII fallback.
 */
class PlyGenerator(private val cfg: SharpConfig) {
    fun writeBinaryPly(output: SharpOutput, outFile: File) {
        val nv = output.numVertices
        val header = buildString {
            appendLine("ply")
            appendLine("format binary_little_endian 1.0")
            appendLine("element vertex $nv")
            appendLine("property float x")
            appendLine("property float y")
            appendLine("property float z")
            appendLine("property float nx")
            appendLine("property float ny")
            appendLine("property float nz")
            appendLine("property uchar red")
            appendLine("property uchar green")
            appendLine("property uchar blue")
            appendLine("end_header")
        }
        FileOutputStream(outFile).buffered().use { out ->
            out.write(header.toByteArray(Charsets.US_ASCII))
            val buf = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN)
            fun writeFloat(v: Float) {
                buf.clear()
                buf.putFloat(v)
                out.write(buf.array())
            }
            for (i in 0 until nv) {
                writeFloat(output.vertices[i * 3 + 0])
                writeFloat(output.vertices[i * 3 + 1])
                writeFloat(output.vertices[i * 3 + 2])
                writeFloat(output.normals[i * 3 + 0])
                writeFloat(output.normals[i * 3 + 1])
                writeFloat(output.normals[i * 3 + 2])
                out.write((output.colors[i * 3 + 0] * 255).toInt().coerceIn(0, 255))
                out.write((output.colors[i * 3 + 1] * 255).toInt().coerceIn(0, 255))
                out.write((output.colors[i * 3 + 2] * 255).toInt().coerceIn(0, 255))
            }
        }
        LogUtil.i("PlyGenerator", "Binary PLY written: ${outFile.absolutePath} ($nv vertices)")
    }

    fun writeAsciiPly(output: SharpOutput, outFile: File) {
        val nv = output.numVertices
        BufferedWriter(FileWriter(outFile)).use { w ->
            w.write("ply\n")
            w.write("format ascii 1.0\n")
            w.write("element vertex $nv\n")
            w.write("property float x\nproperty float y\nproperty float z\n")
            w.write("property float nx\nproperty float ny\nproperty float nz\n")
            w.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            w.write("end_header\n")
            for (i in 0 until nv) {
                val r = (output.colors[i * 3 + 0] * 255).toInt().coerceIn(0, 255)
                val g = (output.colors[i * 3 + 1] * 255).toInt().coerceIn(0, 255)
                val b = (output.colors[i * 3 + 2] * 255).toInt().coerceIn(0, 255)
                w.write("${output.vertices[i * 3 + 0]} ${output.vertices[i * 3 + 1]} ${output.vertices[i * 3 + 2]} ")
                w.write("${output.normals[i * 3 + 0]} ${output.normals[i * 3 + 1]} ${output.normals[i * 3 + 2]} ")
                w.write("$r $g $b\n")
            }
        }
    }

    fun generateToDir(context: Context, output: SharpOutput): File {
        val dir = context.getExternalFilesDir(null)?.let { File(it, cfg.plyOutputDir) }
            ?: File(context.filesDir, cfg.plyOutputDir)
        dir.mkdirs()
        val plyFile = File(dir, "sharp_${System.currentTimeMillis()}.ply")
        writeBinaryPly(output, plyFile)
        return plyFile
    }
}

/**
 * WebGL / Three.js viewer - generates HTML that loads PLY.
 */
class WebGLViewer(private val cfg: SharpConfig) {
    fun generateViewerHtml(plyFileName: String): String = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>SHARP 3D Viewer</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { background: #0a0a0f; overflow: hidden; font-family: system-ui, sans-serif; }
canvas { display: block; }
#hud { position: fixed; top: 16px; left: 16px; color: #aab; font-size: 13px;
  background: rgba(10,10,15,0.75); padding: 12px 16px; border-radius: 8px;
  backdrop-filter: blur(8px); border: 1px solid rgba(255,255,255,0.06); }
#hud h3 { color: #fff; margin-bottom: 6px; font-size: 15px; }
</style>
</head>
<body>
<div id="hud"><h3>SHARP 3D</h3><div id="info">Loading PLY...</div></div>
<!-- Three.js r168 -->
<script src="https://cdn.jsdelivr.net/npm/three@0.168.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.168.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.168.0/examples/js/loaders/PLYLoader.js"></script>
<script>
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.01, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(devicePixelRatio);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
document.body.appendChild(renderer.domElement);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.autoRotate = true;
controls.autoRotateSpeed = 1.5;
scene.add(new THREE.AmbientLight(0xffffff, 0.6));
const dirLight = new THREE.DirectionalLight(0xffffff, 1.0);
dirLight.position.set(5, 10, 7);
scene.add(dirLight);
scene.add(new THREE.HemisphereLight(0x8888ff, 0x443322, 0.4));
scene.fog = new THREE.FogExp2(0x0a0a0f, 0.015);
const loader = new THREE.PLYLoader();
loader.load('${plyFileName}', function(geometry){
  geometry.computeVertexNormals();
  const mat = new THREE.PointsMaterial({
    size: 0.005, vertexColors: geometry.hasAttribute('color'),
    color: geometry.hasAttribute('color') ? 0xffffff : 0x88aaff,
    sizeAttenuation: true, transparent: true, opacity: 0.9
  });
  const points = new THREE.Points(geometry, mat);
  const box = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute('position'));
  const center = new THREE.Vector3();
  box.getCenter(center);
  points.position.sub(center);
  scene.add(points);
  const size = box.getSize(new THREE.Vector3()).length();
  camera.position.set(size * 0.5, size * 0.3, size * 0.5);
  camera.lookAt(controls.target);
  controls.update();
  document.getElementById('info').textContent = geometry.attributes.position.count.toLocaleString() + ' vertices - Scroll to zoom | Drag to orbit';
}, function(xhr){ document.getElementById('info').textContent = 'Loading PLY... ' + Math.round(xhr.loaded/xhr.total*100) + '%'; },
  function(err){ document.getElementById('info').textContent = 'Error loading PLY'; console.error(err); });
window.addEventListener('resize', function(){
  camera.aspect = innerWidth/innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});
(function animate(){ requestAnimationFrame(animate); controls.update(); renderer.render(scene, camera); })();
</script>
</body>
</html>
""".trimIndent()

    fun writeViewerBundle(context: Context, plyFile: File): File {
        val dir = plyFile.parentFile ?: context.getExternalFilesDir(null)!!
        val htmlFile = File(dir, "viewer.html")
        htmlFile.writeText(generateViewerHtml(plyFile.name))
        LogUtil.i("WebGLViewer", "Viewer written: ${htmlFile.absolutePath}")
        return htmlFile
    }
}

/**
 * Lightweight HTTP server to serve PLY + viewer for WebView.
 */
class PlyHttpServer(private val port: Int, private val serveDir: File) {
    @Volatile
    private var running = false
    private var serverSocket: java.net.ServerSocket? = null

    fun start() {
        running = true
        serverSocket = java.net.ServerSocket(port)
        LogUtil.i("PlyHttpServer", "Serving ${serveDir.absolutePath} at http://localhost:$port")
        Thread({
            while (running) {
                try {
                    val client = serverSocket!!.accept()
                    Thread { handleClient(client) }.start()
                } catch (_: java.net.SocketException) { break }
            }
        }, "ply-http-server").start()
    }

    fun stop() {
        running = false
        serverSocket?.close()
        serverSocket = null
    }

    fun viewerUrl(): String = "http://localhost:$port/viewer.html"

    private fun handleClient(socket: java.net.Socket) {
        socket.use { s ->
            val reader = s.getInputStream().bufferedReader()
            val out = s.getOutputStream()
            val requestLine = reader.readLine() ?: ""
            var path = requestLine.split(" ").getOrNull(1)?.removePrefix("/") ?: "viewer.html"
            if (path.isEmpty()) path = "viewer.html"
            val file = File(serveDir, path)
            if (!file.exists() || !file.canonicalPath.startsWith(serveDir.canonicalPath)) {
                out.write("HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".toByteArray(Charsets.US_ASCII))
                return
            }
            val contentType = when (file.extension.lowercase()) {
                "html" -> "text/html"
                "ply" -> "application/octet-stream"
                "js" -> "application/javascript"
                else -> "application/octet-stream"
            }
            val bytes = file.readBytes()
            val header = "HTTP/1.1 200 OK\r\nContent-Type: $contentType\r\nContent-Length: ${bytes.size}\r\n\r\n"
            out.write(header.toByteArray(Charsets.US_ASCII))
            out.write(bytes)
            out.flush()
        }
    }
}

/**
 * Top-level orchestrator - ties everything together.
 */
class SharpOrchestrator(
    private val context: Context,
    private val cfg: SharpConfig = SharpConfig()
) {
    private val pipeline = SharpInferencePipeline(context, cfg)
    private val plyGen = PlyGenerator(cfg)
    private val viewer = WebGLViewer(cfg)
    private var server: PlyHttpServer? = null

    fun loadModel(ptePath: String) {
        pipeline.loadModel(ptePath)
    }

    fun run(inputImage: FloatArray, seqLen: Int): SharpResult {
        val output = pipeline.runInference(inputImage, seqLen)
        val plyFile = plyGen.generateToDir(context, output)
        val htmlFile = viewer.writeViewerBundle(context, plyFile)
        val srv = PlyHttpServer(cfg.webViewerPort, plyFile.parentFile!!)
        srv.start()
        server = srv
        return SharpResult(plyFile, htmlFile, srv.viewerUrl(), output.numVertices)
    }

    fun release() {
        server?.stop()
        pipeline.release()
    }
}

data class SharpResult(
    val plyFile: File,
    val htmlFile: File,
    val viewerUrl: String,
    val numVertices: Int
)
