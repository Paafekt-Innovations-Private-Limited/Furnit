package com.furnit.android.utils

import org.json.JSONObject
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Arrays
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class SharpRoomDimensionsV7Result(
    val approach: String,
    val shotType: String,
    val orientationLabel: String,
    val usedFocal: Boolean,
    val tiltDegrees: Float,
    val tiltReliable: Boolean,
    val cuboidRatio: Float,
    val cuboidThreshold: Float,
    val fillWidth: Float,
    val blend: Float,
    val legacyWidth: Float,
    val legacyHeight: Float,
    val legacyDepth: Float,
    val width: Float,
    val height: Float,
    val depth: Float,
    val sceneWidth: Float,
    val sceneHeight: Float,
    val sceneDepth: Float,
    val zMode: Float,
    val zMedian: Float,
    val zMean: Float,
    val band: Float,
    val count: Int,
    val floorDiagonal: Float,
    val trimmedXSpan: Float,
    val trimmedYSpan: Float,
    val trimmedZSpan: Float,
    val rawWidth: Float,
    val rawHeight: Float,
)

object SharpRoomDimensionsV7 {
    private const val TAG = "SharpRoomDimsV7"
    private const val APPROACH = "room_dims_v7_async"
    private const val MAX_HEADER_BYTES = 1024 * 1024

    private data class BinaryPlyVertexLayout(
        val headerByteCount: Int,
        val vertexCount: Int,
        val vertexStride: Int,
        val xOffset: Int,
        val yOffset: Int,
        val zOffset: Int,
    )

    private data class FloorAlignedPcaResult(
        val tiltDegrees: Float,
        val tiltReliable: Boolean,
    )

    fun measure(
        plyFile: File,
        sourceImageWidthPx: Int,
        sourceImageHeightPx: Int,
        cameraExifFile: File? = null,
        treatAsClassicPly: Boolean = plyFile.nameWithoutExtension.endsWith("_classic"),
    ): SharpRoomDimensionsV7Result? {
        val layout = binaryPlyLayout(plyFile) ?: return null
        val xs = FloatArray(layout.vertexCount)
        val ys = FloatArray(layout.vertexCount)
        val depths = FloatArray(layout.vertexCount)
        var validCount = 0

        var minX = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY
        var minZ = Float.POSITIVE_INFINITY
        var maxZ = Float.NEGATIVE_INFINITY

        RandomAccessFile(plyFile, "r").use { raf ->
            val channel = raf.channel
            channel.position(layout.headerByteCount.toLong())
            val batchVertices = max(1, min(4096, layout.vertexCount))
            val buffer = ByteBuffer.allocateDirect(batchVertices * layout.vertexStride).order(ByteOrder.LITTLE_ENDIAN)
            var remaining = layout.vertexCount
            while (remaining > 0) {
                val verticesThisBatch = min(batchVertices, remaining)
                val bytesThisBatch = verticesThisBatch * layout.vertexStride
                buffer.clear()
                buffer.limit(bytesThisBatch)
                while (buffer.hasRemaining()) {
                    if (channel.read(buffer) < 0) break
                }
                if (buffer.position() < bytesThisBatch) break
                buffer.flip()

                for (i in 0 until verticesThisBatch) {
                    val offset = i * layout.vertexStride
                    val storedX = buffer.getFloat(offset + layout.xOffset)
                    val storedY = buffer.getFloat(offset + layout.yOffset)
                    val storedZ = buffer.getFloat(offset + layout.zOffset)
                    if (!storedX.isFinite() || !storedY.isFinite() || !storedZ.isFinite()) continue

                    minX = min(minX, storedX)
                    maxX = max(maxX, storedX)
                    minY = min(minY, storedY)
                    maxY = max(maxY, storedY)
                    minZ = min(minZ, storedZ)
                    maxZ = max(maxZ, storedZ)

                    val normalizedX = storedX
                    val normalizedY = if (treatAsClassicPly) -storedY else storedY
                    val normalizedZ = if (treatAsClassicPly) -storedZ else storedZ
                    val depth = -normalizedZ
                    if (!depth.isFinite() || depth <= 0.01f) continue

                    xs[validCount] = normalizedX
                    ys[validCount] = normalizedY
                    depths[validCount] = depth
                    validCount++
                }
                remaining -= verticesThisBatch
            }
        }

        if (validCount < 64 ||
            !minX.isFinite() || !maxX.isFinite() || !minY.isFinite() || !maxY.isFinite() ||
            !minZ.isFinite() || !maxZ.isFinite()
        ) {
            return null
        }

        val sortedXs = xs.copyOf(validCount)
        val sortedYs = ys.copyOf(validCount)
        val sortedDepths = depths.copyOf(validCount)
        Arrays.sort(sortedXs)
        Arrays.sort(sortedYs)
        Arrays.sort(sortedDepths)

        val xP3 = sortedXs[percentileIndex(0.03f, validCount)]
        val xP97 = sortedXs[percentileIndex(0.97f, validCount)]
        val yP3 = sortedYs[percentileIndex(0.03f, validCount)]
        val yP97 = sortedYs[percentileIndex(0.97f, validCount)]
        val zP3 = sortedDepths[percentileIndex(0.03f, validCount)]
        val zP97 = sortedDepths[percentileIndex(0.97f, validCount)]

        val trimmedXSpan = xP97 - xP3
        val trimmedYSpan = yP97 - yP3
        val trimmedZSpan = zP97 - zP3
        if (!trimmedXSpan.isFinite() || !trimmedYSpan.isFinite() || !trimmedZSpan.isFinite() ||
            trimmedXSpan <= 0.01f || trimmedZSpan <= 0.01f
        ) {
            return null
        }

        val floorDiagonal = hypotF(trimmedXSpan, trimmedZSpan)
        var trimmedDepthCount = 0
        for (i in 0 until validCount) {
            val depth = depths[i]
            if (depth >= zP3 && depth <= zP97) trimmedDepthCount++
        }
        if (trimmedDepthCount < 64) return null

        val binCount = 200
        val binWidth = max(trimmedZSpan / binCount.toFloat(), 1e-4f)
        val histogram = IntArray(binCount)
        for (i in 0 until validCount) {
            val depth = depths[i]
            if (depth < zP3 || depth > zP97) continue
            val bucket = (((depth - zP3) / binWidth).toInt()).coerceIn(0, binCount - 1)
            histogram[bucket] += 1
        }

        var peakIndex = 0
        for (i in 1 until histogram.size) {
            if (histogram[i] > histogram[peakIndex]) peakIndex = i
        }
        val zMode = zP3 + (peakIndex.toFloat() + 0.5f) * binWidth
        val band = max(0.10f * zMode, 1e-4f)

        val backWallXs = FloatArray(validCount)
        val backWallYs = FloatArray(validCount)
        val backWallDepths = FloatArray(validCount)
        var backWallCount = 0
        for (i in 0 until validCount) {
            val depth = depths[i]
            if (abs(depth - zMode) >= band) continue
            backWallXs[backWallCount] = xs[i]
            backWallYs[backWallCount] = ys[i]
            backWallDepths[backWallCount] = depth
            backWallCount++
        }
        if (backWallCount < 64) return null

        Arrays.sort(backWallXs, 0, backWallCount)
        Arrays.sort(backWallYs, 0, backWallCount)
        Arrays.sort(backWallDepths, 0, backWallCount)
        val idx5 = percentileIndex(0.05f, backWallCount)
        val idx95 = percentileIndex(0.95f, backWallCount)
        val rawWidth = backWallXs[idx95] - backWallXs[idx5]
        val rawHeight = backWallYs[idx95] - backWallYs[idx5]
        val zMedian = backWallDepths[backWallCount / 2]
        var zSum = 0f
        for (i in 0 until backWallCount) zSum += backWallDepths[i]
        val zMean = zSum / backWallCount.toFloat()

        val legacyWidth = rawWidth / 1.2f
        val legacyHeight = rawHeight / 1.2f
        val legacyDepth = zMedian * 1.08f

        val focalPx = sourceFocalPx(cameraExifFile, sourceImageWidthPx, sourceImageHeightPx)
        val hasFocal = focalPx > 0.01f
        val imageWidth = sourceImageWidthPx.toFloat()
        val imageHeight = sourceImageHeightPx.toFloat()
        val orientationLabel = if (imageWidth > imageHeight) "LANDSCAPE" else "PORTRAIT"
        val maxSpan = max(trimmedXSpan, max(trimmedYSpan, trimmedZSpan))
        val minSpan = min(trimmedXSpan, min(trimmedYSpan, trimmedZSpan))
        val cuboidRatio = maxSpan / max(minSpan, 1e-6f)
        val imageAspect = if (imageWidth > 0.01f && imageHeight > 0.01f) {
            max(imageWidth, imageHeight) / min(imageWidth, imageHeight)
        } else {
            1f
        }
        val cuboidThreshold = if (imageWidth > imageHeight) 1.50f * imageAspect else 1.45f
        val isCornerShot = cuboidRatio < cuboidThreshold

        val seedValue = ((zMode * 1_000_000f + rawWidth * 1_000f).toDouble()).toLong()
        val pca = estimateFloorAlignedPca(xs, ys, depths, validCount, seedValue)
        val tiltDegrees = pca?.tiltDegrees ?: 8.0f
        val tiltReliable = pca?.tiltReliable ?: false

        val finalWidth: Float
        val finalHeight: Float
        val finalDepth: Float
        val shotType: String
        val fillWidth: Float
        val blend: Float

        if (!hasFocal) {
            shotType = "FALLBACK_NO_FOCAL"
            fillWidth = 0f
            blend = 0f
            finalWidth = legacyWidth
            finalHeight = legacyHeight
            finalDepth = floorDiagonal
        } else if (isCornerShot) {
            shotType = "CORNER"
            fillWidth = 0f
            blend = 0f
            finalWidth = legacyWidth
            finalHeight = if (tiltDegrees < 12f) legacyHeight else rawHeight
            val diagSquared = floorDiagonal * floorDiagonal
            val widthSquared = finalWidth * finalWidth
            finalDepth = if (diagSquared > widthSquared) sqrt(max(0f, diagSquared - widthSquared)) else floorDiagonal
        } else {
            shotType = "STRAIGHT"
            val fovDiagonal = sqrt(max(0f, (imageWidth / focalPx) * (imageWidth / focalPx) + (imageHeight / focalPx) * (imageHeight / focalPx)))
            val maxLateral = 0.08f * fovDiagonal * zP3
            val sceneExtension = 2f * maxLateral
            val fovWidthAtBackWall = imageWidth * zMode / focalPx
            fillWidth = if (fovWidthAtBackWall > 1e-6f) rawWidth / fovWidthAtBackWall else 0f
            blend = ((fillWidth - 0.55f) / 0.20f).coerceIn(0f, 1f)
            val correctedWidth = rawWidth - sceneExtension
            val correctedHeight = rawHeight - sceneExtension
            finalWidth = max(correctedWidth + blend * (rawWidth - correctedWidth), 0.5f)
            finalHeight = if (tiltDegrees < 12f) {
                max(correctedHeight + blend * (rawHeight - correctedHeight), 0.5f)
            } else {
                rawHeight
            }
            finalDepth = floorDiagonal
        }

        return SharpRoomDimensionsV7Result(
            approach = APPROACH,
            shotType = shotType,
            orientationLabel = orientationLabel,
            usedFocal = hasFocal,
            tiltDegrees = tiltDegrees,
            tiltReliable = tiltReliable,
            cuboidRatio = cuboidRatio,
            cuboidThreshold = cuboidThreshold,
            fillWidth = fillWidth,
            blend = blend,
            legacyWidth = legacyWidth,
            legacyHeight = legacyHeight,
            legacyDepth = legacyDepth,
            width = finalWidth,
            height = finalHeight,
            depth = finalDepth,
            sceneWidth = maxX - minX,
            sceneHeight = maxY - minY,
            sceneDepth = maxZ - minZ,
            zMode = zMode,
            zMedian = zMedian,
            zMean = zMean,
            band = band,
            count = backWallCount,
            floorDiagonal = floorDiagonal,
            trimmedXSpan = trimmedXSpan,
            trimmedYSpan = trimmedYSpan,
            trimmedZSpan = trimmedZSpan,
            rawWidth = rawWidth,
            rawHeight = rawHeight,
        )
    }

    fun measureBest(
        plyFile: File,
        sourceImageWidthPx: Int,
        sourceImageHeightPx: Int,
        cameraExifFile: File? = null,
    ): SharpRoomDimensionsV7Result? {
        val filenameLooksClassic = plyFile.nameWithoutExtension.endsWith("_classic")
        val hinted = measure(
            plyFile = plyFile,
            sourceImageWidthPx = sourceImageWidthPx,
            sourceImageHeightPx = sourceImageHeightPx,
            cameraExifFile = cameraExifFile,
            treatAsClassicPly = filenameLooksClassic,
        )
        val alternate = measure(
            plyFile = plyFile,
            sourceImageWidthPx = sourceImageWidthPx,
            sourceImageHeightPx = sourceImageHeightPx,
            cameraExifFile = cameraExifFile,
            treatAsClassicPly = !filenameLooksClassic,
        )
        return bestOf(hinted, alternate)
    }

    private fun bestOf(
        first: SharpRoomDimensionsV7Result?,
        second: SharpRoomDimensionsV7Result?,
    ): SharpRoomDimensionsV7Result? {
        if (first == null) return second
        if (second == null) return first
        val firstScore = measurementScore(first)
        val secondScore = measurementScore(second)
        return if (secondScore > firstScore) second else first
    }

    private fun measurementScore(result: SharpRoomDimensionsV7Result): Float {
        val saneSize =
            result.width.isFinite() && result.height.isFinite() && result.depth.isFinite() &&
                result.width in 0.3f..20f &&
                result.height in 0.3f..8f &&
                result.depth in 0.3f..20f
        val sizeBonus = if (saneSize) 10_000f else 0f
        val focalBonus = if (result.usedFocal) 1_000f else 0f
        return sizeBonus + focalBonus + result.count.toFloat()
    }

    private fun percentileIndex(p: Float, count: Int): Int {
        if (count <= 1) return 0
        return ((count - 1).toFloat() * p).toInt().coerceIn(0, count - 1)
    }

    private fun binaryPlyLayout(file: File): BinaryPlyVertexLayout? {
        if (!file.isFile) return null
        val headerBytes = readHeaderBytes(file) ?: return null
        val headerString = String(headerBytes.first, Charsets.UTF_8)
        val headerByteCount = headerBytes.second
        val lines = headerString.lineSequence()

        var inVertexElement = false
        var vertexCount: Int? = null
        val properties = mutableListOf<Pair<String, Int>>()

        for (rawLine in lines) {
            val line = rawLine.trim()
            if (line.isEmpty()) continue
            val parts = line.split(Regex("\\s+"))
            val keyword = parts.firstOrNull() ?: continue
            if (keyword == "element" && parts.size >= 3) {
                inVertexElement = parts[1] == "vertex"
                if (inVertexElement) {
                    vertexCount = parts[2].toIntOrNull()
                    properties.clear()
                }
                continue
            }
            if (!inVertexElement || keyword != "property" || parts.size < 3) continue
            if (parts[1] == "list") return null
            val width = byteWidth(parts[1]) ?: return null
            properties += parts[2] to width
        }

        val count = vertexCount?.takeIf { it > 0 } ?: return null
        var runningOffset = 0
        var xOffset: Int? = null
        var yOffset: Int? = null
        var zOffset: Int? = null
        for ((name, width) in properties) {
            when (name) {
                "x" -> xOffset = runningOffset
                "y" -> yOffset = runningOffset
                "z" -> zOffset = runningOffset
            }
            runningOffset += width
        }
        val x = xOffset ?: return null
        val y = yOffset ?: return null
        val z = zOffset ?: return null
        if (runningOffset <= 0) return null
        return BinaryPlyVertexLayout(headerByteCount, count, runningOffset, x, y, z)
    }

    private fun readHeaderBytes(file: File): Pair<ByteArray, Int>? {
        val out = ByteArrayOutputStream()
        val buffer = ByteArray(4096)
        file.inputStream().use { input ->
            var total = 0
            while (total < MAX_HEADER_BYTES) {
                val read = input.read(buffer)
                if (read <= 0) break
                out.write(buffer, 0, read)
                total += read
                val bytes = out.toByteArray()
                val text = String(bytes, Charsets.UTF_8)
                val crlf = text.indexOf("end_header\r\n")
                if (crlf >= 0) return bytes.copyOf(crlf + "end_header\r\n".length) to (crlf + "end_header\r\n".length)
                val lf = text.indexOf("end_header\n")
                if (lf >= 0) return bytes.copyOf(lf + "end_header\n".length) to (lf + "end_header\n".length)
            }
        }
        LogUtil.w(TAG, "PLY header not found within $MAX_HEADER_BYTES bytes: ${file.absolutePath}")
        return null
    }

    private fun byteWidth(rawType: String): Int? = when (rawType) {
        "char", "uchar", "int8", "uint8" -> 1
        "short", "ushort", "int16", "uint16" -> 2
        "int", "uint", "float", "int32", "uint32", "float32" -> 4
        "double", "float64" -> 8
        else -> null
    }

    private fun sourceFocalPx(cameraExifFile: File?, sourceWidth: Int, sourceHeight: Int): Float {
        if (cameraExifFile == null || !cameraExifFile.isFile || sourceWidth <= 0 || sourceHeight <= 0) return 0f
        val json = runCatching { JSONObject(cameraExifFile.readText()) }.getOrNull() ?: return 0f
        val directFocalPx = optFloat(json, "focalLengthPx")
        if (directFocalPx != null && directFocalPx > 0.01f) return directFocalPx

        val focal35 = optFloat(json, "focalLength35mmEquivMm")
        val focalMm = optFloat(json, "focalLengthMm")
        val focal35mm = when {
            focal35 != null && focal35 > 0.01f -> focal35
            focalMm != null && focalMm > 0.01f && focalMm < 10f -> focalMm * 8.4f
            focalMm != null && focalMm > 0.01f -> focalMm
            else -> return 0f
        }
        val diagonal = hypotF(sourceWidth.toFloat(), sourceHeight.toFloat())
        val diagonal35mm = hypotF(36f, 24f)
        if (diagonal <= 1f || diagonal35mm <= 0.01f) return 0f
        return focal35mm * diagonal / diagonal35mm
    }

    private fun optFloat(json: JSONObject, key: String): Float? {
        if (!json.has(key)) return null
        val value = json.optDouble(key, Double.NaN)
        return if (value.isNaN()) null else value.toFloat()
    }

    private fun hypotF(a: Float, b: Float): Float = sqrt(a * a + b * b)

    private fun estimateFloorAlignedPca(
        xs: FloatArray,
        ys: FloatArray,
        depths: FloatArray,
        count: Int,
        seed: Long,
    ): FloorAlignedPcaResult? {
        if (count < 128) return null
        val sampleCount = min(50_000, count)
        val ransacIterations = 500
        val epsilon = 0.05f
        val sampledIndices = IntArray(sampleCount)
        val rng = SeedableRng(seed)
        for (i in 0 until sampleCount) {
            sampledIndices[i] = rng.nextIndex(count)
        }

        var bestNx = 0f
        var bestNy = 1f
        var bestNz = 0f
        var bestInliers = 0

        repeat(ransacIterations) {
            val idx0 = sampledIndices[rng.nextIndex(sampleCount)]
            val idx1 = sampledIndices[rng.nextIndex(sampleCount)]
            val idx2 = sampledIndices[rng.nextIndex(sampleCount)]
            if (idx0 == idx1 || idx1 == idx2 || idx0 == idx2) return@repeat

            val ax = xs[idx1] - xs[idx0]
            val ay = ys[idx1] - ys[idx0]
            val az = depths[idx1] - depths[idx0]
            val bx = xs[idx2] - xs[idx0]
            val by = ys[idx2] - ys[idx0]
            val bz = depths[idx2] - depths[idx0]
            var nx = ay * bz - az * by
            var ny = az * bx - ax * bz
            var nz = ax * by - ay * bx
            val len = sqrt(nx * nx + ny * ny + nz * nz)
            if (len <= 1e-6f) return@repeat
            nx /= len
            ny /= len
            nz /= len
            if (abs(ny) <= 0.8f) return@repeat
            if (ny < 0f) {
                nx = -nx
                ny = -ny
                nz = -nz
            }
            val d = -(nx * xs[idx0] + ny * ys[idx0] + nz * depths[idx0])

            var inliers = 0
            for (sampleIndex in sampledIndices) {
                val distance = abs(nx * xs[sampleIndex] + ny * ys[sampleIndex] + nz * depths[sampleIndex] + d)
                if (distance < epsilon) inliers++
            }
            if (inliers > bestInliers) {
                bestInliers = inliers
                bestNx = nx
                bestNy = ny
                bestNz = nz
            }
        }

        val normalLen = sqrt(bestNx * bestNx + bestNy * bestNy + bestNz * bestNz)
        if (normalLen <= 1e-6f) return null
        bestNx /= normalLen
        bestNy /= normalLen
        bestNz /= normalLen

        val cosTheta = bestNy.coerceIn(-1f, 1f)
        val originalTiltDegrees = acos(abs(cosTheta)) * 180f / Math.PI.toFloat()
        val tiltReliable = originalTiltDegrees < 25f && bestInliers > 3000
        return FloorAlignedPcaResult(
            tiltDegrees = if (tiltReliable) originalTiltDegrees else 8.0f,
            tiltReliable = tiltReliable,
        )
    }

    private class SeedableRng(seed: Long) {
        private var state: Long = if (seed == 0L) 1L else seed

        fun nextIndex(bound: Int): Int {
            state = state xor (state shl 13)
            state = state xor (state ushr 7)
            state = state xor (state shl 17)
            return java.lang.Long.remainderUnsigned(state, bound.toLong()).toInt()
        }
    }
}
