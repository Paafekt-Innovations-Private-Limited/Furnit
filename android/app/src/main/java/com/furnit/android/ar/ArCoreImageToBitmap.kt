package com.furnit.android.ar

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.media.Image
import com.furnit.android.utils.LogUtil
import kotlin.math.hypot

private val nv21Scratch = ThreadLocal<ByteArray>()
private val argbScratch = ThreadLocal<IntArray>()

internal data class CopiedYuv420Frame(
    val width: Int,
    val height: Int,
    val y: ByteArray,
    val yRowStride: Int,
    val yPixelStride: Int,
    val u: ByteArray,
    val uRowStride: Int,
    val uPixelStride: Int,
    val v: ByteArray,
    val vRowStride: Int,
    val vPixelStride: Int,
) {
    /** Builds tightly packed NV21 after ARCore's [Image] has already been released. */
    internal fun toNv21(): ByteArray? {
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) return null
        if (!hasPlaneExtent(y, width, height, yRowStride, yPixelStride)) return null
        val uvWidth = width / 2
        val uvHeight = height / 2
        if (!hasPlaneExtent(u, uvWidth, uvHeight, uRowStride, uPixelStride)) return null
        if (!hasPlaneExtent(v, uvWidth, uvHeight, vRowStride, vPixelStride)) return null

        val requiredBytes = width * height * 3 / 2
        val currentNv21 = nv21Scratch.get()
        val nv21 = if (currentNv21 == null || currentNv21.size != requiredBytes) {
            ByteArray(requiredBytes).also(nv21Scratch::set)
        } else {
            currentNv21
        }
        var output = 0
        for (row in 0 until height) {
            val rowStart = row * yRowStride
            if (yPixelStride == 1) {
                y.copyInto(nv21, output, rowStart, rowStart + width)
                output += width
            } else {
                for (column in 0 until width) {
                    nv21[output++] = y[rowStart + column * yPixelStride]
                }
            }
        }
        for (row in 0 until uvHeight) {
            val uRowStart = row * uRowStride
            val vRowStart = row * vRowStride
            for (column in 0 until uvWidth) {
                nv21[output++] = v[vRowStart + column * vPixelStride]
                nv21[output++] = u[uRowStart + column * uPixelStride]
            }
        }
        return nv21
    }

    /** Lossless, stride-aware camera conversion; Swift likewise feeds an uncompressed RGB buffer. */
    fun toBitmap(): Bitmap? {
        return try {
            val pixels = toArgbPixels() ?: return null
            Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888).also { bitmap ->
                bitmap.setPixels(pixels, 0, width, 0, 0, width, height)
            }
        } catch (exception: Exception) {
            LogUtil.e("ArCoreImage", "Copied YUV frame conversion failed: ${exception.message}", exception)
            null
        }
    }

    internal fun toArgbPixels(): IntArray? {
        if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0) return null
        if (!hasPlaneExtent(y, width, height, yRowStride, yPixelStride)) return null
        val uvWidth = width / 2
        val uvHeight = height / 2
        if (!hasPlaneExtent(u, uvWidth, uvHeight, uRowStride, uPixelStride)) return null
        if (!hasPlaneExtent(v, uvWidth, uvHeight, vRowStride, vPixelStride)) return null

        val pixelCount = width * height
        val currentPixels = argbScratch.get()
        val pixels = if (currentPixels == null || currentPixels.size != pixelCount) {
            IntArray(pixelCount).also(argbScratch::set)
        } else {
            currentPixels
        }
        var output = 0
        for (row in 0 until height) {
            val yRowStart = row * yRowStride
            val uRowStart = (row / 2) * uRowStride
            val vRowStart = (row / 2) * vRowStride
            for (column in 0 until width) {
                val luma = y[yRowStart + column * yPixelStride].toInt() and 0xFF
                val chromaColumn = column / 2
                val cb = (u[uRowStart + chromaColumn * uPixelStride].toInt() and 0xFF) - 128
                val cr = (v[vRowStart + chromaColumn * vPixelStride].toInt() and 0xFF) - 128
                val scaledLuma = 298 * (luma - 16).coerceAtLeast(0)
                val red = ((scaledLuma + 409 * cr + 128) shr 8).coerceIn(0, 255)
                val green = ((scaledLuma - 100 * cb - 208 * cr + 128) shr 8).coerceIn(0, 255)
                val blue = ((scaledLuma + 516 * cb + 128) shr 8).coerceIn(0, 255)
                pixels[output++] = (0xFF shl 24) or (red shl 16) or (green shl 8) or blue
            }
        }
        return pixels
    }

    private fun hasPlaneExtent(
        bytes: ByteArray,
        planeWidth: Int,
        planeHeight: Int,
        rowStride: Int,
        pixelStride: Int,
    ): Boolean {
        if (planeWidth <= 0 || planeHeight <= 0 || rowStride <= 0 || pixelStride <= 0) return false
        val lastIndex = (planeHeight - 1L) * rowStride + (planeWidth - 1L) * pixelStride
        return lastIndex >= 0L && lastIndex < bytes.size.toLong()
    }
}

/**
 * Copies the three ARCore planes with bulk [java.nio.ByteBuffer.get] operations. The copy is kept
 * deliberately small on the GL thread; YUV packing/conversion happens after [Image.close].
 */
internal fun Image.copyYuv420Frame(reuse: CopiedYuv420Frame? = null): CopiedYuv420Frame? {
    if (format != ImageFormat.YUV_420_888 || width <= 0 || height <= 0) return null
    val imagePlanes = planes
    if (imagePlanes.size < 3) return null

    fun copyPlane(plane: Image.Plane, reusableBytes: ByteArray?): ByteArray {
        val source = plane.buffer.duplicate()
        source.rewind()
        val destination = reusableBytes?.takeIf { it.size == source.remaining() }
            ?: ByteArray(source.remaining())
        source.get(destination)
        return destination
    }

    return try {
        val yPlane = imagePlanes[0]
        val uPlane = imagePlanes[1]
        val vPlane = imagePlanes[2]
        CopiedYuv420Frame(
            width = width,
            height = height,
            y = copyPlane(yPlane, reuse?.y),
            yRowStride = yPlane.rowStride,
            yPixelStride = yPlane.pixelStride,
            u = copyPlane(uPlane, reuse?.u),
            uRowStride = uPlane.rowStride,
            uPixelStride = uPlane.pixelStride,
            v = copyPlane(vPlane, reuse?.v),
            vRowStride = vPlane.rowStride,
            vPixelStride = vPlane.pixelStride,
        )
    } catch (exception: Exception) {
        LogUtil.e("ArCoreImage", "YUV plane copy failed: ${exception.message}", exception)
        null
    }
}

/**
 * Converts an ARCore [ImageFormat.YUV_420_888] image directly to an uncompressed [Bitmap].
 */
fun Image.yuv420888ToBitmap(): Bitmap? {
    if (format != ImageFormat.YUV_420_888) {
        LogUtil.w("ArCoreImage", "Unexpected image format: $format")
        return null
    }
    return copyYuv420Frame()?.toBitmap()
}

/**
 * Rotates the bitmap so its aspect (wide vs tall) matches the **room photo lock** (`"portrait"` / `"landscape"`),
 * similar to CameraX [androidx.camera.core.ImageAnalysis] + [androidx.camera.core.ImageProxy.toBitmapSafe].
 *
 * ARCore [Frame.acquireCameraImage] stays in sensor/native layout; this aligns segmentation + overlay with the locked
 * activity orientation (mirrors iOS classic camera vs AR display path).
 *
 * @return Pair of (oriented bitmap, inverse matrix mapping **oriented** pixel coords → **raw** camera image coords),
 *         or `(bitmap, null)` when no rotation is applied (intrinsics + bbox already aligned).
 */
fun Bitmap.rotateToMatchLockedRoomPhoto(lockedOrientation: String): Pair<Bitmap, Matrix?> {
    val wantLandscape = lockedOrientation.equals("landscape", ignoreCase = true)
    // Square analysis targets (e.g. 640²) do not change aspect with rotation; CameraX can still leave
    // content 90° off vs a landscape-locked activity. Apply the same 90° we use for WxH mismatch.
    if (width == height) {
        if (!wantLandscape) return Pair(this, null)
        val cx = width / 2f
        val cy = height / 2f
        val m = Matrix()
        m.postRotate(90f, cx, cy)
        val out = Bitmap.createBitmap(this, 0, 0, width, height, m, true)
        val inv = Matrix()
        return if (m.invert(inv)) Pair(out, inv) else Pair(out, null)
    }
    val isLandscape = width > height
    if (wantLandscape == isLandscape) return Pair(this, null)

    val cx = width / 2f
    val cy = height / 2f
    val m = Matrix()
    m.postRotate(90f, cx, cy)
    val out = Bitmap.createBitmap(this, 0, 0, width, height, m, true)
    val inv = Matrix()
    if (!m.invert(inv)) {
        return Pair(out, null)
    }
    return Pair(out, inv)
}

/**
 * Same inverse mapping as [Bitmap.rotateToMatchLockedRoomPhoto] but without allocating bitmaps.
 * Use this on the GL thread every frame; only decode YUV→bitmap when actually sending a frame to segmentation.
 */
fun orientedToRawInverseForDimensions(
    rawWidth: Int,
    rawHeight: Int,
    lockedOrientation: String,
): Matrix? {
    val wantLandscape = lockedOrientation.equals("landscape", ignoreCase = true)
    if (rawWidth == rawHeight) {
        if (!wantLandscape) return null
        val cx = rawWidth / 2f
        val cy = rawHeight / 2f
        val m = Matrix()
        m.postRotate(90f, cx, cy)
        val inv = Matrix()
        return if (m.invert(inv)) inv else null
    }
    val isLandscape = rawWidth > rawHeight
    if (wantLandscape == isLandscape) return null
    val cx = rawWidth / 2f
    val cy = rawHeight / 2f
    val m = Matrix()
    m.postRotate(90f, cx, cy)
    val inv = Matrix()
    return if (m.invert(inv)) inv else null
}

/**
 * Maps a point from **oriented** bitmap pixels (what segmentation sees after [rotateToMatchLockedRoomPhoto])
 * to **raw** [Frame.acquireCameraImage] pixel coordinates for depth, intrinsics, and hit tests.
 *
 * For a 90° quarter-turn (portrait raw ↔ landscape lock or the reverse), the bitmap dimensions swap;
 * a plain [Matrix.postRotate] inverse in raw-sized space does **not** match [Bitmap.createBitmap] output
 * coordinates — use the explicit mapping below (matches CW quarter-turn used in [rotateToMatchLockedRoomPhoto]).
 */
fun mapOrientedImagePixelToRawCameraPixel(
    orientedX: Float,
    orientedY: Float,
    rawWidth: Int,
    rawHeight: Int,
    lockedPhotoOrientation: String,
): Pair<Float, Float> {
    if (rawWidth <= 0 || rawHeight <= 0) return orientedX to orientedY
    val wantLandscape = lockedPhotoOrientation.equals("landscape", ignoreCase = true)
    val rawIsLandscape = rawWidth > rawHeight
    val rawIsPortrait = rawHeight > rawWidth
    val rawIsSquare = rawWidth == rawHeight

    if (rawIsPortrait || rawIsLandscape) {
        if (wantLandscape == rawIsLandscape) {
            return orientedX to orientedY
        }
        val rx = rawWidth - 1f - orientedY
        val ry = orientedX
        return rx to ry
    }

    if (rawIsSquare && wantLandscape) {
        val cx = rawWidth / 2f
        val cy = rawHeight / 2f
        val m = Matrix()
        m.postRotate(90f, cx, cy)
        val inv = Matrix()
        if (m.invert(inv)) {
            val pts = floatArrayOf(orientedX, orientedY)
            inv.mapPoints(pts)
            return pts[0] to pts[1]
        }
    }
    return orientedX to orientedY
}

/** Maps bbox vertical extent (oriented Y axis) to a Euclidean span in raw image pixels. */
fun orientedBboxVerticalExtentInRawPixels(
    centerXOriented: Float,
    centerYOriented: Float,
    heightOriented: Float,
    rawWidth: Int,
    rawHeight: Int,
    lockedPhotoOrientation: String,
): Float {
    if (heightOriented <= 1f) return heightOriented
    val top = mapOrientedImagePixelToRawCameraPixel(
        centerXOriented,
        centerYOriented - heightOriented * 0.5f,
        rawWidth,
        rawHeight,
        lockedPhotoOrientation,
    )
    val bot = mapOrientedImagePixelToRawCameraPixel(
        centerXOriented,
        centerYOriented + heightOriented * 0.5f,
        rawWidth,
        rawHeight,
        lockedPhotoOrientation,
    )
    return hypot((bot.first - top.first).toDouble(), (bot.second - top.second).toDouble()).toFloat()
}
