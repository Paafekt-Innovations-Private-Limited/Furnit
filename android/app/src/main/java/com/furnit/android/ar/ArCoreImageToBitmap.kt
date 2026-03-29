package com.furnit.android.ar

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.furnit.android.utils.LogUtil
import java.io.ByteArrayOutputStream
import kotlin.math.hypot

/**
 * Converts a [Image] in [ImageFormat.YUV_420_888] (ARCore [Frame.acquireCameraImage]) to a JPEG-decoded [Bitmap],
 * same approach as [com.furnit.android.toBitmapSafe] for CameraX [androidx.camera.core.ImageProxy].
 */
fun Image.yuv420888ToBitmap(jpegQuality: Int = 90): Bitmap? {
    if (format != ImageFormat.YUV_420_888) {
        LogUtil.w("ArCoreImage", "Unexpected image format: $format")
        return null
    }
    return try {
        val planes = planes
        if (planes.size < 3) return null
        val yPlane = planes[0]
        val uPlane = planes[1]
        val vPlane = planes[2]
        val yBuffer = yPlane.buffer.duplicate()
        val uBuffer = uPlane.buffer.duplicate()
        val vBuffer = vPlane.buffer.duplicate()
        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride
        val width = this.width
        val height = this.height
        val nv21 = ByteArray(width * height * 3 / 2)
        var pos = 0
        for (row in 0 until height) {
            yBuffer.position(row * yRowStride)
            yBuffer.get(nv21, pos, width)
            pos += width
        }
        val uvHeight = height / 2
        val uvWidth = width / 2
        for (row in 0 until uvHeight) {
            for (col in 0 until uvWidth) {
                val uvIndex = row * uvRowStride + col * uvPixelStride
                vBuffer.position(uvIndex)
                uBuffer.position(uvIndex)
                nv21[pos++] = vBuffer.get()
                nv21[pos++] = uBuffer.get()
            }
        }
        val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, width, height), jpegQuality, out)
        BitmapFactory.decodeByteArray(out.toByteArray(), 0, out.size())
    } catch (e: Exception) {
        LogUtil.e("ArCoreImage", "yuv420888ToBitmap failed: ${e.message}", e)
        null
    }
}

/**
 * Rotates the bitmap so its aspect (wide vs tall) matches the **room photo lock** (`"portrait"` / `"landscape"`),
 * similar to CameraX [androidx.camera.core.ImageAnalysis] + [androidx.camera.core.ImageProxy.toBitmapSafe].
 *
 * ARCore [Frame.acquireCameraImage] stays in sensor/native layout; this aligns YOLO + overlay with the locked
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
 * Use this on the GL thread every frame; only decode YUV→bitmap when actually sending a frame to YOLO.
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
 * Maps a point from **oriented** bitmap pixels (what YOLO sees after [rotateToMatchLockedRoomPhoto])
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

/** @deprecated Use [mapOrientedImagePixelToRawCameraPixel] — matrix inverse was wrong for dimension-swapping 90° rotations. */
fun orientedBboxHeightToRawPixels(
    centerXOriented: Float,
    centerYOriented: Float,
    heightOriented: Float,
    orientedToRawInverse: Matrix?,
): Float {
    if (orientedToRawInverse == null || heightOriented <= 1f) return heightOriented
    val top = floatArrayOf(centerXOriented, centerYOriented - heightOriented * 0.5f)
    val bot = floatArrayOf(centerXOriented, centerYOriented + heightOriented * 0.5f)
    orientedToRawInverse.mapPoints(top)
    orientedToRawInverse.mapPoints(bot)
    return hypot((bot[0] - top[0]).toDouble(), (bot[1] - top[1]).toDouble()).toFloat()
}
