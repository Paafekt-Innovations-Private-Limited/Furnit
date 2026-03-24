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
    if (width == height) return Pair(this, null)
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

/** Maps bbox vertical extent from oriented pixels to raw [Frame.acquireCameraImage] pixels (after [rotateToMatchLockedRoomPhoto]). */
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
