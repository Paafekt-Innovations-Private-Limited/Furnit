package com.furnit.android.services

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream

/**
 * Preprocesses camera images for MobileNetV3 classification.
 * Resizes to 224x224, normalizes with ImageNet mean/std, returns NCHW FloatArray.
 */
object ImagePreprocessor {

    private const val INPUT_SIZE = 224

    // ImageNet normalization constants
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    /**
     * Preprocess a Bitmap for MobileNetV3 inference.
     * Returns FloatArray in NCHW format [1, 3, 224, 224].
     */
    fun preprocessBitmap(bitmap: Bitmap): FloatArray {
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        scaledBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        if (scaledBitmap !== bitmap) {
            scaledBitmap.recycle()
        }

        val floatArray = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
        val channelSize = INPUT_SIZE * INPUT_SIZE

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16) and 0xFF) / 255f
            val g = ((pixel shr 8) and 0xFF) / 255f
            val b = (pixel and 0xFF) / 255f

            floatArray[i] = (r - MEAN[0]) / STD[0]
            floatArray[channelSize + i] = (g - MEAN[1]) / STD[1]
            floatArray[2 * channelSize + i] = (b - MEAN[2]) / STD[2]
        }

        return floatArray
    }

    /**
     * Convert an ImageProxy (YUV_420_888) to a Bitmap.
     */
    fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val yBuffer = imageProxy.planes[0].buffer
        val uBuffer = imageProxy.planes[1].buffer
        val vBuffer = imageProxy.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        val nv21 = ByteArray(ySize + uSize + vSize)
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        val yuvImage = YuvImage(nv21, ImageFormat.NV21, imageProxy.width, imageProxy.height, null)
        val outputStream = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), 80, outputStream)
        val jpegBytes = outputStream.toByteArray()
        return android.graphics.BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
    }
}
