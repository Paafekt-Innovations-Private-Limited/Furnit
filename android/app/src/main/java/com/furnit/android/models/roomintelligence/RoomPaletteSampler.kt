package com.furnit.android.models.roomintelligence

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import java.io.File
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.sqrt

object BitmapStraightSrgbExtractor {
    const val DEFAULT_ALPHA_THRESHOLD = 16

    /**
     * Mean straight-sRGB color. Transparent pixels are ignored and premultiplied channels are
     * converted back to straight channels before averaging.
     */
    @JvmStatic
    fun mean(
        bitmap: Bitmap,
        maxSamples: Int = 16_384,
        alphaThreshold: Int = DEFAULT_ALPHA_THRESHOLD,
    ): StraightSrgbColor? {
        if (bitmap.width <= 0 || bitmap.height <= 0) return null
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)
        return meanPremultipliedArgb(
            pixels = pixels,
            width = bitmap.width,
            height = bitmap.height,
            maxSamples = maxSamples,
            alphaThreshold = alphaThreshold,
        )
    }

    /**
     * Pure counterpart for JVM tests and callers that already have premultiplied ARGB pixels.
     */
    @JvmStatic
    fun meanPremultipliedArgb(
        pixels: IntArray,
        width: Int,
        height: Int,
        maxSamples: Int = 16_384,
        alphaThreshold: Int = DEFAULT_ALPHA_THRESHOLD,
        left: Int = 0,
        top: Int = 0,
        rightExclusive: Int = width,
        bottomExclusive: Int = height,
    ): StraightSrgbColor? {
        require(width >= 0 && height >= 0 && pixels.size >= width * height)
        require(maxSamples > 0)
        require(alphaThreshold in 0..255)

        val boundedLeft = left.coerceIn(0, width)
        val boundedRight = rightExclusive.coerceIn(boundedLeft, width)
        val boundedTop = top.coerceIn(0, height)
        val boundedBottom = bottomExclusive.coerceIn(boundedTop, height)
        val regionWidth = boundedRight - boundedLeft
        val regionHeight = boundedBottom - boundedTop
        if (regionWidth == 0 || regionHeight == 0) return null

        val sampleStride = max(
            1,
            ceil(sqrt(regionWidth.toDouble() * regionHeight / maxSamples.toDouble())).toInt(),
        )
        var redTotal = 0.0
        var greenTotal = 0.0
        var blueTotal = 0.0
        var sampleCount = 0
        var y = boundedTop
        while (y < boundedBottom) {
            var x = boundedLeft
            while (x < boundedRight) {
                val pixel = pixels[y * width + x]
                val alpha = pixel ushr 24 and 0xff
                if (alpha >= alphaThreshold && alpha > 0) {
                    redTotal += unPremultiply(pixel ushr 16 and 0xff, alpha)
                    greenTotal += unPremultiply(pixel ushr 8 and 0xff, alpha)
                    blueTotal += unPremultiply(pixel and 0xff, alpha)
                    sampleCount++
                }
                x += sampleStride
            }
            y += sampleStride
        }
        if (sampleCount == 0) return null
        return StraightSrgbColor(
            red = (redTotal / sampleCount / 255.0).toFloat().coerceIn(0f, 1f),
            green = (greenTotal / sampleCount / 255.0).toFloat().coerceIn(0f, 1f),
            blue = (blueTotal / sampleCount / 255.0).toFloat().coerceIn(0f, 1f),
        )
    }

    private fun unPremultiply(channel: Int, alpha: Int): Double =
        (channel.toDouble() * 255.0 / alpha).coerceIn(0.0, 255.0)
}

object RoomPaletteSampler {
    private const val MAX_DECODE_EDGE = 256
    private const val MAX_COLOR_SAMPLES = 16_384

    /**
     * Samples dedicated generated surfaces when present. For photo-only rooms, top/middle/bottom
     * regions provide ceiling/wall/floor proxies without claiming geometry or free floor space.
     */
    @JvmStatic
    fun sample(roomFolder: File): SurfacePalette {
        if (!roomFolder.isDirectory) return SurfacePalette.EMPTY
        val floorFile = File(roomFolder, "floor.png")
        val wallFile = File(roomFolder, "front_wall.png")
        val ceilingFile = File(roomFolder, "ceiling.png")

        if (floorFile.isFile || ceilingFile.isFile) {
            return SurfacePalette(
                floor = sampleSurface(floorFile),
                walls = sampleSurface(wallFile),
                ceiling = sampleSurface(ceilingFile),
            )
        }

        val sourceFile = listOf(
            File(roomFolder, "source_photo.jpg"),
            File(roomFolder, "front_wall.png"),
        ).firstOrNull(File::isFile) ?: return SurfacePalette.EMPTY
        val sourceBitmap = decodeDownsampled(sourceFile) ?: return SurfacePalette.EMPTY
        return try {
            regionalPalette(
                pixels = IntArray(sourceBitmap.width * sourceBitmap.height).also {
                    sourceBitmap.getPixels(
                        it,
                        0,
                        sourceBitmap.width,
                        0,
                        0,
                        sourceBitmap.width,
                        sourceBitmap.height,
                    )
                },
                width = sourceBitmap.width,
                height = sourceBitmap.height,
            )
        } finally {
            sourceBitmap.recycle()
        }
    }

    /**
     * Pure, bounded-cost regional fallback for tests and pre-decoded image pipelines.
     */
    @JvmStatic
    fun regionalPalette(
        pixels: IntArray,
        width: Int,
        height: Int,
        maxSamplesPerRegion: Int = MAX_COLOR_SAMPLES,
    ): SurfacePalette {
        if (width <= 0 || height <= 0 || pixels.size < width * height) {
            return SurfacePalette.EMPTY
        }
        val topEnd = (height / 3).coerceAtLeast(1)
        val middleEnd = (height * 2 / 3).coerceAtLeast(topEnd + 1).coerceAtMost(height)
        fun region(top: Int, bottom: Int): SurfaceColors? {
            val mean = BitmapStraightSrgbExtractor.meanPremultipliedArgb(
                pixels = pixels,
                width = width,
                height = height,
                maxSamples = maxSamplesPerRegion,
                top = top,
                bottomExclusive = bottom,
            ) ?: return null
            return SurfaceColors(listOf(mean), inferMaterial(mean))
        }
        return SurfacePalette(
            ceiling = region(0, topEnd),
            walls = region(topEnd, middleEnd),
            floor = region(middleEnd, height),
        )
    }

    @JvmStatic
    fun inferMaterial(color: StraightSrgbColor): MaterialHint {
        val normalized = color.clamped()
        val maximum = max(normalized.red, max(normalized.green, normalized.blue))
        val minimum = minOf(normalized.red, normalized.green, normalized.blue)
        val saturation = if (maximum > 0f) (maximum - minimum) / maximum else 0f
        return when {
            maximum > 0.88f && saturation < 0.08f -> MaterialHint.PLASTER
            maximum < 0.16f && saturation < 0.20f -> MaterialHint.CARPET
            normalized.red > normalized.green * 1.08f &&
                normalized.green > normalized.blue * 1.08f &&
                saturation > 0.18f && maximum in 0.25f..0.82f -> MaterialHint.WOOD
            maximum in 0.35f..0.68f && saturation < 0.07f -> MaterialHint.CONCRETE
            else -> MaterialHint.UNKNOWN
        }
    }

    private fun sampleSurface(file: File): SurfaceColors? {
        if (!file.isFile) return null
        val bitmap = decodeDownsampled(file) ?: return null
        return try {
            BitmapStraightSrgbExtractor.mean(bitmap, MAX_COLOR_SAMPLES)?.let { mean ->
                SurfaceColors(listOf(mean), inferMaterial(mean))
            }
        } finally {
            bitmap.recycle()
        }
    }

    private fun decodeDownsampled(file: File): Bitmap? {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, bounds)
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sampleSize = 1
        while (max(bounds.outWidth, bounds.outHeight) / sampleSize > MAX_DECODE_EDGE) {
            sampleSize *= 2
        }
        return BitmapFactory.decodeFile(
            file.absolutePath,
            BitmapFactory.Options().apply {
                inSampleSize = sampleSize
                inPreferredConfig = Bitmap.Config.ARGB_8888
            },
        )
    }
}
