package com.furnit.android.models

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface

/**
 * Photo orientation for room capture: **how the user held the phone** (portrait vs landscape).
 *
 * **Ultra-wide (0.5×) lens:** Many devices store a **landscape-wide** buffer even for upright shots.
 * When the user marks the photo as 0.5× wide-angle, we coerce [LANDSCAPE] → [PORTRAIT] unless they
 * explicitly overrode orientation (SinglePhotoRoom).
 */
enum class PhotoOrientation(val value: String) {
    PORTRAIT("portrait"),
    LANDSCAPE("landscape"),
    SQUARE("square");

    companion object {
        /**
         * Orientation implied by **pixel layout** of the bitmap actually passed to generated room / saved as thumbnail.
         *
         * Use this for room **metadata and generated room viewer viewer** after decode (and optional EXIF rotation).
         */
        fun fromBitmapDimensions(bitmap: Bitmap): PhotoOrientation {
            val w = bitmap.width
            val h = bitmap.height
            return when {
                h > w -> PORTRAIT
                w > h -> LANDSCAPE
                else -> SQUARE
            }
        }

        /**
         * Ultra-wide (0.5×) photos are often encoded wider than tall while the user held the phone vertically.
         * When [ultraWideLens] is true, treat an automatic [LANDSCAPE] classification as [PORTRAIT].
         * Do **not** use this when the user explicitly chose landscape (see SinglePhotoRoom orientation tap).
         */
        fun coercePortraitForUltraWide(orientation: PhotoOrientation, ultraWideLens: Boolean): PhotoOrientation {
            if (!ultraWideLens) return orientation
            return if (orientation == LANDSCAPE) PORTRAIT else orientation
        }

        /**
         * Decode a full-resolution bitmap and apply JPEG/WebP **EXIF orientation** so pixels match what
         * the user sees in the gallery (upright portrait, etc.).
         *
         * [BitmapFactory.decodeStream] ignores EXIF; applying it here keeps generated rooms aligned
         * with the upright photo the user saw in the picker.
         */
        fun loadBitmapApplyingExif(context: Context, uri: Uri, maxDimensionPx: Int = Int.MAX_VALUE): Bitmap? {
            val options = BitmapFactory.Options().apply { inSampleSize = 1 }
            if (maxDimensionPx != Int.MAX_VALUE) {
                val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri).use { stream ->
                    if (stream != null) BitmapFactory.decodeStream(stream, null, bounds)
                }
                options.inSampleSize = calculateInSampleSize(bounds.outWidth, bounds.outHeight, maxDimensionPx)
            }
            val bitmap = context.contentResolver.openInputStream(uri).use { stream ->
                if (stream == null) null else BitmapFactory.decodeStream(stream, null, options)
            } ?: return null
            val rotation = try {
                context.contentResolver.openInputStream(uri).use { stream ->
                    if (stream == null) 0 else ExifInterface(stream).rotationDegrees
                }
            } catch (_: Exception) {
                0
            }
            return applyExifRotation(bitmap, rotation)
        }

        private fun calculateInSampleSize(rawWidth: Int, rawHeight: Int, maxDimensionPx: Int): Int {
            if (rawWidth <= 0 || rawHeight <= 0 || maxDimensionPx <= 0) return 1
            var sample = 1
            var sampledWidth = rawWidth
            var sampledHeight = rawHeight
            while (sampledWidth > maxDimensionPx || sampledHeight > maxDimensionPx) {
                sample *= 2
                sampledWidth /= 2
                sampledHeight /= 2
            }
            return sample.coerceAtLeast(1)
        }

        private fun applyExifRotation(bitmap: Bitmap, rotationDegrees: Int): Bitmap {
            val r = ((rotationDegrees % 360) + 360) % 360
            if (r == 0) return bitmap
            val matrix = Matrix().apply { postRotate(r.toFloat()) }
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            if (!bitmap.isRecycled && rotated != bitmap) bitmap.recycle()
            return rotated
        }
    }

    val isLandscape: Boolean
        get() = this == LANDSCAPE

    val displayName: String
        get() = when (this) {
            PORTRAIT -> "Portrait"
            LANDSCAPE -> "Landscape"
            SQUARE -> "Square"
        }

    val heldDescription: String
        get() = when (this) {
            PORTRAIT -> "held vertically"
            LANDSCAPE -> "held horizontally"
            SQUARE -> ""
        }
}
