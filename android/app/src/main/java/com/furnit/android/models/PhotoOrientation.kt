package com.furnit.android.models

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.content.Context
import android.os.Build

/**
 * Photo orientation enum matching Ultralytics YOLO / exif_size logic:
 * Orientation is determined by the **display** dimensions (width vs height).
 * - Landscape: width > height (aspect_ratio > 1)
 * - Portrait: height > width (aspect_ratio <= 1 for non-square)
 * - Square: width == height
 *
 * When EXIF rotation is 5, 6, 7, or 8 (90° or 270°), width and height are swapped
 * so the comparison reflects the true visual orientation (see ultralytics.data.utils.exif_size).
 */
enum class PhotoOrientation(val value: String) {
    PORTRAIT("portrait"),
    LANDSCAPE("landscape"),
    SQUARE("square");

    companion object {
        // EXIF orientation constants (same as Android ExifInterface)
        private const val ORIENTATION_NORMAL = 1
        private const val ORIENTATION_FLIP_HORIZONTAL = 2
        private const val ORIENTATION_ROTATE_180 = 3
        private const val ORIENTATION_FLIP_VERTICAL = 4
        private const val ORIENTATION_TRANSPOSE = 5
        private const val ORIENTATION_ROTATE_90 = 6
        private const val ORIENTATION_TRANSVERSE = 7
        private const val ORIENTATION_ROTATE_270 = 8

        /** EXIF values that require swapping width/height to get display dimensions (like exif_size). */
        private val EXIF_SWAP_DIMENSIONS = setOf(
            ORIENTATION_TRANSPOSE, ORIENTATION_ROTATE_90,
            ORIENTATION_TRANSVERSE, ORIENTATION_ROTATE_270
        )

        /**
         * Detect orientation using Ultralytics-style logic:
         * 1) Read raw width/height (decode bounds).
         * 2) If EXIF orientation is 5, 6, 7, or 8, swap width and height (display dimensions).
         * 3) Compare: width > height → LANDSCAPE, height > width → PORTRAIT, else SQUARE.
         */
        fun detect(context: Context, uri: Uri): PhotoOrientation {
            var rawWidth = 0
            var rawHeight = 0
            var exifOrientation = ORIENTATION_NORMAL

            try {
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream, null, options)
                    rawWidth = options.outWidth
                    rawHeight = options.outHeight
                }
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                    context.contentResolver.openInputStream(uri)?.use { inputStream ->
                        val exif = android.media.ExifInterface(inputStream)
                        exifOrientation = exif.getAttributeInt(
                            android.media.ExifInterface.TAG_ORIENTATION,
                            android.media.ExifInterface.ORIENTATION_NORMAL
                        )
                    }
                }
            } catch (_: Exception) {
                return PORTRAIT
            }

            if (rawWidth <= 0 || rawHeight <= 0) return PORTRAIT

            val (displayWidth, displayHeight) = if (exifOrientation in EXIF_SWAP_DIMENSIONS) {
                rawHeight to rawWidth
            } else {
                rawWidth to rawHeight
            }

            return when {
                displayWidth > displayHeight -> LANDSCAPE
                displayHeight > displayWidth -> PORTRAIT
                else -> SQUARE
            }
        }

        /**
         * Fallback using raw dimensions only (no EXIF; e.g. when EXIF not available).
         * Landscape: width > height, Portrait: height > width, Square: equal.
         */
        private fun detectFromDimensions(context: Context, uri: Uri): PhotoOrientation {
            try {
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream, null, options)
                    val w = options.outWidth
                    val h = options.outHeight
                    return when {
                        w > h -> LANDSCAPE
                        h > w -> PORTRAIT
                        else -> SQUARE
                    }
                }
            } catch (_: Exception) { }
            return PORTRAIT
        }

        /**
         * Detect from a bitmap (e.g. already decoded). Uses width vs height only.
         * Use detect(uri) when EXIF correction is needed for the source image.
         */
        fun detectFromBitmap(bitmap: Bitmap): PhotoOrientation {
            return when {
                bitmap.width > bitmap.height -> LANDSCAPE
                bitmap.height > bitmap.width -> PORTRAIT
                else -> SQUARE
            }
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
