package com.furnit.android.models

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.content.Context
import android.os.Build
import java.io.InputStream

/**
 * Photo orientation enum matching iOS PhotoOrientation
 * Used to detect if a photo was taken in landscape or portrait mode
 */
enum class PhotoOrientation(val value: String) {
    PORTRAIT("portrait"),
    LANDSCAPE("landscape"),
    SQUARE("square");

    companion object {
        // EXIF orientation constants
        private const val ORIENTATION_NORMAL = 1
        private const val ORIENTATION_FLIP_HORIZONTAL = 2
        private const val ORIENTATION_ROTATE_180 = 3
        private const val ORIENTATION_FLIP_VERTICAL = 4
        private const val ORIENTATION_TRANSPOSE = 5
        private const val ORIENTATION_ROTATE_90 = 6
        private const val ORIENTATION_TRANSVERSE = 7
        private const val ORIENTATION_ROTATE_270 = 8

        /**
         * Detect orientation from EXIF data or dimensions
         * Similar to iOS implementation that reads UIImage.imageOrientation
         */
        fun detect(context: Context, uri: Uri): PhotoOrientation {
            // Try EXIF-based detection for API 24+
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                try {
                    val inputStream = context.contentResolver.openInputStream(uri)
                    if (inputStream != null) {
                        val exif = android.media.ExifInterface(inputStream)
                        val orientation = exif.getAttributeInt(
                            android.media.ExifInterface.TAG_ORIENTATION,
                            android.media.ExifInterface.ORIENTATION_NORMAL
                        )
                        inputStream.close()

                        // Map EXIF orientation to our enum
                        // EXIF orientation values:
                        // 1 (NORMAL) - Landscape (sensor native orientation)
                        // 6 (ROTATE_90) - Portrait (phone held vertically)
                        // 3 (ROTATE_180) - Landscape upside down
                        // 8 (ROTATE_270) - Portrait upside down
                        return when (orientation) {
                            ORIENTATION_NORMAL,
                            ORIENTATION_FLIP_HORIZONTAL,
                            ORIENTATION_ROTATE_180,
                            ORIENTATION_FLIP_VERTICAL -> LANDSCAPE

                            ORIENTATION_ROTATE_90,
                            ORIENTATION_TRANSVERSE,
                            ORIENTATION_ROTATE_270,
                            ORIENTATION_TRANSPOSE -> PORTRAIT

                            else -> detectFromDimensions(context, uri)
                        }
                    }
                } catch (e: Exception) {
                    // Fallback to dimension-based detection
                }
            }
            return detectFromDimensions(context, uri)
        }

        /**
         * Fallback detection using image dimensions
         */
        private fun detectFromDimensions(context: Context, uri: Uri): PhotoOrientation {
            try {
                val options = BitmapFactory.Options().apply {
                    inJustDecodeBounds = true
                }
                val inputStream = context.contentResolver.openInputStream(uri)
                BitmapFactory.decodeStream(inputStream, null, options)
                inputStream?.close()

                val width = options.outWidth
                val height = options.outHeight

                return when {
                    width > height -> LANDSCAPE
                    height > width -> PORTRAIT
                    else -> SQUARE
                }
            } catch (e: Exception) {
                return PORTRAIT
            }
        }

        /**
         * Detect from a bitmap directly
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
