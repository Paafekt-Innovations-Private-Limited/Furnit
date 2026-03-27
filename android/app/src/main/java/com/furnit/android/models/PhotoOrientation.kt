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
 * Uses encoded JPEG width/height plus [ExifInterface.getRotationDegrees] so portrait shots stored
 * as a landscape sensor buffer (common) still classify as portrait when EXIF is present.
 *
 * **Portrait-first bias:** If there is no EXIF rotation (0°) but pixels are stored landscape
 * (width > height), we still treat as **portrait** — typical for upright phone captures when OEM/camera
 * strips or omits orientation. Users can tap the indicator to switch (SinglePhotoRoom). True landscape
 * shots without EXIF can be corrected with the same tap.
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
         * Display width/height after applying EXIF rotation (90/270 swap dimensions; 0/180 keep order).
         */
        private fun displayDimensions(rawWidth: Int, rawHeight: Int, rotationDegrees: Int): Pair<Int, Int> {
            val r = ((rotationDegrees % 360) + 360) % 360
            return if (r == 90 || r == 270) {
                rawHeight to rawWidth
            } else {
                rawWidth to rawHeight
            }
        }

        /**
         * 1) Bounds-decode width/height.
         * 2) Read [ExifInterface.getRotationDegrees] (AndroidX; reliable on all app minSdk levels).
         * 3) Derive display aspect; if still landscape-encoded with **no** rotation, prefer **portrait**
         *    for this app’s primary use case (phone held straight).
         */
        /**
         * Orientation implied by **pixel layout** of the bitmap actually passed to SHARP / saved as thumbnail.
         *
         * Use this for room **metadata and SharpRoom viewer** after decode (and optional EXIF rotation).
         * [detect] on the file URI can disagree: e.g. portrait-first bias when EXIF rotation is 0 but the
         * buffer is still landscape-wide — that caused ~90° tilt (viewer thought portrait, PLY from landscape tensor).
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
         * Same EXIF + encoded-dimension rules as [detect], for a filesystem path (gallery export, SharpInference, temp camera file).
         */
        fun detectFromFile(imagePath: String): PhotoOrientation {
            val rawWidth: Int
            val rawHeight: Int
            val rotationDegrees: Int
            try {
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                BitmapFactory.decodeFile(imagePath, options)
                rawWidth = options.outWidth
                rawHeight = options.outHeight
                rotationDegrees = try {
                    ExifInterface(imagePath).rotationDegrees
                } catch (_: Exception) {
                    0
                }
            } catch (_: Exception) {
                return PORTRAIT
            }
            return orientationFromEncodedPixels(rawWidth, rawHeight, rotationDegrees)
        }

        fun detect(context: Context, uri: Uri): PhotoOrientation {
            val rawWidth: Int
            val rawHeight: Int
            val rotationDegrees: Int
            try {
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream, null, options)
                }
                rawWidth = options.outWidth
                rawHeight = options.outHeight
                rotationDegrees = context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    ExifInterface(inputStream).rotationDegrees
                } ?: 0
            } catch (_: Exception) {
                return PORTRAIT
            }

            return orientationFromEncodedPixels(rawWidth, rawHeight, rotationDegrees)
        }

        /**
         * Classify from **stored** JPEG/WebP pixel dimensions and EXIF rotation (before decode rotates pixels).
         * Matches iOS / Swift metadata path where gallery orientation comes from EXIF, not from decoded bitmap size alone.
         */
        private fun orientationFromEncodedPixels(rawWidth: Int, rawHeight: Int, rotationDegrees: Int): PhotoOrientation {
            if (rawWidth <= 0 || rawHeight <= 0) return PORTRAIT

            var (displayWidth, displayHeight) = displayDimensions(rawWidth, rawHeight, rotationDegrees)

            // No EXIF rotation but file is stored as landscape buffer → assume portrait (upright phone).
            if (rotationDegrees == 0 && rawWidth > rawHeight) {
                displayWidth = rawHeight
                displayHeight = rawWidth
            }

            return when {
                displayWidth > displayHeight -> LANDSCAPE
                displayHeight > displayWidth -> PORTRAIT
                else -> SQUARE
            }
        }

        /**
         * Decode a full-resolution bitmap and apply JPEG/WebP **EXIF orientation** so pixels match what
         * the user sees in the gallery (upright portrait, etc.).
         *
         * [BitmapFactory.decodeStream] ignores EXIF; Vulkan / ExecuTorch SHARP were fed the raw sensor
         * buffer while [SharpRoomActivity] rotated the PLY for **display** orientation → ~90° mismatch
         * for typical portrait camera JPEGs.
         */
        fun loadBitmapApplyingExif(context: Context, uri: Uri): Bitmap? {
            val bitmap = context.contentResolver.openInputStream(uri).use { stream ->
                if (stream == null) null else BitmapFactory.decodeStream(stream)
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

        /** Same as [loadBitmapApplyingExif] for a filesystem path (e.g. SharpInferenceActivity). */
        fun loadBitmapApplyingExifFromFile(imagePath: String): Bitmap? {
            val bitmap = BitmapFactory.decodeFile(imagePath) ?: return null
            val rotation = try {
                ExifInterface(imagePath).rotationDegrees
            } catch (_: Exception) {
                0
            }
            return applyExifRotation(bitmap, rotation)
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
