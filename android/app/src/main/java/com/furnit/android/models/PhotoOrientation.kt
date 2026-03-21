package com.furnit.android.models

import android.content.Context
import android.graphics.BitmapFactory
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
        fun detect(context: Context, uri: Uri): PhotoOrientation {
            var rawWidth = 0
            var rawHeight = 0
            var rotationDegrees = 0

            try {
                val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    BitmapFactory.decodeStream(inputStream, null, options)
                    rawWidth = options.outWidth
                    rawHeight = options.outHeight
                }
                context.contentResolver.openInputStream(uri)?.use { inputStream ->
                    val exif = ExifInterface(inputStream)
                    rotationDegrees = exif.rotationDegrees
                }
            } catch (_: Exception) {
                return PORTRAIT
            }

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
