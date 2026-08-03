package com.furnit.android.utils

import android.content.Context
import androidx.annotation.StringRes
import com.furnit.android.R

/** Localized display labels for detector classes that can reach the furniture UI. */
object FurnitureClassNames {
    @StringRes
    private fun resourceFor(classId: Int): Int? = when (classId) {
        56 -> R.string.furniture_class_chair
        57 -> R.string.furniture_class_couch
        58 -> R.string.furniture_class_potted_plant
        59 -> R.string.furniture_class_bed
        60 -> R.string.furniture_class_dining_table
        61 -> R.string.furniture_class_toilet
        62 -> R.string.furniture_class_tv
        68 -> R.string.furniture_class_microwave
        69 -> R.string.furniture_class_oven
        70 -> R.string.furniture_class_toaster
        71 -> R.string.furniture_class_sink
        72 -> R.string.furniture_class_refrigerator
        else -> null
    }

    fun localized(context: Context, classId: Int, fallback: String): String =
        resourceFor(classId)?.let(context::getString) ?: fallback
}
