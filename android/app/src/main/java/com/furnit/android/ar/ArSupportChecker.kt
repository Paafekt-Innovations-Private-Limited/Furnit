package com.furnit.android.ar

import android.content.Context
import com.google.ar.core.ArCoreApk

/**
 * Whether ARCore is available on this device (installed or installable).
 */
object ArSupportChecker {

    fun isArCoreSupported(context: Context): Boolean {
        return when (ArCoreApk.getInstance().checkAvailability(context)) {
            ArCoreApk.Availability.SUPPORTED_INSTALLED,
            ArCoreApk.Availability.SUPPORTED_APK_TOO_OLD,
            ArCoreApk.Availability.SUPPORTED_NOT_INSTALLED,
            -> true
            else -> false
        }
    }
}
