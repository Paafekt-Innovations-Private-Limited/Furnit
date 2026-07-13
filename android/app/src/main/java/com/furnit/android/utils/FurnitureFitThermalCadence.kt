package com.furnit.android.utils

import android.content.Context
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock

/**
 * Mirrors iOS `FurnitureFitView` RTMDet live thermal cadence:
 * - nominal/fair → 200ms (~5 fps) minimum between inference starts
 * - serious → 400ms (~2.5 fps)
 * - critical → pause inference (keep last overlay)
 *
 * Android [PowerManager] thermal ladder (API 29+):
 * NONE/LIGHT ≈ nominal/fair, MODERATE ≈ serious, SEVERE+ ≈ critical.
 * Below API 29, only the nominal 200ms gate is applied.
 */
class FurnitureFitThermalCadence(
    private val logTag: String = "FurnitureFitThermal",
) {
    companion object {
        const val NOMINAL_INTERVAL_MS = 200L
        const val SERIOUS_INTERVAL_MS = 400L
    }

    private val lock = Any()
    @Volatile private var targetIntervalMs = NOMINAL_INTERVAL_MS
    @Volatile private var pausedForCritical = false
    private var lastInferenceAtMs = 0L
    private var powerManager: PowerManager? = null
    private var thermalListener: PowerManager.OnThermalStatusChangedListener? = null

    val isPausedForThermalCritical: Boolean
        get() = pausedForCritical

    val currentTargetIntervalMs: Long
        get() = targetIntervalMs

    fun start(context: Context) {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            applyStatus(PowerManager.THERMAL_STATUS_NONE)
            return
        }
        val pm = context.applicationContext.getSystemService(Context.POWER_SERVICE) as? PowerManager
            ?: return
        if (thermalListener != null && powerManager === pm) {
            applyStatus(pm.currentThermalStatus)
            return
        }
        stop()
        powerManager = pm
        val listener = PowerManager.OnThermalStatusChangedListener { status ->
            applyStatus(status)
        }
        thermalListener = listener
        pm.addThermalStatusListener(listener)
        applyStatus(pm.currentThermalStatus)
    }

    fun stop() {
        val pm = powerManager
        val listener = thermalListener
        if (pm != null && listener != null && Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            try {
                pm.removeThermalStatusListener(listener)
            } catch (_: Exception) {
            }
        }
        thermalListener = null
        powerManager = null
    }

    /**
     * Returns true and records the start time when inference is allowed.
     * Call on the analysis thread before beginning work; single-threaded analyzers need no extra sync.
     */
    fun tryBeginInference(nowMs: Long = SystemClock.elapsedRealtime()): Boolean {
        if (pausedForCritical) return false
        synchronized(lock) {
            if (nowMs - lastInferenceAtMs < targetIntervalMs) return false
            lastInferenceAtMs = nowMs
            return true
        }
    }

    fun shouldAcceptInference(nowMs: Long = SystemClock.elapsedRealtime()): Boolean {
        if (pausedForCritical) return false
        synchronized(lock) {
            return nowMs - lastInferenceAtMs >= targetIntervalMs
        }
    }

    private fun applyStatus(status: Int) {
        val previousPaused = pausedForCritical
        val previousInterval = targetIntervalMs
        when (status) {
            PowerManager.THERMAL_STATUS_NONE,
            PowerManager.THERMAL_STATUS_LIGHT,
            -> {
                targetIntervalMs = NOMINAL_INTERVAL_MS
                pausedForCritical = false
            }
            PowerManager.THERMAL_STATUS_MODERATE,
            -> {
                targetIntervalMs = SERIOUS_INTERVAL_MS
                pausedForCritical = false
            }
            PowerManager.THERMAL_STATUS_SEVERE,
            PowerManager.THERMAL_STATUS_CRITICAL,
            PowerManager.THERMAL_STATUS_EMERGENCY,
            PowerManager.THERMAL_STATUS_SHUTDOWN,
            -> {
                pausedForCritical = true
            }
            else -> {
                targetIntervalMs = NOMINAL_INTERVAL_MS
                pausedForCritical = false
            }
        }
        if (previousPaused != pausedForCritical || previousInterval != targetIntervalMs) {
            LogUtil.d(
                logTag,
                "thermalStatus=$status → interval=${targetIntervalMs}ms paused=$pausedForCritical",
            )
        }
    }
}
