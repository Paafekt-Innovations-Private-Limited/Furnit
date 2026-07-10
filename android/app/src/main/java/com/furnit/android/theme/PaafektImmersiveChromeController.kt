package com.furnit.android.theme

import android.content.Context
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import androidx.core.view.isVisible

/**
 * Immersive-first viewer chrome: resting (minimal) ↔ summoned (glass toolbar).
 * Auto-hides summoned chrome after ~3s; shorter fades when reduce-motion is on.
 */
class PaafektImmersiveChromeController(
    private val handler: Handler = Handler(Looper.getMainLooper()),
) {
    enum class Phase { RESTING, SUMMONED }

    var phase: Phase = Phase.RESTING
        private set

    var onPhaseChanged: ((Phase) -> Unit)? = null

    private val autoHideRunnable = Runnable { immerse() }
    private val autoHideMs = 3_000L

    fun summon() {
        handler.removeCallbacks(autoHideRunnable)
        if (phase != Phase.SUMMONED) {
            phase = Phase.SUMMONED
            onPhaseChanged?.invoke(phase)
        }
        scheduleAutoHide()
    }

    fun immerse() {
        handler.removeCallbacks(autoHideRunnable)
        if (phase != Phase.RESTING) {
            phase = Phase.RESTING
            onPhaseChanged?.invoke(phase)
        }
    }

    fun toggle() {
        if (phase == Phase.RESTING) summon() else immerse()
    }

    fun noteChromeInteraction() {
        if (phase == Phase.SUMMONED) scheduleAutoHide()
    }

    fun destroy() {
        handler.removeCallbacks(autoHideRunnable)
    }

    private fun scheduleAutoHide() {
        handler.removeCallbacks(autoHideRunnable)
        handler.postDelayed(autoHideRunnable, autoHideMs)
    }

    fun applyPhase(
        context: Context,
        restingViews: List<View>,
        summonedViews: List<View>,
        animate: Boolean = true,
    ) {
        val summoned = phase == Phase.SUMMONED
        val fadeMs = if (context.reduceMotionEnabled()) 150L else 280L
        restingViews.forEach { view ->
            val show = !summoned
            if (view.isVisible != show) {
                if (animate) fadeVisibility(view, show, fadeMs) else view.visibility = if (show) View.VISIBLE else View.GONE
            }
        }
        summonedViews.forEach { view ->
            if (view.isVisible != summoned) {
                if (animate) fadeVisibility(view, summoned, fadeMs) else view.visibility = if (summoned) View.VISIBLE else View.GONE
            }
        }
    }

    private fun fadeVisibility(view: View, show: Boolean, durationMs: Long) {
        view.animate().cancel()
        if (show) {
            view.alpha = 0f
            view.visibility = View.VISIBLE
            view.animate()
                .alpha(1f)
                .setDuration(durationMs)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .start()
        } else {
            view.animate()
                .alpha(0f)
                .setDuration(durationMs)
                .setInterpolator(AccelerateDecelerateInterpolator())
                .withEndAction {
                    view.visibility = View.GONE
                    view.alpha = 1f
                }
                .start()
        }
    }

    private fun Context.reduceMotionEnabled(): Boolean {
        return try {
            Settings.Global.getFloat(contentResolver, Settings.Global.TRANSITION_ANIMATION_SCALE, 1f) == 0f
        } catch (_: Exception) {
            false
        }
    }
}
