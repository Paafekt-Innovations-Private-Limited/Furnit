package com.furnit.android.ar

import android.content.Context
import android.hardware.display.DisplayManager
import android.hardware.display.DisplayManager.DisplayListener
import android.view.Display
import android.view.WindowManager
import com.google.ar.core.Session

/**
 * Syncs display rotation with ARCore [Session.setDisplayGeometry].
 * Ported from ARCore hello_ar_java DisplayRotationHelper.
 */
class DisplayRotationHelper(private val context: Context) : DisplayListener {

    private var viewportWidth = 0
    private var viewportHeight = 0
    private var viewportChanged = false
    private var displayRotation = 0

    private val display: Display = resolveDisplay(context)

    init {
        displayRotation = display.rotation
    }

    /**
     * Do not use [Context.getDisplay]: it NPEs on a [android.content.ContextWrapper] before the
     * Activity is attached (e.g. field initializers). Prefer [DisplayManager] / [WindowManager].
     */
    private fun resolveDisplay(ctx: Context): Display {
        val appCtx = ctx.applicationContext
        appCtx.getSystemService(DisplayManager::class.java)?.getDisplay(Display.DEFAULT_DISPLAY)?.let { return it }
        val wm = appCtx.getSystemService(Context.WINDOW_SERVICE) as? WindowManager
            ?: ctx.getSystemService(Context.WINDOW_SERVICE) as WindowManager
        @Suppress("DEPRECATION")
        return wm.defaultDisplay
    }

    fun onResume() {
        val dm = context.getSystemService(DisplayManager::class.java) ?: return
        dm.registerDisplayListener(this, null)
    }

    fun onPause() {
        val dm = context.getSystemService(DisplayManager::class.java) ?: return
        dm.unregisterDisplayListener(this)
    }

    fun onSurfaceChanged(width: Int, height: Int) {
        viewportWidth = width
        viewportHeight = height
        viewportChanged = true
    }

    /**
     * Call each frame before [Session.update].
     * Must re-run when **either** the GL surface size **or** [Display.getRotation] changes; otherwise
     * the camera image is mapped with wrong aspect (stretched / "widened") or wrong orientation (tilted).
     */
    fun updateSessionIfNeeded(session: Session) {
        val currentRotation = display.rotation
        val needUpdate = viewportChanged || currentRotation != displayRotation
        if (!needUpdate) return
        displayRotation = currentRotation
        if (viewportWidth > 0 && viewportHeight > 0) {
            session.setDisplayGeometry(displayRotation, viewportWidth, viewportHeight)
        }
        viewportChanged = false
    }

    override fun onDisplayAdded(displayId: Int) {}

    override fun onDisplayRemoved(displayId: Int) {}

    override fun onDisplayChanged(displayId: Int) {
        if (displayId == display.displayId) {
            displayRotation = display.rotation
            viewportChanged = true
        }
    }
}
