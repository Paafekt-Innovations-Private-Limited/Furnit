package com.furnit.android.theme

import android.animation.AnimatorListenerAdapter
import android.content.Context
import android.text.TextUtils
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import androidx.core.content.ContextCompat
import com.furnit.android.R

/**
 * Glass-style hint chip for viewer overlays — matches iOS `PaafektHintChip`.
 * Screen-space overlay only; single hint at a time; auto-dismiss after ~3.5s or first interaction.
 */
object PaafektHintViews {
    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()

    fun createChip(context: Context, @DrawableRes iconRes: Int, text: CharSequence): LinearLayout {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = PaafektDrawables.hintChip()
            setPadding(dp(context, 12), dp(context, 8), dp(context, 12), dp(context, 8))
            elevation = 8f
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
        }

        val icon = ImageView(context).apply {
            setImageResource(iconRes)
            imageTintList = ContextCompat.getColorStateList(context, R.color.paafekt_accent)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            layoutParams = LinearLayout.LayoutParams(dp(context, 22), dp(context, 22))
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        row.addView(icon)

        val label = TextView(context).apply {
            this.text = text
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
            setPadding(dp(context, 8), 0, 0, 0)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            includeFontPadding = false
        }
        row.addView(
            label,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )
        return row
    }
}

/**
 * Manages one transient hint chip on the 2D screen overlay layer.
 */
class PaafektHintController(
    private val host: FrameLayout,
    private val topMarginDp: Int = 96,
) {
    private var chipView: View? = null
    private val dismissRunnable = Runnable { hide(animated = true) }
    private var interactionHookInstalled = false

    fun show(
        context: Context,
        @DrawableRes iconRes: Int,
        @StringRes textRes: Int,
        durationMs: Long = 3500L,
    ) {
        hide(animated = false)
        val chip = PaafektHintViews.createChip(context, iconRes, context.getString(textRes))
        chip.alpha = 0f

        val params = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply {
            gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
            topMargin = dp(context, topMarginDp)
            marginStart = dp(context, 16)
            marginEnd = dp(context, 16)
        }

        host.addView(chip, params)
        chipView = chip
        host.bringChildToFront(chip)
        val fadeMs = if (areAnimationsEnabled(context)) 200L else 0L
        if (fadeMs > 0L) {
            chip.animate().alpha(1f).setDuration(fadeMs).start()
        } else {
            chip.alpha = 1f
        }

        host.removeCallbacks(dismissRunnable)
        host.postDelayed(dismissRunnable, durationMs)
        ensureInteractionDismiss(host)
    }

    fun toggle(
        context: Context,
        @DrawableRes iconRes: Int,
        @StringRes textRes: Int,
        durationMs: Long = 3500L,
    ) {
        if (chipView != null) {
            hide(animated = areAnimationsEnabled(context))
        } else {
            show(context, iconRes, textRes, durationMs)
        }
    }

    fun hide(animated: Boolean = true) {
        host.removeCallbacks(dismissRunnable)
        val chip = chipView ?: return
        chipView = null
        if (animated && areAnimationsEnabled(host.context)) {
            chip.animate()
                .alpha(0f)
                .setDuration(150)
                .setListener(object : AnimatorListenerAdapter() {
                    override fun onAnimationEnd(animation: android.animation.Animator) {
                        host.removeView(chip)
                    }
                })
                .start()
        } else {
            host.removeView(chip)
        }
    }

    val isVisible: Boolean
        get() = chipView != null

    fun bringToFront() {
        chipView?.let { host.bringChildToFront(it) }
    }

    private fun ensureInteractionDismiss(root: View) {
        if (interactionHookInstalled) return
        interactionHookInstalled = true
        root.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN && chipView != null) {
                hide(animated = areAnimationsEnabled(root.context))
            }
            false
        }
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()

    private fun areAnimationsEnabled(context: Context): Boolean {
        val duration = android.provider.Settings.Global.getFloat(
            context.contentResolver,
            android.provider.Settings.Global.ANIMATOR_DURATION_SCALE,
            1f,
        )
        return duration > 0f
    }
}
