package com.furnit.android.theme

import android.animation.AnimatorListenerAdapter
import android.content.Context
import android.content.SharedPreferences
import android.graphics.Typeface
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

    /** Gold capsule hero action — icon + label (matches iOS `PaafektViewerHeroButton`). */
    fun createHeroButton(
        context: Context,
        @DrawableRes iconRes: Int,
        label: CharSequence,
        isActive: Boolean = false,
        onClick: () -> Unit,
    ): LinearLayout {
        val row = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER
            background = PaafektDrawables.heroButton(isActive)
            setPadding(dp(context, 16), dp(context, 12), dp(context, 16), dp(context, 12))
            isClickable = true
            isFocusable = true
            setOnClickListener { onClick() }
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = dp(context, 6)
            }
        }

        val icon = ImageView(context).apply {
            setImageResource(iconRes)
            imageTintList = ContextCompat.getColorStateList(context, R.color.paafekt_accent_text)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            layoutParams = LinearLayout.LayoutParams(dp(context, 18), dp(context, 18))
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
        }
        row.addView(icon)

        val title = TextView(context).apply {
            text = label
            textSize = 15f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            setPadding(dp(context, 8), 0, 0, 0)
            maxLines = 1
            ellipsize = TextUtils.TruncateAt.END
            includeFontPadding = false
        }
        row.addView(title)
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
        showPositioned(context, iconRes, textRes, Gravity.TOP or Gravity.CENTER_HORIZONTAL, topMarginDp = topMarginDp, durationMs = durationMs)
    }

    fun showBottomCentered(
        context: Context,
        @DrawableRes iconRes: Int,
        @StringRes textRes: Int,
        bottomMarginDp: Int = 120,
        durationMs: Long = 3500L,
    ) {
        showPositioned(context, iconRes, textRes, Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL, bottomMarginDp = bottomMarginDp, durationMs = durationMs)
    }

    private fun showPositioned(
        context: Context,
        @DrawableRes iconRes: Int,
        @StringRes textRes: Int,
        gravity: Int,
        topMarginDp: Int = 0,
        bottomMarginDp: Int = 0,
        durationMs: Long = 3500L,
    ) {
        hide(animated = false)
        val chip = PaafektHintViews.createChip(context, iconRes, context.getString(textRes))
        chip.alpha = 0f

        val params = FrameLayout.LayoutParams(
            ViewGroup.LayoutParams.WRAP_CONTENT,
            ViewGroup.LayoutParams.WRAP_CONTENT,
        ).apply {
            this.gravity = gravity
            topMargin = dp(context, topMarginDp)
            bottomMargin = dp(context, bottomMarginDp)
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

object PaafektViewerOnboarding {
    const val FIRST_RUN_COACH_SEEN_KEY = "paafekt_viewer_first_run_coach_seen"

    fun prefs(context: Context): SharedPreferences =
        context.getSharedPreferences("paafekt_viewer", Context.MODE_PRIVATE)

    fun hasSeenFirstRunCoach(context: Context): Boolean =
        prefs(context).getBoolean(FIRST_RUN_COACH_SEEN_KEY, false)

    fun markFirstRunCoachSeen(context: Context) {
        prefs(context).edit().putBoolean(FIRST_RUN_COACH_SEEN_KEY, true).apply()
    }
}

/**
 * One-time centered coach mark over a dimmed scrim — matches iOS `PaafektViewerFirstRunCoachMark`.
 */
class PaafektFirstRunCoachMarkController(
    private val host: FrameLayout,
) {
    private var overlayView: View? = null

    fun showIfNeeded(context: Context, onDismissed: () -> Unit) {
        if (PaafektViewerOnboarding.hasSeenFirstRunCoach(context)) {
            onDismissed()
            return
        }
        hide()
        val scrim = FrameLayout(context).apply {
            setBackgroundColor(PaafektColors.background)
            alpha = 0.72f
            isClickable = true
            isFocusable = true
        }

        val card = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            background = PaafektDrawables.secondaryButton()
            setPadding(dp(context, 24), dp(context, 24), dp(context, 24), dp(context, 24))
        }

        val mark = ImageView(context).apply {
            setImageResource(R.drawable.paafekt_login_mark)
            imageTintList = ContextCompat.getColorStateList(context, R.color.paafekt_accent)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            layoutParams = LinearLayout.LayoutParams(dp(context, 56), dp(context, 56))
        }
        card.addView(mark)

        val title = TextView(context).apply {
            text = context.getString(R.string.room_viewer_first_run_title)
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
            setPadding(0, dp(context, 16), 0, dp(context, 8))
        }
        card.addView(title)

        val body = TextView(context).apply {
            text = context.getString(R.string.room_viewer_first_run_body)
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, 0, 0, dp(context, 16))
        }
        card.addView(body)

        val gotIt = TextView(context).apply {
            text = context.getString(R.string.room_viewer_first_run_got_it)
            textSize = 15f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accentText)
            gravity = Gravity.CENTER
            background = PaafektDrawables.heroButton()
            setPadding(dp(context, 32), dp(context, 12), dp(context, 32), dp(context, 12))
            isClickable = true
            isFocusable = true
            setOnClickListener {
                PaafektViewerOnboarding.markFirstRunCoachSeen(context)
                hide()
                onDismissed()
            }
        }
        card.addView(gotIt)

        scrim.addView(
            card,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.CENTER
                marginStart = dp(context, 24)
                marginEnd = dp(context, 24)
            },
        )

        host.addView(
            scrim,
            FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            ),
        )
        host.bringChildToFront(scrim)
        overlayView = scrim
    }

    fun hide() {
        overlayView?.let { host.removeView(it) }
        overlayView = null
    }

    fun bringToFront() {
        overlayView?.let { host.bringChildToFront(it) }
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}
