package com.furnit.android.theme

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.ViewOutlineProvider
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import com.furnit.android.R

/**
 * Viewer toolbar chrome — mirrors iOS `PaafektViewerToolbar` + `Theme.Space`.
 * iOS is source of truth: one glass capsule (center), quiet monoline icons inside,
 * floating back (leading), optional trailing actions outside the capsule.
 */
object PaafektSpace {
    fun xs(context: Context) = dp(context, 4)
    fun sm(context: Context) = dp(context, 8)
    fun md(context: Context) = dp(context, 12)
    fun lg(context: Context) = dp(context, 16)
    fun xl(context: Context) = dp(context, 24)
    fun xxl(context: Context) = dp(context, 32)

    /** Top inset below status bar — matches iOS viewer `.padding()` + safe area (~48dp). */
    fun viewerTopInset(context: Context) = dp(context, 48)

    /** Bottom inset above home indicator for hero bar. */
    fun viewerBottomInset(context: Context) = dp(context, 20)

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}

object PaafektViewerToolbar {

    fun createFloatingBackButton(context: Context, onClick: () -> Unit): TextView {
        return TextView(context).apply {
            text = "‹"
            textSize = 24f
            gravity = Gravity.CENTER
            setTextColor(PaafektColors.textPrimary)
            background = PaafektDrawables.toolbarCircle()
            setOnClickListener { onClick() }
            layoutParams = FrameLayout.LayoutParams(dp(context, 36), dp(context, 36))
        }
    }

    /** Monoline icon inside the shared glass capsule — no per-icon circle (matches iOS). */
    fun createCapsuleIconButton(
        context: Context,
        @DrawableRes iconResId: Int,
        contentDescription: CharSequence? = null,
        isActive: Boolean = false,
        onClick: () -> Unit,
    ): ImageButton {
        return ImageButton(context).apply {
            setImageResource(iconResId)
            imageTintList = ContextCompat.getColorStateList(
                context,
                if (isActive) R.color.paafekt_accent else android.R.color.white,
            )
            background = null
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dp(context, 7), dp(context, 7), dp(context, 7), dp(context, 7))
            layoutParams = LinearLayout.LayoutParams(dp(context, 36), dp(context, 36)).apply {
                marginStart = dp(context, 4)
                marginEnd = dp(context, 4)
            }
            contentDescription?.let { this.contentDescription = it }
            setOnClickListener { onClick() }
        }
    }

    /** Floating circle for trailing actions outside the capsule (recenter, AR sizing). */
    fun createFloatingIconButton(
        context: Context,
        @DrawableRes iconResId: Int,
        contentDescription: CharSequence? = null,
        isActive: Boolean = false,
        activeFillColor: Int? = null,
        onClick: () -> Unit,
    ): ImageButton {
        return ImageButton(context).apply {
            setImageResource(iconResId)
            imageTintList = ContextCompat.getColorStateList(context, android.R.color.white)
            background = if (isActive && activeFillColor != null) {
                android.graphics.drawable.GradientDrawable().apply {
                    shape = android.graphics.drawable.GradientDrawable.OVAL
                    setColor(activeFillColor)
                }
            } else {
                PaafektDrawables.toolbarCircle()
            }
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dp(context, 7), dp(context, 7), dp(context, 7), dp(context, 7))
            layoutParams = LinearLayout.LayoutParams(dp(context, 36), dp(context, 36)).apply {
                marginStart = dp(context, 4)
            }
            contentDescription?.let { this.contentDescription = it }
            setOnClickListener { onClick() }
        }
    }

    fun createToolbarCapsule(context: Context): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = PaafektDrawables.toolbarCapsule()
            setPadding(PaafektSpace.sm(context), dp(context, 4), PaafektSpace.sm(context), dp(context, 4))
        }
    }

    fun createTopChromeRow(context: Context): FrameLayout {
        return FrameLayout(context).apply {
            setPadding(PaafektSpace.lg(context), PaafektSpace.viewerTopInset(context), PaafektSpace.lg(context), 0)
        }
    }

    /**
     * Resting summon affordance — mirrors iOS `PaafektImmersiveGoldSummonButton`:
     * 46dp gold disk, 22dp chevron inset, dark glyph, drop shadow.
     */
    fun createGoldSummonButton(
        context: Context,
        contentDescription: CharSequence,
        onClick: () -> Unit,
    ): ImageButton {
        val size = dp(context, 46)
        val iconInset = dp(context, 12)
        return ImageButton(context, null, 0).apply {
            setImageResource(R.drawable.ic_chevron_up)
            imageTintList = ColorStateList.valueOf(PaafektColors.accentText)
            background = GradientDrawable().apply {
                shape = GradientDrawable.OVAL
                setColor(PaafektColors.accent)
            }
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(iconInset, iconInset, iconInset, iconInset)
            this.contentDescription = contentDescription
            layoutParams = goldSummonButtonLayoutParams(context)
            ViewCompat.setElevation(this, dp(context, 6).toFloat())
            outlineProvider = ViewOutlineProvider.BACKGROUND
            clipToOutline = false
            minimumWidth = size
            minimumHeight = size
            setOnClickListener { onClick() }
        }
    }

    /** Bottom-right placement — iOS `.padding(.horizontal/.bottom, Theme.Space.lg)`. */
    fun goldSummonButtonLayoutParams(
        context: Context,
        systemBarBottomInset: Int = 0,
    ): FrameLayout.LayoutParams {
        return FrameLayout.LayoutParams(dp(context, 46), dp(context, 46)).apply {
            gravity = Gravity.END or Gravity.BOTTOM
            bottomMargin = systemBarBottomInset + PaafektSpace.lg(context)
            marginEnd = PaafektSpace.lg(context)
        }
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}
