package com.furnit.android.theme

import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import androidx.annotation.ColorInt

/**
 * Paafekt design tokens — single source of truth for Android.
 * See docs/PAAFEKT_DESIGN_SYSTEM.md
 */
object PaafektColors {
    @ColorInt val background = Color.parseColor("#0E0F12")
    @ColorInt val surface = Color.parseColor("#1A1C20")
    @ColorInt val surfaceHi = Color.parseColor("#24272D")
    @ColorInt val hairline = Color.parseColor("#14FFFFFF")
    @ColorInt val textPrimary = Color.parseColor("#F4F3EF")
    @ColorInt val textSecondary = Color.parseColor("#9BA0A8")
    @ColorInt val accent = Color.parseColor("#C9A24B")
    @ColorInt val accentPressed = Color.parseColor("#A9853A")
    @ColorInt val accentText = Color.parseColor("#0E0F12")
    @ColorInt val success = Color.parseColor("#3E9E6E")
    @ColorInt val danger = Color.parseColor("#C85A54")
    @ColorInt val viewerCapsule = Color.parseColor("#8C000000")
    /** Surface at ~60% alpha — hint chip fill (fallback when backdrop blur unavailable). */
    @ColorInt val surfaceHint = Color.argb(153, 0x1A, 0x1C, 0x20)
}

object PaafektDimens {
    const val radiusControlDp = 12f
    const val radiusSheetDp = 20f
}

object PaafektDrawables {
    fun toolbarCapsule(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = 999f
        setColor(PaafektColors.viewerCapsule)
        setStroke(1, PaafektColors.hairline)
    }

    fun toolbarCircle(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.OVAL
        setColor(PaafektColors.viewerCapsule)
        setStroke(1, PaafektColors.hairline)
    }

    fun primaryButton(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = PaafektDimens.radiusControlDp
        setColor(PaafektColors.accent)
    }

    fun heroButton(isActive: Boolean = false): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = 999f
        setColor(if (isActive) PaafektColors.accentPressed else PaafektColors.accent)
    }

    /** Summoned toolbar compact hero — gold stroke, optional active fill (iOS `PaafektImmersiveCompactHeroAction`). */
    fun compactHeroButton(isActive: Boolean = false): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = PaafektDimens.radiusControlDp
        setColor(if (isActive) Color.argb(46, 0xC9, 0xA2, 0x4B) else Color.TRANSPARENT)
        setStroke((1.5f * 1f).toInt().coerceAtLeast(2), PaafektColors.accent)
    }

    fun secondaryButton(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = PaafektDimens.radiusControlDp
        setColor(PaafektColors.surface)
        setStroke(1, PaafektColors.hairline)
    }

    fun hintChip(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = 999f
        setColor(PaafektColors.surfaceHint)
        setStroke(1, PaafektColors.hairline)
    }

    fun creationCardPrimary(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = PaafektDimens.radiusSheetDp
        setColor(PaafektColors.surface)
        setStroke(1, PaafektColors.accent)
    }

    fun creationCardSecondary(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = PaafektDimens.radiusSheetDp
        setColor(PaafektColors.surface)
        setStroke(1, PaafektColors.hairline)
    }
}
