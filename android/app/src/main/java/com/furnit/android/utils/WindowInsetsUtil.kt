package com.furnit.android.utils

import android.graphics.Rect
import android.view.View
import android.view.ViewGroup
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat

/**
 * Edge-to-edge is enforced on targetSdk 35+, so activity content draws behind the
 * status and navigation bars. This helper keeps a screen's own padding as the base
 * and adds the system bar insets on top, so top bars stay clear of the phone's
 * notification/charging indicators and scrolled content clears the navigation bar.
 */
object WindowInsetsUtil {

    /**
     * Offsets [contentView]'s padding by the system bar insets, preserving whatever
     * padding the view already had as the base. Safe to call once during setup; the
     * listener re-applies on rotation and multi-window resizes.
     */
    fun applySystemBarInsetsAsPadding(contentView: View) {
        applyInsets(contentView, left = true, top = true, right = true, bottom = true)
    }

    /**
     * Adds only the top (status bar) inset to [contentView]'s existing top padding.
     * Use for chrome anchored to the top of an otherwise full-bleed screen (camera
     * hints, viewer toolbars) where padding the other edges would enlarge the bar.
     */
    fun applyTopInsetAsPadding(contentView: View) {
        applyInsets(contentView, left = false, top = true, right = false, bottom = false)
    }

    /**
     * Adds only the bottom (navigation bar) inset to [contentView]'s existing bottom
     * padding. Use for controls anchored to the bottom of a full-bleed screen.
     */
    fun applyBottomInsetAsPadding(contentView: View) {
        applyInsets(contentView, left = false, top = false, right = false, bottom = true)
    }

    /**
     * Adds the top (status bar) inset to [contentView]'s existing top margin. Use for
     * floating chrome positioned with a top margin rather than padding (e.g. circular
     * back buttons) where padding would reshape the view. The view must already have
     * [android.view.ViewGroup.MarginLayoutParams] assigned before this is called.
     */
    fun applyTopInsetAsTopMargin(contentView: View) {
        val baseTopMargin =
            (contentView.layoutParams as? ViewGroup.MarginLayoutParams)?.topMargin ?: 0
        ViewCompat.setOnApplyWindowInsetsListener(contentView) { view, windowInsets ->
            val systemBarInsets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            (view.layoutParams as? ViewGroup.MarginLayoutParams)?.let { marginParams ->
                marginParams.topMargin = baseTopMargin + systemBarInsets.top
                view.layoutParams = marginParams
            }
            windowInsets
        }
    }

    /**
     * For a scroll container on an edge-to-edge screen: pads by the system bar insets and,
     * when the soft keyboard (IME) is open, extends the bottom padding to cover it and
     * scrolls content into the remaining visible area. Attach to the ScrollView (not its
     * inner column) so the scroll offset respects the reserved keyboard space.
     * Requires the activity to use windowSoftInputMode="adjustResize".
     *
     * @param bringIntoViewWhenImeVisible optional view to keep above the keyboard (e.g. a
     * primary action button below the focused field). Falls back to the focused view.
     */
    fun applyImeAwareInsetsAsPadding(
        scrollView: View,
        bringIntoViewWhenImeVisible: (() -> View?)? = null,
    ) {
        val basePaddingLeft = scrollView.paddingLeft
        val basePaddingTop = scrollView.paddingTop
        val basePaddingRight = scrollView.paddingRight
        val basePaddingBottom = scrollView.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(scrollView) { view, windowInsets ->
            val bars = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = windowInsets.getInsets(WindowInsetsCompat.Type.ime())
            view.setPadding(
                basePaddingLeft + bars.left,
                basePaddingTop + bars.top,
                basePaddingRight + bars.right,
                basePaddingBottom + maxOf(bars.bottom, ime.bottom)
            )
            if (ime.bottom > 0) {
                val target = bringIntoViewWhenImeVisible?.invoke() ?: view.findFocus()
                target?.let { bringViewAboveIme(it) }
            }
            windowInsets
        }
        ViewCompat.requestApplyInsets(scrollView)
    }

    /** Scrolls [target] (and a bit of space below it) into the visible area above the IME. */
    fun bringViewAboveIme(target: View) {
        target.post {
            val extraBelow = (72 * target.resources.displayMetrics.density).toInt()
            target.requestRectangleOnScreen(
                Rect(0, 0, target.width, target.height + extraBelow),
                true,
            )
        }
    }

    private fun applyInsets(
        contentView: View,
        left: Boolean,
        top: Boolean,
        right: Boolean,
        bottom: Boolean
    ) {
        val basePaddingLeft = contentView.paddingLeft
        val basePaddingTop = contentView.paddingTop
        val basePaddingRight = contentView.paddingRight
        val basePaddingBottom = contentView.paddingBottom
        ViewCompat.setOnApplyWindowInsetsListener(contentView) { view, windowInsets ->
            val systemBarInsets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            view.setPadding(
                basePaddingLeft + (if (left) systemBarInsets.left else 0),
                basePaddingTop + (if (top) systemBarInsets.top else 0),
                basePaddingRight + (if (right) systemBarInsets.right else 0),
                basePaddingBottom + (if (bottom) systemBarInsets.bottom else 0)
            )
            windowInsets
        }
    }
}
