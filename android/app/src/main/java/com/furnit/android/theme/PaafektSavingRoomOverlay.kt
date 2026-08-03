package com.furnit.android.theme

import android.content.Context
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.furnit.android.R
import com.google.android.material.progressindicator.CircularProgressIndicator
import com.google.android.material.progressindicator.LinearProgressIndicator

/**
 * Swift `PaafektSavingRoomOverlay` parity — gold ring, house icon, linear bar, percent.
 */
class PaafektSavingRoomOverlay(context: Context) : FrameLayout(context) {

    private val titleView: TextView
    private val subtitleView: TextView
    private val ringProgress: CircularProgressIndicator
    private val linearProgress: LinearProgressIndicator
    private val percentView: TextView

    init {
        val density = resources.displayMetrics.density
        setBackgroundColor(android.graphics.Color.argb(235, 0x0E, 0x0F, 0x12))
        layoutParams = LayoutParams(
            ViewGroup.LayoutParams.MATCH_PARENT,
            ViewGroup.LayoutParams.MATCH_PARENT,
        )

        val card = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            val pad = (32 * density).toInt()
            setPadding(pad, pad, pad, pad)
            background = PaafektDrawables.secondaryButton().apply {
                cornerRadius = PaafektDimens.radiusSheetDp * density
            }
        }

        val ringSize = (80 * density).toInt()
        val ringFrame = FrameLayout(context).apply {
            layoutParams = LinearLayout.LayoutParams(ringSize, ringSize).apply {
                gravity = Gravity.CENTER_HORIZONTAL
            }
        }
        ringProgress = CircularProgressIndicator(context).apply {
            max = 100
            isIndeterminate = false
            indicatorSize = ringSize
            trackThickness = (6 * density).toInt()
            setIndicatorColor(PaafektColors.accent)
            setTrackColor(android.graphics.Color.argb(64, 0xC9, 0xA2, 0x4B))
            layoutParams = LayoutParams(ringSize, ringSize).apply { gravity = Gravity.CENTER }
        }
        ringFrame.addView(ringProgress)
        ringFrame.addView(
            ImageView(context).apply {
                setImageResource(R.drawable.ic_house)
                layoutParams = LayoutParams(
                    (28 * density).toInt(),
                    (28 * density).toInt(),
                ).apply { gravity = Gravity.CENTER }
            },
        )
        card.addView(ringFrame)

        titleView = TextView(context).apply {
            textSize = 18f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            gravity = Gravity.CENTER
            setPadding(0, (24 * density).toInt(), 0, 0)
        }
        card.addView(titleView)

        subtitleView = TextView(context).apply {
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, (8 * density).toInt(), 0, 0)
        }
        card.addView(subtitleView)

        linearProgress = LinearProgressIndicator(context).apply {
            max = 100
            isIndeterminate = false
            trackThickness = (8 * density).toInt()
            setIndicatorColor(PaafektColors.accent)
            setTrackColor(PaafektColors.surfaceHi)
            layoutParams = LinearLayout.LayoutParams(
                (220 * density).toInt(),
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                topMargin = (20 * density).toInt()
                gravity = Gravity.CENTER_HORIZONTAL
            }
        }
        card.addView(linearProgress)

        percentView = TextView(context).apply {
            textSize = 13f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, (8 * density).toInt(), 0, 0)
        }
        card.addView(percentView)

        addView(
            card,
            LayoutParams(
                (340 * density).toInt().coerceAtMost((resources.displayMetrics.widthPixels * 0.9f).toInt()),
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.CENTER },
        )
        setProgress(0f, null)
    }

    fun setTitle(title: CharSequence) {
        titleView.text = title
    }

    fun setProgress(progress: Float, subtitle: CharSequence?) {
        val clamped = progress.coerceIn(0f, 1f)
        val percent = (clamped * 100).toInt()
        ringProgress.setProgress(percent, true)
        linearProgress.setProgress(percent, true)
        percentView.text = context.getString(R.string.common_percentage, percent)
        if (subtitle != null) {
            subtitleView.text = subtitle
            subtitleView.visibility = View.VISIBLE
        }
    }

    companion object {
        private const val TAG = "paafekt_save_overlay"

        fun show(parent: ViewGroup): PaafektSavingRoomOverlay {
            hide(parent)
            val overlay = PaafektSavingRoomOverlay(parent.context)
            overlay.tag = TAG
            parent.addView(overlay)
            parent.bringChildToFront(overlay)
            return overlay
        }

        fun hide(parent: ViewGroup) {
            val existing = parent.findViewWithTag<View>(TAG)
            if (existing != null) {
                parent.removeView(existing)
            }
        }
    }
}
