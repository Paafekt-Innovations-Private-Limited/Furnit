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

/**
 * Swift `PaafektBuildingRoomOverlay` parity — gold ring, house icon, status, accent percent, subtext.
 */
class PaafektBuildingRoomOverlay(context: Context) : FrameLayout(context) {

    private val statusView: TextView
    private val percentView: TextView
    private val ringProgress: CircularProgressIndicator

    init {
        val density = resources.displayMetrics.density
        setBackgroundColor(android.graphics.Color.argb(225, 0x0E, 0x0F, 0x12))
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

        val ringSize = (72 * density).toInt()
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
                    (26 * density).toInt(),
                    (26 * density).toInt(),
                ).apply { gravity = Gravity.CENTER }
            },
        )
        card.addView(ringFrame)

        card.addView(
            TextView(context).apply {
                text = context.getString(R.string.photo_room_building_room)
                textSize = 18f
                setTypeface(null, Typeface.BOLD)
                setTextColor(PaafektColors.textPrimary)
                gravity = Gravity.CENTER
                setPadding(0, (24 * density).toInt(), 0, 0)
            },
        )

        statusView = TextView(context).apply {
            textSize = 15f
            setTextColor(PaafektColors.textSecondary)
            gravity = Gravity.CENTER
            setPadding(0, (8 * density).toInt(), 0, 0)
        }
        card.addView(statusView)

        percentView = TextView(context).apply {
            textSize = 28f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.accent)
            gravity = Gravity.CENTER
            setPadding(0, (16 * density).toInt(), 0, 0)
        }
        card.addView(percentView)

        card.addView(
            TextView(context).apply {
                text = context.getString(R.string.photo_room_building_room_subtext)
                textSize = 13f
                setTextColor(PaafektColors.textSecondary)
                gravity = Gravity.CENTER
                setPadding(0, (12 * density).toInt(), 0, 0)
            },
        )

        addView(
            card,
            LayoutParams(
                (340 * density).toInt().coerceAtMost((resources.displayMetrics.widthPixels * 0.9f).toInt()),
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply { gravity = Gravity.CENTER },
        )
        setProgress(0f, null)
    }

    fun setProgress(progress: Float, statusMessage: CharSequence?) {
        val clamped = progress.coerceIn(0f, 1f)
        val percent = (clamped * 100).toInt()
        ringProgress.setProgress(percent, true)
        percentView.text = "$percent%"
        if (!statusMessage.isNullOrBlank()) {
            statusView.text = statusMessage
            statusView.visibility = View.VISIBLE
        } else {
            statusView.visibility = View.GONE
        }
    }

    companion object {
        private const val TAG = "paafekt_build_overlay"

        fun show(parent: ViewGroup): PaafektBuildingRoomOverlay {
            hide(parent)
            val overlay = PaafektBuildingRoomOverlay(parent.context)
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
