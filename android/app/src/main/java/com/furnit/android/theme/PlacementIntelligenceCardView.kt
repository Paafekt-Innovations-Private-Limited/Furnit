package com.furnit.android.theme

import android.content.Context
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import com.furnit.android.R
import com.furnit.android.models.roomintelligence.AestheticEvaluation
import com.furnit.android.models.roomintelligence.AestheticRecommendationCode
import com.furnit.android.models.roomintelligence.FurnitureDimensions
import com.furnit.android.models.roomintelligence.RoomDimensions
import com.furnit.android.models.roomintelligence.RoomIntelligenceResult
import com.furnit.android.models.roomintelligence.RoomIntelligenceStatus
import kotlin.math.roundToInt

data class PlacementIntelligenceCardState(
    val statusText: String,
    val statusColor: Int,
    val dimensionsText: String?,
    val harmonyScore: Int?,
    val contrastScore: Int?,
    val styleScore: Int?,
    val notes: List<String>,
)

object PlacementIntelligenceCardMapper {
    fun map(
        context: Context,
        result: RoomIntelligenceResult,
        roomDimensions: RoomDimensions?,
        furnitureDimensions: FurnitureDimensions?,
    ): PlacementIntelligenceCardState {
        val statusText = context.getString(
            when (result.status) {
                RoomIntelligenceStatus.MEASURING -> R.string.placement_intelligence_measuring
                RoomIntelligenceStatus.STYLE_ONLY -> R.string.placement_intelligence_style_only
                RoomIntelligenceStatus.FITS_BY_DIMENSIONS -> R.string.placement_intelligence_fits
                RoomIntelligenceStatus.DOES_NOT_FIT -> R.string.placement_intelligence_does_not_fit
            },
        )
        val statusColor = when (result.status) {
            RoomIntelligenceStatus.FITS_BY_DIMENSIONS -> PaafektColors.success
            RoomIntelligenceStatus.DOES_NOT_FIT -> PaafektColors.danger
            RoomIntelligenceStatus.MEASURING -> Color.CYAN
            RoomIntelligenceStatus.STYLE_ONLY -> Color.CYAN
        }
        val availableAesthetic = result.aesthetic as? AestheticEvaluation.Available
        val notes = mutableListOf<String>()
        if (furnitureDimensions == null) {
            notes += context.getString(R.string.placement_intelligence_measure_with_ar)
        } else {
            notes += context.getString(R.string.placement_intelligence_depth_derived)
            if (result.status == RoomIntelligenceStatus.DOES_NOT_FIT && roomDimensions != null) {
                if (furnitureDimensions.heightMeters > roomDimensions.heightMeters) {
                    notes += context.getString(R.string.placement_intelligence_recommendation_height)
                }
                if (furnitureDimensions.widthMeters > maxOf(roomDimensions.widthMeters, roomDimensions.depthMeters)) {
                    notes += context.getString(R.string.placement_intelligence_recommendation_width)
                }
                if (furnitureDimensions.depthMeters > maxOf(roomDimensions.widthMeters, roomDimensions.depthMeters)) {
                    notes += context.getString(R.string.placement_intelligence_recommendation_depth)
                }
            }
            if (result.dimensionFit?.rotatedFootprint == true) {
                notes += context.getString(R.string.placement_intelligence_recommendation_rotate)
            }
        }
        if (availableAesthetic == null) {
            notes += context.getString(R.string.placement_intelligence_aesthetic_unavailable)
        } else {
            availableAesthetic.recommendations.forEach { code ->
                notes += context.getString(recommendationString(code))
            }
        }
        return PlacementIntelligenceCardState(
            statusText = statusText,
            statusColor = statusColor,
            dimensionsText = furnitureDimensions?.let {
                context.getString(
                    R.string.placement_intelligence_dimensions,
                    it.widthMeters,
                    it.heightMeters,
                    it.depthMeters,
                )
            },
            harmonyScore = availableAesthetic?.harmonyScore?.times(100f)?.roundToInt(),
            contrastScore = availableAesthetic?.contrastScore?.times(100f)?.roundToInt(),
            styleScore = availableAesthetic?.styleCompatibilityScore?.times(100f)?.roundToInt(),
            notes = notes.distinct(),
        )
    }

    private fun recommendationString(code: AestheticRecommendationCode): Int = when (code) {
        AestheticRecommendationCode.AESTHETIC_UNAVAILABLE ->
            R.string.placement_intelligence_aesthetic_unavailable
        AestheticRecommendationCode.USE_NEUTRAL_BRIDGE ->
            R.string.placement_intelligence_recommendation_neutral_bridge
        AestheticRecommendationCode.ANALOGOUS_HARMONY ->
            R.string.placement_intelligence_recommendation_analogous
        AestheticRecommendationCode.COMPLEMENTARY_FOCAL_POINT ->
            R.string.placement_intelligence_recommendation_complementary
        AestheticRecommendationCode.INCREASE_CONTRAST ->
            R.string.placement_intelligence_recommendation_contrast
        AestheticRecommendationCode.SOFTEN_CONTRAST ->
            R.string.placement_intelligence_recommendation_soften_contrast
        AestheticRecommendationCode.STYLE_MISMATCH ->
            R.string.placement_intelligence_recommendation_style
        AestheticRecommendationCode.BROADLY_COMPATIBLE ->
            R.string.placement_intelligence_recommendation_compatible
    }
}

/**
 * Reusable viewer chip and expandable card matching the iOS placement-intelligence control.
 * It deliberately receives display-ready state so the evaluator remains independent of Android resources.
 */
class PlacementIntelligenceCardView(context: Context) : LinearLayout(context) {
    private val expandedCard: LinearLayout
    private val chipCircle: ImageView
    private val statusLabel: TextView
    private val detailsContainer: LinearLayout
    private val dimensionsLabel: TextView
    private val scoresLabel: TextView
    private val notesLabel: TextView
    private var expanded = false

    init {
        orientation = VERTICAL
        gravity = Gravity.CENTER_HORIZONTAL
        visibility = GONE
        importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO

        expandedCard = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(dp(16), dp(12), dp(16), dp(12))
            background = cardBackground()
            elevation = dp(12).toFloat()
            visibility = GONE
        }
        val header = LinearLayout(context).apply {
            orientation = HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        header.addView(TextView(context).apply {
            text = context.getString(R.string.placement_intelligence_title)
            textSize = 13f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(PaafektColors.textPrimary)
        }, LayoutParams(0, LayoutParams.WRAP_CONTENT, 1f))
        statusLabel = TextView(context).apply {
            textSize = 11f
            typeface = Typeface.DEFAULT_BOLD
            setTextColor(PaafektColors.textSecondary)
        }
        header.addView(statusLabel)
        expandedCard.addView(header)

        detailsContainer = LinearLayout(context).apply {
            orientation = VERTICAL
            setPadding(0, dp(10), 0, 0)
        }
        dimensionsLabel = detailLabel()
        scoresLabel = detailLabel()
        notesLabel = detailLabel()
        detailsContainer.addView(dimensionsLabel)
        detailsContainer.addView(scoresLabel)
        detailsContainer.addView(notesLabel)
        expandedCard.addView(detailsContainer)
        addView(
            expandedCard,
            LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.WRAP_CONTENT),
        )

        val chipTouchTarget = FrameLayout(context).apply {
            isClickable = true
            isFocusable = true
            elevation = dp(12).toFloat()
            importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_YES
        }
        chipCircle = ImageView(context).apply {
            setImageResource(R.drawable.ic_placement_intelligence)
            scaleType = ImageView.ScaleType.CENTER_INSIDE
            setPadding(dp(12), dp(12), dp(12), dp(12))
            background = chipBackground(PaafektColors.textSecondary)
        }
        chipTouchTarget.addView(
            chipCircle,
            FrameLayout.LayoutParams(dp(40), dp(40), Gravity.CENTER),
        )
        chipTouchTarget.setOnClickListener { setExpanded(!expanded) }
        addView(
            chipTouchTarget,
            LayoutParams(dp(48), dp(48)).apply {
                topMargin = dp(8)
                gravity = Gravity.CENTER_HORIZONTAL
            },
        )
        updateAccessibilityDescription()
    }

    fun render(state: PlacementIntelligenceCardState) {
        visibility = VISIBLE
        statusLabel.text = state.statusText
        statusLabel.setTextColor(state.statusColor)
        chipCircle.background = chipBackground(state.statusColor)

        dimensionsLabel.text = state.dimensionsText.orEmpty()
        dimensionsLabel.visibility = if (state.dimensionsText.isNullOrBlank()) GONE else VISIBLE

        val scores = buildList {
            state.harmonyScore?.let {
                add(context.getString(R.string.placement_intelligence_harmony) + " " +
                    context.getString(R.string.placement_intelligence_score, it))
            }
            state.contrastScore?.let {
                add(context.getString(R.string.placement_intelligence_contrast) + " " +
                    context.getString(R.string.placement_intelligence_score, it))
            }
            state.styleScore?.let {
                add(context.getString(R.string.placement_intelligence_style_fit) + " " +
                    context.getString(R.string.placement_intelligence_score, it))
            }
        }
        scoresLabel.text = scores.joinToString("  ·  ")
        scoresLabel.visibility = if (scores.isEmpty()) GONE else VISIBLE

        notesLabel.text = state.notes.joinToString("\n") { "• $it" }
        notesLabel.visibility = if (state.notes.isEmpty()) GONE else VISIBLE
        updateAccessibilityDescription()
    }

    fun clear() {
        visibility = GONE
        statusLabel.text = ""
        dimensionsLabel.text = ""
        scoresLabel.text = ""
        notesLabel.text = ""
        setExpanded(false)
    }

    fun setExpanded(expand: Boolean) {
        expanded = expand
        expandedCard.visibility = if (expand) VISIBLE else GONE
        updateAccessibilityDescription()
    }

    fun isExpanded(): Boolean = expanded

    private fun updateAccessibilityDescription() {
        val accessibilityDescription = context.getString(
            if (expanded) {
                R.string.placement_intelligence_collapse
            } else {
                R.string.placement_intelligence_expand
            },
        )
        getChildAt(childCount - 1)?.contentDescription = accessibilityDescription
    }

    private fun detailLabel(): TextView = TextView(context).apply {
        textSize = 12f
        setTextColor(PaafektColors.textSecondary)
        setLineSpacing(0f, 1.15f)
        setPadding(0, 0, 0, dp(6))
    }

    private fun cardBackground(): GradientDrawable = GradientDrawable().apply {
        shape = GradientDrawable.RECTANGLE
        cornerRadius = dp(18).toFloat()
        setColor(Color.argb(230, 26, 28, 32))
        setStroke(dp(1), PaafektColors.hairline)
    }

    private fun chipBackground(ringColor: Int): GradientDrawable =
        GradientDrawable(
            GradientDrawable.Orientation.TL_BR,
            intArrayOf(Color.rgb(56, 56, 56), Color.rgb(31, 31, 31)),
        ).apply {
            shape = GradientDrawable.OVAL
            setStroke(dp(3), ringColor)
        }

    private fun dp(value: Int): Int =
        (value * resources.displayMetrics.density).toInt()

    companion object {
        fun viewerLayoutParams(context: Context, topMarginDp: Int = 92): FrameLayout.LayoutParams =
            FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.TOP or Gravity.CENTER_HORIZONTAL
                marginStart = PaafektSpace.lg(context)
                marginEnd = PaafektSpace.lg(context)
                topMargin = (topMarginDp * context.resources.displayMetrics.density).toInt()
            }
    }
}
