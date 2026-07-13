package com.furnit.android.theme

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.PorterDuff
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
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
                marginStart = dp(context, 6)
                marginEnd = dp(context, 6)
            }
            contentDescription?.let { this.contentDescription = it }
            setOnClickListener { onClick() }
        }
    }

    /**
     * Resting summon affordance — quieter glass disk (bottom-leading). Fit FAB owns bottom-trailing gold.
     */
    fun createQuietSummonButton(
        context: Context,
        contentDescription: CharSequence,
        onClick: () -> Unit,
    ): FrameLayout {
        val size = dp(context, 38)
        val iconSize = dp(context, 18)
        return FrameLayout(context).apply {
            layoutParams = quietSummonButtonLayoutParams(context)
            background = PaafektDrawables.toolbarCircle()
            alpha = 0.92f
            ViewCompat.setElevation(this, dp(context, 4).toFloat())

            addView(
                ImageView(context).apply {
                    setImageResource(R.drawable.ic_chevron_up)
                    imageTintList = ColorStateList.valueOf(PaafektColors.textSecondary)
                    imageTintMode = PorterDuff.Mode.SRC_IN
                    scaleType = ImageView.ScaleType.FIT_CENTER
                    importantForAccessibility = View.IMPORTANT_FOR_ACCESSIBILITY_NO
                },
                FrameLayout.LayoutParams(iconSize, iconSize, Gravity.CENTER),
            )

            minimumWidth = size
            minimumHeight = size
            this.contentDescription = contentDescription
            isClickable = true
            isFocusable = true
            setOnClickListener { onClick() }
        }
    }

    /** Bottom-leading placement for quiet summon. */
    fun quietSummonButtonLayoutParams(
        context: Context,
        systemBarBottomInset: Int = 0,
    ): FrameLayout.LayoutParams {
        return FrameLayout.LayoutParams(dp(context, 38), dp(context, 38)).apply {
            gravity = Gravity.START or Gravity.BOTTOM
            bottomMargin = systemBarBottomInset + PaafektSpace.lg(context)
            marginStart = PaafektSpace.lg(context)
        }
    }

    fun setPersistentFitFabActive(button: LinearLayout, active: Boolean) {
        if (button.tag == active) return
        button.tag = active
        button.background = PaafektDrawables.heroButton(active)
    }

    class PersistentPrimaryActionsHolder(
        val container: LinearLayout,
        val fitButton: LinearLayout?,
        val saveButton: LinearLayout?,
    )

    enum class MorphingPrimaryAction {
        FIT_ENTER,
        FIT_EXIT_ACTIVE,
        SEGMENT,
        DONE,
    }

    object MorphingPrimaryActionResolver {
        fun resolve(
            showingFurnitureFit: Boolean,
            showFullVideoWithIdentifications: Boolean,
            segmentationModeSegmentSelected: Boolean,
            hasSelectedObject: Boolean,
        ): MorphingPrimaryAction {
            if (!showingFurnitureFit) return MorphingPrimaryAction.FIT_ENTER
            if (segmentationModeSegmentSelected) return MorphingPrimaryAction.DONE
            if (showFullVideoWithIdentifications && hasSelectedObject) return MorphingPrimaryAction.SEGMENT
            return MorphingPrimaryAction.FIT_EXIT_ACTIVE
        }
    }

    fun updateMorphingPrimaryFitButton(
        button: LinearLayout?,
        action: MorphingPrimaryAction,
        segmentEnabled: Boolean = true,
    ) {
        button ?: return
        val context = button.context
        val iconView = (button.getChildAt(0) as? ImageView)
        val labelView = (button.getChildAt(1) as? TextView)
        when (action) {
            MorphingPrimaryAction.FIT_ENTER, MorphingPrimaryAction.FIT_EXIT_ACTIVE -> {
                iconView?.visibility = View.VISIBLE
                iconView?.setImageResource(R.drawable.ic_ai)
                labelView?.text = context.getString(R.string.room_viewer_immersive_fit_short)
                button.contentDescription = context.getString(R.string.room_viewer_immersive_fit_short)
            }
            MorphingPrimaryAction.SEGMENT -> {
                iconView?.visibility = View.GONE
                labelView?.text = context.getString(R.string.segment_furniture_action)
                button.contentDescription = context.getString(R.string.segment_furniture_action)
            }
            MorphingPrimaryAction.DONE -> {
                iconView?.visibility = View.GONE
                labelView?.text = context.getString(R.string.room_viewer_segmentation_done)
                button.contentDescription = context.getString(R.string.room_viewer_segmentation_done)
            }
        }
        val isActive = action == MorphingPrimaryAction.FIT_EXIT_ACTIVE
        setPersistentFitFabActive(button, isActive)
        button.isEnabled = action != MorphingPrimaryAction.SEGMENT || segmentEnabled
        button.alpha = if (button.isEnabled) 1f else 0.5f
    }

    /**
     * Bottom-trailing row for persistent Fit (+ optional Save in creation flow).
     * Save is trailing (dominant); Fit sits to its left.
     */
    fun createPersistentPrimaryActionsRow(
        context: Context,
        showFit: Boolean,
        showSave: Boolean,
        fitLabel: CharSequence,
        saveLabel: CharSequence,
        onFit: () -> Unit,
        onSave: () -> Unit,
        systemBarBottomInset: Int = 0,
    ): PersistentPrimaryActionsHolder {
        val container = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END or Gravity.CENTER_VERTICAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.END or Gravity.BOTTOM
                bottomMargin = systemBarBottomInset + PaafektSpace.lg(context)
                marginEnd = PaafektSpace.lg(context)
            }
        }

        var fitButton: LinearLayout? = null
        var saveButton: LinearLayout? = null

        if (showFit) {
            fitButton = createPersistentActionButton(
                context,
                R.drawable.ic_ai,
                fitLabel,
                onFit,
            )
            container.addView(
                fitButton,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ),
            )
        }
        if (showSave) {
            saveButton = createPersistentActionButton(
                context,
                R.drawable.ic_download,
                saveLabel,
                onSave,
            ).apply {
                contentDescription = context.getString(R.string.common_save)
            }
            container.addView(
                saveButton,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply {
                    if (showFit) {
                        marginStart = PaafektSpace.sm(context)
                    }
                },
            )
        }

        return PersistentPrimaryActionsHolder(container, fitButton, saveButton)
    }

    private fun createPersistentActionButton(
        context: Context,
        @DrawableRes iconRes: Int,
        label: CharSequence,
        onClick: () -> Unit,
    ): LinearLayout {
        return PaafektHintViews.createHeroButton(
            context,
            iconRes,
            label,
            isActive = false,
            onClick = onClick,
        ).apply {
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
            tag = false
        }
    }

    fun updatePersistentPrimaryActionsInsets(
        holder: PersistentPrimaryActionsHolder?,
        systemBarBottomInset: Int,
    ) {
        holder ?: return
        (holder.container.layoutParams as? FrameLayout.LayoutParams)?.let { lp ->
            lp.bottomMargin = systemBarBottomInset + PaafektSpace.lg(holder.container.context)
            holder.container.layoutParams = lp
        }
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}

/**
 * Bottom summoned toolbar — iOS `PaafektImmersiveSummonedToolbar` (glass capsule + nav icons + compact Fit/Capture).
 */
class ImmersiveSummonedToolbarHolder(
    val root: FrameLayout,
    private val fullVideoButton: ImageButton?,
    private val arSizingButton: ImageButton?,
) {
    fun setFullVideoVisible(visible: Boolean) {
        fullVideoButton?.visibility = if (visible) View.VISIBLE else View.GONE
    }

    fun setFullVideoActive(active: Boolean) {
        fullVideoButton?.let { button ->
            button.imageTintList = ColorStateList.valueOf(
                if (active) PaafektColors.accent else PaafektColors.textPrimary,
            )
        }
    }

    fun setArSizingVisible(visible: Boolean) {
        arSizingButton?.visibility = if (visible) View.VISIBLE else View.GONE
    }

    fun setArSizingActive(active: Boolean) {
        arSizingButton?.let { button ->
            button.imageTintList = ColorStateList.valueOf(
                if (active) PaafektColors.accent else PaafektColors.textPrimary,
            )
        }
    }
}

object PaafektImmersiveSummonedToolbar {

    fun createBottomChrome(
        context: Context,
        onRecenter: () -> Unit,
        onRuler: () -> Unit,
        onPinchHint: () -> Unit,
        onDisplayAllHelpers: () -> Unit,
        onFullVideo: () -> Unit,
        onArSizing: () -> Unit,
        onCapture: () -> Unit,
        includeFurnitureFitExtras: Boolean = true,
    ): ImmersiveSummonedToolbarHolder {
        val outer = FrameLayout(context).apply {
            setPadding(
                PaafektSpace.lg(context),
                0,
                PaafektSpace.lg(context),
                PaafektSpace.lg(context),
            )
        }

        val column = LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER_HORIZONTAL
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.BOTTOM
            }
        }

        val capsule = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = PaafektDrawables.toolbarCapsule()
            setPadding(PaafektSpace.sm(context), dp(context, 4), PaafektSpace.sm(context), dp(context, 4))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                gravity = Gravity.CENTER_HORIZONTAL
            }
        }

        val navRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        navRow.addView(
            PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_viewfinder,
                contentDescription = context.getString(R.string.room_viewer_recenter),
                onClick = onRecenter,
            ),
        )
        navRow.addView(
            PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_ruler,
                contentDescription = context.getString(R.string.faq_measurement_pill),
                onClick = onRuler,
            ),
        )
        navRow.addView(
            PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_gesture_pinch,
                contentDescription = context.getString(R.string.room_viewer_navigation_teaching_hint),
                onClick = onPinchHint,
            ),
        )
        navRow.addView(
            PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_grid_3x3,
                contentDescription = context.getString(R.string.room_viewer_display_all_helpers),
                onClick = onDisplayAllHelpers,
            ),
        )

        var fullVideoButton: ImageButton? = null
        var arSizingButton: ImageButton? = null
        if (includeFurnitureFitExtras) {
            fullVideoButton = PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_text_viewfinder,
                contentDescription = context.getString(R.string.room_viewer_full_video_with_identifications),
                onClick = onFullVideo,
            ).apply { visibility = View.GONE }
            navRow.addView(fullVideoButton)

            arSizingButton = PaafektViewerToolbar.createCapsuleIconButton(
                context,
                R.drawable.ic_square_resize,
                contentDescription = context.getString(R.string.room_viewer_ar_sizing_enable),
                onClick = onArSizing,
            ).apply { visibility = View.GONE }
            navRow.addView(arSizingButton)
        }

        capsule.addView(
            navRow,
            LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ),
        )

        capsule.addView(
            View(context),
            LinearLayout.LayoutParams(0, 0, 1f),
        )

        val heroRow = LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val captureButton = PaafektHintViews.createCompactHeroAction(
            context,
            R.drawable.ic_snapshot,
            context.getString(R.string.room_viewer_immersive_capture_short),
            onClick = onCapture,
        )
        heroRow.addView(captureButton)

        capsule.addView(heroRow)
        column.addView(capsule)
        outer.addView(column)

        return ImmersiveSummonedToolbarHolder(outer, fullVideoButton, arSizingButton)
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}
