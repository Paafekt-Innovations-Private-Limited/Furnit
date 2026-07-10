package com.furnit.android.theme

import android.content.Context
import android.content.res.ColorStateList
import android.graphics.Typeface
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.SeekBar
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.appcompat.widget.SwitchCompat
import androidx.core.content.ContextCompat
import com.furnit.android.R

/** Shared list/settings screen chrome — Paafekt gold-on-dark tokens. */
object PaafektScreenViews {

    fun createScreenScrollView(context: Context): ScrollView {
        return ScrollView(context).apply {
            setBackgroundColor(PaafektColors.background)
            isFillViewport = true
        }
    }

    fun createScreenColumn(context: Context): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(
                PaafektSpace.lg(context),
                PaafektSpace.viewerTopInset(context),
                PaafektSpace.lg(context),
                PaafektSpace.xl(context),
            )
        }
    }

    fun createBackButton(context: Context, label: String, onClick: () -> Unit): TextView {
        return TextView(context).apply {
            text = label
            textSize = 16f
            setTextColor(PaafektColors.accent)
            setPadding(0, 0, 0, PaafektSpace.lg(context))
            setOnClickListener { onClick() }
        }
    }

    fun createScreenTitle(context: Context, title: String): TextView {
        return TextView(context).apply {
            text = title
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textPrimary)
            setPadding(0, 0, 0, PaafektSpace.xl(context))
        }
    }

    fun createSectionCard(context: Context): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.VERTICAL
            background = PaafektDrawables.secondaryButton()
            setPadding(
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
            )
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                bottomMargin = PaafektSpace.md(context)
            }
        }
    }

    fun createSectionLabel(context: Context, text: String): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 11f
            setTypeface(null, Typeface.BOLD)
            setTextColor(PaafektColors.textSecondary)
            letterSpacing = 0.06f
            setPadding(0, 0, 0, PaafektSpace.sm(context))
        }
    }

    fun createPrimaryLabel(context: Context, text: String): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 16f
            setTextColor(PaafektColors.textPrimary)
        }
    }

    fun createSecondaryLabel(context: Context, text: String): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 12f
            setTextColor(PaafektColors.textSecondary)
        }
    }

    fun createCaptionLabel(context: Context, text: String): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 11f
            setTextColor(PaafektColors.textSecondary)
            setPadding(0, PaafektSpace.sm(context), 0, 0)
        }
    }

    fun createDivider(context: Context): View {
        return View(context).apply {
            setBackgroundColor(PaafektColors.hairline)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                1,
            ).apply {
                topMargin = PaafektSpace.sm(context)
                bottomMargin = PaafektSpace.sm(context)
            }
        }
    }

    fun createLinkLabel(context: Context, text: String, onClick: () -> Unit): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 16f
            setTextColor(PaafektColors.accent)
            setPadding(0, PaafektSpace.sm(context), 0, PaafektSpace.sm(context))
            setOnClickListener { onClick() }
        }
    }

    fun createWarningChip(context: Context, text: String): TextView {
        return TextView(context).apply {
            this.text = text
            textSize = 14f
            setTextColor(PaafektColors.textPrimary)
            setPadding(
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
            )
            background = PaafektDrawables.creationCardPrimary()
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                bottomMargin = PaafektSpace.lg(context)
            }
        }
    }

    fun createSectionHeader(context: Context, @DrawableRes iconRes: Int, title: String): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, PaafektSpace.sm(context))
            addView(
                ImageView(context).apply {
                    setImageResource(iconRes)
                    imageTintList = ColorStateList.valueOf(PaafektColors.accent)
                    layoutParams = LinearLayout.LayoutParams(
                        dp(context, 22),
                        dp(context, 22),
                    ).apply { marginEnd = PaafektSpace.sm(context) }
                },
            )
            addView(
                TextView(context).apply {
                    this.text = title
                    textSize = 16f
                    setTypeface(null, Typeface.BOLD)
                    setTextColor(PaafektColors.textPrimary)
                },
            )
        }
    }

    fun createNavRow(
        context: Context,
        title: String,
        subtitle: String?,
        onClick: () -> Unit,
    ): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, PaafektSpace.md(context), 0, PaafektSpace.md(context))
            isClickable = true
            isFocusable = true
            setOnClickListener { onClick() }
            addView(
                LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    addView(createPrimaryLabel(context, title))
                    subtitle?.let { addView(createSecondaryLabel(context, it)) }
                },
            )
            addView(
                TextView(context).apply {
                    text = "›"
                    textSize = 20f
                    setTextColor(PaafektColors.textSecondary)
                },
            )
        }
    }

    fun createToggleRow(
        context: Context,
        title: String,
        subtitle: String?,
        checked: Boolean,
        onChanged: (Boolean) -> Unit,
    ): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, PaafektSpace.sm(context), 0, PaafektSpace.sm(context))
            addView(
                LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    addView(createPrimaryLabel(context, title))
                    subtitle?.let { addView(createSecondaryLabel(context, it)) }
                },
            )
            addView(createSwitch(context, checked, onChanged))
        }
    }

    fun createSwitch(context: Context, checked: Boolean, onChanged: (Boolean) -> Unit): SwitchCompat {
        return SwitchCompat(context).apply {
            isChecked = checked
            trackTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked),
                ),
                intArrayOf(PaafektColors.surfaceHi, PaafektColors.accent),
            )
            thumbTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked),
                ),
                intArrayOf(PaafektColors.textSecondary, PaafektColors.accentText),
            )
            setOnCheckedChangeListener { _, isChecked -> onChanged(isChecked) }
        }
    }

    fun styleSeekBarGold(seekBar: SeekBar) {
        seekBar.progressTintList = ColorStateList.valueOf(PaafektColors.accent)
        seekBar.thumbTintList = ColorStateList.valueOf(PaafektColors.accent)
        seekBar.progressBackgroundTintList = ColorStateList.valueOf(PaafektColors.surfaceHi)
    }

    fun createDangerButton(context: Context, text: String, onClick: () -> Unit): Button {
        return Button(context).apply {
            this.text = text
            textSize = 16f
            setTextColor(PaafektColors.textPrimary)
            backgroundTintList = ColorStateList.valueOf(PaafektColors.danger)
            setPadding(
                PaafektSpace.xl(context),
                PaafektSpace.md(context),
                PaafektSpace.xl(context),
                PaafektSpace.md(context),
            )
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ).apply {
                topMargin = PaafektSpace.xl(context)
            }
            setOnClickListener { onClick() }
        }
    }

    fun createPrimaryActionRow(context: Context, title: String, subtitle: String, onClick: () -> Unit): LinearLayout {
        return LinearLayout(context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = PaafektDrawables.primaryButton()
            setPadding(
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
                PaafektSpace.lg(context),
                PaafektSpace.md(context),
            )
            setOnClickListener { onClick() }
            addView(
                LinearLayout(context).apply {
                    orientation = LinearLayout.VERTICAL
                    layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
                    addView(
                        TextView(context).apply {
                            text = title
                            textSize = 16f
                            setTypeface(null, Typeface.BOLD)
                            setTextColor(PaafektColors.accentText)
                        },
                    )
                    addView(
                        TextView(context).apply {
                            text = subtitle
                            textSize = 13f
                            setTextColor(PaafektColors.accentText)
                            alpha = 0.85f
                        },
                    )
                },
            )
            addView(
                TextView(context).apply {
                    text = "↗"
                    textSize = 16f
                    setTextColor(PaafektColors.accentText)
                },
            )
        }
    }

    private fun dp(context: Context, value: Int): Int =
        (value * context.resources.displayMetrics.density).toInt()
}
