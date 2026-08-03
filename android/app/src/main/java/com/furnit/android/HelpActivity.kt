package com.furnit.android

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.theme.PaafektColors
import com.furnit.android.theme.PaafektScreenViews
import com.furnit.android.theme.PaafektSpace

/**
 * Help & Support — Paafekt-styled FAQ + contact (mirrors iOS SupportView).
 */
class HelpActivity : AppCompatActivity() {

    private val expandedFAQs = mutableSetOf<String>()

    data class FAQItem(
        val questionResId: Int,
        val answerResId: Int,
    ) {
        fun getId(context: Context): String = context.getString(questionResId)
    }

    data class FAQSection(
        val titleResId: Int,
        val iconResId: Int,
        val items: List<FAQItem>,
    )

    private val faqSections by lazy {
        listOf(
            FAQSection(
                titleResId = R.string.faq_room_creation,
                iconResId = R.drawable.ic_camera,
                items = listOf(
                    FAQItem(R.string.faq_how_to_create, R.string.faq_how_to_create_answer),
                    FAQItem(R.string.faq_how_to_take_photo, R.string.faq_how_to_take_photo_answer),
                    FAQItem(R.string.faq_best_photos, R.string.faq_best_photos_answer),
                    FAQItem(R.string.faq_how_many_rooms, R.string.faq_how_many_rooms_answer),
                    FAQItem(R.string.faq_how_to_view, R.string.faq_how_to_view_answer),
                    FAQItem(R.string.faq_how_to_navigate, R.string.faq_how_to_navigate_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_ai_room_manual_setup,
                iconResId = R.drawable.ic_house,
                items = listOf(
                    FAQItem(R.string.faq_two_methods, R.string.faq_two_methods_answer),
                    FAQItem(R.string.faq_what_is_ai_room, R.string.faq_what_is_ai_room_answer),
                    FAQItem(R.string.faq_what_is_manual_room, R.string.faq_what_is_manual_room_answer),
                    FAQItem(R.string.faq_which_method_better, R.string.faq_which_method_better_answer),
                    FAQItem(R.string.faq_depth_aware_room_photo, R.string.faq_depth_aware_room_photo_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_ai_features,
                iconResId = R.drawable.ic_ai,
                items = listOf(
                    FAQItem(R.string.faq_what_is_brain_icon, R.string.faq_what_is_brain_icon_answer),
                    FAQItem(R.string.faq_what_is_viewfinder_button, R.string.faq_what_is_viewfinder_button_answer),
                    FAQItem(R.string.faq_what_is_segmentation, R.string.faq_what_is_segmentation_answer),
                    FAQItem(R.string.faq_how_to_segment, R.string.faq_how_to_segment_answer),
                    FAQItem(R.string.faq_ar_assisted_sizing, R.string.faq_ar_assisted_sizing_answer),
                    FAQItem(R.string.faq_measurement_pill, R.string.faq_measurement_pill_answer),
                    FAQItem(R.string.faq_reset_overlay_scale, R.string.faq_reset_overlay_scale_answer),
                    FAQItem(R.string.faq_pinch_overlay_size, R.string.faq_pinch_overlay_size_answer),
                    FAQItem(R.string.faq_how_to_place, R.string.faq_how_to_place_answer),
                    FAQItem(R.string.faq_multiple_pieces, R.string.faq_multiple_pieces_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_placement_intelligence,
                iconResId = R.drawable.ic_ai,
                items = listOf(
                    FAQItem(
                        R.string.faq_what_is_placement_intelligence,
                        R.string.faq_what_is_placement_intelligence_answer,
                    ),
                    FAQItem(R.string.faq_free_floor_location, R.string.faq_free_floor_location_answer),
                    FAQItem(R.string.faq_dimensional_fit, R.string.faq_dimensional_fit_answer),
                    FAQItem(R.string.faq_room_fitment, R.string.faq_room_fitment_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_aesthetic_guidance,
                iconResId = R.drawable.ic_ai,
                items = listOf(
                    FAQItem(R.string.faq_aesthetic_score, R.string.faq_aesthetic_score_answer),
                    FAQItem(
                        R.string.faq_furniture_color_aesthetic,
                        R.string.faq_furniture_color_aesthetic_answer,
                    ),
                    FAQItem(
                        R.string.faq_what_do_harmony_contrast_mean,
                        R.string.faq_what_do_harmony_contrast_mean_answer,
                    ),
                    FAQItem(
                        R.string.faq_aesthetic_unavailable_question,
                        R.string.faq_aesthetic_unavailable_answer,
                    ),
                    FAQItem(
                        R.string.faq_when_does_aesthetic_score_low,
                        R.string.faq_when_does_aesthetic_score_low_answer,
                    ),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_measurements_accuracy,
                iconResId = R.drawable.ic_ruler,
                items = listOf(
                    FAQItem(R.string.faq_accuracy, R.string.faq_accuracy_answer),
                    FAQItem(R.string.faq_adjust_dimensions, R.string.faq_adjust_dimensions_answer),
                    FAQItem(R.string.faq_purchase_reliance, R.string.faq_purchase_reliance_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_saved_rooms_storage,
                iconResId = R.drawable.ic_house,
                items = listOf(
                    FAQItem(R.string.faq_how_to_save_room, R.string.faq_how_to_save_room_answer),
                    FAQItem(R.string.faq_where_rooms_saved, R.string.faq_where_rooms_saved_answer),
                    FAQItem(R.string.faq_uninstall_rooms, R.string.faq_uninstall_rooms_answer),
                    FAQItem(R.string.faq_what_is_memory_display, R.string.faq_what_is_memory_display_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_sharing_exporting,
                iconResId = R.drawable.ic_share,
                items = listOf(
                    FAQItem(R.string.faq_export_room, R.string.faq_export_room_answer),
                    FAQItem(R.string.faq_how_to_screenshot, R.string.faq_how_to_screenshot_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_privacy_account,
                iconResId = R.drawable.ic_lock_shield,
                items = listOf(
                    FAQItem(R.string.faq_are_photos_uploaded, R.string.faq_are_photos_uploaded_answer),
                    FAQItem(R.string.faq_firebase_data, R.string.faq_firebase_data_answer),
                    FAQItem(R.string.faq_delete_account_effect, R.string.faq_delete_account_effect_answer),
                ),
            ),
            FAQSection(
                titleResId = R.string.faq_troubleshooting_support,
                iconResId = R.drawable.ic_help,
                items = listOf(
                    FAQItem(R.string.faq_generation_failing, R.string.faq_generation_failing_answer),
                    FAQItem(R.string.faq_not_detected, R.string.faq_not_detected_answer),
                    FAQItem(R.string.faq_what_do_arrows_do, R.string.faq_what_do_arrows_do_answer),
                    FAQItem(R.string.faq_contact_support, R.string.faq_contact_support_answer),
                ),
            ),
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = PaafektScreenViews.createScreenColumn(this)

        layout.addView(
            PaafektScreenViews.createBackButton(this, PaafektScreenViews.backLabel(this)) { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.help_title)))
        layout.addView(
            PaafektScreenViews.createWarningChip(this, getString(R.string.help_measurement_warning)),
        )

        for (section in faqSections) {
            layout.addView(createFAQSection(section))
        }
        layout.addView(createContactSupportSection())

        PaafektScreenViews.createScreenScrollView(this).apply {
            addView(layout)
            setContentView(this)
        }
    }

    private fun createFAQSection(section: FAQSection): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(
                PaafektScreenViews.createSectionHeader(
                    this@HelpActivity,
                    section.iconResId,
                    getString(section.titleResId),
                ),
            )
            addView(PaafektScreenViews.createDivider(this@HelpActivity))
            for (item in section.items) {
                addView(createFAQItem(item))
            }
        }
    }

    private fun createFAQItem(item: FAQItem): LinearLayout {
        val itemId = item.getId(this)
        val questionStr = getString(item.questionResId)
        val answerStr = getString(item.answerResId)

        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, PaafektSpace.md(this@HelpActivity), 0, 0)

            val questionLayout = LinearLayout(this@HelpActivity).apply {
                orientation = LinearLayout.HORIZONTAL
                gravity = Gravity.CENTER_VERTICAL
            }

            val questionText = TextView(this@HelpActivity).apply {
                text = questionStr
                textSize = 14f
                setTextColor(PaafektColors.textPrimary)
                layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
            }
            questionLayout.addView(questionText)

            val chevron = TextView(this@HelpActivity).apply {
                text = if (expandedFAQs.contains(itemId)) "▲" else "▼"
                textSize = 11f
                setTextColor(PaafektColors.accent)
            }
            questionLayout.addView(chevron)

            val answerText = TextView(this@HelpActivity).apply {
                text = answerStr
                textSize = 13f
                setTextColor(PaafektColors.textSecondary)
                setPadding(0, PaafektSpace.md(this@HelpActivity), 0, 0)
                visibility = if (expandedFAQs.contains(itemId)) View.VISIBLE else View.GONE
            }

            questionLayout.setOnClickListener {
                val isExpanded = expandedFAQs.contains(itemId)
                if (isExpanded) {
                    expandedFAQs.remove(itemId)
                    answerText.visibility = View.GONE
                    chevron.text = "▼"
                } else {
                    expandedFAQs.add(itemId)
                    answerText.visibility = View.VISIBLE
                    chevron.text = "▲"
                }
            }

            addView(questionLayout)
            addView(answerText)
            addView(PaafektScreenViews.createDivider(this@HelpActivity))
        }
    }

    private fun createContactSupportSection(): LinearLayout {
        return PaafektScreenViews.createSectionCard(this).apply {
            addView(
                PaafektScreenViews.createSectionHeader(
                    this@HelpActivity,
                    R.drawable.ic_help,
                    getString(R.string.help_contact_support),
                ),
            )
            addView(PaafektScreenViews.createDivider(this@HelpActivity))
            addView(PaafektScreenViews.createSecondaryLabel(this@HelpActivity, getString(R.string.help_cant_find)))
            addView(
                PaafektScreenViews.createCaptionLabel(this@HelpActivity, getString(R.string.help_contact_description))
                    .apply { setPadding(0, 0, 0, PaafektSpace.lg(this@HelpActivity)) },
            )
            addView(
                PaafektScreenViews.createPrimaryActionRow(
                    this@HelpActivity,
                    getString(R.string.help_email_support),
                    "support@paafekt.com",
                ) { openEmailComposer() },
            )
            addView(
                PaafektScreenViews.createLinkLabel(this@HelpActivity, getString(R.string.help_copy_email)) {
                    copyEmailToClipboard()
                },
            )
        }
    }

    private fun openEmailComposer() {
        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_EMAIL, arrayOf("support@paafekt.com"))
            putExtra(Intent.EXTRA_SUBJECT, getString(R.string.help_email_subject))
            putExtra(Intent.EXTRA_TEXT, getString(R.string.help_email_body))
        }
        try {
            startActivity(Intent.createChooser(intent, getString(R.string.help_send_email)))
        } catch (_: Exception) {
            Toast.makeText(this, getString(R.string.no_email_app), Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyEmailToClipboard() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        clipboard.setPrimaryClip(
            ClipData.newPlainText(getString(R.string.help_email_clipboard_label), "support@paafekt.com"),
        )
        Toast.makeText(this, getString(R.string.email_copied_clipboard), Toast.LENGTH_SHORT).show()
    }
}
