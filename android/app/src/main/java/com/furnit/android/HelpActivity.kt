package com.furnit.android

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

/**
 * Help & Support Activity with FAQ sections and contact support
 * Mirrors the iOS SupportView implementation
 * Uses localized strings from strings.xml
 */
class HelpActivity : AppCompatActivity() {

    // Track expanded FAQ items
    private val expandedFAQs = mutableSetOf<String>()

    // FAQ Data Models
    data class FAQItem(
        val questionResId: Int,
        val answerResId: Int
    ) {
        fun getId(context: Context): String = context.getString(questionResId)
    }

    data class FAQSection(
        val titleResId: Int,
        val icon: String,
        val items: List<FAQItem>
    )

    // FAQ Content using string resources
    private val faqSections by lazy {
        listOf(
            FAQSection(
                titleResId = R.string.faq_room_creation,
                icon = "camera",
                items = listOf(
                    FAQItem(R.string.faq_how_to_create, R.string.faq_how_to_create_answer),
                    FAQItem(R.string.faq_how_to_take_photo, R.string.faq_how_to_take_photo_answer),
                    FAQItem(R.string.faq_depth_aware_room_photo, R.string.faq_depth_aware_room_photo_answer),
                    FAQItem(R.string.faq_two_methods, R.string.faq_two_methods_answer),
                    FAQItem(R.string.faq_what_is_ai_room, R.string.faq_what_is_ai_room_answer),
                    FAQItem(R.string.faq_what_is_manual_room, R.string.faq_what_is_manual_room_answer),
                    FAQItem(R.string.faq_which_method_better, R.string.faq_which_method_better_answer),
                    FAQItem(R.string.faq_best_photos, R.string.faq_best_photos_answer),
                    FAQItem(R.string.faq_generation_failing, R.string.faq_generation_failing_answer),
                    FAQItem(R.string.faq_how_many_rooms, R.string.faq_how_many_rooms_answer),
                    FAQItem(R.string.faq_how_to_save_room, R.string.faq_how_to_save_room_answer)
                )
            ),
            FAQSection(
                titleResId = R.string.faq_ai_features,
                icon = "brain",
                items = listOf(
                    FAQItem(R.string.faq_what_is_brain_icon, R.string.faq_what_is_brain_icon_answer),
                    FAQItem(R.string.faq_how_to_screenshot, R.string.faq_how_to_screenshot_answer),
                    FAQItem(R.string.faq_what_is_segmentation, R.string.faq_what_is_segmentation_answer),
                    FAQItem(R.string.faq_how_to_segment, R.string.faq_how_to_segment_answer),
                    FAQItem(R.string.faq_not_detected, R.string.faq_not_detected_answer)
                )
            ),
            FAQSection(
                titleResId = R.string.faq_furniture_measurements,
                icon = "ruler",
                items = listOf(
                    FAQItem(R.string.faq_ar_assisted_sizing, R.string.faq_ar_assisted_sizing_answer),
                    FAQItem(R.string.faq_measurement_pill, R.string.faq_measurement_pill_answer),
                    FAQItem(R.string.faq_reset_overlay_scale, R.string.faq_reset_overlay_scale_answer),
                    FAQItem(R.string.faq_how_to_place, R.string.faq_how_to_place_answer),
                    FAQItem(R.string.faq_multiple_pieces, R.string.faq_multiple_pieces_answer),
                    FAQItem(R.string.faq_room_fitment, R.string.faq_room_fitment_answer),
                )
            ),
            FAQSection(
                titleResId = R.string.faq_room_controls,
                icon = "cube",
                items = listOf(
                    FAQItem(R.string.faq_how_to_view, R.string.faq_how_to_view_answer),
                    FAQItem(R.string.faq_how_to_navigate, R.string.faq_how_to_navigate_answer),
                    FAQItem(R.string.faq_what_do_arrows_do, R.string.faq_what_do_arrows_do_answer),
                    FAQItem(R.string.faq_what_is_memory_display, R.string.faq_what_is_memory_display_answer),
                    FAQItem(R.string.faq_sample_room, R.string.faq_sample_room_answer),
                    FAQItem(R.string.faq_accuracy, R.string.faq_accuracy_answer),
                    FAQItem(R.string.faq_adjust_dimensions, R.string.faq_adjust_dimensions_answer)
                )
            )
        )
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F5F5F5"))
        }

        val mainLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 48, 0, 32)
        }

        // Header with back button
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(32, 0, 32, 24)
        }

        val backBtn = TextView(this).apply {
            text = "< ${getString(R.string.common_back)}"
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setOnClickListener { finish() }
        }
        headerLayout.addView(backBtn)

        val titleSpacer = View(this).apply {
            layoutParams = LinearLayout.LayoutParams(0, 1, 1f)
        }
        headerLayout.addView(titleSpacer)

        mainLayout.addView(headerLayout)

        // Title
        val title = TextView(this).apply {
            text = getString(R.string.help_title)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(32, 0, 32, 24)
        }
        mainLayout.addView(title)

        // FAQ Sections
        for (section in faqSections) {
            mainLayout.addView(createFAQSection(section))
        }

        // Contact Support Section
        mainLayout.addView(createContactSupportSection())

        scrollView.addView(mainLayout)
        setContentView(scrollView)
    }

    private fun createFAQSection(section: FAQSection): LinearLayout {
        val sectionLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(24, 16, 24, 16)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }
        }

        // Section header with icon
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, 12)
        }

        val iconText = TextView(this).apply {
            text = when (section.icon) {
                "camera" -> "\uD83D\uDCF7"  // Camera emoji
                "brain" -> "\uD83E\uDDE0"   // Brain emoji
                "ruler" -> "\uD83D\uDCCF"   // Straight ruler emoji
                "cube" -> "\uD83D\uDDBC"    // Cube emoji
                else -> "\u2753"            // Question mark
            }
            textSize = 18f
            setPadding(0, 0, 12, 0)
        }
        headerLayout.addView(iconText)

        val sectionTitle = TextView(this).apply {
            text = getString(section.titleResId)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
        }
        headerLayout.addView(sectionTitle)

        sectionLayout.addView(headerLayout)

        // Divider
        val divider = View(this).apply {
            setBackgroundColor(Color.parseColor("#E0E0E0"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                1
            ).apply { setMargins(0, 0, 0, 8) }
        }
        sectionLayout.addView(divider)

        // FAQ Items
        for (item in section.items) {
            sectionLayout.addView(createFAQItem(item))
        }

        return sectionLayout
    }

    private fun createFAQItem(item: FAQItem): LinearLayout {
        val itemId = item.getId(this)
        val questionStr = getString(item.questionResId)
        val answerStr = getString(item.answerResId)

        val itemLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 12, 0, 12)
        }

        // Question row (clickable)
        val questionLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }

        val questionText = TextView(this).apply {
            text = questionStr
            textSize = 14f
            setTextColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        questionLayout.addView(questionText)

        val chevron = TextView(this).apply {
            text = if (expandedFAQs.contains(itemId)) "▲" else "▼"
            textSize = 12f
            setTextColor(Color.parseColor("#999999"))
            tag = "chevron_$itemId"
        }
        questionLayout.addView(chevron)

        itemLayout.addView(questionLayout)

        // Answer (initially hidden unless expanded)
        val answerText = TextView(this).apply {
            text = answerStr
            textSize = 13f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 12, 0, 0)
            visibility = if (expandedFAQs.contains(itemId)) View.VISIBLE else View.GONE
            tag = "answer_$itemId"
        }
        itemLayout.addView(answerText)

        // Click handler for expand/collapse
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

        // Divider at bottom
        val divider = View(this).apply {
            setBackgroundColor(Color.parseColor("#F0F0F0"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                1
            ).apply { setMargins(0, 12, 0, 0) }
        }
        itemLayout.addView(divider)

        return itemLayout
    }

    private fun createContactSupportSection(): LinearLayout {
        val sectionLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(24, 16, 24, 24)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }
        }

        // Section header
        val headerLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 0, 0, 12)
        }

        val iconText = TextView(this).apply {
            text = "\uD83C\uDFA7"  // Headphones emoji
            textSize = 18f
            setPadding(0, 0, 12, 0)
        }
        headerLayout.addView(iconText)

        val sectionTitle = TextView(this).apply {
            text = getString(R.string.help_contact_support)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
        }
        headerLayout.addView(sectionTitle)

        sectionLayout.addView(headerLayout)

        // Divider
        val divider = View(this).apply {
            setBackgroundColor(Color.parseColor("#E0E0E0"))
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                1
            ).apply { setMargins(0, 0, 0, 16) }
        }
        sectionLayout.addView(divider)

        // Description texts
        val cantFindText = TextView(this).apply {
            text = getString(R.string.help_cant_find)
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 0, 0, 8)
        }
        sectionLayout.addView(cantFindText)

        val descriptionText = TextView(this).apply {
            text = getString(R.string.help_contact_description)
            textSize = 13f
            setTextColor(Color.parseColor("#999999"))
            setPadding(0, 0, 0, 24)
        }
        sectionLayout.addView(descriptionText)

        // Email support button
        val emailButton = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(24, 20, 24, 20)

            val bgDrawable = GradientDrawable().apply {
                setColor(Color.parseColor("#E8F5E9"))
                cornerRadius = 12f * resources.displayMetrics.density
            }
            background = bgDrawable

            setOnClickListener { openEmailComposer() }
        }

        val emailIcon = TextView(this).apply {
            text = "\u2709"  // Envelope
            textSize = 24f
            setTextColor(Color.parseColor("#4CAF50"))
            setPadding(0, 0, 16, 0)
        }
        emailButton.addView(emailIcon)

        val emailTextLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        val emailTitle = TextView(this).apply {
            text = getString(R.string.help_email_support)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#4CAF50"))
        }
        emailTextLayout.addView(emailTitle)

        val emailAddress = TextView(this).apply {
            text = "support@paafekt.com"
            textSize = 13f
            setTextColor(Color.parseColor("#666666"))
        }
        emailTextLayout.addView(emailAddress)

        emailButton.addView(emailTextLayout)

        val arrowIcon = TextView(this).apply {
            text = "↗"
            textSize = 14f
            setTextColor(Color.parseColor("#999999"))
        }
        emailButton.addView(arrowIcon)

        sectionLayout.addView(emailButton)

        // Copy email button
        val copyButton = TextView(this).apply {
            text = "\uD83D\uDCCB ${getString(R.string.help_copy_email)}"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 20, 0, 0)
            setOnClickListener { copyEmailToClipboard() }
        }
        sectionLayout.addView(copyButton)

        return sectionLayout
    }

    private fun openEmailComposer() {
        val email = "support@paafekt.com"
        val subject = "Paafekt App Support"
        val body = "Hi Paafekt Support Team,\n\nI need help with:\n\n"

        val intent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_EMAIL, arrayOf(email))
            putExtra(Intent.EXTRA_SUBJECT, subject)
            putExtra(Intent.EXTRA_TEXT, body)
        }

        try {
            startActivity(Intent.createChooser(intent, "Send email"))
        } catch (e: Exception) {
            Toast.makeText(this, getString(R.string.no_email_app), Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyEmailToClipboard() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("email", "support@paafekt.com")
        clipboard.setPrimaryClip(clip)
        Toast.makeText(this, getString(R.string.email_copied_clipboard), Toast.LENGTH_SHORT).show()
    }
}
