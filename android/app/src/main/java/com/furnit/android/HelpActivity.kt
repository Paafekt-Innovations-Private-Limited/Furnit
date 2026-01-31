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
import android.view.animation.AnimationUtils
import android.widget.*
import androidx.appcompat.app.AppCompatActivity

/**
 * Help & Support Activity with FAQ sections and contact support
 * Mirrors the iOS SupportView implementation
 */
class HelpActivity : AppCompatActivity() {

    // Track expanded FAQ items
    private val expandedFAQs = mutableSetOf<String>()

    // FAQ Data Models
    data class FAQItem(
        val question: String,
        val answer: String
    ) {
        val id: String get() = question
    }

    data class FAQSection(
        val title: String,
        val icon: String,
        val items: List<FAQItem>
    )

    // FAQ Content - matching iOS
    private val faqSections = listOf(
        FAQSection(
            title = "Room Creation",
            icon = "camera",
            items = listOf(
                FAQItem(
                    "How do I create a 3D room?",
                    "Tap the photo icon in the top-left corner of the home screen, then take or select a photo of your room. You'll see two options to choose from for creating your 3D room."
                ),
                FAQItem(
                    "What are the two room creation options?",
                    "When you select a photo, you can choose between: 1) AI-Powered 3D Room - automatically creates a realistic 3D room using artificial intelligence, or 2) Manual Setup - lets you draw the room boundaries yourself for more control."
                ),
                FAQItem(
                    "What is AI-Powered 3D Room?",
                    "This option uses smart technology to automatically turn your photo into a 3D room you can walk through. Just pick a photo and the app does the rest - no drawing or adjusting needed. It works like magic!"
                ),
                FAQItem(
                    "What is Manual Setup?",
                    "With Manual Setup, you draw the outline of your room's walls on the photo. This gives you more control over exactly how the 3D room looks. It's great when you want to fine-tune the room shape yourself."
                ),
                FAQItem(
                    "Which method should I choose?",
                    "Try AI-Powered first - it's faster and works great for most rooms. If the result doesn't look quite right, use Manual Setup to draw the walls exactly where you want them. You can always try both and see which one you prefer!"
                ),
                FAQItem(
                    "What kind of photos work best?",
                    "For best results, take photos in good lighting with the camera held level. Try to capture the entire room including floors, walls, and ceiling edges. Avoid blurry or dark photos."
                ),
                FAQItem(
                    "Why is my room generation failing?",
                    "Room generation may fail if the photo is too dark, blurry, or doesn't show enough room features. Try taking a new photo with better lighting and a wider angle."
                ),
                FAQItem(
                    "How many rooms can I create?",
                    "You can create up to 1000 rooms. Delete older rooms to make space for new ones. The room count and total storage used are shown at the top of your home screen."
                ),
                FAQItem(
                    "How do I save a room?",
                    "After generating a 3D room, tap the save icon (download arrow) in the toolbar to save it with a custom name. Rooms that aren't saved will be deleted when you tap Back."
                )
            )
        ),
        FAQSection(
            title = "AI Features",
            icon = "brain",
            items = listOf(
                FAQItem(
                    "What does the brain icon do?",
                    "The brain icon activates SmartyPants - an AI-powered object detection feature. It uses your camera to identify furniture and objects in real-time, showing labels and bounding boxes around detected items."
                ),
                FAQItem(
                    "How do I take a screenshot?",
                    "When SmartyPants is active (brain icon is green), a share icon appears on the bottom-right. Tap it to save a screenshot of the view with AI detections to your Photos library."
                ),
                FAQItem(
                    "What is furniture segmentation?",
                    "Furniture segmentation uses AI to identify and separate furniture items in your photos, allowing you to see how each piece would fit in your 3D room."
                ),
                FAQItem(
                    "How do I segment furniture from a photo?",
                    "When viewing your 3D room, tap on the brain icon to activate SmartyPants. Point your camera at furniture to see real-time detection with labels."
                ),
                FAQItem(
                    "Why isn't my furniture being detected?",
                    "Object detection works best with clear, well-lit environments where furniture is clearly visible. Make sure the object isn't partially hidden or too far from the camera."
                )
            )
        ),
        FAQSection(
            title = "3D Room Controls",
            icon = "cube",
            items = listOf(
                FAQItem(
                    "How do I view my 3D room?",
                    "Tap on any room in your home screen to open the 3D viewer. Use touch gestures to rotate and zoom, or use the joystick at the bottom to move around inside the room."
                ),
                FAQItem(
                    "How do I navigate inside the room?",
                    "Use the joystick at the bottom center of the screen to move around. Drag it in any direction to fly through the room. Use the recenter button (viewfinder icon) to reset the camera position."
                ),
                FAQItem(
                    "What does the MB number mean?",
                    "The MB (megabytes) shown in rooms indicates the file size. This helps you manage storage. The total storage used by all rooms is shown at the top of the home screen."
                ),
                FAQItem(
                    "Can I use a sample room instead of my own?",
                    "Yes! The app provides sample rooms (Vintage Living Room, Cozy Living Room) for you to experiment with. Access them from the home screen to try out features without creating your own room first."
                ),
                FAQItem(
                    "How accurate is the 3D model?",
                    "The 3D model provides a visual approximation of your room's layout. The dimensions shown are relative units from the AI model, not exact real-world measurements."
                ),
                FAQItem(
                    "Can I adjust room dimensions?",
                    "For USDZ rooms, go to Settings and look for Room Dimensions options. For AI-generated (PLY) rooms, the dimensions are determined by the AI model."
                )
            )
        )
    )

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
            text = "< Back"
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
            text = "Help & Support"
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
                "cube" -> "\uD83D\uDDBC"    // Cube emoji
                else -> "\u2753"            // Question mark
            }
            textSize = 18f
            setPadding(0, 0, 12, 0)
        }
        headerLayout.addView(iconText)

        val sectionTitle = TextView(this).apply {
            text = section.title
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
            text = item.question
            textSize = 14f
            setTextColor(Color.parseColor("#333333"))
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        questionLayout.addView(questionText)

        val chevron = TextView(this).apply {
            text = if (expandedFAQs.contains(item.id)) "▲" else "▼"
            textSize = 12f
            setTextColor(Color.parseColor("#999999"))
            tag = "chevron_${item.id}"
        }
        questionLayout.addView(chevron)

        itemLayout.addView(questionLayout)

        // Answer (initially hidden unless expanded)
        val answerText = TextView(this).apply {
            text = item.answer
            textSize = 13f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 12, 0, 0)
            visibility = if (expandedFAQs.contains(item.id)) View.VISIBLE else View.GONE
            tag = "answer_${item.id}"
        }
        itemLayout.addView(answerText)

        // Click handler for expand/collapse
        questionLayout.setOnClickListener {
            val isExpanded = expandedFAQs.contains(item.id)
            if (isExpanded) {
                expandedFAQs.remove(item.id)
                answerText.visibility = View.GONE
                chevron.text = "▼"
            } else {
                expandedFAQs.add(item.id)
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
            text = "Contact Support"
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
            text = "Can't find what you're looking for?"
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 0, 0, 8)
        }
        sectionLayout.addView(cantFindText)

        val descriptionText = TextView(this).apply {
            text = "Our support team is here to help. Send us an email and we'll get back to you as soon as possible."
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
            text = "Email Support"
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
            text = "\uD83D\uDCCB Copy Email Address"
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
            Toast.makeText(this, "No email app found", Toast.LENGTH_SHORT).show()
        }
    }

    private fun copyEmailToClipboard() {
        val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
        val clip = ClipData.newPlainText("email", "support@paafekt.com")
        clipboard.setPrimaryClip(clip)
        Toast.makeText(this, "Email copied to clipboard", Toast.LENGTH_SHORT).show()
    }
}
