package com.furnit.android

import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * In-app credits / acknowledgements screen for third-party AI and ML tools.
 * Keeps acknowledgements separate from open-source licenses for clearer legal structure.
 */
class CreditsActivity : AppCompatActivity() {

    private val appleUrl = "https://www.apple.com/"
    private val openAiUrl = "https://openai.com/"
    private val anthropicUrl = "https://www.anthropic.com/"
    private val lumaUrl = "https://lumalabs.ai/"
    private val ultralyticsUrl = "https://www.ultralytics.com/"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val scrollView = ScrollView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT,
            )
            setPadding(24, 24, 24, 24)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            )
        }

        val titleView = TextView(this).apply {
            text = getString(R.string.credits_title)
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 24)
        }
        layout.addView(titleView)

        addSection(layout, getString(R.string.credits_intro), isBold = true)
        addSection(layout, getString(R.string.credits_disclaimer))
        addSection(layout, getString(R.string.credits_apple_title), getString(R.string.credits_apple_body), appleUrl)
        addSection(layout, getString(R.string.credits_openai_title), getString(R.string.credits_openai_body), openAiUrl)
        addSection(layout, getString(R.string.credits_anthropic_title), getString(R.string.credits_anthropic_body), anthropicUrl)
        addSection(layout, getString(R.string.credits_luma_title), getString(R.string.credits_luma_body), lumaUrl)
        addSection(layout, getString(R.string.credits_ultralytics_title), getString(R.string.credits_ultralytics_body), ultralyticsUrl)

        scrollView.addView(layout)
        setContentView(scrollView)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = getString(R.string.credits_title)
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private fun addSection(
        parent: LinearLayout,
        title: String,
        body: String? = null,
        websiteUrl: String? = null,
        isBold: Boolean = false,
    ) {
        val section = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, 20)
        }

        val titleView = TextView(this).apply {
            text = title
            textSize = 16f
            setTypeface(null, if (isBold) Typeface.BOLD else Typeface.NORMAL)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 6)
        }
        section.addView(titleView)

        if (!body.isNullOrEmpty()) {
            val bodyView = TextView(this).apply {
                text = body
                textSize = 14f
                setTextColor(Color.parseColor("#666666"))
            }
            section.addView(bodyView)
        }

        if (!websiteUrl.isNullOrEmpty()) {
            val linkView = TextView(this).apply {
                text = getString(R.string.credits_visit_website)
                textSize = 14f
                setTextColor(Color.parseColor("#007AFF"))
                setPadding(0, 6, 0, 0)
                setOnClickListener {
                    try {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(websiteUrl)))
                    } catch (_: Exception) {
                    }
                }
            }
            section.addView(linkView)
        }

        parent.addView(section)
    }
}
