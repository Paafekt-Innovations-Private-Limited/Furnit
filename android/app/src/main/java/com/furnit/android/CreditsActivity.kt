package com.furnit.android

import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.theme.PaafektScreenViews

class CreditsActivity : AppCompatActivity() {

    private val appleUrl = "https://www.apple.com/"
    private val googleUrl = "https://about.google/"
    private val openAiUrl = "https://openai.com/"
    private val anthropicUrl = "https://www.anthropic.com/"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = PaafektScreenViews.createScreenColumn(this)
        layout.addView(
            PaafektScreenViews.createBackButton(this, "‹ ${getString(R.string.common_back)}") { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.credits_title)))

        addSection(layout, getString(R.string.credits_intro), isBold = true)
        addSection(layout, getString(R.string.credits_disclaimer))
        addSection(layout, getString(R.string.credits_apple_title), getString(R.string.credits_apple_body), appleUrl)
        addSection(layout, getString(R.string.credits_google_title), getString(R.string.credits_google_body), googleUrl)
        addSection(layout, getString(R.string.credits_openai_title), getString(R.string.credits_openai_body), openAiUrl)
        addSection(layout, getString(R.string.credits_anthropic_title), getString(R.string.credits_anthropic_body), anthropicUrl)

        PaafektScreenViews.createScreenScrollView(this).apply {
            addView(layout)
            setContentView(this)
        }
    }

    private fun addSection(
        parent: LinearLayout,
        title: String,
        body: String? = null,
        websiteUrl: String? = null,
        isBold: Boolean = false,
    ) {
        val card = PaafektScreenViews.createSectionCard(this)
        card.addView(
            TextView(this).apply {
                text = title
                textSize = 16f
                setTypeface(null, if (isBold) Typeface.BOLD else Typeface.NORMAL)
                setTextColor(com.furnit.android.theme.PaafektColors.textPrimary)
            },
        )
        if (!body.isNullOrEmpty()) {
            card.addView(PaafektScreenViews.createSecondaryLabel(this, body))
        }
        if (!websiteUrl.isNullOrEmpty()) {
            card.addView(
                PaafektScreenViews.createLinkLabel(this, getString(R.string.credits_visit_website)) {
                    try {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(websiteUrl)))
                    } catch (_: Exception) {
                    }
                },
            )
        }
        parent.addView(card)
    }
}
