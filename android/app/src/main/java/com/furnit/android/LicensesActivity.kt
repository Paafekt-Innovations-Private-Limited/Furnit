package com.furnit.android

import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.theme.PaafektScreenViews

class LicensesActivity : AppCompatActivity() {

    private val urlMit = "https://opensource.org/licenses/MIT"
    private val urlApache2 = "https://www.apache.org/licenses/LICENSE-2.0"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = PaafektScreenViews.createScreenColumn(this)
        layout.addView(
            PaafektScreenViews.createBackButton(this, "‹ ${getString(R.string.common_back)}") { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.licenses_title)))

        addSection(layout, getString(R.string.licenses_phase1_notice), isBold = true)
        addSection(layout, getString(R.string.licenses_open_source_section), getString(R.string.licenses_open_source_intro))
        addSection(layout, getString(R.string.licenses_depth_anything_title), getString(R.string.licenses_depth_anything), licenseUrl = urlApache2)
        addSection(layout, getString(R.string.licenses_geo_calib_title), getString(R.string.licenses_geo_calib), licenseUrl = urlApache2)
        addSection(layout, getString(R.string.licenses_firebase_title), getString(R.string.licenses_firebase), licenseUrl = urlApache2)
        addSection(layout, getString(R.string.licenses_rtmdet_title), getString(R.string.licenses_rtmdet), licenseUrl = urlApache2)
        addSection(layout, getString(R.string.licenses_three_title), getString(R.string.licenses_three), licenseUrl = urlMit)

        PaafektScreenViews.createScreenScrollView(this).apply {
            addView(layout)
            setContentView(this)
        }
    }

    private fun addSection(
        parent: LinearLayout,
        title: String,
        body: String? = null,
        isBold: Boolean = false,
        licenseUrl: String? = null,
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
        if (!licenseUrl.isNullOrEmpty()) {
            card.addView(
                PaafektScreenViews.createLinkLabel(this, getString(R.string.licenses_view_full_license)) {
                    try {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(licenseUrl)))
                    } catch (_: Exception) {
                    }
                },
            )
        }
        parent.addView(card)
    }
}
