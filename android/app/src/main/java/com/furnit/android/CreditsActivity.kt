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

    private data class CreditEntry(
        val titleRes: Int,
        val bodyRes: Int,
        val websiteUrl: String,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = PaafektScreenViews.createScreenColumn(this)
        layout.addView(
            PaafektScreenViews.createBackButton(this, "‹ ${getString(R.string.common_back)}") { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.credits_title)))

        addSection(layout, getString(R.string.credits_intro), isBold = true)
        addSection(layout, getString(R.string.credits_disclaimer))

        val creditEntries = listOf(
            CreditEntry(R.string.credits_apple_title, R.string.credits_apple_body, "https://www.apple.com/"),
            CreditEntry(R.string.credits_google_title, R.string.credits_google_body, "https://about.google/"),
            CreditEntry(
                R.string.credits_depth_anything_title,
                R.string.credits_depth_anything_body,
                "https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf",
            ),
            CreditEntry(
                R.string.credits_geo_calib_title,
                R.string.credits_geo_calib_body,
                "https://github.com/cvg/GeoCalib",
            ),
            CreditEntry(
                R.string.credits_rtmdet_title,
                R.string.credits_rtmdet_body,
                "https://github.com/open-mmlab/mmdetection",
            ),
            CreditEntry(
                R.string.credits_onnx_runtime_title,
                R.string.credits_onnx_runtime_body,
                "https://onnxruntime.ai/",
            ),
            CreditEntry(
                R.string.credits_filament_title,
                R.string.credits_filament_body,
                "https://google.github.io/filament/",
            ),
            CreditEntry(R.string.credits_three_title, R.string.credits_three_body, "https://threejs.org/"),
            CreditEntry(
                R.string.credits_hypersim_title,
                R.string.credits_hypersim_body,
                "https://github.com/apple/ml-hypersim",
            ),
            CreditEntry(R.string.credits_coco_title, R.string.credits_coco_body, "https://cocodataset.org/"),
            CreditEntry(R.string.credits_openai_title, R.string.credits_openai_body, "https://openai.com/"),
            CreditEntry(R.string.credits_anthropic_title, R.string.credits_anthropic_body, "https://www.anthropic.com/"),
        )

        for (entry in creditEntries) {
            addSection(
                layout,
                getString(entry.titleRes),
                getString(entry.bodyRes),
                entry.websiteUrl,
            )
        }

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
