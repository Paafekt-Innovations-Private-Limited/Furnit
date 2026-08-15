package com.furnit.android

import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.widget.ScrollView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.furnit.android.theme.PaafektScreenViews
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class LicensesActivity : AppCompatActivity() {

    private val urlCcBy4 = "https://creativecommons.org/licenses/by/4.0/legalcode"
    private val urlHypersim = "https://github.com/apple/ml-hypersim"
    private val urlCoco = "https://cocodataset.org/#termsofuse"

    private val apacheLicense = BundledDocument(
        labelResource = R.string.licenses_view_full_license,
        assetPath = "legal/APACHE-2.0.txt",
    )
    private val liteRtLicense = BundledDocument(
        labelResource = R.string.licenses_view_full_license,
        assetPath = "legal/LITERT-LICENSE.txt",
    )
    private val thirdPartyNotices = BundledDocument(
        labelResource = R.string.licenses_view_notices,
        assetPath = "legal/THIRD_PARTY_NOTICES.txt",
    )
    private val onnxRuntimeLicense = BundledDocument(
        labelResource = R.string.licenses_view_full_license,
        assetPath = "legal/ONNXRUNTIME-MIT.txt",
    )
    private val onnxRuntimeNotices = BundledDocument(
        labelResource = R.string.licenses_view_notices,
        assetPath = "legal/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt",
    )
    private val threeJsLicense = BundledDocument(
        labelResource = R.string.licenses_view_full_license,
        assetPath = "legal/THREEJS-MIT.txt",
    )
    private val recaptchaNotices = BundledDocument(
        labelResource = R.string.licenses_view_notices,
        assetPath = "legal/RECAPTCHA-THIRD-PARTY-LICENSES.txt",
    )

    private data class BundledDocument(
        val labelResource: Int,
        val assetPath: String,
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val layout = PaafektScreenViews.createScreenColumn(this)
        layout.addView(
            PaafektScreenViews.createBackButton(this, PaafektScreenViews.backLabel(this)) { finish() },
        )
        layout.addView(PaafektScreenViews.createScreenTitle(this, getString(R.string.licenses_title)))

        addSection(
            layout,
            getString(R.string.licenses_open_source_section),
            getString(R.string.licenses_open_source_intro),
            bundledDocument = thirdPartyNotices,
        )
        addSection(
            layout,
            getString(R.string.licenses_depth_anything_title),
            getString(R.string.licenses_depth_anything),
            bundledDocument = apacheLicense,
        )
        addSection(layout, getString(R.string.licenses_hypersim_title), getString(R.string.licenses_hypersim), licenseUrl = urlHypersim)
        addSection(
            layout,
            getString(R.string.licenses_geo_calib_title),
            getString(R.string.licenses_geo_calib),
            licenseLinks = listOf(
                getString(R.string.licenses_view_cc_by_4) to urlCcBy4,
            ),
            bundledDocument = apacheLicense.copy(labelResource = R.string.licenses_view_apache_2_code),
        )
        addSection(
            layout,
            getString(R.string.licenses_android_platform_title),
            getString(R.string.licenses_android_platform),
            bundledDocument = apacheLicense,
        )
        addSection(
            layout,
            getString(R.string.licenses_other_runtime_title),
            getString(R.string.licenses_other_runtime),
            bundledDocument = liteRtLicense,
        )
        addSection(layout, getString(R.string.licenses_arcore_title), getString(R.string.licenses_arcore), licenseUrl = "https://developers.google.com/ar/develop/terms")
        addSection(
            layout,
            getString(R.string.licenses_firebase_title),
            getString(R.string.licenses_firebase),
            bundledDocument = apacheLicense,
        )
        addSection(
            layout,
            getString(R.string.licenses_recaptcha_title),
            getString(R.string.licenses_recaptcha),
            bundledDocument = recaptchaNotices,
        )
        addSection(
            layout,
            getString(R.string.licenses_rtmdet_title),
            getString(R.string.licenses_rtmdet),
            bundledDocument = apacheLicense,
        )
        addSection(layout, getString(R.string.licenses_coco_title), getString(R.string.licenses_coco), licenseUrl = urlCoco)
        addSection(
            layout,
            getString(R.string.licenses_onnx_runtime_title),
            getString(R.string.licenses_onnx_runtime),
            bundledDocuments = listOf(onnxRuntimeLicense, onnxRuntimeNotices),
        )
        addSection(
            layout,
            getString(R.string.licenses_filament_title),
            getString(R.string.licenses_filament),
            bundledDocument = apacheLicense,
        )
        addSection(
            layout,
            getString(R.string.licenses_three_title),
            getString(R.string.licenses_three),
            bundledDocument = threeJsLicense,
        )

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
        licenseLinks: List<Pair<String, String>> = emptyList(),
        bundledDocument: BundledDocument? = null,
        bundledDocuments: List<BundledDocument> = emptyList(),
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
        val links = when {
            licenseLinks.isNotEmpty() -> licenseLinks
            !licenseUrl.isNullOrEmpty() -> listOf(getString(R.string.licenses_view_full_license) to licenseUrl)
            else -> emptyList()
        }
        for ((label, url) in links) {
            card.addView(
                PaafektScreenViews.createLinkLabel(this, label) {
                    try {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
                    } catch (_: Exception) {
                    }
                },
            )
        }
        for (document in listOfNotNull(bundledDocument) + bundledDocuments) {
            card.addView(
                PaafektScreenViews.createLinkLabel(this, getString(document.labelResource)) {
                    showBundledDocument(title, document.assetPath)
                },
            )
        }
        parent.addView(card)
    }

    private fun showBundledDocument(title: String, assetPath: String) {
        lifecycleScope.launch {
            val documentText = withContext(Dispatchers.IO) {
                runCatching {
                    assets.open(assetPath).bufferedReader(Charsets.UTF_8).use { reader ->
                        reader.readText()
                    }
                }.getOrNull()
            } ?: getString(R.string.licenses_document_unavailable)

            val textView = TextView(this@LicensesActivity).apply {
                text = documentText
                textSize = 12f
                typeface = Typeface.MONOSPACE
                setTextColor(com.furnit.android.theme.PaafektColors.textPrimary)
                setTextIsSelectable(true)
                val padding = (16 * resources.displayMetrics.density).toInt()
                setPadding(padding, padding, padding, padding)
            }
            val scrollView = ScrollView(this@LicensesActivity).apply {
                addView(textView)
            }
            AlertDialog.Builder(this@LicensesActivity)
                .setTitle(title)
                .setView(scrollView)
                .setPositiveButton(android.R.string.ok, null)
                .show()
        }
    }
}
