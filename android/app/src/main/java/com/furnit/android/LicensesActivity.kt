package com.furnit.android

import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.os.Bundle
import android.view.Gravity
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

/**
 * In-app Licenses & Attributions screen.
 * Shows Phase 1 non-commercial notice and Open Source Licenses (YOLO11, Sharp ML, Firebase).
 */
class LicensesActivity : AppCompatActivity() {

    private val urlAgpl = "https://www.gnu.org/licenses/agpl-3.0.html"
    private val urlMit = "https://opensource.org/licenses/MIT"
    private val urlApache2 = "https://www.apache.org/licenses/LICENSE-2.0"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val scrollView = ScrollView(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setPadding(24, 24, 24, 24)
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            )
        }

        val titleView = TextView(this).apply {
            text = getString(R.string.licenses_title)
            textSize = 20f
            setTypeface(null, Typeface.BOLD)
            setTextColor(android.graphics.Color.parseColor("#333333"))
            setPadding(0, 0, 0, 24)
        }
        layout.addView(titleView)

        addSection(layout, getString(R.string.licenses_phase1_notice), isBold = true)

        addSection(layout, getString(R.string.licenses_open_source_section), getString(R.string.licenses_open_source_intro))
        addSection(layout, getString(R.string.licenses_yoloe_title), getString(R.string.licenses_yoloe), licenseUrl = urlAgpl)
        addSection(layout, getString(R.string.licenses_sharp_title), getString(R.string.licenses_sharp), licenseUrl = urlMit)
        addSection(layout, getString(R.string.licenses_firebase_title), getString(R.string.licenses_firebase), licenseUrl = urlApache2)

        scrollView.addView(layout)
        setContentView(scrollView)
        supportActionBar?.setDisplayHomeAsUpEnabled(true)
        title = getString(R.string.licenses_title)
    }

    override fun onSupportNavigateUp(): Boolean {
        finish()
        return true
    }

    private fun addSection(
        parent: LinearLayout,
        title: String,
        body: String? = null,
        isBold: Boolean = false,
        licenseUrl: String? = null
    ) {
        val section = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(0, 0, 0, 20)
        }
        val titleView = TextView(this).apply {
            text = title
            textSize = 16f
            setTypeface(null, if (isBold) Typeface.BOLD else Typeface.NORMAL)
            setTextColor(android.graphics.Color.parseColor("#333333"))
            setPadding(0, 0, 0, 6)
        }
        section.addView(titleView)
        if (!body.isNullOrEmpty()) {
            val bodyView = TextView(this).apply {
                text = body
                textSize = 14f
                setTextColor(android.graphics.Color.parseColor("#666666"))
                setPadding(0, 0, 0, 0)
            }
            section.addView(bodyView)
        }
        if (!licenseUrl.isNullOrEmpty()) {
            val linkView = TextView(this).apply {
                text = getString(R.string.licenses_view_full_license)
                textSize = 14f
                setTextColor(android.graphics.Color.parseColor("#007AFF"))
                setPadding(0, 6, 0, 0)
                setOnClickListener {
                    try {
                        startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(licenseUrl)))
                    } catch (_: Exception) { }
                }
            }
            section.addView(linkView)
        }
        parent.addView(section)
    }
}
