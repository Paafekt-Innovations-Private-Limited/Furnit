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
 * Shows Phase 1 non-commercial notice and ML attributions (YOLO-E, SHARP).
 */
class LicensesActivity : AppCompatActivity() {

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
        addSection(layout, getString(R.string.licenses_yoloe_title), getString(R.string.licenses_yoloe))
        addSection(layout, getString(R.string.licenses_sharp_title), getString(R.string.licenses_sharp))

        val linkView = TextView(this).apply {
            text = getString(R.string.licenses_full_online)
            textSize = 14f
            setTextColor(android.graphics.Color.parseColor("#007AFF"))
            setPadding(0, 24, 0, 0)
            setOnClickListener {
                try {
                    startActivity(Intent(Intent.ACTION_VIEW, Uri.parse("https://paafekt.com/licenses")))
                } catch (_: Exception) { }
            }
        }
        layout.addView(linkView)

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
        isBold: Boolean = false
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
        parent.addView(section)
    }
}
