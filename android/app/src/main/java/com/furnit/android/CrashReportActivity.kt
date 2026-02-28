package com.furnit.android

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity

/**
 * Shown when the app crashes. Runs in a separate process so it can display
 * after the main process is killed. Offers "Submit Report" (email) and "Copy Details".
 */
class CrashReportActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val message = intent.getStringExtra(EXTRA_CRASH_MESSAGE) ?: ""
        val stackTrace = intent.getStringExtra(EXTRA_CRASH_STACKTRACE) ?: ""
        val fullDetails = buildReportBody(message, stackTrace)

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.crash_report_title))
            .setMessage(getString(R.string.crash_report_message))
            .setCancelable(false)
            .setPositiveButton(getString(R.string.crash_report_submit_report)) { _, _ ->
                sendReport(fullDetails)
            }
            .setNeutralButton(getString(R.string.crash_report_copy_details)) { _, _ ->
                copyToClipboard(fullDetails)
                Toast.makeText(this, getString(R.string.crash_report_copy_details), Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(android.R.string.cancel) { _, _ ->
                finish()
            }
            .setOnDismissListener { finish() }
            .show()
    }

    private fun buildReportBody(message: String, stackTrace: String): String {
        val appVersion = try {
            packageManager.getPackageInfo(packageName, 0).versionName ?: "unknown"
        } catch (_: Exception) {
            "unknown"
        }
        return """
            Paafekt crash report

            Version: $appVersion
            Exception: $message

            $stackTrace
        """.trimIndent()
    }

    private fun sendReport(body: String) {
        val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_EMAIL, arrayOf(SUPPORT_EMAIL))
            putExtra(Intent.EXTRA_SUBJECT, "Paafekt crash report")
            putExtra(Intent.EXTRA_TEXT, body)
        }
        if (packageManager.resolveActivity(emailIntent, 0) != null) {
            startActivity(Intent.createChooser(emailIntent, getString(R.string.crash_report_submit_report)))
        } else {
            Toast.makeText(this, getString(R.string.crash_report_email_not_configured_message), Toast.LENGTH_LONG).show()
            copyToClipboard(body)
        }
        finish()
    }

    private fun copyToClipboard(text: String) {
        (getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager)?.setPrimaryClip(
            ClipData.newPlainText("Crash report", text)
        )
    }

    companion object {
        private const val SUPPORT_EMAIL = "support@paafekt.com"
        const val EXTRA_CRASH_MESSAGE = "crash_message"
        const val EXTRA_CRASH_STACKTRACE = "crash_stacktrace"
    }
}
