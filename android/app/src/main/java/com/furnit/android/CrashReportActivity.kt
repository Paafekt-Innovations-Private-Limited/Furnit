package com.furnit.android

import android.os.Bundle
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.utils.CrashReporter

/**
 * Shown when the app crashes. Runs in a separate process so it can display
 * after the main process is killed. Offers "Submit Report" (email) and "Copy Details".
 *
 * Email body / support address are shared with [com.furnit.android.utils.CrashReporter] (Swift parity for handled errors).
 */
class CrashReportActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val message = intent.getStringExtra(EXTRA_CRASH_MESSAGE) ?: ""
        val stackTrace = intent.getStringExtra(EXTRA_CRASH_STACKTRACE) ?: ""
        val fullDetails = CrashReporter.buildUncaughtReportBody(this, message, stackTrace)

        AlertDialog.Builder(this)
            .setTitle(getString(R.string.crash_report_title))
            .setMessage(getString(R.string.crash_report_message))
            .setCancelable(false)
            .setPositiveButton(getString(R.string.crash_report_submit_report)) { _, _ ->
                sendReport(fullDetails)
            }
            .setNeutralButton(getString(R.string.crash_report_copy_details)) { _, _ ->
                CrashReporter.copyReportToClipboard(this, fullDetails)
                Toast.makeText(this, getString(R.string.crash_report_copy_details), Toast.LENGTH_SHORT).show()
            }
            .setNegativeButton(android.R.string.cancel) { _, _ ->
                finish()
            }
            .setOnDismissListener { finish() }
            .show()
    }

    private fun sendReport(body: String) {
        if (!CrashReporter.sendReportEmail(this, body)) {
            Toast.makeText(this, getString(R.string.crash_report_email_not_configured_message), Toast.LENGTH_LONG).show()
            CrashReporter.copyReportToClipboard(this, body)
        }
        finish()
    }

    companion object {
        const val EXTRA_CRASH_MESSAGE = "crash_message"
        const val EXTRA_CRASH_STACKTRACE = "crash_stacktrace"
    }
}
