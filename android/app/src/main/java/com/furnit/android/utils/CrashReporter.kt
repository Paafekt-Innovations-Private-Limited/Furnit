package com.furnit.android.utils

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.net.Uri
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.fragment.app.Fragment
import com.furnit.android.R

/**
 * Parity with iOS [Furnit/Utilities/CrashReporter.swift]:
 * - **Uncaught** crashes → [com.furnit.android.CrashReportActivity] (separate `:crash` process) still handles those.
 * - **Caught** errors → call [report] from an `Activity` `catch` block (alert + mailto + copy), same support address as Swift.
 */
object CrashReporter {

    const val SUPPORT_EMAIL = "support@paafekt.com"

    fun appVersionName(context: Context): String = try {
        context.packageManager.getPackageInfo(context.packageName, 0).versionName ?: "unknown"
    } catch (_: Exception) {
        "unknown"
    }

    fun buildUncaughtReportBody(context: Context, message: String, stackTrace: String): String =
        """
            Paafekt crash report

            Version: ${appVersionName(context)}
            Exception: $message

            $stackTrace
        """.trimIndent()

    fun buildHandledReportBody(context: Context, userContext: String, throwable: Throwable): String =
        """
            Paafekt crash report (handled error)

            Version: ${appVersionName(context)}
            Context: $userContext
            Exception: ${throwable.message ?: throwable.toString()}

            ${throwable.stackTraceToString()}
        """.trimIndent()

    /**
     * @return true if a mail handler was started
     */
    fun sendReportEmail(
        context: Context,
        body: String,
        subject: String = "Paafekt crash report",
    ): Boolean {
        val emailIntent = Intent(Intent.ACTION_SENDTO).apply {
            data = Uri.parse("mailto:")
            putExtra(Intent.EXTRA_EMAIL, arrayOf(SUPPORT_EMAIL))
            putExtra(Intent.EXTRA_SUBJECT, subject)
            putExtra(Intent.EXTRA_TEXT, body)
        }
        val pm = context.packageManager
        val canHandle = emailIntent.resolveActivity(pm) != null
        return if (canHandle) {
            context.startActivity(
                Intent.createChooser(emailIntent, context.getString(R.string.crash_report_submit_report))
            )
            true
        } else {
            false
        }
    }

    fun copyReportToClipboard(context: Context, body: String) {
        (context.getSystemService(Context.CLIPBOARD_SERVICE) as? ClipboardManager)?.setPrimaryClip(
            ClipData.newPlainText("Crash report", body)
        )
    }

    /** Same as [report] for [Fragment] hosts (e.g. [FurnitureFitFragment]). */
    fun report(fragment: Fragment, throwable: Throwable, userContext: String) {
        val act = fragment.activity as? AppCompatActivity ?: return
        report(act, throwable, userContext)
    }

    /**
     * Same idea as Swift `CrashReporter.shared.report(_:context:)` — use in `catch` when you want the user to email support.
     */
    fun report(activity: AppCompatActivity, throwable: Throwable, userContext: String) {
        activity.runOnUiThread {
            val body = buildHandledReportBody(activity, userContext, throwable)
            val msg = buildString {
                append(activity.getString(R.string.crash_report_message))
                append("\n\n")
                append(userContext)
                throwable.message?.let { append("\n").append(it) }
            }
            AlertDialog.Builder(activity)
                .setTitle(R.string.crash_report_title)
                .setMessage(msg)
                .setPositiveButton(R.string.crash_report_submit_report) { _, _ ->
                    if (!sendReportEmail(
                            activity,
                            body,
                            subject = "Paafekt crash report: $userContext",
                        )
                    ) {
                        Toast.makeText(
                            activity,
                            activity.getString(R.string.crash_report_email_not_configured_message),
                            Toast.LENGTH_LONG,
                        ).show()
                        copyReportToClipboard(activity, body)
                    }
                }
                .setNeutralButton(R.string.crash_report_copy_details) { _, _ ->
                    copyReportToClipboard(activity, body)
                    Toast.makeText(
                        activity,
                        activity.getString(R.string.crash_report_copy_details),
                        Toast.LENGTH_SHORT,
                    ).show()
                }
                .setNegativeButton(android.R.string.cancel, null)
                .show()
        }
    }
}
