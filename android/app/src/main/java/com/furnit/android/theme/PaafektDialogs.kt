package com.furnit.android.theme

import android.app.Activity
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AlertDialog
import com.furnit.android.R
import com.google.android.material.button.MaterialButton
import com.google.android.material.snackbar.Snackbar

/**
 * Paafekt-styled dialogs and snackbars — see docs/PAAFEKT_DESIGN_SYSTEM.md §9.
 */
object PaafektDialogs {

    fun showNameRoomDialog(
        activity: Activity,
        initialName: String = "",
        title: String = activity.getString(R.string.room_viewer_name_your_room),
        placeholder: String = activity.getString(R.string.room_viewer_room_name_placeholder),
        onSave: (String, dismiss: () -> Unit) -> Unit,
    ) {
        val density = activity.resources.displayMetrics.density
        val pad = (20 * density).toInt()
        val maxWidth = (420 * density).toInt()

        val input = EditText(activity).apply {
            setText(initialName)
            setSelection(initialName.length)
            hint = placeholder
            setHintTextColor(PaafektColors.textSecondary)
            setTextColor(PaafektColors.textPrimary)
            textSize = 16f
            inputType = InputType.TYPE_CLASS_TEXT or InputType.TYPE_TEXT_FLAG_CAP_WORDS
            setPadding(pad, pad, pad, pad)
            background = PaafektDrawables.secondaryButton()
            imeOptions = EditorInfo.IME_ACTION_DONE
        }

        val buttonRow = LinearLayout(activity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.END
        }

        lateinit var dialog: AlertDialog

        val cancelButton = MaterialButton(
            activity,
            null,
            com.google.android.material.R.attr.materialButtonOutlinedStyle,
        ).apply {
            text = activity.getString(R.string.common_cancel)
            setTextColor(PaafektColors.textPrimary)
            strokeColor = android.content.res.ColorStateList.valueOf(PaafektColors.hairline)
            setOnClickListener { dialog.dismiss() }
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f).apply {
                marginEnd = (8 * density).toInt()
            }
        }

        val saveButton = MaterialButton(activity).apply {
            text = activity.getString(R.string.common_save)
            setTextColor(PaafektColors.accentText)
            backgroundTintList = android.content.res.ColorStateList.valueOf(PaafektColors.accent)
            setOnClickListener {
                onSave(input.text?.toString()?.trim().orEmpty()) { dialog.dismiss() }
            }
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        buttonRow.addView(cancelButton)
        buttonRow.addView(saveButton)

        val content = LinearLayout(activity).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
            addView(input, LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ))
            addView(
                buttonRow,
                LinearLayout.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.WRAP_CONTENT,
                ).apply { topMargin = (16 * density).toInt() },
            )
        }

        val scroll = ScrollView(activity).apply {
            addView(content, FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT,
            ))
        }

        dialog = AlertDialog.Builder(activity, R.style.DarkDialogTheme)
            .setTitle(title)
            .setView(scroll)
            .create()

        dialog.window?.setSoftInputMode(WindowManager.LayoutParams.SOFT_INPUT_ADJUST_RESIZE)
        dialog.show()
        dialog.window?.setLayout(
            minOf((activity.resources.displayMetrics.widthPixels * 0.92f).toInt(), maxWidth),
            ViewGroup.LayoutParams.WRAP_CONTENT,
        )
        input.requestFocus()
    }

    fun showDeleteRoomDialog(
        activity: Activity,
        onDelete: () -> Unit,
    ) {
        AlertDialog.Builder(activity, R.style.DarkDialogTheme)
            .setTitle(R.string.delete_room_title)
            .setMessage(R.string.delete_room_message)
            .setNegativeButton(R.string.common_cancel, null)
            .setPositiveButton(R.string.common_delete) { _, _ -> onDelete() }
            .show()
    }
}

object PaafektSnackbar {

    fun showRoomSaved(rootView: View, roomName: String) {
        val message = rootView.context.getString(R.string.room_viewer_save_success, roomName)
        val snackbar = Snackbar.make(rootView, message, Snackbar.LENGTH_LONG)
        val snackView = snackbar.view
        snackView.setBackgroundColor(PaafektColors.surface)
        snackView.elevation = 12f

        val textView = snackView.findViewById<TextView>(com.google.android.material.R.id.snackbar_text)
        textView.setTextColor(PaafektColors.textPrimary)
        textView.maxLines = 2

        val density = rootView.resources.displayMetrics.density
        val row = LinearLayout(rootView.context).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding((4 * density).toInt(), 0, 0, 0)
        }
        val check = ImageView(rootView.context).apply {
            setImageResource(R.drawable.ic_check)
            layoutParams = LinearLayout.LayoutParams(
                (22 * density).toInt(),
                (22 * density).toInt(),
            ).apply { marginEnd = (10 * density).toInt() }
        }
        (textView.parent as? ViewGroup)?.let { parent ->
            val index = parent.indexOfChild(textView)
            parent.removeView(textView)
            row.addView(check)
            row.addView(textView, LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f))
            parent.addView(row, index)
        }

        snackbar.show()
    }
}
