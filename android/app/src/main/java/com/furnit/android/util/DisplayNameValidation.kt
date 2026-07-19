package com.furnit.android.util

/** Shared rule for login name and room name: ≥3 characters with at least one letter. */
object DisplayNameValidation {
    fun isValid(raw: String): Boolean {
        val trimmed = raw.trim()
        if (trimmed.length < 3) return false
        return trimmed.any { it.isLetter() }
    }
}
