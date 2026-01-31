package com.furnit.android

import android.app.Application
import android.util.Log
import com.furnit.android.utils.DebugLogger
import com.google.firebase.FirebaseApp

/**
 * FurnitApplication - Application class for initializing Firebase
 */
class FurnitApplication : Application() {

    companion object {
        private const val TAG = "FurnitApp"
    }

    override fun onCreate() {
        super.onCreate()

        // Initialize DebugLogger
        DebugLogger.init(this)

        // Initialize Firebase
        try {
            FirebaseApp.initializeApp(this)
            Log.d(TAG, "Firebase initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize Firebase", e)
        }
    }
}
