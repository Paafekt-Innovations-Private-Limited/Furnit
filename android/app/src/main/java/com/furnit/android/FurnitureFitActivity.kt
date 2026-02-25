package com.furnit.android

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class FurnitureFitActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "FurnitureFitActivity"
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            loadFragment()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Lock orientation based on room's photo orientation (no auto-rotate)
        val photoOrientation = intent.getStringExtra("PHOTO_ORIENTATION") ?: "portrait"
        requestedOrientation = if (photoOrientation == "landscape") {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            loadFragment()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun loadFragment() {
        val fragment = FurnitureFitFragment()

        // Pass room info to fragment so the 3D background matches the opened room when room.glb exists.
        // ROOM_FOLDER = absolute path to room folder. No fallback: if no room.glb, no 3D background is shown.
        var roomId = intent.getStringExtra("ROOM_ID")
        val roomName = intent.getStringExtra("ROOM_NAME")
        var roomFolder = intent.getStringExtra("ROOM_FOLDER")
        // Resolve to absolute path so fragment can find room.glb
        if (roomFolder != null && roomFolder.isNotBlank()) {
            val f = File(roomFolder)
            if (!f.isAbsolute) {
                roomFolder = File(filesDir, roomFolder).absolutePath
            }
        }
        Log.d(TAG, "Brain opened with ROOM_ID=$roomId ROOM_NAME=$roomName ROOM_FOLDER=$roomFolder")

        fragment.arguments = Bundle().apply {
            roomId?.let { putString("ROOM_ID", it) }
            roomName?.let { putString("ROOM_NAME", it) }
            roomFolder?.let { putString("ROOM_FOLDER", it) }
        }

        supportFragmentManager.beginTransaction()
            .replace(android.R.id.content, fragment)
            .commit()
    }
}
