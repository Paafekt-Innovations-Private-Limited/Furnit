package com.furnit.android

import android.Manifest
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import android.os.Bundle
import com.furnit.android.utils.LogUtil
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import java.io.File

class FurnitureFitActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "FurnitureFitActivity"
        const val EXTRA_ENABLE_AR_ASSISTED_SIZING = "enable_ar_assisted_sizing"
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            ensureFragmentLoaded()
        } else {
            Toast.makeText(this, getString(R.string.camera_permission_required), Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        // Lock orientation based on room's photo orientation (no auto-rotate)
        val photoOrientation = intent.getStringExtra("PHOTO_ORIENTATION") ?: "portrait"
        requestedOrientation = if (photoOrientation == "landscape") {
            ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE
        } else {
            ActivityInfo.SCREEN_ORIENTATION_PORTRAIT
        }

        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_furniture_fit)

        if (savedInstanceState != null) {
            return
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED) {
            ensureFragmentLoaded()
        } else {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun ensureFragmentLoaded() {
        if (supportFragmentManager.findFragmentById(R.id.furniture_fit_container) != null) {
            return
        }
        loadFragment()
    }

    private fun loadFragment() {
        val fragment = FurnitureFitFragment()

        // Pass room info to fragment so the 3D background matches the opened room (same camera framing as SharpRoom).
        var roomId = intent.getStringExtra("ROOM_ID")
        val roomName = intent.getStringExtra("ROOM_NAME")
        var roomFolder = intent.getStringExtra("ROOM_FOLDER")
        val roomWidth = intent.getFloatExtra("ROOM_WIDTH", RoomDefaults.widthMeters(this))
        val roomHeight = intent.getFloatExtra("ROOM_HEIGHT", RoomDefaults.heightMeters(this))
        val roomDepth = intent.getFloatExtra("ROOM_DEPTH", RoomDefaults.depthMeters(this))
        val photoOrientation = intent.getStringExtra("PHOTO_ORIENTATION") ?: "portrait"
        val enableArAssistedSizing = intent.getBooleanExtra(EXTRA_ENABLE_AR_ASSISTED_SIZING, false)
        if (roomFolder != null && roomFolder.isNotBlank()) {
            val f = File(roomFolder)
            if (!f.isAbsolute) {
                roomFolder = File(filesDir, roomFolder).absolutePath
            }
        }
        LogUtil.d(
            TAG,
            "Brain opened with ROOM_ID=$roomId ROOM_FOLDER=$roomFolder dims=${roomWidth}x${roomHeight}x${roomDepth} orientation=$photoOrientation arAssist=$enableArAssistedSizing",
        )

        fragment.arguments = Bundle().apply {
            roomId?.let { putString("ROOM_ID", it) }
            roomName?.let { putString("ROOM_NAME", it) }
            roomFolder?.let { putString("ROOM_FOLDER", it) }
            putFloat("ROOM_WIDTH", roomWidth)
            putFloat("ROOM_HEIGHT", roomHeight)
            putFloat("ROOM_DEPTH", roomDepth)
            putString("PHOTO_ORIENTATION", photoOrientation)
            putBoolean(EXTRA_ENABLE_AR_ASSISTED_SIZING, enableArAssistedSizing)
        }

        supportFragmentManager.beginTransaction()
            .replace(R.id.furniture_fit_container, fragment)
            .commit()
    }
}
