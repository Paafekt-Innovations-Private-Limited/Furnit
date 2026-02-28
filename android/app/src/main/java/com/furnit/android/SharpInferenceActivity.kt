package com.furnit.android

import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import com.furnit.android.utils.LogUtil
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.furnit.android.services.SharpService
import java.io.InputStream

/**
 * Runs SHARP inference on an image provided by another app (e.g. BeeWare).
 * Caller passes EXTRA_IMAGE_URI (content URI). On success, returns RESULT_OK with EXTRA_PLY_PATH.
 * Used so BeeWare can run Sharp ML by starting this activity with startActivityForResult.
 */
class SharpInferenceActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sharp_inference)
        val progressBar = findViewById<ProgressBar>(R.id.sharp_progress)
        val statusText = findViewById<TextView>(R.id.sharp_status)

        val imageUri = intent.getStringExtra(EXTRA_IMAGE_URI)?.let { Uri.parse(it) }
        val imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        if (imageUri == null && imagePath.isNullOrBlank()) {
            Toast.makeText(this, getString(R.string.sharp_inference_no_image), Toast.LENGTH_SHORT).show()
            setResult(RESULT_CANCELED)
            finish()
            return
        }
        val bitmap = when {
            imageUri != null -> loadBitmapFromUri(imageUri)
            else -> BitmapFactory.decodeFile(imagePath)
        }
        if (bitmap == null) {
            Toast.makeText(this, getString(R.string.failed_load_image), Toast.LENGTH_SHORT).show()
            setResult(RESULT_CANCELED)
            finish()
            return
        }

        statusText.text = getString(R.string.sharp_inference_loading_model)
        SharpService.getInstance(this).generateGaussians(bitmap, object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                runOnUiThread {
                    progressBar.progress = (progress * 100).toInt().coerceIn(0, 100)
                    statusText.text = message
                }
            }
            override fun onComplete(result: SharpService.GenerationResult) {
                runOnUiThread {
                    LogUtil.d(TAG, "Sharp inference done: ${result.classicPlyFile.absolutePath}")
                    setResult(RESULT_OK, android.content.Intent().putExtra(EXTRA_PLY_PATH, result.classicPlyFile.absolutePath))
                    finish()
                }
            }
            override fun onError(message: String) {
                runOnUiThread {
                    LogUtil.e(TAG, "Sharp inference error: $message")
                    Toast.makeText(this@SharpInferenceActivity, message, Toast.LENGTH_LONG).show()
                    setResult(RESULT_CANCELED)
                    finish()
                }
            }
        })
    }

    private fun loadBitmapFromUri(uri: Uri): android.graphics.Bitmap? {
        return try {
            contentResolver.openInputStream(uri).use { stream: InputStream? ->
                stream?.let { BitmapFactory.decodeStream(it) }
            }
        } catch (e: Exception) {
            LogUtil.e(TAG, "Failed to load image from URI", e)
            null
        }
    }

        companion object {
        private const val TAG = "SharpInference"
        const val EXTRA_IMAGE_URI = "image_uri"
        const val EXTRA_IMAGE_PATH = "image_path"
        const val EXTRA_PLY_PATH = "ply_path"
    }
}
