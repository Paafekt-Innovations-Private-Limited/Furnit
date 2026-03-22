package com.furnit.android

import android.net.Uri
import android.os.Bundle
import com.furnit.android.utils.CrashReporter
import com.furnit.android.utils.LogUtil
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.furnit.android.models.PhotoOrientation
import com.furnit.android.services.SharpService

/**
 * Runs SHARP inference on an image provided by another app (e.g. BeeWare).
 * Caller passes EXTRA_IMAGE_URI (content URI). On success, returns RESULT_OK with EXTRA_PLY_PATH.
 * Used so BeeWare can run Sharp ML by starting this activity with startActivityForResult.
 */
class SharpInferenceActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sharp_inference)
        val progressBar = findViewById<LinearProgressIndicator>(R.id.sharp_progress)
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
            imageUri != null -> PhotoOrientation.loadBitmapApplyingExif(this, imageUri)
            else -> PhotoOrientation.loadBitmapApplyingExifFromFile(imagePath!!)
        }
        if (bitmap == null) {
            Toast.makeText(this, getString(R.string.failed_load_image), Toast.LENGTH_SHORT).show()
            CrashReporter.report(
                this,
                IllegalStateException("loadBitmapApplyingExif returned null"),
                "Sharp inference — load image",
            )
            setResult(RESULT_CANCELED)
            finish()
            return
        }

        statusText.text = getString(R.string.sharp_inference_loading_model)
        SharpService.getInstance(this).generateGaussians(
            bitmap,
            object : SharpService.ProgressCallback {
            override fun onProgress(progress: Float, message: String) {
                runOnUiThread {
                    progressBar.setProgress((progress * 100).toInt().coerceIn(0, 100), true)
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
                    CrashReporter.report(
                        this@SharpInferenceActivity,
                        RuntimeException(message),
                        "Sharp inference — SHARP generation",
                    )
                    setResult(RESULT_CANCELED)
                    finish()
                }
            }
        },
            sourcePhotoUri = imageUri,
            sourcePhotoPath = imagePath,
        )
    }

    companion object {
        private const val TAG = "SharpInference"
        const val EXTRA_IMAGE_URI = "image_uri"
        const val EXTRA_IMAGE_PATH = "image_path"
        const val EXTRA_PLY_PATH = "ply_path"
    }
}
