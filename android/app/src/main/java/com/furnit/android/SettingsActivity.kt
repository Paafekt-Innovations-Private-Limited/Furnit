package com.furnit.android

import android.content.Intent
import android.content.SharedPreferences
import android.net.Uri
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.*
import android.content.res.ColorStateList
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.widget.SwitchCompat
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.models.QualitySettings
import com.furnit.android.services.BackendConfig
import com.furnit.android.utils.DebugLogger

class SettingsActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences
    private lateinit var authManager: AuthenticationManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        prefs = getSharedPreferences("furnit_prefs", MODE_PRIVATE)
        authManager = AuthenticationManager.getInstance(this)

        val scrollView = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F5F5F5"))
        }

        val layout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 48, 32, 32)
        }

        // Back button
        val backBtn = TextView(this).apply {
            text = "< ${getString(R.string.common_back)}"
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 0, 24)
            setOnClickListener { finish() }
        }
        layout.addView(backBtn)

        // Title
        val title = TextView(this).apply {
            text = getString(R.string.settings_title)
            textSize = 24f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 0, 0, 32)
        }
        layout.addView(title)

        // User info section
        val userSection = createSection(getString(R.string.settings_account))
        val userInfo = TextView(this).apply {
            val userName = authManager.getUserName()
            val userPhone = authManager.getUserPhone()
            text = if (userName.isNotEmpty()) "$userName\n$userPhone" else userPhone.ifEmpty { getString(R.string.settings_not_signed_in) }
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 8, 0, 16)
        }
        userSection.addView(userInfo)
        layout.addView(userSection)

        // Quality settings section
        val qualitySection = createSection(getString(R.string.settings_rendering_quality))

        val rg = RadioGroup(this).apply {
            orientation = RadioGroup.VERTICAL
        }
        val lowId = View.generateViewId()
        val medId = View.generateViewId()
        val highId = View.generateViewId()

        val low = RadioButton(this).apply {
            text = "${getString(R.string.quality_low)} - ${getString(R.string.quality_low_description)}"
            id = lowId
            setTextColor(Color.parseColor("#333333"))
        }
        val med = RadioButton(this).apply {
            text = "${getString(R.string.quality_medium)} - ${getString(R.string.quality_medium_description)}"
            id = medId
            setTextColor(Color.parseColor("#333333"))
        }
        val high = RadioButton(this).apply {
            text = "${getString(R.string.quality_high)} - ${getString(R.string.quality_high_description)}"
            id = highId
            setTextColor(Color.parseColor("#333333"))
        }
        rg.addView(low)
        rg.addView(med)
        rg.addView(high)

        val current = prefs.getString("quality", QualitySettings.MEDIUM)
        when (current) {
            QualitySettings.LOW -> rg.check(lowId)
            QualitySettings.MEDIUM -> rg.check(medId)
            QualitySettings.HIGH -> rg.check(highId)
        }

        rg.setOnCheckedChangeListener { _, checkedId ->
            val v = when (checkedId) {
                lowId -> QualitySettings.LOW
                medId -> QualitySettings.MEDIUM
                highId -> QualitySettings.HIGH
                else -> QualitySettings.MEDIUM
            }
            prefs.edit().putString("quality", v).apply()
        }

        qualitySection.addView(rg)
        layout.addView(qualitySection)

        // Room Viewer settings section
        val viewerSection = createSection(getString(R.string.settings_room_viewer))

        // Auto-orbit toggle (default OFF)
        val autoOrbitLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        val autoOrbitLabel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }
        val autoOrbitTitle = TextView(this).apply {
            text = getString(R.string.settings_auto_orbit)
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
        }
        val autoOrbitDesc = TextView(this).apply {
            text = getString(R.string.settings_auto_orbit_description)
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        autoOrbitLabel.addView(autoOrbitTitle)
        autoOrbitLabel.addView(autoOrbitDesc)

        val autoOrbitSwitch = createStyledSwitch(prefs.getBoolean("auto_orbit_enabled", false)) { isChecked ->
            prefs.edit().putBoolean("auto_orbit_enabled", isChecked).apply()
        }

        autoOrbitLayout.addView(autoOrbitLabel)
        autoOrbitLayout.addView(autoOrbitSwitch)
        viewerSection.addView(autoOrbitLayout)
        layout.addView(viewerSection)

        // Developer Settings section (matches iOS)
        val developerSection = createSection(getString(R.string.settings_developer))

        // Debug mode toggle
        val debugLayout = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, 8, 0, 8)
        }

        val debugLabel = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        val debugIcon = TextView(this).apply {
            text = "\uD83D\uDC1E" // Bug/ladybug emoji
            textSize = 20f
            setPadding(0, 0, 12, 0)
        }

        val debugTextContainer = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
        }
        debugTextContainer.addView(debugIcon)

        val debugTextLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, ViewGroup.LayoutParams.WRAP_CONTENT, 1f)
        }

        val debugTitle = TextView(this).apply {
            text = getString(R.string.settings_debug_mode)
            textSize = 16f
            setTextColor(Color.parseColor("#333333"))
        }
        val debugDesc = TextView(this).apply {
            text = getString(R.string.settings_debug_mode_description)
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        debugTextLayout.addView(debugTitle)
        debugTextLayout.addView(debugDesc)
        debugTextContainer.addView(debugTextLayout)
        debugLabel.addView(debugTextContainer)

        // Initialize DebugLogger
        DebugLogger.init(this)

        val debugSwitch = createStyledSwitch(DebugLogger.isDebugMode) { isChecked ->
            DebugLogger.setDebugMode(isChecked)
        }

        debugLayout.addView(debugLabel)
        debugLayout.addView(debugSwitch)
        developerSection.addView(debugLayout)

        // Inference Backend selection (3-way radio group)
        val backendTitle = TextView(this).apply {
            text = "Inference Backend"
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 16, 0, 8)
        }
        developerSection.addView(backendTitle)

        // Migrate old boolean pref to new string pref
        val currentBackend = migrateBackendPref()

        val backendRadioGroup = RadioGroup(this).apply {
            orientation = RadioGroup.VERTICAL
        }

        val onnxRadioId = View.generateViewId()
        val onnxFp16RadioId = View.generateViewId()
        val onnxInt8RadioId = View.generateViewId()
        val ncnnRadioId = View.generateViewId()
        val executorchRadioId = View.generateViewId()
        val executorchFp16RadioId = View.generateViewId()
        val executorchInt8RadioId = View.generateViewId()
        val litertRadioId = View.generateViewId()

        val onnxRadio = RadioButton(this).apply {
            id = onnxRadioId
            text = "ONNX (default)"
            setTextColor(Color.parseColor("#333333"))
        }
        val onnxRadioDesc = TextView(this).apply {
            text = "Standard inference with ONNX Runtime (FP32)"
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
        }

        val onnxFp16Radio = RadioButton(this).apply {
            id = onnxFp16RadioId
            text = if (BackendConfig.ENABLE_ONNX_FP16) "ONNX FP16" else "ONNX FP16 (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_ONNX_FP16
            alpha = if (BackendConfig.ENABLE_ONNX_FP16) 1.0f else 0.5f
        }
        val onnxFp16RadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_ONNX_FP16) {
                "FP16 split (50% smaller, faster on ARM)"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_ONNX_FP16) 1.0f else 0.5f
        }

        val onnxInt8Radio = RadioButton(this).apply {
            id = onnxInt8RadioId
            text = if (BackendConfig.ENABLE_ONNX_INT8) "ONNX INT8" else "ONNX INT8 (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_ONNX_INT8
            alpha = if (BackendConfig.ENABLE_ONNX_INT8) 1.0f else 0.5f
        }
        val onnxInt8RadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_ONNX_INT8) {
                "Single INT8 model (~700 MB, experimental quality)"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_ONNX_INT8) 1.0f else 0.5f
        }

        val ncnnRadio = RadioButton(this).apply {
            id = ncnnRadioId
            text = if (BackendConfig.ENABLE_NCNN) "NCNN" else "NCNN (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_NCNN
            alpha = if (BackendConfig.ENABLE_NCNN) 1.0f else 0.5f
        }
        val ncnnRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_NCNN) {
                "Faster room generation (requires NCNN model files)"
            } else {
                "Disabled in this build (wrappers kept, ONNX only)"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_NCNN) 1.0f else 0.5f
        }

        val executorchRadio = RadioButton(this).apply {
            id = executorchRadioId
            text = if (BackendConfig.ENABLE_EXECUTORCH) "ExecuTorch" else "ExecuTorch (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_EXECUTORCH
            alpha = if (BackendConfig.ENABLE_EXECUTORCH) 1.0f else 0.5f
        }
        val executorchRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_EXECUTORCH) {
                "PyTorch on-device inference (ExecuTorch)"
            } else {
                "Disabled in this build (wrappers kept, ONNX only)"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_EXECUTORCH) 1.0f else 0.5f
        }

        val executorchFp16Radio = RadioButton(this).apply {
            id = executorchFp16RadioId
            text = if (BackendConfig.ENABLE_EXECUTORCH_FP16) "ExecuTorch FP16" else "ExecuTorch FP16 (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_EXECUTORCH_FP16
            alpha = if (BackendConfig.ENABLE_EXECUTORCH_FP16) 1.0f else 0.5f
        }
        val executorchFp16RadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_EXECUTORCH_FP16) {
                "FP16 split (50% smaller, XNNPACK)"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_EXECUTORCH_FP16) 1.0f else 0.5f
        }

        val executorchInt8Radio = RadioButton(this).apply {
            id = executorchInt8RadioId
            text = if (BackendConfig.ENABLE_EXECUTORCH_INT8) "ExecuTorch INT8" else "ExecuTorch INT8 (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_EXECUTORCH_INT8
            alpha = if (BackendConfig.ENABLE_EXECUTORCH_INT8) 1.0f else 0.5f
        }
        val executorchInt8RadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_EXECUTORCH_INT8) {
                "INT8 quantized, single model (~600MB, XNNPACK)"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_EXECUTORCH_INT8) 1.0f else 0.5f
        }

        val litertRadio = RadioButton(this).apply {
            id = litertRadioId
            text = if (BackendConfig.ENABLE_LITERT) "LiteRT" else "LiteRT (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_LITERT
            alpha = if (BackendConfig.ENABLE_LITERT) 1.0f else 0.5f
        }
        val litertRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_LITERT) {
                "TFLite FP16 + GPU delegate (best quality, GPU-accelerated)"
            } else {
                "Disabled in this build (wrappers kept, ONNX only)"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_LITERT) 1.0f else 0.5f
        }

        backendRadioGroup.addView(onnxRadio)
        backendRadioGroup.addView(onnxRadioDesc)
        backendRadioGroup.addView(onnxFp16Radio)
        backendRadioGroup.addView(onnxFp16RadioDesc)
        backendRadioGroup.addView(onnxInt8Radio)
        backendRadioGroup.addView(onnxInt8RadioDesc)
        backendRadioGroup.addView(ncnnRadio)
        backendRadioGroup.addView(ncnnRadioDesc)
        backendRadioGroup.addView(executorchRadio)
        backendRadioGroup.addView(executorchRadioDesc)
        backendRadioGroup.addView(executorchFp16Radio)
        backendRadioGroup.addView(executorchFp16RadioDesc)
        backendRadioGroup.addView(executorchInt8Radio)
        backendRadioGroup.addView(executorchInt8RadioDesc)
        backendRadioGroup.addView(litertRadio)
        backendRadioGroup.addView(litertRadioDesc)

        val pythonRadioId = View.generateViewId()
        val pythonRadio = RadioButton(this).apply {
            id = pythonRadioId
            text = if (BackendConfig.ENABLE_PYTHON) "Python (PyTorch)" else "Python (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_PYTHON
            alpha = if (BackendConfig.ENABLE_PYTHON) 1.0f else 0.5f
        }
        val pythonRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_PYTHON) {
                "Native PyTorch via Chaquopy — same code as Mac/PC, no conversion"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_PYTHON) 1.0f else 0.5f
        }
        backendRadioGroup.addView(pythonRadio)
        backendRadioGroup.addView(pythonRadioDesc)

        val torchMobileRadioId = View.generateViewId()
        val torchMobileRadio = RadioButton(this).apply {
            id = torchMobileRadioId
            text = if (BackendConfig.ENABLE_TORCH_MOBILE) "PyTorch Mobile" else "PyTorch Mobile (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_TORCH_MOBILE
            alpha = if (BackendConfig.ENABLE_TORCH_MOBILE) 1.0f else 0.5f
        }
        val torchMobileRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_TORCH_MOBILE) {
                "Direct .ptl model — same weights as Python, no conversion"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_TORCH_MOBILE) 1.0f else 0.5f
        }
        backendRadioGroup.addView(torchMobileRadio)
        backendRadioGroup.addView(torchMobileRadioDesc)

        val nativePtRadioId = View.generateViewId()
        val nativePtRadio = RadioButton(this).apply {
            id = nativePtRadioId
            text = if (BackendConfig.ENABLE_NATIVE_PT) "Native .pt" else "Native .pt (disabled)"
            setTextColor(Color.parseColor("#333333"))
            isEnabled = BackendConfig.ENABLE_NATIVE_PT
            alpha = if (BackendConfig.ENABLE_NATIVE_PT) 1.0f else 0.5f
        }
        val nativePtRadioDesc = TextView(this).apply {
            text = if (BackendConfig.ENABLE_NATIVE_PT) {
                "TorchScript + LibTorch native — FP32, internal storage, no fallback"
            } else {
                "Disabled in this build"
            }
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(48, 0, 0, 8)
            alpha = if (BackendConfig.ENABLE_NATIVE_PT) 1.0f else 0.5f
        }
        backendRadioGroup.addView(nativePtRadio)
        backendRadioGroup.addView(nativePtRadioDesc)

        when (currentBackend) {
            "onnx" -> backendRadioGroup.check(onnxRadioId)
            "onnx_fp16" -> backendRadioGroup.check(onnxFp16RadioId)
            "onnx_int8" -> backendRadioGroup.check(onnxInt8RadioId)
            "ncnn" -> backendRadioGroup.check(ncnnRadioId)
            "executorch" -> backendRadioGroup.check(executorchRadioId)
            "executorch_fp16" -> backendRadioGroup.check(executorchFp16RadioId)
            "executorch_int8" -> backendRadioGroup.check(executorchInt8RadioId)
            "litert" -> backendRadioGroup.check(litertRadioId)
            "python" -> backendRadioGroup.check(pythonRadioId)
            "torch_mobile" -> backendRadioGroup.check(torchMobileRadioId)
            "native_pt" -> backendRadioGroup.check(nativePtRadioId)
            else -> backendRadioGroup.check(executorchInt8RadioId)
        }

        backendRadioGroup.setOnCheckedChangeListener { _, checkedId ->
            val backend = when (checkedId) {
                onnxRadioId -> "onnx"
                onnxFp16RadioId -> "onnx_fp16"
                onnxInt8RadioId -> "onnx_int8"
                ncnnRadioId -> "ncnn"
                executorchRadioId -> "executorch"
                executorchFp16RadioId -> "executorch_fp16"
                executorchInt8RadioId -> "executorch_int8"
                litertRadioId -> "litert"
                pythonRadioId -> "python"
                torchMobileRadioId -> "torch_mobile"
                nativePtRadioId -> "native_pt"
                else -> "executorch_int8"
            }
            prefs.edit().putString("inference_backend", backend).apply()
        }

        developerSection.addView(backendRadioGroup)

        // Developer section footer
        val developerFooter = TextView(this).apply {
            text = getString(R.string.settings_developer_footer)
            textSize = 11f
            setTextColor(Color.parseColor("#999999"))
            setPadding(0, 8, 0, 0)
        }
        developerSection.addView(developerFooter)

        layout.addView(developerSection)

        // Legal section
        val legalSection = createSection(getString(R.string.settings_legal))

        val privacyButton = TextView(this).apply {
            text = getString(R.string.settings_privacy_policy)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                openUrl("https://paafekt.com/privacy")
            }
        }
        legalSection.addView(privacyButton)

        val termsButton = TextView(this).apply {
            text = getString(R.string.settings_terms_of_service)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                openUrl("https://paafekt.com/terms")
            }
        }
        legalSection.addView(termsButton)

        val licenseButton = TextView(this).apply {
            text = getString(R.string.settings_licenses)
            textSize = 16f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 8, 0, 8)
            setOnClickListener {
                startActivity(Intent(this@SettingsActivity, LicensesActivity::class.java))
            }
        }
        legalSection.addView(licenseButton)

        layout.addView(legalSection)

        // App info section
        val appSection = createSection(getString(R.string.profile_about))
        val versionText = TextView(this).apply {
            text = "Furnit v1.0.0"
            textSize = 14f
            setTextColor(Color.parseColor("#666666"))
        }
        appSection.addView(versionText)
        layout.addView(appSection)

        // Logout button
        val logoutBtn = Button(this).apply {
            text = getString(R.string.settings_sign_out)
            textSize = 16f
            setTextColor(Color.WHITE)
            setBackgroundColor(Color.parseColor("#F44336"))
            setPadding(24, 16, 24, 16)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 32, 0, 0) }

            setOnClickListener {
                showLogoutConfirmation()
            }
        }
        layout.addView(logoutBtn)

        scrollView.addView(layout)
        setContentView(scrollView)
    }

    private fun createSection(title: String): LinearLayout {
        return LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(Color.WHITE)
            setPadding(24, 16, 24, 16)
            layoutParams = LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.WRAP_CONTENT
            ).apply { setMargins(0, 0, 0, 16) }

            val titleView = TextView(this@SettingsActivity).apply {
                text = title
                textSize = 12f
                setTypeface(null, Typeface.BOLD)
                setTextColor(Color.parseColor("#999999"))
                setPadding(0, 0, 0, 8)
            }
            addView(titleView)
        }
    }

    private fun showLogoutConfirmation() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.settings_sign_out))
            .setMessage(getString(R.string.settings_sign_out_confirm))
            .setPositiveButton(getString(R.string.settings_sign_out)) { _, _ ->
                authManager.logout()
                navigateToLogin()
            }
            .setNegativeButton(getString(R.string.common_cancel), null)
            .show()
    }

    private fun navigateToLogin() {
        val intent = Intent(this, LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        finish()
    }

    private fun openUrl(url: String) {
        try {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            startActivity(intent)
        } catch (e: Exception) {
            // Ignore if no browser available
        }
    }

    private fun migrateBackendPref(): String {
        // If new pref exists, use it
        val existingBackend = prefs.getString("inference_backend", null)
        if (existingBackend != null) {
            val normalized = BackendConfig.normalize(existingBackend)
            if (normalized != existingBackend) {
                prefs.edit().putString("inference_backend", normalized).apply()
            }
            return normalized
        }

        // Migrate old boolean pref (default new installs to ExecuTorch INT8 for easy testing)
        val useNcnn = prefs.getBoolean("use_ncnn_backend", false)
        val backend = BackendConfig.normalize(if (useNcnn) "ncnn" else "executorch_int8")
        prefs.edit()
            .putString("inference_backend", backend)
            .remove("use_ncnn_backend")
            .apply()
        return backend
    }

    private fun createStyledSwitch(checked: Boolean, onChanged: (Boolean) -> Unit): SwitchCompat {
        return SwitchCompat(this).apply {
            isChecked = checked
            // Track colors: gray when off, green when on
            trackTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked)
                ),
                intArrayOf(
                    Color.parseColor("#CCCCCC"),  // Off - gray border
                    Color.parseColor("#81C784")   // On - light green
                )
            )
            // Thumb colors: white when off, green when on
            thumbTintList = ColorStateList(
                arrayOf(
                    intArrayOf(-android.R.attr.state_checked),
                    intArrayOf(android.R.attr.state_checked)
                ),
                intArrayOf(
                    Color.parseColor("#FFFFFF"),  // Off - white
                    Color.parseColor("#4CAF50")   // On - green
                )
            )
            setOnCheckedChangeListener { _, isChecked -> onChanged(isChecked) }
        }
    }
}
