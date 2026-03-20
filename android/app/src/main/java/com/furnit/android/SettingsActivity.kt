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
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.furnit.android.auth.AuthenticationManager
import com.furnit.android.auth.LoginActivity
import com.furnit.android.models.QualitySettings
import com.furnit.android.services.BackendConfig
import com.furnit.android.utils.DebugLogger
import com.furnit.android.utils.Part1OnlyTest

class SettingsActivity : AppCompatActivity() {
    private lateinit var prefs: SharedPreferences
    private lateinit var authManager: AuthenticationManager
    private lateinit var part1WarmupStatusView: TextView

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
            text = getString(R.string.settings_inference_backend)
            textSize = 16f
            setTypeface(null, Typeface.BOLD)
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 16, 0, 8)
        }
        developerSection.addView(backendTitle)

        migrateBackendPref()
        // Choice: ExecuTorch INT8 (Vulkan) vs CPU ExecuTorch INT8 — both C++, single Part4b
        val executorchChoiceLayout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 8, 0, 16) }
        val executorchChoiceRg = RadioGroup(this).apply { orientation = RadioGroup.VERTICAL; setPadding(0, 8, 0, 0) }
        val vulkanId = View.generateViewId()
        val cpuId = View.generateViewId()
        val useCpuExecutorch = prefs.getBoolean("executorch_int8_use_cpu_stable", false)
        val vulkanRb = RadioButton(this).apply {
            id = vulkanId
            text = "ExecuTorch INT8 (Vulkan)"
            setTextColor(Color.parseColor("#333333"))
        }
        val cpuRb = RadioButton(this).apply {
            id = cpuId
            text = "CPU ExecuTorch INT8"
            setTextColor(Color.parseColor("#333333"))
        }
        executorchChoiceRg.addView(vulkanRb)
        executorchChoiceRg.addView(cpuRb)
        if (useCpuExecutorch) executorchChoiceRg.check(cpuId) else executorchChoiceRg.check(vulkanId)
        executorchChoiceRg.setOnCheckedChangeListener { _, checkedId ->
            prefs.edit().putBoolean("executorch_int8_use_cpu_stable", checkedId == cpuId).apply()
        }
        executorchChoiceLayout.addView(executorchChoiceRg)
        val executorchChoiceDesc = TextView(this).apply {
            text = "Vulkan = GPU; CPU = portable Part1/2/3/4. Both C++ only."
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, 8, 0, 0)
        }
        executorchChoiceLayout.addView(executorchChoiceDesc)
        // Part4b: when ON, prefer single decoder if sharp_split_part4b_int8/fp16/.pte is on device (with v2, also push tile_b4); avoids INT8 tiled "foggy squares". OFF = tiled tile_b4 / tile_00 first.
        val stablePart4b = prefs.getBoolean("executorch_int8_stable_part4b", true)
        val stablePart4bLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val stablePart4bLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        stablePart4bLabel.addView(TextView(this).apply { text = getString(R.string.settings_stable_part4b); textSize = 14f; setTextColor(Color.parseColor("#222222")) })
        stablePart4bLabel.addView(TextView(this).apply { text = getString(R.string.settings_stable_part4b_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) })
        val stablePart4bSwitch = createStyledSwitch(stablePart4b) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_stable_part4b", isChecked).apply()
        }
        stablePart4bLayout.addView(stablePart4bLabel)
        stablePart4bLayout.addView(stablePart4bSwitch)
        executorchChoiceLayout.addView(stablePart4bLayout)
        // Part4b single on CPU: use FP32 single decoder for clean output (slower).
        val part4bSingleOnCpu = prefs.getBoolean("executorch_int8_part4b_single_on_cpu", true)
        val part4bSingleOnCpuLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val part4bSingleOnCpuLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        part4bSingleOnCpuLabel.addView(TextView(this).apply { text = getString(R.string.settings_part4b_single_on_cpu); textSize = 14f; setTextColor(Color.parseColor("#222222")) })
        part4bSingleOnCpuLabel.addView(TextView(this).apply { text = getString(R.string.settings_part4b_single_on_cpu_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) })
        val part4bSingleOnCpuSwitch = createStyledSwitch(part4bSingleOnCpu) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_part4b_single_on_cpu", isChecked).apply()
        }
        part4bSingleOnCpuLayout.addView(part4bSingleOnCpuLabel)
        part4bSingleOnCpuLayout.addView(part4bSingleOnCpuSwitch)
        executorchChoiceLayout.addView(part4bSingleOnCpuLayout)
        // Part1+2: single-patch only avoids batch-4 SIGSEGV on some devices.
        val part12SingleOnly = prefs.getBoolean("executorch_int8_part12_single_only", true)
        val part12SingleLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val part12SingleLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        part12SingleLabel.addView(TextView(this).apply { text = getString(R.string.settings_part12_single_only); textSize = 14f; setTextColor(Color.parseColor("#222222")) })
        part12SingleLabel.addView(TextView(this).apply { text = getString(R.string.settings_part12_single_only_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) })
        val part12SingleSwitch = createStyledSwitch(part12SingleOnly) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_part12_single_only", isChecked).apply()
        }
        part12SingleLayout.addView(part12SingleLabel)
        part12SingleLayout.addView(part12SingleSwitch)
        executorchChoiceLayout.addView(part12SingleLayout)
        // Part1+2: 25 patches only skips 0.5x+0.25x to avoid SIGSEGV at patch 25/35.
        val part12_25Only = prefs.getBoolean("executorch_int8_part12_25_only", true)
        val part12_25Layout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val part12_25Label = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        part12_25Label.addView(TextView(this).apply { text = getString(R.string.settings_part12_25_only); textSize = 14f; setTextColor(Color.parseColor("#222222")) })
        part12_25Label.addView(TextView(this).apply { text = getString(R.string.settings_part12_25_only_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) })
        val part12_25Switch = createStyledSwitch(part12_25Only) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_part12_25_only", isChecked).apply()
        }
        part12_25Layout.addView(part12_25Label)
        part12_25Layout.addView(part12_25Switch)
        executorchChoiceLayout.addView(part12_25Layout)
        // Tiled Part4b NDC: try ON if output shows misaligned 4×4 patches (transposed export).
        val swapTileNdc = prefs.getBoolean("executorch_int8_swap_tile_ndc_xy", false)
        val swapTileNdcLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 0, 0, 12) }
        val swapTileNdcLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        swapTileNdcLabel.addView(TextView(this).apply { text = getString(R.string.settings_swap_tile_ndc_xy); textSize = 14f; setTextColor(Color.parseColor("#222222")) })
        swapTileNdcLabel.addView(TextView(this).apply { text = getString(R.string.settings_swap_tile_ndc_xy_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) })
        val swapTileNdcSwitch = createStyledSwitch(swapTileNdc) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_swap_tile_ndc_xy", isChecked).apply()
        }
        swapTileNdcLayout.addView(swapTileNdcLabel)
        swapTileNdcLayout.addView(swapTileNdcSwitch)
        executorchChoiceLayout.addView(swapTileNdcLayout)
        developerSection.addView(executorchChoiceLayout)

        // Max Gaussians (splat count limit)
        val maxGaussLayout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 12, 0, 12) }
        val maxGaussTitle = TextView(this).apply { text = getString(R.string.settings_max_gaussians); textSize = 14f; setTextColor(Color.parseColor("#222222")) }
        val maxGaussDesc = TextView(this).apply { text = getString(R.string.settings_max_gaussians_description); textSize = 12f; setTextColor(Color.parseColor("#666666")) }
        maxGaussLayout.addView(maxGaussTitle)
        maxGaussLayout.addView(maxGaussDesc)

        val maxGaussRg = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL; setPadding(0, 8, 0, 0) }
        val mgUnlimitedId = View.generateViewId()
        val mg300kId = View.generateViewId()
        val mg500kId = View.generateViewId()
        val mgUnlimited = RadioButton(this).apply { id = mgUnlimitedId; text = "All"; setTextColor(Color.parseColor("#333333")) }
        val mg300k = RadioButton(this).apply { id = mg300kId; text = "300k"; setTextColor(Color.parseColor("#333333")) }
        val mg500k = RadioButton(this).apply { id = mg500kId; text = "500k"; setTextColor(Color.parseColor("#333333")) }
        maxGaussRg.addView(mgUnlimited)
        maxGaussRg.addView(mg300k)
        maxGaussRg.addView(mg500k)

        val currentMaxG = prefs.getInt("executorch_int8_max_gaussians", 500000)
        when (currentMaxG) {
            300000 -> maxGaussRg.check(mg300kId)
            500000 -> maxGaussRg.check(mg500kId)
            0 -> maxGaussRg.check(mgUnlimitedId)
            else -> maxGaussRg.check(mg500kId)
        }
        maxGaussRg.setOnCheckedChangeListener { _, checkedId ->
            val v = when (checkedId) {
                mg300kId -> 300000
                mg500kId -> 500000
                else -> 0
            }
            prefs.edit().putInt("executorch_int8_max_gaussians", v).apply()
        }
        maxGaussLayout.addView(maxGaussRg)
        developerSection.addView(maxGaussLayout)

        val implLayout = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 12, 0, 12) }
        // Use 1280×1280 as intermediate then upscale to 1536 (may help Vulkan / reduce memory)
        val use1280 = prefs.getBoolean("executorch_int8_use_1280", false)
        val use1280Layout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val use1280Label = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        val use1280Title = TextView(this).apply { text = "Use 1280×1280"; textSize = 14f; setTextColor(Color.parseColor("#222222")) }
        val use1280Desc = TextView(this).apply { text = "Process at 1280 then upscale to 1536 (may help Vulkan on some devices)"; textSize = 12f; setTextColor(Color.parseColor("#666666")) }
        use1280Label.addView(use1280Title)
        use1280Label.addView(use1280Desc)
        val use1280Switch = createStyledSwitch(use1280) { isChecked ->
            prefs.edit().putBoolean("executorch_int8_use_1280", isChecked).apply()
        }
        use1280Layout.addView(use1280Label)
        use1280Layout.addView(use1280Switch)
        implLayout.addView(use1280Layout)

        // Vulkan 1536x1536 support test (logs to logcat)
        val vulkanTestLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val vulkanTestLabel = TextView(this).apply {
            text = "Run Vulkan 1536 test"
            textSize = 14f
            setTextColor(Color.parseColor("#222222"))
            setPadding(0, 0, 0, 0)
        }
        val vulkanTestBtn = TextView(this).apply {
            text = "Run"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(24, 0, 0, 0)
            setOnClickListener {
                com.furnit.android.utils.Vulkan1536Test.runAndLog()
                android.widget.Toast.makeText(this@SettingsActivity, "Check logcat tag Vulkan1536Test", android.widget.Toast.LENGTH_SHORT).show()
            }
        }
        vulkanTestLayout.addView(vulkanTestLabel)
        vulkanTestLayout.addView(vulkanTestBtn)
        implLayout.addView(vulkanTestLayout)

        // Vulkan & ExecuTorch diagnostics: device, extensions, sync (synchronization2), shader note
        val vulkanDiagLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val vulkanDiagLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        val vulkanDiagTitle = TextView(this).apply { text = "Vulkan & ExecuTorch diagnostics"; textSize = 14f; setTextColor(Color.parseColor("#222222")) }
        val vulkanDiagDesc = TextView(this).apply { text = "Log device, extensions, sync (VK_KHR_synchronization2), shader note. See logcat tag VulkanDiag."; textSize = 12f; setTextColor(Color.parseColor("#666666")) }
        vulkanDiagLabel.addView(vulkanDiagTitle)
        vulkanDiagLabel.addView(vulkanDiagDesc)
        val vulkanDiagBtn = TextView(this).apply {
            text = "Run"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(24, 0, 0, 0)
            setOnClickListener {
                val adbCmd = com.furnit.android.utils.Vulkan1536Test.runDiagnosticsAndLog()
                android.widget.Toast.makeText(this@SettingsActivity, "Logged. To see: $adbCmd", android.widget.Toast.LENGTH_LONG).show()
            }
        }
        vulkanDiagLayout.addView(vulkanDiagLabel)
        vulkanDiagLayout.addView(vulkanDiagBtn)
        implLayout.addView(vulkanDiagLayout)

        // Part1 warmup (manual): run one forward first; status line updates when done (prefs + Part1Warmup logcat)
        val part1WarmupVertical = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, 12, 0, 0) }
        val part1WarmupRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        val part1WarmupLabelCol = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
        }
        part1WarmupLabelCol.addView(TextView(this).apply {
            text = "Part1 warmup"
            textSize = 14f
            setTextColor(Color.parseColor("#222222"))
        })
        part1WarmupLabelCol.addView(TextView(this).apply {
            text = "Load Part1 once + 2× forward (Vulkan cache). Keep app open (same PID). Then Run reuses Module."
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        })
        val part1WarmupBtn = TextView(this).apply {
            text = "Warmup"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(24, 0, 0, 0)
            setOnClickListener {
                part1WarmupStatusView.text = "Warmup status: starting…"
                lifecycleScope.launch(Dispatchers.Default) {
                    val result = Part1OnlyTest.runWarmupFromSettings(this@SettingsActivity)
                    withContext(Dispatchers.Main) {
                        part1WarmupStatusView.text = Part1OnlyTest.getWarmupStatusSummary(this@SettingsActivity)
                        Toast.makeText(this@SettingsActivity, result.userMessage, Toast.LENGTH_LONG).show()
                    }
                }
            }
        }
        part1WarmupRow.addView(part1WarmupLabelCol)
        part1WarmupRow.addView(part1WarmupBtn)
        part1WarmupVertical.addView(part1WarmupRow)
        part1WarmupStatusView = TextView(this).apply {
            text = Part1OnlyTest.getWarmupStatusSummary(this@SettingsActivity)
            textSize = 12f
            setTextColor(Color.parseColor("#333333"))
            setPadding(0, 8, 0, 0)
        }
        part1WarmupVertical.addView(part1WarmupStatusView)
        part1WarmupVertical.addView(TextView(this).apply {
            text = "Release Part1 cache"
            textSize = 12f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 4, 0, 0)
            setOnClickListener {
                Part1OnlyTest.releaseCachedPart1Module()
                part1WarmupStatusView.text = Part1OnlyTest.getWarmupStatusSummary(this@SettingsActivity)
                Toast.makeText(this@SettingsActivity, "Part1 Module released from memory", Toast.LENGTH_SHORT).show()
            }
        })
        implLayout.addView(part1WarmupVertical)

        // Part1 only test: load part1_test_patch_f32.bin, run sharp_split_part1.pte, log outputs (tag Part1Test)
        val part1OnlyLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; setPadding(0, 12, 0, 12) }
        val part1OnlyLabel = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f) }
        val part1OnlyTitle = TextView(this).apply { text = "Part1 only test"; textSize = 14f; setTextColor(Color.parseColor("#222222")) }
        val part1OnlyDesc = TextView(this).apply {
            text = "Run: one forward + golden stats. Benchmark 3×: same Module, same patch, logs P1_BENCH durations (Vulkan perf). " +
                "Portable .pte = fast CPU baseline. adb: adb logcat -d | grep P1_BENCH"
            textSize = 12f
            setTextColor(Color.parseColor("#666666"))
        }
        part1OnlyLabel.addView(part1OnlyTitle)
        part1OnlyLabel.addView(part1OnlyDesc)
        val part1OnlyButtons = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(24, 0, 0, 0)
        }
        part1OnlyButtons.addView(TextView(this).apply {
            text = "Run"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setPadding(0, 0, 0, 8)
            setOnClickListener {
                lifecycleScope.launch(Dispatchers.Default) {
                    val msg = Part1OnlyTest.run(this@SettingsActivity)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@SettingsActivity, msg, Toast.LENGTH_LONG).show()
                    }
                }
            }
        })
        part1OnlyButtons.addView(TextView(this).apply {
            text = "Benchmark 3×"
            textSize = 14f
            setTextColor(Color.parseColor("#007AFF"))
            setOnClickListener {
                lifecycleScope.launch(Dispatchers.Default) {
                    val msg = Part1OnlyTest.runTripleForwardBenchmark(this@SettingsActivity)
                    withContext(Dispatchers.Main) {
                        Toast.makeText(this@SettingsActivity, msg, Toast.LENGTH_LONG).show()
                    }
                }
            }
        })
        part1OnlyLayout.addView(part1OnlyLabel)
        part1OnlyLayout.addView(part1OnlyButtons)
        implLayout.addView(part1OnlyLayout)

        developerSection.addView(implLayout)

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
            text = getString(R.string.app_version_display)
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

    override fun onResume() {
        super.onResume()
        if (::part1WarmupStatusView.isInitialized) {
            part1WarmupStatusView.text = Part1OnlyTest.getWarmupStatusSummary(this)
        }
    }
}
