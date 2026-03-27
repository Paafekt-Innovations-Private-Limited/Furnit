package com.furnit.android.utils

import android.content.Context
import android.util.Log
import com.furnit.android.BuildConfig
import com.furnit.android.services.ExecutorchInt8Sharp
import com.furnit.android.services.SharpExecuTorchSplitModelNames
import org.pytorch.executorch.EValue
import org.pytorch.executorch.Module
import org.pytorch.executorch.Tensor
import org.json.JSONObject
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Isolated Part4 tile_00 test using the Vulkan-safe split export.
 *
 * stage_pre + decoder_head + raw_heads stay on Vulkan.
 * init/base + compose stay portable because those paths create rank-5 tensors.
 */
object Part4OnlyTest {
    private const val TAG = "Part4Test"
    const val P4_BENCH_MARKER = "P4_BENCH"
    const val P4_DECODER_COMPARE_MARKER = "P4_DECODER_COMPARE"
    const val P4_DECODER_CHUNK_BENCH_MARKER = "P4_DECODER_CHUNK_BENCH"
    const val P4_DECODER_STACK_BENCH_MARKER = "P4_DECODER_STACK_BENCH"
    const val P4_LATENT0_COMPARE_MARKER = "P4_LATENT0_COMPARE"
    private val part4ActionRunning = AtomicBoolean(false)

    private const val TILE_BATCH = 1
    private const val TILE_IMAGE_SIZE = 384
    private const val FEATURE_DIM = 1024
    private const val TILE_LATENT_SIZE = 24
    private const val TILE_X1_SIZE = 12
    private const val TILE_X2_SIZE = 6
    private const val PARAMS_PER_GAUSSIAN = 14

    private data class ModelPaths(
        val stagePre: File,
        val decoderOnly: File?,
        val disparityHead: File?,
        val decoderSeed: File?,
        val decoderMergeX1: File?,
        val decoderMergeX0: File?,
        val decoderMergeLatent1: File?,
        val decoderMergeLatent0: File?,
        val decoderMergeLatent0Prefuse: File?,
        val decoderMergeLatent0Postfuse: File?,
        val decoderMergeLatent0PrefusePortable: File?,
        val decoderMergeLatent0PostfusePortable: File?,
        val decoderHead: File,
        val decoderHeadPortable: File?,
        val initBase: File,
        val rawHeads: File,
        val compose: File,
    )

    private data class ArtifactDiagnostics(
        val artifact: String,
        val modelVariant: String?,
        val surgeryGroups: Int?,
        val modifiedLayerCount: Int?,
        val estimatedHotpathParamReductionPct: Double?,
        val delegateCallCount: Int?,
        val kernelCallCount: Int?,
        val mixedDelegateAndKernelCalls: Boolean?,
        val insertedTransitionCount: Int?,
        val transitionLineCount: Int?,
        val widthPackedHits: Int?,
        val channelsPackedHits: Int?,
        val highLayoutChurnSuspected: Boolean?,
    )

    private data class StaticInputs(
        val image: EValue,
        val latent0: EValue,
        val latent1: EValue,
        val x0Feat: EValue,
        val x1Feat: EValue,
        val x2Feat: EValue,
        val xLowres: EValue,
    )

    private fun modelsDirs(context: Context): List<File> {
        val list = mutableListOf<File>()
        list.add(File(context.filesDir, ExecutorchInt8Sharp.MODELS_SUBDIR_CPU_VULKAN_HYBRID).also { it.mkdirs() })
        context.getExternalFilesDir(ExecutorchInt8Sharp.MODELS_SUBDIR_CPU_VULKAN_HYBRID)?.let { list.add(it) }
        list.add(File(context.filesDir, "models_cpu").also { it.mkdirs() })
        context.getExternalFilesDir("models_cpu")?.let { list.add(it) }
        list.add(File("/data/local/tmp/furnit"))
        return list
    }

    private fun findFile(context: Context, filename: String): File? {
        for (dir in modelsDirs(context)) {
            val file = File(dir, filename)
            if (file.exists() && file.length() > 0L) return file
        }
        return null
    }

    private fun resolveModelPaths(context: Context): Result<ModelPaths> {
        if (!BuildConfig.EXECUTORCH_USE_VULKAN_AAR) {
            return Result.failure(IllegalStateException("Vulkan Part4 test only applies to the etVulkan APK"))
        }
        val stagePre = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_PRE_VULKAN)
            ?: return Result.failure(
                IllegalStateException("Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_STAGE_PRE_VULKAN}"),
            )
        val decoderOnly = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_ONLY)
        val disparityHead = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DISPARITY_HEAD)
        val decoderSeed = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_SEED)
        val decoderMergeX1 = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_X1)
        val decoderMergeX0 = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_X0)
        val decoderMergeLatent1 = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT1)
        val decoderMergeLatent0 = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0)
        val decoderMergeLatent0Prefuse =
            findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE)
        val decoderMergeLatent0Postfuse =
            findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE)
        val decoderMergeLatent0PrefusePortable =
            findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE_PORTABLE)
        val decoderMergeLatent0PostfusePortable =
            findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE_PORTABLE)
        val decoderHead = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_HEAD)
            ?: return Result.failure(
                IllegalStateException("Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_HEAD}"),
            )
        val decoderHeadPortable = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_HEAD_PORTABLE)
        val initBase = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_INIT_BASE)
            ?: return Result.failure(
                IllegalStateException("Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_INIT_BASE}"),
            )
        val rawHeads = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_RAW_HEADS_VULKAN)
            ?: return Result.failure(
                IllegalStateException("Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_RAW_HEADS_VULKAN}"),
            )
        val compose = findFile(context, SharpExecuTorchSplitModelNames.PART4B_TILE_00_COMPOSE)
            ?: return Result.failure(
                IllegalStateException("Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_COMPOSE}"),
            )
        return Result.success(
            ModelPaths(
                stagePre = stagePre,
                decoderOnly = decoderOnly,
                disparityHead = disparityHead,
                decoderSeed = decoderSeed,
                decoderMergeX1 = decoderMergeX1,
                decoderMergeX0 = decoderMergeX0,
                decoderMergeLatent1 = decoderMergeLatent1,
                decoderMergeLatent0 = decoderMergeLatent0,
                decoderMergeLatent0Prefuse = decoderMergeLatent0Prefuse,
                decoderMergeLatent0Postfuse = decoderMergeLatent0Postfuse,
                decoderMergeLatent0PrefusePortable = decoderMergeLatent0PrefusePortable,
                decoderMergeLatent0PostfusePortable = decoderMergeLatent0PostfusePortable,
                decoderHead = decoderHead,
                decoderHeadPortable = decoderHeadPortable,
                initBase = initBase,
                rawHeads = rawHeads,
                compose = compose,
            )
        )
    }

    private fun classifyFailure(t: Throwable): String {
        val msg = buildString {
            append(t.message ?: "")
            append(' ')
            append(t.cause?.message ?: "")
        }.lowercase()
        return when {
            "could not find shaderinfo" in msg || "shaderregistry" in msg || "get_shader_info" in msg -> "missing_vulkan_shader"
            "resource not found" in msg || "0x20" in msg -> "resource_not_found"
            "vk_error" in msg || "vulkan" in msg || "backend failed" in msg || "delegate init failed" in msg -> "vulkan_runtime_failure"
            t is OutOfMemoryError -> "oom"
            else -> "unknown"
        }
    }

    private fun createStaticInputs(): StaticInputs {
        val image = EValue.from(
            Tensor.fromBlob(
                FloatArray(TILE_BATCH * 3 * TILE_IMAGE_SIZE * TILE_IMAGE_SIZE),
                longArrayOf(TILE_BATCH.toLong(), 3, TILE_IMAGE_SIZE.toLong(), TILE_IMAGE_SIZE.toLong())
            )
        )
        val latentShape = longArrayOf(TILE_BATCH.toLong(), FEATURE_DIM.toLong(), TILE_LATENT_SIZE.toLong(), TILE_LATENT_SIZE.toLong())
        val x1Shape = longArrayOf(TILE_BATCH.toLong(), FEATURE_DIM.toLong(), TILE_X1_SIZE.toLong(), TILE_X1_SIZE.toLong())
        val x2Shape = longArrayOf(TILE_BATCH.toLong(), FEATURE_DIM.toLong(), TILE_X2_SIZE.toLong(), TILE_X2_SIZE.toLong())
        val latent0 = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_LATENT_SIZE * TILE_LATENT_SIZE), latentShape))
        val latent1 = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_LATENT_SIZE * TILE_LATENT_SIZE), latentShape))
        val x0Feat = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_LATENT_SIZE * TILE_LATENT_SIZE), latentShape))
        val x1Feat = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_X1_SIZE * TILE_X1_SIZE), x1Shape))
        val x2Feat = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_X2_SIZE * TILE_X2_SIZE), x2Shape))
        val xLowres = EValue.from(Tensor.fromBlob(FloatArray(TILE_BATCH * FEATURE_DIM * TILE_X2_SIZE * TILE_X2_SIZE), x2Shape))
        return StaticInputs(image, latent0, latent1, x0Feat, x1Feat, x2Feat, xLowres)
    }

    private fun shapeString(tensor: Tensor): String =
        tensor.shape().joinToString(prefix = "[", postfix = "]")

    private fun logTensorShape(label: String, value: EValue) {
        val tensor = value.toTensor()
        Log.i(TAG, "Part4 tensor: $label shape=${shapeString(tensor)} rank=${tensor.shape().size}")
    }

    private fun logInputShapes(inputs: StaticInputs) {
        logTensorShape("image", inputs.image)
        logTensorShape("latent0", inputs.latent0)
        logTensorShape("latent1", inputs.latent1)
        logTensorShape("x0Feat", inputs.x0Feat)
        logTensorShape("x1Feat", inputs.x1Feat)
        logTensorShape("x2Feat", inputs.x2Feat)
        logTensorShape("xLowres", inputs.xLowres)
    }

    private fun cloneAsEValue(label: String, value: EValue): EValue {
        val tensor = value.toTensor()
        val shape = tensor.shape()
        Log.i(TAG, "Part4 clone: $label shape=${shapeString(tensor)} rank=${shape.size}")
        return EValue.from(Tensor.fromBlob(tensor.getDataAsFloatArray(), shape))
    }

    private fun gaussianCountFromPacked(output: Tensor): Int =
        (output.numel() / PARAMS_PER_GAUSSIAN).toInt().coerceAtLeast(0)

    private fun loadArtifactDiagnostics(context: Context, modelFile: File): ArtifactDiagnostics? {
        val manifestFile = findFile(context, modelFile.name + ".manifest.json") ?: return null
        return try {
            val json = JSONObject(manifestFile.readText())
            val runtime = json.optJSONObject("runtime_partitioning")
            val exportLog = json.optJSONObject("export_log")
            val layoutStrings = json.optJSONObject("layout_string_search")
            val surgery = json.optJSONObject("model_surgery")
            ArtifactDiagnostics(
                artifact = modelFile.name,
                modelVariant = json.optString("model_variant").takeIf { it.isNotBlank() },
                surgeryGroups = surgery?.optInt("groups"),
                modifiedLayerCount = surgery?.optInt("modified_layer_count"),
                estimatedHotpathParamReductionPct = surgery?.optDouble("estimated_hotpath_param_reduction_pct"),
                delegateCallCount = runtime?.optInt("delegate_call_count"),
                kernelCallCount = runtime?.optInt("kernel_call_count"),
                mixedDelegateAndKernelCalls = runtime?.optBoolean("mixed_delegate_and_kernel_calls"),
                insertedTransitionCount = exportLog?.optInt("inserted_transition_count"),
                transitionLineCount = exportLog?.optInt("transition_line_count"),
                widthPackedHits = exportLog?.optInt("width_packed_hits") ?: layoutStrings?.optInt("width_packed_hits"),
                channelsPackedHits = exportLog?.optInt("channels_packed_hits") ?: layoutStrings?.optInt("channels_packed_hits"),
                highLayoutChurnSuspected = exportLog?.optBoolean("high_layout_churn_suspected")
                    ?: layoutStrings?.optBoolean("high_layout_churn_suspected"),
            )
        } catch (t: Throwable) {
            Log.w(TAG, "Failed to parse diagnostics manifest for ${modelFile.name}: ${t.message}")
            null
        }
    }

    private fun artifactDiagnosticLine(diagnostics: ArtifactDiagnostics): String {
        val delegateCalls = diagnostics.delegateCallCount?.toString() ?: "?"
        val kernelCalls = diagnostics.kernelCallCount?.toString() ?: "?"
        val mixed = diagnostics.mixedDelegateAndKernelCalls?.toString() ?: "?"
        val insertedTransitions = diagnostics.insertedTransitionCount?.toString() ?: "?"
        val transitionLines = diagnostics.transitionLineCount?.toString() ?: "?"
        val widthPacked = diagnostics.widthPackedHits?.toString() ?: "?"
        val channelsPacked = diagnostics.channelsPackedHits?.toString() ?: "?"
        val layoutChurn = diagnostics.highLayoutChurnSuspected?.toString() ?: "?"
        val variant = diagnostics.modelVariant ?: "?"
        val groups = diagnostics.surgeryGroups?.toString() ?: "?"
        val modifiedLayers = diagnostics.modifiedLayerCount?.toString() ?: "?"
        val hotpathReduction = diagnostics.estimatedHotpathParamReductionPct?.let { "%.2f".format(it) } ?: "?"
        return "${diagnostics.artifact}: variant=$variant groups=$groups modified_layers=$modifiedLayers " +
            "hotpath_param_cut_pct=$hotpathReduction delegate=$delegateCalls kernel=$kernelCalls mixed=$mixed " +
            "inserted_transitions=$insertedTransitions transition_lines=$transitionLines " +
            "width_packed=$widthPacked channels_packed=$channelsPacked layout_churn=$layoutChurn"
    }

    private inline fun runExclusive(actionName: String, block: () -> String): String {
        if (!part4ActionRunning.compareAndSet(false, true)) {
            val msg = "Another Part4 action is already running. Wait for it to finish and try again."
            Log.w(TAG, "$actionName skipped: another Part4 action is already running")
            return msg
        }
        return try {
            Log.i(TAG, "$actionName begin")
            block()
        } finally {
            part4ActionRunning.set(false)
            Log.i(TAG, "$actionName end")
        }
    }

    @JvmStatic
    fun inspectDiagnostics(context: Context): String {
        return runExclusive("inspectDiagnostics") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            val lines = mutableListOf(
                "Part4 tile_00 fine split diagnostics",
                "Graph breaks suspect when delegate>0 and kernel>0.",
                "Layout/storage copy churn suspect when inserted_transitions>0 or packed-layout hits are non-zero.",
            )
            val artifacts = listOf(
                modelPaths.stagePre,
                modelPaths.decoderOnly,
                modelPaths.disparityHead,
                modelPaths.decoderSeed,
                modelPaths.decoderMergeX1,
                modelPaths.decoderMergeX0,
                modelPaths.decoderMergeLatent1,
                modelPaths.decoderMergeLatent0,
                modelPaths.decoderMergeLatent0Prefuse,
                modelPaths.decoderMergeLatent0Postfuse,
                modelPaths.decoderMergeLatent0PrefusePortable,
                modelPaths.decoderMergeLatent0PostfusePortable,
                modelPaths.decoderHead,
                modelPaths.decoderHeadPortable,
                modelPaths.initBase,
                modelPaths.rawHeads,
                modelPaths.compose,
            )
            for (artifact in artifacts) {
                if (artifact == null) continue
                val diagnostics = loadArtifactDiagnostics(context, artifact)
                val line = diagnostics?.let(::artifactDiagnosticLine) ?: "${artifact.name}: manifest missing"
                Log.i(TAG, "PART4_DIAG $line")
                lines.add(line)
            }
            lines.joinToString("\n")
        }
    }

    private data class DecoderHeadPreparedInputs(
        val latent0Up: EValue,
        val latent1Up: EValue,
        val x0Up: EValue,
        val x1Up: EValue,
        val xFused: EValue,
    )

    private fun cloneStagePreOutputsForDecoder(stagePreOutputs: Array<EValue>, label: String): DecoderHeadPreparedInputs {
        return DecoderHeadPreparedInputs(
            latent0Up = cloneAsEValue("${label}_latent0_up", stagePreOutputs[0]),
            latent1Up = cloneAsEValue("${label}_latent1_up", stagePreOutputs[1]),
            x0Up = cloneAsEValue("${label}_x0_up", stagePreOutputs[2]),
            x1Up = cloneAsEValue("${label}_x1_up", stagePreOutputs[3]),
            xFused = cloneAsEValue("${label}_x_fused", stagePreOutputs[4]),
        )
    }

    private fun runDecoderHeadOnly(
        module: Module,
        preparedInputs: DecoderHeadPreparedInputs,
        stageTag: String,
    ): Pair<Long, Array<EValue>> {
        val startedAt = System.currentTimeMillis()
        Log.i(TAG, "Part4 step: forward $stageTag begin")
        val outputs = module.forward(
            preparedInputs.latent0Up,
            preparedInputs.latent1Up,
            preparedInputs.x0Up,
            preparedInputs.x1Up,
            preparedInputs.xFused
        )
        val durationMs = System.currentTimeMillis() - startedAt
        Log.i(TAG, "Part4 step: forward $stageTag end outputs=${outputs.size} duration_ms=$durationMs")
        outputs.forEachIndexed { index, value -> logTensorShape("$stageTag[$index]", value) }
        return Pair(durationMs, outputs)
    }

    private fun runDecoderOnly(
        module: Module,
        preparedInputs: DecoderHeadPreparedInputs,
        stageTag: String,
    ): Pair<Long, EValue> {
        val startedAt = System.currentTimeMillis()
        Log.i(TAG, "Part4 step: forward $stageTag begin")
        val outputs = module.forward(
            preparedInputs.latent0Up,
            preparedInputs.latent1Up,
            preparedInputs.x0Up,
            preparedInputs.x1Up,
            preparedInputs.xFused
        )
        val durationMs = System.currentTimeMillis() - startedAt
        Log.i(TAG, "Part4 step: forward $stageTag end outputs=${outputs.size} duration_ms=$durationMs")
        outputs.forEachIndexed { index, value -> logTensorShape("$stageTag[$index]", value) }
        return Pair(durationMs, outputs[0])
    }

    private fun runDisparityHeadOnly(
        module: Module,
        decoderFeatures: EValue,
        stageTag: String,
    ): Pair<Long, EValue> {
        val startedAt = System.currentTimeMillis()
        Log.i(TAG, "Part4 step: forward $stageTag begin")
        val outputs = module.forward(decoderFeatures)
        val durationMs = System.currentTimeMillis() - startedAt
        Log.i(TAG, "Part4 step: forward $stageTag end outputs=${outputs.size} duration_ms=$durationMs")
        outputs.forEachIndexed { index, value -> logTensorShape("$stageTag[$index]", value) }
        return Pair(durationMs, outputs[0])
    }

    private fun runSingleTensorStage(
        module: Module,
        input: EValue,
        stageTag: String,
    ): Pair<Long, EValue> {
        val startedAt = System.currentTimeMillis()
        Log.i(TAG, "Part4 step: forward $stageTag begin")
        val outputs = module.forward(input)
        val durationMs = System.currentTimeMillis() - startedAt
        Log.i(TAG, "Part4 step: forward $stageTag end outputs=${outputs.size} duration_ms=$durationMs")
        outputs.forEachIndexed { index, value -> logTensorShape("$stageTag[$index]", value) }
        return Pair(durationMs, outputs[0])
    }

    private fun runTwoInputTensorStage(
        module: Module,
        first: EValue,
        second: EValue,
        stageTag: String,
    ): Pair<Long, EValue> {
        val startedAt = System.currentTimeMillis()
        Log.i(TAG, "Part4 step: forward $stageTag begin")
        val outputs = module.forward(first, second)
        val durationMs = System.currentTimeMillis() - startedAt
        Log.i(TAG, "Part4 step: forward $stageTag end outputs=${outputs.size} duration_ms=$durationMs")
        outputs.forEachIndexed { index, value -> logTensorShape("$stageTag[$index]", value) }
        return Pair(durationMs, outputs[0])
    }

    private fun runSplitPass(
        stagePreModule: Module,
        decoderHeadModule: Module,
        initBaseModule: Module,
        rawHeadsModule: Module,
        composeModule: Module,
        inputs: StaticInputs,
    ): Pair<Long, Int> {
        val startedAt = System.currentTimeMillis()
        logInputShapes(inputs)

        Log.i(TAG, "Part4 step: forward stage_pre begin")
        val stagePreOutputs = stagePreModule.forward(
            inputs.image,
            inputs.latent0,
            inputs.latent1,
            inputs.x0Feat,
            inputs.x1Feat,
            inputs.x2Feat,
            inputs.xLowres
        )
        Log.i(TAG, "Part4 step: forward stage_pre end outputs=${stagePreOutputs.size}")
        stagePreOutputs.forEachIndexed { index, value -> logTensorShape("stage_pre[$index]", value) }

        val latent0Up = cloneAsEValue("latent0_up", stagePreOutputs[0])
        val latent1Up = cloneAsEValue("latent1_up", stagePreOutputs[1])
        val x0Up = cloneAsEValue("x0_up", stagePreOutputs[2])
        val x1Up = cloneAsEValue("x1_up", stagePreOutputs[3])
        val xFused = cloneAsEValue("x_fused", stagePreOutputs[4])

        Log.i(TAG, "Part4 step: forward decoder_head begin")
        val decoderHeadOutputs = decoderHeadModule.forward(
            latent0Up,
            latent1Up,
            x0Up,
            x1Up,
            xFused
        )
        Log.i(TAG, "Part4 step: forward decoder_head end outputs=${decoderHeadOutputs.size}")
        decoderHeadOutputs.forEachIndexed { index, value -> logTensorShape("decoder_head[$index]", value) }

        val disparity = cloneAsEValue("disparity", decoderHeadOutputs[0])
        val decoderFeatures = cloneAsEValue("decoder_features", decoderHeadOutputs[1])
        Log.i(TAG, "Part4 step: forward init_base begin")
        val initBaseOutputs = initBaseModule.forward(inputs.image, disparity)
        Log.i(TAG, "Part4 step: forward init_base end outputs=${initBaseOutputs.size}")
        initBaseOutputs.forEachIndexed { index, value -> logTensorShape("init_base[$index]", value) }

        val featureInput = cloneAsEValue("feature_input", initBaseOutputs[0])

        Log.i(TAG, "Part4 step: forward raw_heads begin")
        val rawHeadOutputs = rawHeadsModule.forward(
            featureInput,
            latent0Up,
            latent1Up,
            x0Up,
            x1Up,
            xFused,
            decoderFeatures
        )
        Log.i(TAG, "Part4 step: forward raw_heads end outputs=${rawHeadOutputs.size}")
        rawHeadOutputs.forEachIndexed { index, value -> logTensorShape("raw_heads[$index]", value) }

        val geometryRaw = cloneAsEValue("geometry_raw", rawHeadOutputs[0])
        val textureRaw = cloneAsEValue("texture_raw", rawHeadOutputs[1])
        val meanXNdc = cloneAsEValue("mean_x_ndc", initBaseOutputs[1])
        val meanYNdc = cloneAsEValue("mean_y_ndc", initBaseOutputs[2])
        val meanInverseZNdc = cloneAsEValue("mean_inverse_z_ndc", initBaseOutputs[3])
        val scales = cloneAsEValue("scales", initBaseOutputs[4])
        val quaternions = cloneAsEValue("quaternions", initBaseOutputs[5])
        val colors = cloneAsEValue("colors", initBaseOutputs[6])
        val opacities = cloneAsEValue("opacities", initBaseOutputs[7])
        val globalScale = cloneAsEValue("global_scale", initBaseOutputs[8])

        Log.i(TAG, "Part4 step: forward compose begin")
        val packedTensor = composeModule.forward(
            geometryRaw,
            textureRaw,
            meanXNdc,
            meanYNdc,
            meanInverseZNdc,
            scales,
            quaternions,
            colors,
            opacities,
            globalScale
        )[0].toTensor()
        Log.i(TAG, "Part4 step: forward compose end output_shape=${shapeString(packedTensor)}")

        return Pair(System.currentTimeMillis() - startedAt, gaussianCountFromPacked(packedTensor))
    }

    @JvmStatic
    fun run(context: Context): String {
        return runExclusive("run") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            var stagePreModule: Module? = null
            var decoderHeadModule: Module? = null
            var initBaseModule: Module? = null
            var rawHeadsModule: Module? = null
            var composeModule: Module? = null
            try {
                val inputs = createStaticInputs()
                Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_head=${modelPaths.decoderHead.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT init_base=${modelPaths.initBase.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT raw_heads=${modelPaths.rawHeads.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT compose=${modelPaths.compose.absolutePath}")

                Log.i(TAG, "Part4 step: load stage_pre begin")
                stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 step: load stage_pre end")
                Log.i(TAG, "Part4 step: load decoder_head begin")
                decoderHeadModule = Module.load(modelPaths.decoderHead.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 step: load decoder_head end")
                Log.i(TAG, "Part4 step: load init_base begin")
                initBaseModule = Module.load(modelPaths.initBase.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 step: load init_base end")
                Log.i(TAG, "Part4 step: load raw_heads begin")
                rawHeadsModule = Module.load(modelPaths.rawHeads.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 step: load raw_heads end")
                Log.i(TAG, "Part4 step: load compose begin")
                composeModule = Module.load(modelPaths.compose.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 step: load compose end")

                val (durationMs, gaussianCount) = runSplitPass(
                    stagePreModule,
                    decoderHeadModule,
                    initBaseModule,
                    rawHeadsModule,
                    composeModule,
                    inputs,
                )
                Log.i(TAG, "Part4 tile_00 split synthetic forward OK in ${durationMs}ms gaussians=$gaussianCount")
                "Part4 tile_00 split test OK in ${durationMs}ms. See logcat tag $TAG."
            } catch (t: Throwable) {
                val kind = classifyFailure(t)
                Log.e(TAG, "Part4 tile_00 split synthetic forward failed kind=$kind msg=${t.message}", t)
                "Part4 tile_00 split test failed ($kind): ${t.message ?: t::class.java.simpleName}"
            } finally {
                try { stagePreModule?.destroy() } catch (_: Throwable) { }
                try { decoderHeadModule?.destroy() } catch (_: Throwable) { }
                try { initBaseModule?.destroy() } catch (_: Throwable) { }
                try { rawHeadsModule?.destroy() } catch (_: Throwable) { }
                try { composeModule?.destroy() } catch (_: Throwable) { }
            }
        }
    }

    @JvmStatic
    fun runTripleForwardBenchmark(context: Context): String {
        return runExclusive("runTripleForwardBenchmark") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            var stagePreModule: Module? = null
            var decoderHeadModule: Module? = null
            var initBaseModule: Module? = null
            var rawHeadsModule: Module? = null
            var composeModule: Module? = null
            try {
                val inputs = createStaticInputs()
                Log.i(TAG, "Part4 bench: load stage_pre begin")
                stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 bench: load stage_pre end")
                Log.i(TAG, "Part4 bench: load decoder_head begin")
                decoderHeadModule = Module.load(modelPaths.decoderHead.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 bench: load decoder_head end")
                Log.i(TAG, "Part4 bench: load init_base begin")
                initBaseModule = Module.load(modelPaths.initBase.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 bench: load init_base end")
                Log.i(TAG, "Part4 bench: load raw_heads begin")
                rawHeadsModule = Module.load(modelPaths.rawHeads.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 bench: load raw_heads end")
                Log.i(TAG, "Part4 bench: load compose begin")
                composeModule = Module.load(modelPaths.compose.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 bench: load compose end")

                val times = LongArray(3)
                var gaussianCount: Int
                repeat(3) { index ->
                    val (durationMs, gaussians) = runSplitPass(
                        stagePreModule,
                        decoderHeadModule,
                        initBaseModule,
                        rawHeadsModule,
                        composeModule,
                        inputs,
                    )
                    times[index] = durationMs
                    gaussianCount = gaussians
                    Log.i(TAG, "$P4_BENCH_MARKER iter=${index + 1} total_ms=$durationMs gaussians=$gaussianCount")
                }
                "Part4 tile_00 split benchmark done: ${times.joinToString(", ")} ms. Grep $P4_BENCH_MARKER in logcat."
            } catch (t: Throwable) {
                val kind = classifyFailure(t)
                Log.e(TAG, "Part4 tile_00 split benchmark failed kind=$kind msg=${t.message}", t)
                "Part4 tile_00 split benchmark failed ($kind): ${t.message ?: t::class.java.simpleName}"
            } finally {
                try { stagePreModule?.destroy() } catch (_: Throwable) { }
                try { decoderHeadModule?.destroy() } catch (_: Throwable) { }
                try { initBaseModule?.destroy() } catch (_: Throwable) { }
                try { rawHeadsModule?.destroy() } catch (_: Throwable) { }
                try { composeModule?.destroy() } catch (_: Throwable) { }
            }
        }
    }

    @JvmStatic
    fun compareDecoderHeadBackends(context: Context): String {
        return runExclusive("compareDecoderHeadBackends") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            val portablePath = modelPaths.decoderHeadPortable
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_HEAD_PORTABLE}"
            var stagePreModule: Module? = null
            var decoderHeadVulkanModule: Module? = null
            var decoderHeadPortableModule: Module? = null
            try {
                val inputs = createStaticInputs()
                Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_head_vulkan=${modelPaths.decoderHead.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_head_portable=${portablePath.absolutePath}")

                Log.i(TAG, "Part4 compare: load stage_pre begin")
                stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 compare: load stage_pre end")
                Log.i(TAG, "Part4 compare: load decoder_head_vulkan begin")
                decoderHeadVulkanModule = Module.load(modelPaths.decoderHead.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 compare: load decoder_head_vulkan end")
                Log.i(TAG, "Part4 compare: load decoder_head_portable begin")
                decoderHeadPortableModule = Module.load(portablePath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 compare: load decoder_head_portable end")

                val vulkanTimes = LongArray(3)
                val portableTimes = LongArray(3)
                val stagePreTimes = LongArray(3)
                repeat(3) { index ->
                    logInputShapes(inputs)
                    val stagePreStartedAt = System.currentTimeMillis()
                    Log.i(TAG, "Part4 compare: forward stage_pre begin")
                    val stagePreOutputs = stagePreModule.forward(
                        inputs.image,
                        inputs.latent0,
                        inputs.latent1,
                        inputs.x0Feat,
                        inputs.x1Feat,
                        inputs.x2Feat,
                        inputs.xLowres
                    )
                    stagePreTimes[index] = System.currentTimeMillis() - stagePreStartedAt
                    Log.i(TAG, "Part4 compare: forward stage_pre end outputs=${stagePreOutputs.size} duration_ms=${stagePreTimes[index]}")
                    stagePreOutputs.forEachIndexed { outputIndex, value -> logTensorShape("compare_stage_pre[$outputIndex]", value) }

                    val vulkanInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "compare_vulkan")
                    val portableInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "compare_portable")
                    val (vulkanDuration, vulkanOutputs) = runDecoderHeadOnly(
                        decoderHeadVulkanModule,
                        vulkanInputs,
                        "decoder_head_vulkan"
                    )
                    val (portableDuration, _) = runDecoderHeadOnly(
                        decoderHeadPortableModule,
                        portableInputs,
                        "decoder_head_portable"
                    )
                    vulkanTimes[index] = vulkanDuration
                    portableTimes[index] = portableDuration

                    val disparityShape = shapeString(vulkanOutputs[0].toTensor())
                    val featuresShape = shapeString(vulkanOutputs[1].toTensor())
                    val speedup = if (vulkanDuration > 0L) portableDuration.toDouble() / vulkanDuration.toDouble() else Double.NaN
                    Log.i(
                        TAG,
                        "$P4_DECODER_COMPARE_MARKER iter=${index + 1} stage_pre_ms=${stagePreTimes[index]} " +
                            "vulkan_ms=$vulkanDuration portable_ms=$portableDuration portable_over_vulkan=%.3f " +
                            "disparity_shape=%s features_shape=%s".format(speedup, disparityShape, featuresShape)
                    )
                }
                "Decoder_head compare done. stage_pre=${stagePreTimes.joinToString(", ")} ms, " +
                    "vulkan=${vulkanTimes.joinToString(", ")} ms, portable=${portableTimes.joinToString(", ")} ms. " +
                    "Grep $P4_DECODER_COMPARE_MARKER in logcat."
            } catch (t: Throwable) {
                val kind = classifyFailure(t)
                Log.e(TAG, "Part4 decoder_head compare failed kind=$kind msg=${t.message}", t)
                "Part4 decoder_head compare failed ($kind): ${t.message ?: t::class.java.simpleName}"
            } finally {
                try { stagePreModule?.destroy() } catch (_: Throwable) { }
                try { decoderHeadVulkanModule?.destroy() } catch (_: Throwable) { }
                try { decoderHeadPortableModule?.destroy() } catch (_: Throwable) { }
            }
        }
    }

    @JvmStatic
    fun compareLatent0MergeBackends(context: Context): String {
        return runExclusive("compareLatent0MergeBackends") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            val decoderSeedPath = modelPaths.decoderSeed
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_SEED}"
            val decoderMergeX1Path = modelPaths.decoderMergeX1
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_X1}"
            val decoderMergeX0Path = modelPaths.decoderMergeX0
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_X0}"
            val decoderMergeLatent1Path = modelPaths.decoderMergeLatent1
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT1}"
            val decoderMergeLatent0PrefusePath = modelPaths.decoderMergeLatent0Prefuse
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE}"
            val decoderMergeLatent0PostfusePath = modelPaths.decoderMergeLatent0Postfuse
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE}"
            val decoderMergeLatent0PrefusePortablePath = modelPaths.decoderMergeLatent0PrefusePortable
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_PREFUSE_PORTABLE}"
            val decoderMergeLatent0PostfusePortablePath = modelPaths.decoderMergeLatent0PostfusePortable
                ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_MERGE_LATENT0_POSTFUSE_PORTABLE}"
            var stagePreModule: Module? = null
            var decoderSeedModule: Module? = null
            var decoderMergeX1Module: Module? = null
            var decoderMergeX0Module: Module? = null
            var decoderMergeLatent1Module: Module? = null
            var decoderMergeLatent0PrefuseModule: Module? = null
            var decoderMergeLatent0PrefusePortableModule: Module? = null
            var decoderMergeLatent0PostfuseModule: Module? = null
            var decoderMergeLatent0PostfusePortableModule: Module? = null
            try {
                val inputs = createStaticInputs()
                Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_seed=${decoderSeedPath.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_x1=${decoderMergeX1Path.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_x0=${decoderMergeX0Path.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent1=${decoderMergeLatent1Path.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_prefuse_vulkan=${decoderMergeLatent0PrefusePath.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_prefuse_portable=${decoderMergeLatent0PrefusePortablePath.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_postfuse_vulkan=${decoderMergeLatent0PostfusePath.absolutePath}")
                Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_postfuse_portable=${decoderMergeLatent0PostfusePortablePath.absolutePath}")

                Log.i(TAG, "Part4 latent0 compare: load stage_pre begin")
                stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load stage_pre end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_seed begin")
                decoderSeedModule = Module.load(decoderSeedPath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_seed end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_x1 begin")
                decoderMergeX1Module = Module.load(decoderMergeX1Path.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_x1 end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_x0 begin")
                decoderMergeX0Module = Module.load(decoderMergeX0Path.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_x0 end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent1 begin")
                decoderMergeLatent1Module = Module.load(decoderMergeLatent1Path.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent1 end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_prefuse_vulkan begin")
                decoderMergeLatent0PrefuseModule = Module.load(decoderMergeLatent0PrefusePath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_prefuse_vulkan end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_prefuse_portable begin")
                decoderMergeLatent0PrefusePortableModule = Module.load(decoderMergeLatent0PrefusePortablePath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_prefuse_portable end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_postfuse_vulkan begin")
                decoderMergeLatent0PostfuseModule = Module.load(decoderMergeLatent0PostfusePath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_postfuse_vulkan end")
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_postfuse_portable begin")
                decoderMergeLatent0PostfusePortableModule = Module.load(decoderMergeLatent0PostfusePortablePath.absolutePath, Module.LOAD_MODE_MMAP)
                Log.i(TAG, "Part4 latent0 compare: load decoder_merge_latent0_postfuse_portable end")

                val loadedStagePreModule = stagePreModule ?: error("stage_pre not loaded")
                val loadedDecoderSeedModule = decoderSeedModule ?: error("decoder_seed not loaded")
                val loadedDecoderMergeX1Module = decoderMergeX1Module ?: error("decoder_merge_x1 not loaded")
                val loadedDecoderMergeX0Module = decoderMergeX0Module ?: error("decoder_merge_x0 not loaded")
                val loadedDecoderMergeLatent1Module = decoderMergeLatent1Module ?: error("decoder_merge_latent1 not loaded")
                val loadedDecoderMergeLatent0PrefuseModule = decoderMergeLatent0PrefuseModule ?: error("decoder_merge_latent0_prefuse_vulkan not loaded")
                val loadedDecoderMergeLatent0PrefusePortableModule = decoderMergeLatent0PrefusePortableModule ?: error("decoder_merge_latent0_prefuse_portable not loaded")
                val loadedDecoderMergeLatent0PostfuseModule = decoderMergeLatent0PostfuseModule ?: error("decoder_merge_latent0_postfuse_vulkan not loaded")
                val loadedDecoderMergeLatent0PostfusePortableModule = decoderMergeLatent0PostfusePortableModule ?: error("decoder_merge_latent0_postfuse_portable not loaded")

                val stagePreTimes = LongArray(3)
                val upstreamTimes = LongArray(3)
                val prefuseVulkanTimes = LongArray(3)
                val prefusePortableTimes = LongArray(3)
                val postfuseVulkanTimes = LongArray(3)
                val postfusePortableTimes = LongArray(3)
                repeat(3) { index ->
                    logInputShapes(inputs)
                    val stagePreStartedAt = System.currentTimeMillis()
                    Log.i(TAG, "Part4 latent0 compare: forward stage_pre begin")
                    val stagePreOutputs = loadedStagePreModule.forward(
                        inputs.image,
                        inputs.latent0,
                        inputs.latent1,
                        inputs.x0Feat,
                        inputs.x1Feat,
                        inputs.x2Feat,
                        inputs.xLowres
                    )
                    stagePreTimes[index] = System.currentTimeMillis() - stagePreStartedAt
                    Log.i(TAG, "Part4 latent0 compare: forward stage_pre end outputs=${stagePreOutputs.size} duration_ms=${stagePreTimes[index]}")
                    stagePreOutputs.forEachIndexed { outputIndex, value -> logTensorShape("compare_latent0_stage_pre[$outputIndex]", value) }

                    val preparedInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "compare_latent0")
                    val upstreamStartedAt = System.currentTimeMillis()
                    val (_, decoderSeed) = runSingleTensorStage(
                        loadedDecoderSeedModule,
                        preparedInputs.xFused,
                        "compare_latent0_decoder_seed"
                    )
                    val (_, decoder48) = runTwoInputTensorStage(
                        loadedDecoderMergeX1Module,
                        cloneAsEValue("compare_latent0_decoder_seed_out", decoderSeed),
                        preparedInputs.x1Up,
                        "compare_latent0_decoder_merge_x1"
                    )
                    val (_, decoder96) = runTwoInputTensorStage(
                        loadedDecoderMergeX0Module,
                        cloneAsEValue("compare_latent0_decoder_48", decoder48),
                        preparedInputs.x0Up,
                        "compare_latent0_decoder_merge_x0"
                    )
                    val (_, decoder192Up) = runTwoInputTensorStage(
                        loadedDecoderMergeLatent1Module,
                        cloneAsEValue("compare_latent0_decoder_96", decoder96),
                        preparedInputs.latent1Up,
                        "compare_latent0_decoder_merge_latent1"
                    )
                    upstreamTimes[index] = System.currentTimeMillis() - upstreamStartedAt

                    val (prefuseVulkanDuration, prefuseVulkanOutput) = runTwoInputTensorStage(
                        loadedDecoderMergeLatent0PrefuseModule,
                        cloneAsEValue("compare_latent0_decoder_192_up_vulkan", decoder192Up),
                        cloneAsEValue("compare_latent0_latent0_up_vulkan", preparedInputs.latent0Up),
                        "decoder_merge_latent0_prefuse_vulkan"
                    )
                    val (prefusePortableDuration, prefusePortableOutput) = runTwoInputTensorStage(
                        loadedDecoderMergeLatent0PrefusePortableModule,
                        cloneAsEValue("compare_latent0_decoder_192_up_portable", decoder192Up),
                        cloneAsEValue("compare_latent0_latent0_up_portable", preparedInputs.latent0Up),
                        "decoder_merge_latent0_prefuse_portable"
                    )
                    val (postfuseVulkanDuration, postfuseVulkanOutput) = runSingleTensorStage(
                        loadedDecoderMergeLatent0PostfuseModule,
                        cloneAsEValue("compare_latent0_prefuse_out_vulkan", prefuseVulkanOutput),
                        "decoder_merge_latent0_postfuse_vulkan"
                    )
                    val (postfusePortableDuration, postfusePortableOutput) = runSingleTensorStage(
                        loadedDecoderMergeLatent0PostfusePortableModule,
                        cloneAsEValue("compare_latent0_prefuse_out_portable", prefusePortableOutput),
                        "decoder_merge_latent0_postfuse_portable"
                    )

                    prefuseVulkanTimes[index] = prefuseVulkanDuration
                    prefusePortableTimes[index] = prefusePortableDuration
                    postfuseVulkanTimes[index] = postfuseVulkanDuration
                    postfusePortableTimes[index] = postfusePortableDuration

                    val prefuseRatio = if (prefuseVulkanDuration > 0L) prefusePortableDuration.toDouble() / prefuseVulkanDuration.toDouble() else Double.NaN
                    val postfuseRatio = if (postfuseVulkanDuration > 0L) postfusePortableDuration.toDouble() / postfuseVulkanDuration.toDouble() else Double.NaN
                    Log.i(
                        TAG,
                        "$P4_LATENT0_COMPARE_MARKER iter=${index + 1} stage_pre_ms=${stagePreTimes[index]} upstream_ms=${upstreamTimes[index]} " +
                            "prefuse_vulkan_ms=$prefuseVulkanDuration prefuse_portable_ms=$prefusePortableDuration prefuse_portable_over_vulkan=%.3f ".format(prefuseRatio) +
                            "postfuse_vulkan_ms=$postfuseVulkanDuration postfuse_portable_ms=$postfusePortableDuration postfuse_portable_over_vulkan=%.3f ".format(postfuseRatio) +
                            "prefuse_shape=${shapeString(prefuseVulkanOutput.toTensor())} postfuse_shape=${shapeString(postfuseVulkanOutput.toTensor())} " +
                            "postfuse_portable_shape=${shapeString(postfusePortableOutput.toTensor())}"
                    )
                }
                "Latent0 compare done. stage_pre=${stagePreTimes.joinToString(", ")} ms, upstream=${upstreamTimes.joinToString(", ")} ms, " +
                    "prefuse_vulkan=${prefuseVulkanTimes.joinToString(", ")} ms, prefuse_portable=${prefusePortableTimes.joinToString(", ")} ms, " +
                    "postfuse_vulkan=${postfuseVulkanTimes.joinToString(", ")} ms, postfuse_portable=${postfusePortableTimes.joinToString(", ")} ms. " +
                    "Grep $P4_LATENT0_COMPARE_MARKER in logcat."
            } catch (t: Throwable) {
                val kind = classifyFailure(t)
                Log.e(TAG, "Part4 latent0 compare failed kind=$kind msg=${t.message}", t)
                "Part4 latent0 compare failed ($kind): ${t.message ?: t::class.java.simpleName}"
            } finally {
                try { stagePreModule?.destroy() } catch (_: Throwable) { }
                try { decoderSeedModule?.destroy() } catch (_: Throwable) { }
                try { decoderMergeX1Module?.destroy() } catch (_: Throwable) { }
                try { decoderMergeX0Module?.destroy() } catch (_: Throwable) { }
                try { decoderMergeLatent1Module?.destroy() } catch (_: Throwable) { }
                try { decoderMergeLatent0PrefuseModule?.destroy() } catch (_: Throwable) { }
                try { decoderMergeLatent0PrefusePortableModule?.destroy() } catch (_: Throwable) { }
                try { decoderMergeLatent0PostfuseModule?.destroy() } catch (_: Throwable) { }
                try { decoderMergeLatent0PostfusePortableModule?.destroy() } catch (_: Throwable) { }
            }
        }
    }

    @JvmStatic
    fun benchmarkDecoderHeadChunks(context: Context): String {
        return runExclusive("benchmarkDecoderHeadChunks") {
            val modelPaths = resolveModelPaths(context).getOrElse { return@runExclusive it.message ?: "Part4 model resolution failed" }
            val haveFineStack = listOf(
                modelPaths.decoderSeed,
                modelPaths.decoderMergeX1,
                modelPaths.decoderMergeX0,
                modelPaths.decoderMergeLatent1,
                modelPaths.decoderMergeLatent0,
                modelPaths.disparityHead,
            ).all { it != null }
            val haveFineLatent0Split = listOf(
                modelPaths.decoderMergeLatent0Prefuse,
                modelPaths.decoderMergeLatent0Postfuse,
            ).all { it != null }

            if (haveFineStack && haveFineLatent0Split) {
                val decoderSeedPath = modelPaths.decoderSeed!!
                val decoderMergeX1Path = modelPaths.decoderMergeX1!!
                val decoderMergeX0Path = modelPaths.decoderMergeX0!!
                val decoderMergeLatent1Path = modelPaths.decoderMergeLatent1!!
                val decoderMergeLatent0PrefusePath = modelPaths.decoderMergeLatent0Prefuse!!
                val decoderMergeLatent0PostfusePath = modelPaths.decoderMergeLatent0Postfuse!!
                val disparityHeadPath = modelPaths.disparityHead!!
                var stagePreModule: Module? = null
                var decoderSeedModule: Module? = null
                var decoderMergeX1Module: Module? = null
                var decoderMergeX0Module: Module? = null
                var decoderMergeLatent1Module: Module? = null
                var decoderMergeLatent0PrefuseModule: Module? = null
                var decoderMergeLatent0PostfuseModule: Module? = null
                var disparityHeadModule: Module? = null
                try {
                    val inputs = createStaticInputs()
                    Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_seed=${decoderSeedPath.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_x1=${decoderMergeX1Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_x0=${decoderMergeX0Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent1=${decoderMergeLatent1Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_prefuse=${decoderMergeLatent0PrefusePath.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0_postfuse=${decoderMergeLatent0PostfusePath.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT disparity_head=${disparityHeadPath.absolutePath}")

                    Log.i(TAG, "Part4 decoder stack bench: load stage_pre begin")
                    stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load stage_pre end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_seed begin")
                    decoderSeedModule = Module.load(decoderSeedPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_seed end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x1 begin")
                    decoderMergeX1Module = Module.load(decoderMergeX1Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x1 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x0 begin")
                    decoderMergeX0Module = Module.load(decoderMergeX0Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x0 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent1 begin")
                    decoderMergeLatent1Module = Module.load(decoderMergeLatent1Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent1 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0_prefuse begin")
                    decoderMergeLatent0PrefuseModule = Module.load(decoderMergeLatent0PrefusePath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0_prefuse end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0_postfuse begin")
                    decoderMergeLatent0PostfuseModule = Module.load(decoderMergeLatent0PostfusePath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0_postfuse end")
                    Log.i(TAG, "Part4 decoder stack bench: load disparity_head begin")
                    disparityHeadModule = Module.load(disparityHeadPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load disparity_head end")

                    val loadedStagePreModule = stagePreModule ?: error("stage_pre not loaded")
                    val loadedDecoderSeedModule = decoderSeedModule ?: error("decoder_seed not loaded")
                    val loadedDecoderMergeX1Module = decoderMergeX1Module ?: error("decoder_merge_x1 not loaded")
                    val loadedDecoderMergeX0Module = decoderMergeX0Module ?: error("decoder_merge_x0 not loaded")
                    val loadedDecoderMergeLatent1Module = decoderMergeLatent1Module ?: error("decoder_merge_latent1 not loaded")
                    val loadedDecoderMergeLatent0PrefuseModule = decoderMergeLatent0PrefuseModule ?: error("decoder_merge_latent0_prefuse not loaded")
                    val loadedDecoderMergeLatent0PostfuseModule = decoderMergeLatent0PostfuseModule ?: error("decoder_merge_latent0_postfuse not loaded")
                    val loadedDisparityHeadModule = disparityHeadModule ?: error("disparity_head not loaded")

                    val stagePreTimes = LongArray(3)
                    val decoderSeedTimes = LongArray(3)
                    val decoderMergeX1Times = LongArray(3)
                    val decoderMergeX0Times = LongArray(3)
                    val decoderMergeLatent1Times = LongArray(3)
                    val decoderMergeLatent0PrefuseTimes = LongArray(3)
                    val decoderMergeLatent0PostfuseTimes = LongArray(3)
                    val disparityHeadTimes = LongArray(3)
                    repeat(3) { index ->
                        logInputShapes(inputs)
                        val stagePreStartedAt = System.currentTimeMillis()
                        Log.i(TAG, "Part4 decoder stack bench: forward stage_pre begin")
                        val stagePreOutputs = loadedStagePreModule.forward(
                            inputs.image,
                            inputs.latent0,
                            inputs.latent1,
                            inputs.x0Feat,
                            inputs.x1Feat,
                            inputs.x2Feat,
                            inputs.xLowres
                        )
                        stagePreTimes[index] = System.currentTimeMillis() - stagePreStartedAt
                        Log.i(
                            TAG,
                            "Part4 decoder stack bench: forward stage_pre end outputs=${stagePreOutputs.size} duration_ms=${stagePreTimes[index]}"
                        )
                        stagePreOutputs.forEachIndexed { outputIndex, value -> logTensorShape("stack_stage_pre[$outputIndex]", value) }

                        val preparedInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "stack_decoder")
                        val (decoderSeedDuration, decoderSeed) = runSingleTensorStage(
                            loadedDecoderSeedModule,
                            preparedInputs.xFused,
                            "decoder_seed"
                        )
                        val (decoderMergeX1Duration, decoder48) = runTwoInputTensorStage(
                            loadedDecoderMergeX1Module,
                            cloneAsEValue("stack_decoder_seed", decoderSeed),
                            preparedInputs.x1Up,
                            "decoder_merge_x1"
                        )
                        val (decoderMergeX0Duration, decoder96) = runTwoInputTensorStage(
                            loadedDecoderMergeX0Module,
                            cloneAsEValue("stack_decoder_48", decoder48),
                            preparedInputs.x0Up,
                            "decoder_merge_x0"
                        )
                        val (decoderMergeLatent1Duration, decoder192Up) = runTwoInputTensorStage(
                            loadedDecoderMergeLatent1Module,
                            cloneAsEValue("stack_decoder_96", decoder96),
                            preparedInputs.latent1Up,
                            "decoder_merge_latent1"
                        )
                        val (decoderMergeLatent0PrefuseDuration, decoder192Prefused) = runTwoInputTensorStage(
                            loadedDecoderMergeLatent0PrefuseModule,
                            cloneAsEValue("stack_decoder_192_up", decoder192Up),
                            preparedInputs.latent0Up,
                            "decoder_merge_latent0_prefuse"
                        )
                        val (decoderMergeLatent0PostfuseDuration, decoderFeatures) = runSingleTensorStage(
                            loadedDecoderMergeLatent0PostfuseModule,
                            cloneAsEValue("stack_decoder_192_prefused", decoder192Prefused),
                            "decoder_merge_latent0_postfuse"
                        )
                        val (disparityHeadDuration, disparity) = runDisparityHeadOnly(
                            loadedDisparityHeadModule,
                            cloneAsEValue("stack_decoder_final", decoderFeatures),
                            "disparity_head"
                        )

                        decoderSeedTimes[index] = decoderSeedDuration
                        decoderMergeX1Times[index] = decoderMergeX1Duration
                        decoderMergeX0Times[index] = decoderMergeX0Duration
                        decoderMergeLatent1Times[index] = decoderMergeLatent1Duration
                        decoderMergeLatent0PrefuseTimes[index] = decoderMergeLatent0PrefuseDuration
                        decoderMergeLatent0PostfuseTimes[index] = decoderMergeLatent0PostfuseDuration
                        disparityHeadTimes[index] = disparityHeadDuration

                        val decoderMergeLatent0Duration = decoderMergeLatent0PrefuseDuration + decoderMergeLatent0PostfuseDuration
                        val splitTotal = decoderSeedDuration +
                            decoderMergeX1Duration +
                            decoderMergeX0Duration +
                            decoderMergeLatent1Duration +
                            decoderMergeLatent0Duration +
                            disparityHeadDuration
                        Log.i(
                            TAG,
                            "$P4_DECODER_STACK_BENCH_MARKER iter=${index + 1} stage_pre_ms=${stagePreTimes[index]} " +
                                "decoder_seed_ms=$decoderSeedDuration decoder_merge_x1_ms=$decoderMergeX1Duration " +
                                "decoder_merge_x0_ms=$decoderMergeX0Duration decoder_merge_latent1_ms=$decoderMergeLatent1Duration " +
                                "decoder_merge_latent0_prefuse_ms=$decoderMergeLatent0PrefuseDuration " +
                                "decoder_merge_latent0_postfuse_ms=$decoderMergeLatent0PostfuseDuration " +
                                "decoder_merge_latent0_ms=$decoderMergeLatent0Duration disparity_head_ms=$disparityHeadDuration " +
                                "split_total_ms=$splitTotal decoder_features_shape=${shapeString(decoderFeatures.toTensor())} " +
                                "disparity_shape=${shapeString(disparity.toTensor())}"
                        )
                    }
                    "Decoder stack benchmark done. stage_pre=${stagePreTimes.joinToString(", ")} ms, " +
                        "decoder_seed=${decoderSeedTimes.joinToString(", ")} ms, " +
                        "merge_x1=${decoderMergeX1Times.joinToString(", ")} ms, " +
                        "merge_x0=${decoderMergeX0Times.joinToString(", ")} ms, " +
                        "merge_latent1=${decoderMergeLatent1Times.joinToString(", ")} ms, " +
                        "merge_latent0_prefuse=${decoderMergeLatent0PrefuseTimes.joinToString(", ")} ms, " +
                        "merge_latent0_postfuse=${decoderMergeLatent0PostfuseTimes.joinToString(", ")} ms, " +
                        "disparity_head=${disparityHeadTimes.joinToString(", ")} ms. " +
                        "Grep $P4_DECODER_STACK_BENCH_MARKER in logcat."
                } catch (t: Throwable) {
                    val kind = classifyFailure(t)
                    Log.e(TAG, "Part4 decoder stack benchmark failed kind=$kind msg=${t.message}", t)
                    "Part4 decoder stack benchmark failed ($kind): ${t.message ?: t::class.java.simpleName}"
                } finally {
                    try { stagePreModule?.destroy() } catch (_: Throwable) { }
                    try { decoderSeedModule?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeX1Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeX0Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeLatent1Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeLatent0PrefuseModule?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeLatent0PostfuseModule?.destroy() } catch (_: Throwable) { }
                    try { disparityHeadModule?.destroy() } catch (_: Throwable) { }
                }
            } else if (haveFineStack) {
                val decoderSeedPath = modelPaths.decoderSeed!!
                val decoderMergeX1Path = modelPaths.decoderMergeX1!!
                val decoderMergeX0Path = modelPaths.decoderMergeX0!!
                val decoderMergeLatent1Path = modelPaths.decoderMergeLatent1!!
                val decoderMergeLatent0Path = modelPaths.decoderMergeLatent0!!
                val disparityHeadPath = modelPaths.disparityHead!!
                var stagePreModule: Module? = null
                var decoderSeedModule: Module? = null
                var decoderMergeX1Module: Module? = null
                var decoderMergeX0Module: Module? = null
                var decoderMergeLatent1Module: Module? = null
                var decoderMergeLatent0Module: Module? = null
                var disparityHeadModule: Module? = null
                try {
                    val inputs = createStaticInputs()
                    Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_seed=${decoderSeedPath.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_x1=${decoderMergeX1Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_x0=${decoderMergeX0Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent1=${decoderMergeLatent1Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_merge_latent0=${decoderMergeLatent0Path.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT disparity_head=${disparityHeadPath.absolutePath}")

                    Log.i(TAG, "Part4 decoder stack bench: load stage_pre begin")
                    stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load stage_pre end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_seed begin")
                    decoderSeedModule = Module.load(decoderSeedPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_seed end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x1 begin")
                    decoderMergeX1Module = Module.load(decoderMergeX1Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x1 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x0 begin")
                    decoderMergeX0Module = Module.load(decoderMergeX0Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_x0 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent1 begin")
                    decoderMergeLatent1Module = Module.load(decoderMergeLatent1Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent1 end")
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0 begin")
                    decoderMergeLatent0Module = Module.load(decoderMergeLatent0Path.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load decoder_merge_latent0 end")
                    Log.i(TAG, "Part4 decoder stack bench: load disparity_head begin")
                    disparityHeadModule = Module.load(disparityHeadPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder stack bench: load disparity_head end")

                    val loadedStagePreModule = stagePreModule ?: error("stage_pre not loaded")
                    val loadedDecoderSeedModule = decoderSeedModule ?: error("decoder_seed not loaded")
                    val loadedDecoderMergeX1Module = decoderMergeX1Module ?: error("decoder_merge_x1 not loaded")
                    val loadedDecoderMergeX0Module = decoderMergeX0Module ?: error("decoder_merge_x0 not loaded")
                    val loadedDecoderMergeLatent1Module = decoderMergeLatent1Module ?: error("decoder_merge_latent1 not loaded")
                    val loadedDecoderMergeLatent0Module = decoderMergeLatent0Module ?: error("decoder_merge_latent0 not loaded")
                    val loadedDisparityHeadModule = disparityHeadModule ?: error("disparity_head not loaded")

                    val stagePreTimes = LongArray(3)
                    val decoderSeedTimes = LongArray(3)
                    val decoderMergeX1Times = LongArray(3)
                    val decoderMergeX0Times = LongArray(3)
                    val decoderMergeLatent1Times = LongArray(3)
                    val decoderMergeLatent0Times = LongArray(3)
                    val disparityHeadTimes = LongArray(3)
                    repeat(3) { index ->
                        logInputShapes(inputs)
                        val stagePreStartedAt = System.currentTimeMillis()
                        Log.i(TAG, "Part4 decoder stack bench: forward stage_pre begin")
                        val stagePreOutputs = loadedStagePreModule.forward(
                            inputs.image,
                            inputs.latent0,
                            inputs.latent1,
                            inputs.x0Feat,
                            inputs.x1Feat,
                            inputs.x2Feat,
                            inputs.xLowres
                        )
                        stagePreTimes[index] = System.currentTimeMillis() - stagePreStartedAt
                        Log.i(
                            TAG,
                            "Part4 decoder stack bench: forward stage_pre end outputs=${stagePreOutputs.size} duration_ms=${stagePreTimes[index]}"
                        )
                        stagePreOutputs.forEachIndexed { outputIndex, value -> logTensorShape("stack_stage_pre[$outputIndex]", value) }

                        val preparedInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "stack_decoder")
                        val (decoderSeedDuration, decoderSeed) = runSingleTensorStage(
                            loadedDecoderSeedModule,
                            preparedInputs.xFused,
                            "decoder_seed"
                        )
                        val (decoderMergeX1Duration, decoder48) = runTwoInputTensorStage(
                            loadedDecoderMergeX1Module,
                            cloneAsEValue("stack_decoder_seed", decoderSeed),
                            preparedInputs.x1Up,
                            "decoder_merge_x1"
                        )
                        val (decoderMergeX0Duration, decoder96) = runTwoInputTensorStage(
                            loadedDecoderMergeX0Module,
                            cloneAsEValue("stack_decoder_48", decoder48),
                            preparedInputs.x0Up,
                            "decoder_merge_x0"
                        )
                        val (decoderMergeLatent1Duration, decoder192Up) = runTwoInputTensorStage(
                            loadedDecoderMergeLatent1Module,
                            cloneAsEValue("stack_decoder_96", decoder96),
                            preparedInputs.latent1Up,
                            "decoder_merge_latent1"
                        )
                        val (decoderMergeLatent0Duration, decoderFeatures) = runTwoInputTensorStage(
                            loadedDecoderMergeLatent0Module,
                            cloneAsEValue("stack_decoder_192_up", decoder192Up),
                            preparedInputs.latent0Up,
                            "decoder_merge_latent0"
                        )
                        val (disparityHeadDuration, disparity) = runDisparityHeadOnly(
                            loadedDisparityHeadModule,
                            cloneAsEValue("stack_decoder_final", decoderFeatures),
                            "disparity_head"
                        )

                        decoderSeedTimes[index] = decoderSeedDuration
                        decoderMergeX1Times[index] = decoderMergeX1Duration
                        decoderMergeX0Times[index] = decoderMergeX0Duration
                        decoderMergeLatent1Times[index] = decoderMergeLatent1Duration
                        decoderMergeLatent0Times[index] = decoderMergeLatent0Duration
                        disparityHeadTimes[index] = disparityHeadDuration

                        val splitTotal = decoderSeedDuration +
                            decoderMergeX1Duration +
                            decoderMergeX0Duration +
                            decoderMergeLatent1Duration +
                            decoderMergeLatent0Duration +
                            disparityHeadDuration
                        Log.i(
                            TAG,
                            "$P4_DECODER_STACK_BENCH_MARKER iter=${index + 1} stage_pre_ms=${stagePreTimes[index]} " +
                                "decoder_seed_ms=$decoderSeedDuration decoder_merge_x1_ms=$decoderMergeX1Duration " +
                                "decoder_merge_x0_ms=$decoderMergeX0Duration decoder_merge_latent1_ms=$decoderMergeLatent1Duration " +
                                "decoder_merge_latent0_ms=$decoderMergeLatent0Duration disparity_head_ms=$disparityHeadDuration " +
                                "split_total_ms=$splitTotal decoder_features_shape=${shapeString(decoderFeatures.toTensor())} " +
                                "disparity_shape=${shapeString(disparity.toTensor())}"
                        )
                    }
                    "Decoder stack benchmark done. stage_pre=${stagePreTimes.joinToString(", ")} ms, " +
                        "decoder_seed=${decoderSeedTimes.joinToString(", ")} ms, " +
                        "merge_x1=${decoderMergeX1Times.joinToString(", ")} ms, " +
                        "merge_x0=${decoderMergeX0Times.joinToString(", ")} ms, " +
                        "merge_latent1=${decoderMergeLatent1Times.joinToString(", ")} ms, " +
                        "merge_latent0=${decoderMergeLatent0Times.joinToString(", ")} ms, " +
                        "disparity_head=${disparityHeadTimes.joinToString(", ")} ms. " +
                        "Grep $P4_DECODER_STACK_BENCH_MARKER in logcat."
                } catch (t: Throwable) {
                    val kind = classifyFailure(t)
                    Log.e(TAG, "Part4 decoder stack benchmark failed kind=$kind msg=${t.message}", t)
                    "Part4 decoder stack benchmark failed ($kind): ${t.message ?: t::class.java.simpleName}"
                } finally {
                    try { stagePreModule?.destroy() } catch (_: Throwable) { }
                    try { decoderSeedModule?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeX1Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeX0Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeLatent1Module?.destroy() } catch (_: Throwable) { }
                    try { decoderMergeLatent0Module?.destroy() } catch (_: Throwable) { }
                    try { disparityHeadModule?.destroy() } catch (_: Throwable) { }
                }
            } else {
                val decoderOnlyPath = modelPaths.decoderOnly
                    ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DECODER_ONLY}"
                val disparityHeadPath = modelPaths.disparityHead
                    ?: return@runExclusive "Missing ${SharpExecuTorchSplitModelNames.PART4B_TILE_00_DISPARITY_HEAD}"
                var stagePreModule: Module? = null
                var decoderOnlyModule: Module? = null
                var disparityHeadModule: Module? = null
                try {
                    val inputs = createStaticInputs()
                    Log.i(TAG, "PART4_ARTIFACT stage_pre=${modelPaths.stagePre.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT decoder_only=${decoderOnlyPath.absolutePath}")
                    Log.i(TAG, "PART4_ARTIFACT disparity_head=${disparityHeadPath.absolutePath}")

                    Log.i(TAG, "Part4 decoder chunk bench: load stage_pre begin")
                    stagePreModule = Module.load(modelPaths.stagePre.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder chunk bench: load stage_pre end")
                    Log.i(TAG, "Part4 decoder chunk bench: load decoder_only begin")
                    decoderOnlyModule = Module.load(decoderOnlyPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder chunk bench: load decoder_only end")
                    Log.i(TAG, "Part4 decoder chunk bench: load disparity_head begin")
                    disparityHeadModule = Module.load(disparityHeadPath.absolutePath, Module.LOAD_MODE_MMAP)
                    Log.i(TAG, "Part4 decoder chunk bench: load disparity_head end")

                    val stagePreTimes = LongArray(3)
                    val decoderOnlyTimes = LongArray(3)
                    val disparityHeadTimes = LongArray(3)
                    repeat(3) { index ->
                        logInputShapes(inputs)
                        val stagePreStartedAt = System.currentTimeMillis()
                        Log.i(TAG, "Part4 decoder chunk bench: forward stage_pre begin")
                        val stagePreOutputs = stagePreModule.forward(
                            inputs.image,
                            inputs.latent0,
                            inputs.latent1,
                            inputs.x0Feat,
                            inputs.x1Feat,
                            inputs.x2Feat,
                            inputs.xLowres
                        )
                        stagePreTimes[index] = System.currentTimeMillis() - stagePreStartedAt
                        Log.i(
                            TAG,
                            "Part4 decoder chunk bench: forward stage_pre end outputs=${stagePreOutputs.size} duration_ms=${stagePreTimes[index]}"
                        )
                        stagePreOutputs.forEachIndexed { outputIndex, value -> logTensorShape("chunk_stage_pre[$outputIndex]", value) }

                        val preparedInputs = cloneStagePreOutputsForDecoder(stagePreOutputs, "chunk_decoder")
                        val (decoderOnlyDuration, decoderFeatures) = runDecoderOnly(
                            decoderOnlyModule,
                            preparedInputs,
                            "decoder_only"
                        )
                        val (disparityHeadDuration, disparity) = runDisparityHeadOnly(
                            disparityHeadModule,
                            cloneAsEValue("chunk_decoder_features", decoderFeatures),
                            "disparity_head"
                        )
                        decoderOnlyTimes[index] = decoderOnlyDuration
                        disparityHeadTimes[index] = disparityHeadDuration

                        val decoderFeaturesShape = shapeString(decoderFeatures.toTensor())
                        val disparityShape = shapeString(disparity.toTensor())
                        val splitTotal = decoderOnlyDuration + disparityHeadDuration
                        Log.i(
                            TAG,
                            "$P4_DECODER_CHUNK_BENCH_MARKER iter=${index + 1} stage_pre_ms=${stagePreTimes[index]} " +
                                "decoder_only_ms=$decoderOnlyDuration disparity_head_ms=$disparityHeadDuration " +
                                "split_total_ms=$splitTotal decoder_features_shape=$decoderFeaturesShape disparity_shape=$disparityShape"
                        )
                    }
                    "Decoder_head chunk benchmark done. stage_pre=${stagePreTimes.joinToString(", ")} ms, " +
                        "decoder_only=${decoderOnlyTimes.joinToString(", ")} ms, " +
                        "disparity_head=${disparityHeadTimes.joinToString(", ")} ms. " +
                        "Grep $P4_DECODER_CHUNK_BENCH_MARKER in logcat."
                } catch (t: Throwable) {
                    val kind = classifyFailure(t)
                    Log.e(TAG, "Part4 decoder chunk benchmark failed kind=$kind msg=${t.message}", t)
                    "Part4 decoder chunk benchmark failed ($kind): ${t.message ?: t::class.java.simpleName}"
                } finally {
                    try { stagePreModule?.destroy() } catch (_: Throwable) { }
                    try { decoderOnlyModule?.destroy() } catch (_: Throwable) { }
                    try { disparityHeadModule?.destroy() } catch (_: Throwable) { }
                }
            }
        }
    }
}
