package com.furnit.android.utils

import com.furnit.android.utils.LogUtil
import io.github.sceneview.SceneView
import io.github.sceneview.environment.Environment
import io.github.sceneview.loaders.EnvironmentLoader
import io.github.sceneview.math.Direction
import kotlin.math.sqrt

/**
 * Indoor PBR lighting for bundled/user GLB rooms in SceneView/Filament.
 *
 * RealityKit applies default image-based lighting automatically; Filament does not unless
 * configured. Diffuse surfaces need [com.google.android.filament.IndirectLight]; metallic /
 * glossy surfaces also need a packaged skybox (or equivalent reflection environment) or they render
 * near-black even when the matte wall next to them is correctly lit.
 */
object RoomSceneLighting {

    private const val TAG = "RoomSceneLighting"

    /** Tuned for dielectric interior materials (metallic=0). */
    private const val INDOOR_IBL_INTENSITY = 12_000f

    /** Softer key fill now that IBL carries most of the room light. */
    private const val INDOOR_DIRECTIONAL_INTENSITY = 35_000f

    /** Higher f-stop darkens the scene to match iOS RealityKit tonemapping. */
    private const val INDOOR_APERTURE = 22f
    private const val INDOOR_SHUTTER_SPEED = 1f / 125f
    private const val INDOOR_ISO = 100f

    /** Bundled with SceneView AAR; merged into the app APK assets. */
    private const val NEUTRAL_IBL_ASSET = "environments/neutral/neutral_ibl.ktx"
    private const val NEUTRAL_SKYBOX_ASSET = "environments/neutral/neutral_skybox.ktx"

    /** App-specific indoor probe (optional). Drop cmgen KTX files here to override neutral. */
    private const val INDOOR_IBL_ASSET = "environments/indoor/indoor_ibl.ktx"
    private const val INDOOR_SKYBOX_ASSET = "environments/indoor/indoor_skybox.ktx"

    private val INDOOR_KEY_LIGHT_DIRECTION: Direction = normalizeDirection(0.25f, -0.88f, -0.38f)

    fun applyIndoorPbrLighting(sceneView: SceneView) {
        applyEnvironment(sceneView)
        applyCameraExposure(sceneView)
        applyKeyLight(sceneView)
    }

    private fun applyEnvironment(sceneView: SceneView) {
        val loader = sceneView.environmentLoader
        val baseEnvironment = runCatching {
            loadKtxEnvironment(sceneView, loader)
        }.getOrElse { error ->
            LogUtil.w(TAG, "Failed to load KTX environment; boosting existing indirect light", error)
            boostExistingIndirectLight(sceneView)
            return
        }

        val environment = baseEnvironment
        if (environment.skybox == null) {
            // The reflection texture belongs to IndirectLight. Reusing it as a Skybox can
            // double-destroy the native Filament texture when environments are replaced.
            LogUtil.w(TAG, "KTX environment has no packaged skybox; retaining IBL only")
        }
        sceneView.environment = environment

        environment.indirectLight?.setIntensity(INDOOR_IBL_INTENSITY)
        sceneView.indirectLight = environment.indirectLight
        sceneView.skybox = environment.skybox

        LogUtil.d(
            TAG,
            "Environment applied iblIntensity=$INDOOR_IBL_INTENSITY " +
                "indirectLight=${environment.indirectLight != null} " +
                "skybox=${environment.skybox != null} " +
                "reflections=${environment.indirectLight?.reflectionsTexture != null}",
        )
    }

    private fun loadKtxEnvironment(sceneView: SceneView, loader: EnvironmentLoader): Environment {
        return if (assetExists(sceneView, INDOOR_IBL_ASSET)) {
            loader.createKTX1Environment(
                iblAssetFile = INDOOR_IBL_ASSET,
                skyboxAssetFile = INDOOR_SKYBOX_ASSET.takeIf { assetExists(sceneView, it) },
            )
        } else {
            loader.createKTX1Environment(
                iblAssetFile = NEUTRAL_IBL_ASSET,
                skyboxAssetFile = NEUTRAL_SKYBOX_ASSET,
            )
        }
    }

    private fun boostExistingIndirectLight(sceneView: SceneView) {
        sceneView.indirectLight?.setIntensity(INDOOR_IBL_INTENSITY)
    }

    private fun applyCameraExposure(sceneView: SceneView) {
        sceneView.cameraNode.setExposure(INDOOR_APERTURE, INDOOR_SHUTTER_SPEED, INDOOR_ISO)
        LogUtil.d(
            TAG,
            "Camera exposure f/$INDOOR_APERTURE @ ${INDOOR_SHUTTER_SPEED}s ISO $INDOOR_ISO",
        )
    }

    private fun applyKeyLight(sceneView: SceneView) {
        sceneView.mainLightNode?.apply {
            intensity = INDOOR_DIRECTIONAL_INTENSITY
            lightDirection = INDOOR_KEY_LIGHT_DIRECTION
            isShadowCaster = true
        }
        LogUtil.d(
            TAG,
            "Key light intensity=$INDOOR_DIRECTIONAL_INTENSITY direction=$INDOOR_KEY_LIGHT_DIRECTION",
        )
    }

    private fun assetExists(sceneView: SceneView, assetPath: String): Boolean {
        return runCatching {
            sceneView.context.assets.open(assetPath).close()
            true
        }.getOrDefault(false)
    }

    private fun normalizeDirection(x: Float, y: Float, z: Float): Direction {
        val length = sqrt(x * x + y * y + z * z)
        return Direction(x / length, y / length, z / length)
    }
}
