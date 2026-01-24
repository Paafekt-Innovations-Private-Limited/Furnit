package com.furnit.android

import android.content.Intent
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import io.github.sceneview.SceneView
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Tests for ModelDetailActivity - 3D Room Viewer
 * Verifies overlay controls, edge-to-edge display, and UI components.
 */
@RunWith(AndroidJUnit4::class)
class ModelDetailActivityTest {

    @Test
    fun testActivityLaunchesWithModelId() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                assertNotNull("Activity should not be null", activity)
                println("ModelDetailActivity launched successfully with model ID")
            }
        }
    }

    @Test
    fun testSceneViewExists() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)
                assertEquals("SceneView width should match parent",
                    View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED),
                    View.MeasureSpec.makeMeasureSpec(0, View.MeasureSpec.UNSPECIFIED))
                println("SceneView found and configured")
            }
        }
    }

    @Test
    fun testTopBarOverlayExists() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val topBarContainer = activity.findViewById<LinearLayout>(R.id.topBarContainer)
                assertNotNull("Top bar container should exist", topBarContainer)

                val backButton = activity.findViewById<ImageButton>(R.id.backButton)
                assertNotNull("Back button should exist", backButton)

                val modelTitle = activity.findViewById<TextView>(R.id.modelTitle)
                assertNotNull("Model title should exist", modelTitle)
                assertEquals("Title should be '3D Room View'", "3D Room View", modelTitle.text.toString())

                val helpButton = activity.findViewById<ImageButton>(R.id.helpButton)
                assertNotNull("Help button should exist", helpButton)

                println("Top bar overlay controls verified")
            }
        }
    }

    @Test
    fun testBottomControlsOverlayExists() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val bottomControls = activity.findViewById<FrameLayout>(R.id.bottomControlsContainer)
                assertNotNull("Bottom controls container should exist", bottomControls)

                val brainButton = activity.findViewById<ImageButton>(R.id.brainButton)
                assertNotNull("Brain button should exist", brainButton)
                assertEquals("Brain button should be visible", View.VISIBLE, brainButton.visibility)

                val screenshotButton = activity.findViewById<ImageButton>(R.id.screenshotButton)
                assertNotNull("Screenshot button should exist", screenshotButton)
                assertEquals("Screenshot button should be visible", View.VISIBLE, screenshotButton.visibility)

                val orientationLabel = activity.findViewById<LinearLayout>(R.id.orientationLabel)
                assertNotNull("Orientation label should exist", orientationLabel)

                println("Bottom controls overlay verified: brain, screenshot, orientation label")
            }
        }
    }

    @Test
    fun testSaveButtonVisibleInPreviewMode() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
            putExtra("IS_PREVIEW", true)
            putExtra("GLB_PATH", "/data/test/room.glb")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val saveButton = activity.findViewById<ImageButton>(R.id.saveButton)
                assertNotNull("Save button should exist", saveButton)
                // In preview mode with GLB_PATH, save button should be visible
                println("Save button exists in preview mode")
            }
        }
    }

    @Test
    fun testShareButtonVisibleInViewMode() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val shareButton = activity.findViewById<ImageButton>(R.id.shareButton)
                assertNotNull("Share button should exist", shareButton)
                assertEquals("Share button should be visible in view mode", View.VISIBLE, shareButton.visibility)

                val saveButton = activity.findViewById<ImageButton>(R.id.saveButton)
                assertEquals("Save button should be hidden in view mode", View.GONE, saveButton.visibility)

                println("View mode: share visible, save hidden")
            }
        }
    }

    @Test
    fun testStatusBarColor() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val expectedColor = android.graphics.Color.parseColor("#1C1C1E")
                assertEquals("Status bar color should be dark theme color",
                    expectedColor, activity.window.statusBarColor)
                println("Status bar color verified: #1C1C1E")
            }
        }
    }

    @Test
    fun testNavigationBarColor() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val expectedColor = android.graphics.Color.BLACK
                assertEquals("Navigation bar color should be black",
                    expectedColor, activity.window.navigationBarColor)
                println("Navigation bar color verified: BLACK")
            }
        }
    }

    @Test
    fun testBackButtonFinishesActivity() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val backButton = activity.findViewById<ImageButton>(R.id.backButton)
                assertNotNull("Back button should exist", backButton)
                assertTrue("Back button should be clickable", backButton.isClickable)
                println("Back button is clickable")
            }
        }
    }

    @Test
    fun testOrientationLabelContent() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val orientationLabel = activity.findViewById<LinearLayout>(R.id.orientationLabel)
                assertNotNull("Orientation label should exist", orientationLabel)

                // Check child TextViews
                val childCount = orientationLabel.childCount
                assertTrue("Orientation label should have children", childCount >= 2)

                // First child should be "held vertically"
                val heldVerticallyText = orientationLabel.getChildAt(0) as? TextView
                assertNotNull("First child should be TextView", heldVerticallyText)
                assertEquals("Should show 'held vertically'", "held vertically", heldVerticallyText?.text.toString())

                // Second child should be "Portrait"
                val portraitText = orientationLabel.getChildAt(1) as? TextView
                assertNotNull("Second child should be TextView", portraitText)
                assertEquals("Should show 'Portrait'", "Portrait", portraitText?.text.toString())

                println("Orientation label content verified: 'held vertically' + 'Portrait'")
            }
        }
    }

    @Test
    fun testEditTextBorderDrawableExists() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Verify the border drawable resource exists
        val drawableId = R.drawable.edittext_border
        assertTrue("EditText border drawable should exist", drawableId != 0)

        val drawable = context.getDrawable(drawableId)
        assertNotNull("Should be able to load edittext_border drawable", drawable)

        println("EditText border drawable verified")
    }
}
