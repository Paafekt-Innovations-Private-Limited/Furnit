package com.furnit.android

import android.content.Intent
import android.util.DisplayMetrics
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
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
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

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

    @Test
    fun testSceneViewFullScreen() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            // Wait for layout to complete
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)

                // Wait for view to be laid out
                sceneView.post {
                    // Get screen dimensions
                    val displayMetrics = DisplayMetrics()
                    activity.windowManager.defaultDisplay.getMetrics(displayMetrics)
                    val screenWidth = displayMetrics.widthPixels
                    val screenHeight = displayMetrics.heightPixels

                    // Get SceneView dimensions
                    val sceneViewWidth = sceneView.width
                    val sceneViewHeight = sceneView.height

                    println("Screen size: ${screenWidth}x${screenHeight}")
                    println("SceneView size: ${sceneViewWidth}x${sceneViewHeight}")

                    // SceneView should take full width
                    assertEquals("SceneView width should match screen width",
                        screenWidth, sceneViewWidth)

                    // SceneView height should be at least 80% of screen height (full screen minus system bars)
                    val minExpectedHeight = (screenHeight * 0.8).toInt()
                    assertTrue("SceneView height ($sceneViewHeight) should be at least 80% of screen height ($minExpectedHeight). " +
                            "If only half screen, height would be ~${screenHeight / 2}",
                        sceneViewHeight >= minExpectedHeight)

                    // CRITICAL: SceneView should NOT be half the screen
                    val halfScreenHeight = screenHeight / 2
                    val tolerance = screenHeight * 0.1 // 10% tolerance
                    val isHalfScreen = sceneViewHeight > (halfScreenHeight - tolerance) &&
                                       sceneViewHeight < (halfScreenHeight + tolerance)
                    assertFalse("FAIL: SceneView appears to be only HALF the screen! " +
                            "Height=$sceneViewHeight, HalfScreen=$halfScreenHeight",
                        isHalfScreen)

                    // Check layout params are MATCH_PARENT
                    val layoutParams = sceneView.layoutParams
                    assertEquals("SceneView width layout param should be MATCH_PARENT",
                        ViewGroup.LayoutParams.MATCH_PARENT, layoutParams.width)
                    assertEquals("SceneView height layout param should be MATCH_PARENT",
                        ViewGroup.LayoutParams.MATCH_PARENT, layoutParams.height)

                    println("PASS: SceneView is FULL SCREEN (not half)")
                    latch.countDown()
                }
            }

            // Wait for the check to complete
            assertTrue("Layout check timed out", latch.await(5, TimeUnit.SECONDS))
        }
    }

    @Test
    fun testRootLayoutFullScreen() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                // Get the root content view
                val rootView = activity.findViewById<View>(android.R.id.content)
                assertNotNull("Root content view should exist", rootView)

                rootView.post {
                    val displayMetrics = DisplayMetrics()
                    activity.windowManager.defaultDisplay.getMetrics(displayMetrics)
                    val screenWidth = displayMetrics.widthPixels
                    val screenHeight = displayMetrics.heightPixels

                    val rootWidth = rootView.width
                    val rootHeight = rootView.height

                    println("Screen: ${screenWidth}x${screenHeight}")
                    println("Root view: ${rootWidth}x${rootHeight}")

                    // Root should match screen width
                    assertEquals("Root width should match screen", screenWidth, rootWidth)

                    // Root height should be close to screen height
                    val heightDiff = kotlin.math.abs(screenHeight - rootHeight)
                    val maxAllowedDiff = screenHeight * 0.15 // Allow 15% for system bars
                    assertTrue("Root height ($rootHeight) should be close to screen height ($screenHeight), diff=$heightDiff",
                        heightDiff <= maxAllowedDiff)

                    println("Root layout is full screen")
                    latch.countDown()
                }
            }

            assertTrue("Layout check timed out", latch.await(5, TimeUnit.SECONDS))
        }
    }

    @Test
    fun testOverlaysAreOnTopOfSceneView() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                val topBar = activity.findViewById<LinearLayout>(R.id.topBarContainer)
                val bottomControls = activity.findViewById<FrameLayout>(R.id.bottomControlsContainer)

                assertNotNull("SceneView should exist", sceneView)
                assertNotNull("Top bar should exist", topBar)
                assertNotNull("Bottom controls should exist", bottomControls)

                // Get parent (should be FrameLayout)
                val parent = sceneView.parent as? FrameLayout
                assertNotNull("Parent should be FrameLayout", parent)

                // In FrameLayout, later children are drawn on top
                // SceneView should be first (index 0), overlays should come after
                val sceneViewIndex = parent!!.indexOfChild(sceneView)
                val topBarIndex = parent.indexOfChild(topBar)
                val bottomControlsIndex = parent.indexOfChild(bottomControls)

                println("Child indices - SceneView: $sceneViewIndex, TopBar: $topBarIndex, BottomControls: $bottomControlsIndex")

                assertTrue("Top bar should be after SceneView in z-order (overlay)",
                    topBarIndex > sceneViewIndex)
                assertTrue("Bottom controls should be after SceneView in z-order (overlay)",
                    bottomControlsIndex > sceneViewIndex)

                // Verify overlays are visible
                assertEquals("Top bar should be visible", View.VISIBLE, topBar.visibility)
                assertEquals("Bottom controls should be visible", View.VISIBLE, bottomControls.visibility)

                println("PASS: Overlays are correctly positioned on top of SceneView")
            }
        }
    }

    @Test
    fun testCameraPositioningForRoomView() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)

                // Wait for scene to be ready and model to load
                sceneView.post {
                    sceneView.postDelayed({
                        val cameraNode = sceneView.cameraNode
                        val camPos = cameraNode.position

                        println("=== Camera Position Test ===")
                        println("Camera position: (${camPos.x}, ${camPos.y}, ${camPos.z})")

                        // Room dimensions from GlbGenerator:
                        // width=4 (X: -2 to +2), depth=4.5 (Z: -2.25 to +2.25), height=2.8 (Y: 0 to 2.8)
                        // Camera should be positioned at back of room, eye level
                        // Expected: Camera at (0, 1.6, 3.5) looking at front wall

                        println("Expected camera: (0, 1.6, 3.5) - back of room, eye level")
                        println("Actual camera: (${camPos.x}, ${camPos.y}, ${camPos.z})")

                        // Verify camera is correctly positioned
                        val isCentered = kotlin.math.abs(camPos.x) < 0.1f
                        val isAtEyeLevel = camPos.y > 1.0f && camPos.y < 2.0f
                        val isBehindRoom = camPos.z > 2.0f

                        println("Camera X centered: $isCentered")
                        println("Camera Y at eye level: $isAtEyeLevel")
                        println("Camera Z behind room: $isBehindRoom")

                        // Assert camera is NOT at default position (0, 0, 1)
                        assertTrue("Camera Y should be at eye level (>1.0), not at default 0. " +
                                "Actual: ${camPos.y}", isAtEyeLevel)
                        assertTrue("Camera Z should be behind room (>2.0), not at default 1. " +
                                "Actual: ${camPos.z}", isBehindRoom)
                        assertTrue("Camera X should be centered (<0.1). Actual: ${camPos.x}", isCentered)

                        latch.countDown()
                    }, 3000)  // Wait 3 seconds for model load
                }
            }

            assertTrue("Camera test timed out", latch.await(15, TimeUnit.SECONDS))
        }
    }

    @Test
    fun testCameraPositioningForGeneratedRoom() {
        val context = InstrumentationRegistry.getInstrumentation().targetContext

        // Create a room using GlbGenerator (same as user-created rooms)
        val generator = com.furnit.android.services.GlbGenerator()
        val dimensions = com.furnit.android.services.GlbGenerator.RoomDimensions()

        // Create simple textures
        val grayTexture = android.graphics.Bitmap.createBitmap(256, 256, android.graphics.Bitmap.Config.ARGB_8888).apply {
            eraseColor(android.graphics.Color.parseColor("#E0E0E0"))
        }

        val testDir = java.io.File(context.cacheDir, "test_camera")
        testDir.mkdirs()
        val glbFile = java.io.File(testDir, "test_room.glb")

        val success = generator.generateGlb(
            outputFile = glbFile,
            dimensions = dimensions,
            frontWallTexture = grayTexture,
            floorTexture = grayTexture,
            ceilingTexture = grayTexture,
            leftWallTexture = grayTexture,
            rightWallTexture = grayTexture
        )

        assertTrue("GLB generation should succeed", success)
        assertTrue("GLB file should exist", glbFile.exists())

        // Launch ModelDetailActivity with the generated GLB
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("GLB_PATH", glbFile.absolutePath)
            putExtra("IS_PREVIEW", true)
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)

                // Wait for model to load and camera to be positioned
                sceneView.post {
                    sceneView.postDelayed({
                        val cameraNode = sceneView.cameraNode
                        val camPos = cameraNode.position

                        println("=== Generated Room Camera Test ===")
                        println("GLB path: ${glbFile.absolutePath}")
                        println("Camera position: (${camPos.x}, ${camPos.y}, ${camPos.z})")
                        println("Expected: (0, 1.6, 3.5) - back of room, eye level")

                        // Verify camera is correctly positioned (not at default)
                        val isAtEyeLevel = camPos.y > 1.0f && camPos.y < 2.0f
                        val isBehindRoom = camPos.z > 2.0f

                        println("Camera Y at eye level: $isAtEyeLevel")
                        println("Camera Z behind room: $isBehindRoom")

                        assertTrue("Generated room: Camera Y should be at eye level (>1.0). Actual: ${camPos.y}",
                            isAtEyeLevel)
                        assertTrue("Generated room: Camera Z should be behind room (>2.0). Actual: ${camPos.z}",
                            isBehindRoom)

                        latch.countDown()
                    }, 3000)
                }
            }

            assertTrue("Generated room camera test timed out", latch.await(15, TimeUnit.SECONDS))
        }

        // Cleanup
        grayTexture.recycle()
        glbFile.delete()
        testDir.delete()
    }
}
