package com.furnit.android

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.util.DisplayMetrics
import android.util.Log
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
import com.furnit.android.services.GlbGenerator
import io.github.sceneview.SceneView
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Tests for ModelDetailActivity - 3D Room Viewer
 * Verifies overlay controls, edge-to-edge display, camera positioning, and UI components.
 *
 * IMPORTANT: Uses @Before/@After for proper cleanup of test data.
 */
@RunWith(AndroidJUnit4::class)
class ModelDetailActivityTest {

    companion object {
        private const val TAG = "ModelDetailActivityTest"
    }

    private lateinit var context: android.content.Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
        Log.d(TAG, "=== Test Setup: Cleaning test data ===")

        // Clean up any leftover test data before each test
        TestCleanup.cleanAll(context)

        // List any remaining test data
        val remainingData = TestCleanup.listTestData(context)
        if (remainingData.isNotEmpty()) {
            Log.w(TAG, "Remaining test data after cleanup:")
            remainingData.forEach { Log.w(TAG, "  $it") }
        }
    }

    @After
    fun teardown() {
        Log.d(TAG, "=== Test Teardown: Cleaning test data ===")

        // Clean up test data after each test
        TestCleanup.cleanAll(context)
    }

    @Test
    fun testActivityLaunchesWithModelId() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                assertNotNull("Activity should not be null", activity)
                Log.d(TAG, "ModelDetailActivity launched successfully with model ID")
            }
        }
    }

    @Test
    fun testSceneViewExists() {
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
                Log.d(TAG, "SceneView found and configured")
            }
        }
    }

    @Test
    fun testTopBarOverlayExists() {
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

                Log.d(TAG, "Top bar overlay controls verified")
            }
        }
    }

    @Test
    fun testSaveButtonVisibleInPreviewMode() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
            putExtra("IS_PREVIEW", true)
            putExtra("GLB_PATH", "/data/test/room.glb")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val saveButton = activity.findViewById<ImageButton>(R.id.saveButton)
                assertNotNull("Save button should exist", saveButton)
                Log.d(TAG, "Save button exists in preview mode")
            }
        }
    }

    @Test
    fun testShareButtonVisibleInViewMode() {
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

                Log.d(TAG, "View mode: share visible, save hidden")
            }
        }
    }

    @Test
    fun testStatusBarColor() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val expectedColor = Color.parseColor("#1C1C1E")
                assertEquals("Status bar color should be dark theme color",
                    expectedColor, activity.window.statusBarColor)
                Log.d(TAG, "Status bar color verified: #1C1C1E")
            }
        }
    }

    @Test
    fun testNavigationBarColor() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val expectedColor = Color.BLACK
                assertEquals("Navigation bar color should be black",
                    expectedColor, activity.window.navigationBarColor)
                Log.d(TAG, "Navigation bar color verified: BLACK")
            }
        }
    }

    @Test
    fun testBackButtonFinishesActivity() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val backButton = activity.findViewById<ImageButton>(R.id.backButton)
                assertNotNull("Back button should exist", backButton)
                assertTrue("Back button should be clickable", backButton.isClickable)
                Log.d(TAG, "Back button is clickable")
            }
        }
    }

    @Test
    fun testEditTextBorderDrawableExists() {
        val drawableId = R.drawable.edittext_border
        assertTrue("EditText border drawable should exist", drawableId != 0)

        val drawable = context.getDrawable(drawableId)
        assertNotNull("Should be able to load edittext_border drawable", drawable)

        Log.d(TAG, "EditText border drawable verified")
    }

    @Test
    fun testSceneViewFullScreen() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)

                sceneView.post {
                    val displayMetrics = DisplayMetrics()
                    activity.windowManager.defaultDisplay.getMetrics(displayMetrics)
                    val screenWidth = displayMetrics.widthPixels
                    val screenHeight = displayMetrics.heightPixels

                    val sceneViewWidth = sceneView.width
                    val sceneViewHeight = sceneView.height

                    Log.d(TAG, "Screen size: ${screenWidth}x${screenHeight}")
                    Log.d(TAG, "SceneView size: ${sceneViewWidth}x${sceneViewHeight}")

                    assertEquals("SceneView width should match screen width",
                        screenWidth, sceneViewWidth)

                    val minExpectedHeight = (screenHeight * 0.8).toInt()
                    assertTrue("SceneView height ($sceneViewHeight) should be at least 80% of screen height ($minExpectedHeight). " +
                            "If only half screen, height would be ~${screenHeight / 2}",
                        sceneViewHeight >= minExpectedHeight)

                    // CRITICAL: SceneView should NOT be half the screen
                    val halfScreenHeight = screenHeight / 2
                    val tolerance = screenHeight * 0.1
                    val isHalfScreen = sceneViewHeight > (halfScreenHeight - tolerance) &&
                                       sceneViewHeight < (halfScreenHeight + tolerance)
                    assertFalse("FAIL: SceneView appears to be only HALF the screen! " +
                            "Height=$sceneViewHeight, HalfScreen=$halfScreenHeight",
                        isHalfScreen)

                    val layoutParams = sceneView.layoutParams
                    assertEquals("SceneView width layout param should be MATCH_PARENT",
                        ViewGroup.LayoutParams.MATCH_PARENT, layoutParams.width)
                    assertEquals("SceneView height layout param should be MATCH_PARENT",
                        ViewGroup.LayoutParams.MATCH_PARENT, layoutParams.height)

                    Log.d(TAG, "PASS: SceneView is FULL SCREEN (not half)")
                    latch.countDown()
                }
            }

            assertTrue("Layout check timed out", latch.await(5, TimeUnit.SECONDS))
        }
    }

    @Test
    fun testRootLayoutFullScreen() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val rootView = activity.findViewById<View>(android.R.id.content)
                assertNotNull("Root content view should exist", rootView)

                rootView.post {
                    val displayMetrics = DisplayMetrics()
                    activity.windowManager.defaultDisplay.getMetrics(displayMetrics)
                    val screenWidth = displayMetrics.widthPixels
                    val screenHeight = displayMetrics.heightPixels

                    val rootWidth = rootView.width
                    val rootHeight = rootView.height

                    Log.d(TAG, "Screen: ${screenWidth}x${screenHeight}")
                    Log.d(TAG, "Root view: ${rootWidth}x${rootHeight}")

                    assertEquals("Root width should match screen", screenWidth, rootWidth)

                    val heightDiff = kotlin.math.abs(screenHeight - rootHeight)
                    val maxAllowedDiff = screenHeight * 0.15
                    assertTrue("Root height ($rootHeight) should be close to screen height ($screenHeight), diff=$heightDiff",
                        heightDiff <= maxAllowedDiff)

                    Log.d(TAG, "Root layout is full screen")
                    latch.countDown()
                }
            }

            assertTrue("Layout check timed out", latch.await(5, TimeUnit.SECONDS))
        }
    }

    @Test
    fun testTopBarOverlayOnTopOfSceneView() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                val topBar = activity.findViewById<LinearLayout>(R.id.topBarContainer)

                assertNotNull("SceneView should exist", sceneView)
                assertNotNull("Top bar should exist", topBar)

                val parent = sceneView.parent as? FrameLayout
                assertNotNull("Parent should be FrameLayout", parent)

                val sceneViewIndex = parent!!.indexOfChild(sceneView)
                val topBarIndex = parent.indexOfChild(topBar)

                Log.d(TAG, "Child indices - SceneView: $sceneViewIndex, TopBar: $topBarIndex")

                assertTrue("Top bar should be after SceneView in z-order (overlay)",
                    topBarIndex > sceneViewIndex)

                assertEquals("Top bar should be visible", View.VISIBLE, topBar.visibility)

                Log.d(TAG, "PASS: Top bar correctly positioned on top of SceneView")
            }
        }
    }

    @Test
    fun testCameraPositioningForRoomView() {
        val intent = Intent(context, ModelDetailActivity::class.java).apply {
            putExtra("MODEL_ID", "vintage")
        }

        ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
            val latch = CountDownLatch(1)

            scenario.onActivity { activity ->
                val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                assertNotNull("SceneView should exist", sceneView)

                sceneView.post {
                    sceneView.postDelayed({
                        val cameraNode = sceneView.cameraNode
                        val camPos = cameraNode.position

                        Log.d(TAG, "=== Camera Position Test (Vintage Room) ===")
                        Log.d(TAG, "Camera position: (${camPos.x}, ${camPos.y}, ${camPos.z})")
                        Log.d(TAG, "Expected: back of room, eye level (~1.6), behind room (>2.0)")

                        val isCentered = kotlin.math.abs(camPos.x) < 0.5f
                        val isAtEyeLevel = camPos.y > 1.0f && camPos.y < 2.5f
                        val isBehindRoom = camPos.z > 2.0f

                        // CRITICAL ASSERTIONS - These should FAIL if camera isn't positioned correctly
                        assertFalse("FAIL: Camera Y is at default position (0). " +
                                "Camera should be at eye level. Actual: ${camPos.y}",
                            camPos.y < 0.5f)

                        assertFalse("FAIL: Camera Z is at default position (1). " +
                                "Camera should be behind room. Actual: ${camPos.z}",
                            camPos.z < 1.5f)

                        assertTrue("Camera Y should be at eye level (1.0-2.5). Actual: ${camPos.y}", isAtEyeLevel)
                        assertTrue("Camera Z should be behind room (>2.0). Actual: ${camPos.z}", isBehindRoom)
                        assertTrue("Camera X should be roughly centered. Actual: ${camPos.x}", isCentered)

                        Log.d(TAG, "PASS: Camera positioned correctly")
                        latch.countDown()
                    }, 3000)
                }
            }

            assertTrue("Camera test timed out", latch.await(15, TimeUnit.SECONDS))
        }
    }

    /**
     * CRITICAL TEST: Tests camera positioning for user-generated rooms.
     * This is the test that should catch the "half screen" issue.
     *
     * The issue was that camera position was being reset by SceneView's manipulator
     * after we set it. The fix was to set camera position IMMEDIATELY after adding
     * the model, not in a delayed callback.
     */
    @Test
    fun testCameraPositioningForGeneratedRoom() {
        Log.d(TAG, "=== Generated Room Camera Test START ===")

        // Create a room using GlbGenerator (same as user-created rooms)
        val generator = GlbGenerator()
        val dimensions = GlbGenerator.RoomDimensions()

        // Create simple textures
        val grayTexture = Bitmap.createBitmap(256, 256, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.parseColor("#E0E0E0"))
        }

        val testDir = File(context.cacheDir, "test_camera")
        testDir.mkdirs()
        val glbFile = File(testDir, "test_room.glb")

        Log.d(TAG, "Creating test GLB at: ${glbFile.absolutePath}")

        try {
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
            Log.d(TAG, "GLB file created: ${glbFile.length()} bytes")

            // Launch ModelDetailActivity with the generated GLB
            val intent = Intent(context, ModelDetailActivity::class.java).apply {
                putExtra("GLB_PATH", glbFile.absolutePath)
                putExtra("IS_PREVIEW", true)
            }

            ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
                val latch = CountDownLatch(1)
                var testError: AssertionError? = null

                scenario.onActivity { activity ->
                    val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                    assertNotNull("SceneView should exist", sceneView)

                    sceneView.post {
                        sceneView.postDelayed({
                            try {
                                val cameraNode = sceneView.cameraNode
                                val camPos = cameraNode.position

                                Log.d(TAG, "=== Camera Position Check ===")
                                Log.d(TAG, "GLB path: ${glbFile.absolutePath}")
                                Log.d(TAG, "Camera position: (${camPos.x}, ${camPos.y}, ${camPos.z})")
                                Log.d(TAG, "Expected: X~0, Y~1.6 (eye level), Z>2.0 (behind room)")

                                // CRITICAL: Detect default camera position which causes half-screen issue
                                val isDefaultPosition = camPos.y < 0.5f && camPos.z < 1.5f
                                if (isDefaultPosition) {
                                    Log.e(TAG, "FAIL: Camera at DEFAULT position (0, 0, 1)!")
                                    Log.e(TAG, "This causes the room to appear at half screen.")
                                }

                                assertFalse("Camera should NOT be at default Y position (near 0). " +
                                        "This causes half-screen display. Actual Y: ${camPos.y}",
                                    camPos.y < 0.5f)

                                assertFalse("Camera should NOT be at default Z position (near 1). " +
                                        "Camera should be behind room. Actual Z: ${camPos.z}",
                                    camPos.z < 1.5f)

                                val isAtEyeLevel = camPos.y > 1.0f && camPos.y < 2.5f
                                val isBehindRoom = camPos.z > 2.0f

                                assertTrue("Generated room: Camera Y should be at eye level (1.0-2.5). Actual: ${camPos.y}",
                                    isAtEyeLevel)
                                assertTrue("Generated room: Camera Z should be behind room (>2.0). Actual: ${camPos.z}",
                                    isBehindRoom)

                                Log.d(TAG, "PASS: Generated room camera positioned correctly")
                            } catch (e: AssertionError) {
                                testError = e
                            } finally {
                                latch.countDown()
                            }
                        }, 3000)
                    }
                }

                assertTrue("Generated room camera test timed out", latch.await(15, TimeUnit.SECONDS))
                testError?.let { throw it }
            }
        } finally {
            // Cleanup - always runs
            Log.d(TAG, "Cleaning up test files")
            grayTexture.recycle()
            if (glbFile.exists()) glbFile.delete()
            if (testDir.exists()) testDir.delete()
        }
    }

    /**
     * Test that simulates the complete user flow:
     * 1. Create room via GlbGenerator
     * 2. Open in ModelDetailActivity (preview mode)
     * 3. Verify camera is positioned correctly (not at default)
     * 4. Verify SceneView fills the screen
     */
    @Test
    fun testCompleteRoomCreationAndViewingFlow() {
        Log.d(TAG, "=== Complete Room Flow Test START ===")

        // Step 1: Create a room (simulating SinglePhotoRoomReconstructor output)
        val generator = GlbGenerator()
        val dimensions = GlbGenerator.RoomDimensions(
            width = 4.0f,
            depth = 4.5f,
            height = 2.8f
        )

        val testTexture = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.parseColor("#D4C4B0"))  // Beige color like room walls
        }

        val roomsDir = File(context.filesDir, "rooms")
        roomsDir.mkdirs()
        val roomFolder = File(roomsDir, "room_test_${System.currentTimeMillis()}")
        roomFolder.mkdirs()
        val glbFile = File(roomFolder, "room.glb")

        try {
            val success = generator.generateGlb(
                outputFile = glbFile,
                dimensions = dimensions,
                frontWallTexture = testTexture,
                floorTexture = testTexture,
                ceilingTexture = testTexture,
                leftWallTexture = testTexture,
                rightWallTexture = testTexture
            )

            assertTrue("Room GLB generation should succeed", success)
            assertTrue("Room GLB file should exist", glbFile.exists())
            Log.d(TAG, "Room created at: ${glbFile.absolutePath}")

            // Step 2: Open in ModelDetailActivity
            val intent = Intent(context, ModelDetailActivity::class.java).apply {
                putExtra("GLB_PATH", glbFile.absolutePath)
                putExtra("IS_PREVIEW", true)
            }

            ActivityScenario.launch<ModelDetailActivity>(intent).use { scenario ->
                val latch = CountDownLatch(1)
                var testError: AssertionError? = null

                scenario.onActivity { activity ->
                    val sceneView = activity.findViewById<SceneView>(R.id.sceneView)
                    assertNotNull("SceneView should exist", sceneView)

                    sceneView.post {
                        sceneView.postDelayed({
                            try {
                                // Step 3: Verify camera position
                                val camPos = sceneView.cameraNode.position
                                Log.d(TAG, "Camera position: (${camPos.x}, ${camPos.y}, ${camPos.z})")

                                assertFalse("Camera Y should not be at default (0). Actual: ${camPos.y}",
                                    camPos.y < 0.5f)
                                assertFalse("Camera Z should not be at default (1). Actual: ${camPos.z}",
                                    camPos.z < 1.5f)

                                // Step 4: Verify SceneView fills screen
                                val displayMetrics = DisplayMetrics()
                                activity.windowManager.defaultDisplay.getMetrics(displayMetrics)
                                val screenHeight = displayMetrics.heightPixels
                                val sceneViewHeight = sceneView.height

                                Log.d(TAG, "SceneView height: $sceneViewHeight, Screen height: $screenHeight")

                                val halfScreen = screenHeight / 2
                                val isHalfScreen = sceneViewHeight < (halfScreen * 1.2)
                                assertFalse("SceneView should NOT be half screen. Height: $sceneViewHeight, HalfScreen: $halfScreen",
                                    isHalfScreen)

                                Log.d(TAG, "PASS: Complete room flow test passed")
                            } catch (e: AssertionError) {
                                testError = e
                            } finally {
                                latch.countDown()
                            }
                        }, 3000)
                    }
                }

                assertTrue("Test timed out", latch.await(15, TimeUnit.SECONDS))
                testError?.let { throw it }
            }
        } finally {
            // Cleanup
            testTexture.recycle()
            if (glbFile.exists()) glbFile.delete()
            if (roomFolder.exists()) roomFolder.deleteRecursively()
            Log.d(TAG, "Test cleanup complete")
        }
    }
}
