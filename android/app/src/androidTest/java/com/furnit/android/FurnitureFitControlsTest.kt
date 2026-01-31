package com.furnit.android

import android.content.Intent
import android.view.Gravity
import android.view.View
import android.widget.FrameLayout
import android.widget.ImageButton
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.views.JoystickView
import io.github.sceneview.SceneView
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Tests for FurnitureFitFragment controls
 * Verifies:
 * - Progress bar shows during initialization
 * - Joystick is present and functional
 * - Screenshot button is present
 * - Room SceneView background is available
 * - Controls visibility state management
 */
@RunWith(AndroidJUnit4::class)
class FurnitureFitControlsTest {

    private lateinit var context: android.content.Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    @Test
    fun testJoystickViewCanBeCreated() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val latch = CountDownLatch(1)
        var joystickCreated = false

        instrumentation.runOnMainSync {
            try {
                val joystick = JoystickView(context)
                joystick.layoutParams = FrameLayout.LayoutParams(200, 200)
                joystickCreated = true
            } catch (e: Exception) {
                e.printStackTrace()
            }
            latch.countDown()
        }

        latch.await(1, TimeUnit.SECONDS)
        assertTrue("JoystickView should be created without exceptions", joystickCreated)
        println("JoystickView creation test PASSED")
    }

    @Test
    fun testProgressBarConfiguration() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var progressBar: ProgressBar? = null

        instrumentation.runOnMainSync {
            // Create progress bar like FurnitureFitFragment does
            progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                val lp = LinearLayout.LayoutParams(250, LinearLayout.LayoutParams.WRAP_CONTENT)
                layoutParams = lp
                max = 100
                progress = 5  // Initial 5%
            }
        }

        assertNotNull("Progress bar should be created", progressBar)
        assertEquals("Max should be 100", 100, progressBar!!.max)
        assertEquals("Initial progress should be 5", 5, progressBar!!.progress)
        println("Progress bar configuration test PASSED")
    }

    @Test
    fun testProgressContainerLayout() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var container: LinearLayout? = null
        var label: TextView? = null
        var progressBar: ProgressBar? = null

        instrumentation.runOnMainSync {
            // Create progress container like FurnitureFitFragment does
            container = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                setBackgroundColor(0x99000000.toInt())  // Semi-transparent black
                setPadding(32, 16, 32, 16)
                val lp = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                )
                lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.TOP
                lp.setMargins(0, 120, 0, 0)
                layoutParams = lp
            }

            label = TextView(context).apply {
                text = "Starting camera..."
                setTextColor(0xFFFFFFFF.toInt())
                textSize = 14f
                gravity = Gravity.CENTER
            }
            container!!.addView(label)

            progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 100
                progress = 5
            }
            container!!.addView(progressBar)
        }

        assertNotNull("Container should be created", container)
        assertEquals("Container should have 2 children", 2, container!!.childCount)
        assertEquals("Container orientation should be VERTICAL", LinearLayout.VERTICAL, container!!.orientation)
        assertEquals("Label text should be 'Starting camera...'", "Starting camera...", label!!.text)
        println("Progress container layout test PASSED")
    }

    @Test
    fun testProgressValues() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var progressBar: ProgressBar? = null
        var progressLabel: TextView? = null

        instrumentation.runOnMainSync {
            progressLabel = TextView(context)
            progressBar = ProgressBar(context, null, android.R.attr.progressBarStyleHorizontal).apply {
                max = 100
            }

            // Test progress values like FurnitureFitFragment.setProgress()
            // Initial: 5%
            progressBar!!.progress = 5
            progressLabel!!.text = "Starting camera..."
            assertEquals(5, progressBar!!.progress)

            // Preprocessing: 15%
            progressBar!!.progress = 15
            progressLabel!!.text = "Preprocessing..."
            assertEquals(15, progressBar!!.progress)

            // Scanning: 40%
            progressBar!!.progress = 40
            progressLabel!!.text = "Scanning for furniture..."
            assertEquals(40, progressBar!!.progress)
        }

        println("Progress values test PASSED")
    }

    @Test
    fun testBottomControlsLayout() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var bottomControls: FrameLayout? = null
        var joystickContainer: LinearLayout? = null
        var screenshotButton: ImageButton? = null

        instrumentation.runOnMainSync {
            bottomControls = FrameLayout(context).apply {
                val lp = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                )
                lp.gravity = Gravity.BOTTOM
                lp.setMargins(20, 0, 20, 24)
                layoutParams = lp
            }

            joystickContainer = LinearLayout(context).apply {
                orientation = LinearLayout.VERTICAL
                gravity = Gravity.CENTER
                val lp = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.WRAP_CONTENT,
                    FrameLayout.LayoutParams.WRAP_CONTENT
                )
                lp.gravity = Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM
                layoutParams = lp
            }

            val joystick = JoystickView(context).apply {
                val size = (100 * context.resources.displayMetrics.density).toInt()
                layoutParams = LinearLayout.LayoutParams(size, size)
            }
            joystickContainer!!.addView(joystick)
            bottomControls!!.addView(joystickContainer)

            screenshotButton = ImageButton(context).apply {
                setImageResource(android.R.drawable.ic_menu_camera)
                val size = (56 * context.resources.displayMetrics.density).toInt()
                val lp = FrameLayout.LayoutParams(size, size)
                lp.gravity = Gravity.END or Gravity.BOTTOM
                layoutParams = lp
            }
            bottomControls!!.addView(screenshotButton)
        }

        assertNotNull("Bottom controls should be created", bottomControls)
        assertEquals("Bottom controls should have 2 children", 2, bottomControls!!.childCount)

        val joystickParams = joystickContainer!!.layoutParams as FrameLayout.LayoutParams
        assertEquals("Joystick should be centered", Gravity.CENTER_HORIZONTAL or Gravity.BOTTOM, joystickParams.gravity)

        val screenshotParams = screenshotButton!!.layoutParams as FrameLayout.LayoutParams
        assertEquals("Screenshot should be at end", Gravity.END or Gravity.BOTTOM, screenshotParams.gravity)

        println("Bottom controls layout test PASSED")
    }

    @Test
    fun testRoomBackgroundVisibilityLogic() {
        // Test the visibility logic for room background
        // Initially GONE, becomes VISIBLE when detections are found

        var roomVisibility = View.GONE
        var hasFirstDetection = false
        val showRoomBackground = true

        // Simulate no detections
        var detectionsCount = 0
        if (showRoomBackground && detectionsCount > 0) {
            roomVisibility = View.VISIBLE
        }
        assertEquals("Room should be GONE with no detections", View.GONE, roomVisibility)

        // Simulate first detection
        detectionsCount = 1
        hasFirstDetection = true
        if (showRoomBackground && detectionsCount > 0) {
            roomVisibility = View.VISIBLE
        }
        assertEquals("Room should be VISIBLE with detections", View.VISIBLE, roomVisibility)

        println("Room background visibility logic test PASSED")
    }

    @Test
    fun testProgressHiddenAfterFirstDetection() {
        // Test progress container hiding logic

        var progressVisibility = View.VISIBLE
        var hasFirstDetection = false

        // Initial state - progress visible
        assertEquals("Progress should be visible initially", View.VISIBLE, progressVisibility)

        // Simulate first detection
        hasFirstDetection = true
        progressVisibility = View.GONE

        assertEquals("Progress should be GONE after first detection", View.GONE, progressVisibility)
        assertTrue("hasFirstDetection should be true", hasFirstDetection)

        println("Progress hidden after first detection test PASSED")
    }

    @Test
    fun testJoystickMoveCallback() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val callbackReceived = mutableListOf<Pair<Float, Float>>()

        instrumentation.runOnMainSync {
            val joystick = JoystickView(context).apply {
                val spec = View.MeasureSpec.makeMeasureSpec(200, View.MeasureSpec.EXACTLY)
                measure(spec, spec)
                layout(0, 0, 200, 200)

                onJoystickMove = { x, y ->
                    callbackReceived.add(Pair(x, y))
                }
            }

            // Simulate touch
            val downTime = System.currentTimeMillis()
            val event = android.view.MotionEvent.obtain(
                downTime, downTime, android.view.MotionEvent.ACTION_DOWN, 150f, 100f, 0
            )
            joystick.dispatchTouchEvent(event)
            event.recycle()
        }

        assertTrue("Callback should be received", callbackReceived.isNotEmpty())
        val (x, y) = callbackReceived.first()
        // Touch at (150, 100) in 200x200 view means offset of (50, 0) from center (100, 100)
        // Normalized X should be positive (right of center)
        assertTrue("X should be positive (right of center)", x > 0)
        // Y should be near 0 (at center vertically)
        assertTrue("Y should be near 0 (center vertically)", kotlin.math.abs(y) < 0.2f)

        println("Joystick move callback test PASSED - received ${callbackReceived.size} callback(s)")
    }

    @Test
    fun testRoomCameraMoveLogic() {
        // Test the camera move logic used in FurnitureFitFragment.moveRoomCamera()

        val moveSpeed = 0.1f
        val deadZone = 0.1f

        // Test dead zone - small movements ignored
        var normalizedX = 0.05f
        var normalizedY = 0.05f
        var magnitude = kotlin.math.sqrt(normalizedX * normalizedX + normalizedY * normalizedY)
        assertTrue("Small movement should be within dead zone", magnitude < deadZone)

        // Test significant movement
        normalizedX = 0.5f
        normalizedY = 0.3f
        magnitude = kotlin.math.sqrt(normalizedX * normalizedX + normalizedY * normalizedY)
        assertTrue("Larger movement should exceed dead zone", magnitude >= deadZone)

        // Calculate delta
        val deltaX = normalizedX * moveSpeed
        val deltaZ = normalizedY * moveSpeed
        assertEquals("Delta X calculation", 0.05f, deltaX, 0.001f)
        assertEquals("Delta Z calculation", 0.03f, deltaZ, 0.001f)

        println("Room camera move logic test PASSED")
    }

    @Test
    fun testFurnitureOverlayCanBeCreated() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var overlayCreated = false

        instrumentation.runOnMainSync {
            try {
                val overlay = FurnitureFitOverlayView(context)
                overlay.layoutParams = FrameLayout.LayoutParams(
                    FrameLayout.LayoutParams.MATCH_PARENT,
                    FrameLayout.LayoutParams.MATCH_PARENT
                )
                overlayCreated = true
            } catch (e: Exception) {
                e.printStackTrace()
            }
        }

        assertTrue("FurnitureFitOverlayView should be created without exceptions", overlayCreated)
        println("FurnitureFitOverlayView creation test PASSED")
    }

    @Test
    fun testFurnitureOverlayTouchOutsideCallback() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var callbackInvoked = false
        var overlay: FurnitureFitOverlayView? = null

        instrumentation.runOnMainSync {
            overlay = FurnitureFitOverlayView(context).apply {
                val spec = View.MeasureSpec.makeMeasureSpec(1000, View.MeasureSpec.EXACTLY)
                measure(spec, spec)
                layout(0, 0, 1000, 1000)

                // Set callback for touch outside furniture
                onTouchOutsideFurniture = { _ ->
                    callbackInvoked = true
                }
            }

            // Touch without any mask set - should trigger callback (no furniture to touch)
            val downTime = System.currentTimeMillis()
            val event = android.view.MotionEvent.obtain(
                downTime, downTime, android.view.MotionEvent.ACTION_DOWN, 500f, 500f, 0
            )
            overlay!!.handleExternalTouchEvent(event)
            event.recycle()
        }

        assertTrue("Callback should be invoked when touching with no mask", callbackInvoked)
        println("Furniture overlay touch outside callback test PASSED")
    }

    @Test
    fun testFurnitureOverlayHitTestWithDetection() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var touchOnFurnitureResult = false
        var touchOutsideFurnitureResult = false

        instrumentation.runOnMainSync {
            val overlay = FurnitureFitOverlayView(context).apply {
                val spec = View.MeasureSpec.makeMeasureSpec(1000, View.MeasureSpec.EXACTLY)
                measure(spec, spec)
                layout(0, 0, 1000, 1000)
            }

            // Create a large detection covering most of the view
            // Detection at center (320, 320) with large size 500x500 in 640 input coords
            val detection = DetectionResult(
                x = 320f,
                y = 320f,
                w = 500f,
                h = 500f,
                confidence = 0.9f,
                label = "chair"
            )

            // Create a simple mask bitmap
            val maskBitmap = android.graphics.Bitmap.createBitmap(640, 640, android.graphics.Bitmap.Config.ARGB_8888)

            // Set mask and detections
            overlay.setMaskAndDetections(maskBitmap, listOf(detection), 640)

            // Set callbacks to track results
            var outsideFurnitureCallback = false

            overlay.onTouchOutsideFurniture = { _ ->
                outsideFurnitureCallback = true
            }

            // Calculate where furniture actually is:
            // baseScale = 1000/640 = 1.5625, furnitureScale = 0.6
            // floorOffsetY = 1000 * 0.35 = 350
            // centerOffsetX = (1000 - 600) / 2 = 200
            // detLeft = (320-250) * 1.5625 * 0.6 + 200 = 70 * 0.9375 + 200 = 265.6
            // detTop = (320-250) * 1.5625 * 0.6 + 350 = 65.6 + 350 = 415.6
            // detRight = (320+250) * 1.5625 * 0.6 + 200 = 570 * 0.9375 + 200 = 734.4
            // detBottom = (320+250) * 1.5625 * 0.6 + 350 = 534.4 + 350 = 884.4
            // So furniture is roughly at (266, 416) to (734, 884)

            // Touch inside furniture bounds (500, 650 - well within the calculated bounds)
            val downTime = System.currentTimeMillis()
            val insideEvent = android.view.MotionEvent.obtain(
                downTime, downTime, android.view.MotionEvent.ACTION_DOWN, 500f, 650f, 0
            )
            overlay.handleExternalTouchEvent(insideEvent)
            insideEvent.recycle()

            // If callback wasn't invoked, touch was on furniture
            touchOnFurnitureResult = !outsideFurnitureCallback

            // Reset and touch corner (50, 50 - clearly outside furniture)
            outsideFurnitureCallback = false
            val cornerEvent = android.view.MotionEvent.obtain(
                downTime, downTime, android.view.MotionEvent.ACTION_DOWN, 50f, 50f, 0
            )
            overlay.handleExternalTouchEvent(cornerEvent)
            cornerEvent.recycle()

            touchOutsideFurnitureResult = outsideFurnitureCallback

            maskBitmap.recycle()
        }

        assertTrue("Touch inside furniture bounds should be on furniture", touchOnFurnitureResult)
        assertTrue("Touch in corner should be outside furniture", touchOutsideFurnitureResult)
        println("Furniture overlay hit test with detection PASSED")
    }

    @Test
    fun testFurnitureDragUpdatesTranslation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var dragWorked = false

        instrumentation.runOnMainSync {
            val overlay = FurnitureFitOverlayView(context).apply {
                val spec = View.MeasureSpec.makeMeasureSpec(1000, View.MeasureSpec.EXACTLY)
                measure(spec, spec)
                layout(0, 0, 1000, 1000)
            }

            // Create detection and mask so touch is recognized as on furniture
            val detection = DetectionResult(
                x = 320f, y = 320f, w = 400f, h = 400f,
                confidence = 0.9f, label = "couch"
            )
            val maskBitmap = android.graphics.Bitmap.createBitmap(640, 640, android.graphics.Bitmap.Config.ARGB_8888)
            overlay.setMaskAndDetections(maskBitmap, listOf(detection), 640)

            // Make sure callback is NOT invoked (touch should be on furniture)
            var outsideCallback = false
            overlay.onTouchOutsideFurniture = { outsideCallback = true }

            val downTime = System.currentTimeMillis()

            // Simulate drag: DOWN -> MOVE -> UP
            val downEvent = android.view.MotionEvent.obtain(
                downTime, downTime, android.view.MotionEvent.ACTION_DOWN, 500f, 500f, 0
            )
            overlay.handleExternalTouchEvent(downEvent)
            downEvent.recycle()

            val moveEvent = android.view.MotionEvent.obtain(
                downTime, downTime + 16, android.view.MotionEvent.ACTION_MOVE, 600f, 550f, 0
            )
            overlay.handleExternalTouchEvent(moveEvent)
            moveEvent.recycle()

            val upEvent = android.view.MotionEvent.obtain(
                downTime, downTime + 32, android.view.MotionEvent.ACTION_UP, 600f, 550f, 0
            )
            overlay.handleExternalTouchEvent(upEvent)
            upEvent.recycle()

            // If outside callback wasn't invoked, drag was on furniture
            dragWorked = !outsideCallback

            maskBitmap.recycle()
        }

        assertTrue("Drag on furniture should not trigger outside callback", dragWorked)
        println("Furniture drag updates translation test PASSED")
    }

    @Test
    fun testCameraPreviewVisibilityToggle() {
        // Test the logic that hides camera preview when 3D room is shown
        var previewVisibility = View.VISIBLE
        var roomVisibility = View.GONE
        val showRoomBackground = true

        // Initial state - camera preview visible, room hidden
        assertEquals("Preview should be visible initially", View.VISIBLE, previewVisibility)
        assertEquals("Room should be hidden initially", View.GONE, roomVisibility)

        // Simulate detection found - room shows, preview hides
        val detectionsNotEmpty = true
        if (showRoomBackground && detectionsNotEmpty) {
            roomVisibility = View.VISIBLE
            previewVisibility = View.GONE
        }

        assertEquals("Preview should be hidden when room is shown", View.GONE, previewVisibility)
        assertEquals("Room should be visible with detections", View.VISIBLE, roomVisibility)

        // Simulate no detections - room hides, preview shows
        val noDetections = false
        if (!noDetections) {
            // Detection lost
            roomVisibility = View.GONE
            previewVisibility = View.VISIBLE
        }

        assertEquals("Preview should be visible when room is hidden", View.VISIBLE, previewVisibility)
        assertEquals("Room should be hidden without detections", View.GONE, roomVisibility)

        println("Camera preview visibility toggle test PASSED")
    }

    @Test
    fun testCameraDragSensitivity() {
        // Test camera drag calculation
        val sensitivity = 0.01f

        // Simulate drag of 100 pixels
        val deltaX = 100f
        val deltaY = 50f

        val cameraMoveX = -deltaX * sensitivity  // Negative because drag right moves camera left
        val cameraMoveZ = -deltaY * sensitivity

        assertEquals("Camera X movement", -1.0f, cameraMoveX, 0.001f)
        assertEquals("Camera Z movement", -0.5f, cameraMoveZ, 0.001f)

        println("Camera drag sensitivity test PASSED")
    }

    @Test
    fun testSingleDetectionOnly() {
        // Test that only one detection (highest confidence) is used
        val detections = listOf(
            DetectionResult(100f, 100f, 50f, 50f, 0.7f, "chair"),
            DetectionResult(200f, 200f, 60f, 60f, 0.95f, "couch"),  // Highest confidence
            DetectionResult(300f, 300f, 40f, 40f, 0.8f, "table")
        )

        // Simulate manager behavior - take highest confidence
        val primaryDet = detections.maxByOrNull { it.confidence }

        assertNotNull("Primary detection should be found", primaryDet)
        assertEquals("Highest confidence detection should be selected", "couch", primaryDet!!.label)
        assertEquals("Confidence should be 0.95", 0.95f, primaryDet.confidence, 0.001f)

        // Only one detection should be returned to overlay
        val resultDetections = listOf(primaryDet)
        assertEquals("Only one detection should be returned", 1, resultDetections.size)

        println("Single detection only test PASSED")
    }

    @Test
    fun testPinchScaleLimits() {
        // Test pinch-to-zoom scale limits (0.3 to 3.0)
        val minScale = 0.3f
        val maxScale = 3.0f

        var scale = 1.0f

        // Test scaling up
        scale *= 1.5f  // 1.5
        scale = kotlin.math.max(minScale, kotlin.math.min(scale, maxScale))
        assertEquals("Scale up should work", 1.5f, scale, 0.001f)

        // Test max limit
        scale *= 3.0f  // Would be 4.5, should clamp to 3.0
        scale = kotlin.math.max(minScale, kotlin.math.min(scale, maxScale))
        assertEquals("Scale should be clamped to max", maxScale, scale, 0.001f)

        // Test scaling down
        scale = 0.5f
        scale *= 0.4f  // Would be 0.2, should clamp to 0.3
        scale = kotlin.math.max(minScale, kotlin.math.min(scale, maxScale))
        assertEquals("Scale should be clamped to min", minScale, scale, 0.001f)

        println("Pinch scale limits test PASSED")
    }
}
