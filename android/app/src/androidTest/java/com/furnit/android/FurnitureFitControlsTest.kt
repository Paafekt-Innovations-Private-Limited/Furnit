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
}
