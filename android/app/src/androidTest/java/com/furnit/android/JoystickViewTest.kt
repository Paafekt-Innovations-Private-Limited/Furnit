package com.furnit.android

import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.furnit.android.views.JoystickView
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Tests for JoystickView - Virtual joystick for camera control
 * Verifies:
 * - Touch handling (ACTION_DOWN, ACTION_MOVE, ACTION_UP)
 * - Normalized output values (-1 to 1)
 * - Knob constraint within maxDistance
 * - Reset to center on release
 * - Callback invocation
 */
@RunWith(AndroidJUnit4::class)
class JoystickViewTest {

    private lateinit var context: android.content.Context
    private lateinit var joystickView: JoystickView

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    private fun createJoystickView(): JoystickView {
        val joystick = JoystickView(context)
        joystick.layoutParams = FrameLayout.LayoutParams(200, 200)
        // Force layout to set dimensions
        val spec = View.MeasureSpec.makeMeasureSpec(200, View.MeasureSpec.EXACTLY)
        joystick.measure(spec, spec)
        joystick.layout(0, 0, 200, 200)
        return joystick
    }

    @Test
    fun testJoystickViewCreation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        instrumentation.runOnMainSync {
            joystickView = createJoystickView()
            assertNotNull("JoystickView should be created", joystickView)
            assertEquals("Width should be set", 200, joystickView.width)
            assertEquals("Height should be set", 200, joystickView.height)
        }
    }

    @Test
    fun testCallbackInvocation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val latch = CountDownLatch(1)
        var receivedX = 0f
        var receivedY = 0f

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                receivedX = x
                receivedY = y
                latch.countDown()
            }

            // Simulate touch at center (should give 0, 0)
            val downTime = System.currentTimeMillis()
            val eventDown = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 100f, 100f, 0)
            joystickView.dispatchTouchEvent(eventDown)
            eventDown.recycle()
        }

        assertTrue("Callback should be invoked", latch.await(1, TimeUnit.SECONDS))
        assertEquals("X should be near 0 at center", 0f, receivedX, 0.1f)
        assertEquals("Y should be near 0 at center", 0f, receivedY, 0.1f)
    }

    @Test
    fun testNormalizedOutputRange() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val values = mutableListOf<Pair<Float, Float>>()

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                values.add(Pair(x, y))
            }

            val downTime = System.currentTimeMillis()

            // Touch at top (should give negative Y in view coords)
            val eventTop = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 100f, 10f, 0)
            joystickView.dispatchTouchEvent(eventTop)
            eventTop.recycle()

            // Touch at bottom
            val eventBottom = MotionEvent.obtain(downTime, downTime + 100, MotionEvent.ACTION_MOVE, 100f, 190f, 0)
            joystickView.dispatchTouchEvent(eventBottom)
            eventBottom.recycle()

            // Touch at left
            val eventLeft = MotionEvent.obtain(downTime, downTime + 200, MotionEvent.ACTION_MOVE, 10f, 100f, 0)
            joystickView.dispatchTouchEvent(eventLeft)
            eventLeft.recycle()

            // Touch at right
            val eventRight = MotionEvent.obtain(downTime, downTime + 300, MotionEvent.ACTION_MOVE, 190f, 100f, 0)
            joystickView.dispatchTouchEvent(eventRight)
            eventRight.recycle()
        }

        // Verify all values are in normalized range
        for ((x, y) in values) {
            assertTrue("X ($x) should be >= -1", x >= -1.0f)
            assertTrue("X ($x) should be <= 1", x <= 1.0f)
            assertTrue("Y ($y) should be >= -1", y >= -1.0f)
            assertTrue("Y ($y) should be <= 1", y <= 1.0f)
        }

        println("JoystickView normalized output test PASSED")
        println("Values received: $values")
    }

    @Test
    fun testResetOnRelease() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var finalX = Float.NaN
        var finalY = Float.NaN

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                finalX = x
                finalY = y
            }

            val downTime = System.currentTimeMillis()

            // Touch at offset position
            val eventDown = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 150f, 150f, 0)
            joystickView.dispatchTouchEvent(eventDown)
            eventDown.recycle()

            // Release touch - should reset to center (0, 0)
            val eventUp = MotionEvent.obtain(downTime, downTime + 100, MotionEvent.ACTION_UP, 150f, 150f, 0)
            joystickView.dispatchTouchEvent(eventUp)
            eventUp.recycle()
        }

        assertEquals("X should reset to 0 on release", 0f, finalX, 0.01f)
        assertEquals("Y should reset to 0 on release", 0f, finalY, 0.01f)
        println("JoystickView reset on release test PASSED")
    }

    @Test
    fun testKnobConstraint() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var maxX = 0f
        var maxY = 0f

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                if (kotlin.math.abs(x) > kotlin.math.abs(maxX)) maxX = x
                if (kotlin.math.abs(y) > kotlin.math.abs(maxY)) maxY = y
            }

            val downTime = System.currentTimeMillis()

            // Touch way outside the joystick bounds (should be constrained)
            val eventFar = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 500f, 500f, 0)
            joystickView.dispatchTouchEvent(eventFar)
            eventFar.recycle()
        }

        // Even with touch far outside, values should be constrained to -1 to 1
        assertTrue("X should be constrained to <= 1", maxX <= 1.0f)
        assertTrue("Y should be constrained to <= 1", maxY <= 1.0f)
        println("JoystickView knob constraint test PASSED - maxX=$maxX, maxY=$maxY")
    }

    @Test
    fun testActionCancel() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var finalX = Float.NaN
        var finalY = Float.NaN

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                finalX = x
                finalY = y
            }

            val downTime = System.currentTimeMillis()

            // Touch at offset position
            val eventDown = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 150f, 150f, 0)
            joystickView.dispatchTouchEvent(eventDown)
            eventDown.recycle()

            // Cancel touch - should also reset to center
            val eventCancel = MotionEvent.obtain(downTime, downTime + 100, MotionEvent.ACTION_CANCEL, 150f, 150f, 0)
            joystickView.dispatchTouchEvent(eventCancel)
            eventCancel.recycle()
        }

        assertEquals("X should reset to 0 on cancel", 0f, finalX, 0.01f)
        assertEquals("Y should reset to 0 on cancel", 0f, finalY, 0.01f)
        println("JoystickView ACTION_CANCEL test PASSED")
    }

    @Test
    fun testMoveEvent() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        val moveValues = mutableListOf<Pair<Float, Float>>()

        instrumentation.runOnMainSync {
            joystickView = createJoystickView()

            joystickView.onJoystickMove = { x, y ->
                moveValues.add(Pair(x, y))
            }

            val downTime = System.currentTimeMillis()

            // Initial touch at center
            val eventDown = MotionEvent.obtain(downTime, downTime, MotionEvent.ACTION_DOWN, 100f, 100f, 0)
            joystickView.dispatchTouchEvent(eventDown)
            eventDown.recycle()

            // Move to different positions
            for (i in 0..4) {
                val offset = i * 10f
                val eventMove = MotionEvent.obtain(downTime, downTime + 100 + i * 50, MotionEvent.ACTION_MOVE, 100f + offset, 100f + offset, 0)
                joystickView.dispatchTouchEvent(eventMove)
                eventMove.recycle()
            }
        }

        assertTrue("Should receive multiple move callbacks", moveValues.size >= 5)

        // Verify values increase as we move away from center
        var prevMagnitude = 0f
        for (i in 1 until moveValues.size) {
            val (x, y) = moveValues[i]
            val magnitude = kotlin.math.sqrt(x * x + y * y)
            assertTrue("Magnitude should increase or stay same as we move away", magnitude >= prevMagnitude - 0.1f)
            prevMagnitude = magnitude
        }

        println("JoystickView move event test PASSED - received ${moveValues.size} callbacks")
    }
}
