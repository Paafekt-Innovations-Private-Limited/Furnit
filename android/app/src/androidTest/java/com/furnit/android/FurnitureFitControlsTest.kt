package com.furnit.android

import android.content.Context
import android.graphics.Bitmap
import android.view.MotionEvent
import android.view.View
import android.widget.FrameLayout
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class FurnitureFitControlsTest {
    private lateinit var context: Context

    @Before
    fun setup() {
        context = InstrumentationRegistry.getInstrumentation().targetContext
    }

    @Test
    fun testFurnitureOverlayCanBeCreated() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var overlayCreated = false

        instrumentation.runOnMainSync {
            val overlay = FurnitureFitOverlayView(context)
            overlay.layoutParams = FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            )
            overlayCreated = true
        }

        assertTrue("FurnitureFitOverlayView should be created without exceptions", overlayCreated)
    }

    @Test
    fun testFurnitureOverlayTouchOutsideCallback() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var callbackInvoked = false

        instrumentation.runOnMainSync {
            val overlay = measuredOverlay()
            overlay.onTouchOutsideFurniture = { _ -> callbackInvoked = true }

            touch(overlay, MotionEvent.ACTION_DOWN, 500f, 500f)
        }

        assertTrue("Touching without a mask should invoke the outside callback", callbackInvoked)
    }

    @Test
    fun testFurnitureOverlayHitTestWithDetection() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var touchOnFurniture = false
        var touchOutsideFurniture = false

        instrumentation.runOnMainSync {
            val overlay = measuredOverlay()
            val maskBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            val detection = DetectionResult(
                x = 320f,
                y = 320f,
                w = 500f,
                h = 500f,
                confidence = 0.9f,
                label = "chair",
                classId = 0,
            )
            overlay.setMaskAndDetections(maskBitmap, listOf(detection), 640)

            var outsideCallbackInvoked = false
            overlay.onTouchOutsideFurniture = { _ -> outsideCallbackInvoked = true }
            touch(overlay, MotionEvent.ACTION_DOWN, 500f, 650f)
            touchOnFurniture = !outsideCallbackInvoked

            outsideCallbackInvoked = false
            touch(overlay, MotionEvent.ACTION_DOWN, 50f, 50f)
            touchOutsideFurniture = outsideCallbackInvoked
            maskBitmap.recycle()
        }

        assertTrue("Touch inside furniture bounds should hit furniture", touchOnFurniture)
        assertTrue("Touch in the corner should be outside furniture", touchOutsideFurniture)
    }

    @Test
    fun testFurnitureDragUpdatesTranslation() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var dragStayedOnFurniture = false

        instrumentation.runOnMainSync {
            val overlay = measuredOverlay()
            val maskBitmap = Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888)
            val detection = DetectionResult(
                x = 320f,
                y = 320f,
                w = 400f,
                h = 400f,
                confidence = 0.9f,
                label = "couch",
                classId = 0,
            )
            overlay.setMaskAndDetections(maskBitmap, listOf(detection), 640)

            var outsideCallbackInvoked = false
            overlay.onTouchOutsideFurniture = { _ -> outsideCallbackInvoked = true }
            val downTime = System.currentTimeMillis()
            dispatchTouch(overlay, downTime, downTime, MotionEvent.ACTION_DOWN, 500f, 500f)
            dispatchTouch(overlay, downTime, downTime + 16, MotionEvent.ACTION_MOVE, 600f, 550f)
            dispatchTouch(overlay, downTime, downTime + 32, MotionEvent.ACTION_UP, 600f, 550f)
            dragStayedOnFurniture = !outsideCallbackInvoked
            maskBitmap.recycle()
        }

        assertTrue("Dragging furniture should not trigger the outside callback", dragStayedOnFurniture)
    }

    private fun measuredOverlay(): FurnitureFitOverlayView {
        return FurnitureFitOverlayView(context).apply {
            val measureSpec = View.MeasureSpec.makeMeasureSpec(1000, View.MeasureSpec.EXACTLY)
            measure(measureSpec, measureSpec)
            layout(0, 0, 1000, 1000)
        }
    }

    private fun touch(overlay: FurnitureFitOverlayView, action: Int, x: Float, y: Float) {
        val eventTime = System.currentTimeMillis()
        dispatchTouch(overlay, eventTime, eventTime, action, x, y)
    }

    private fun dispatchTouch(
        overlay: FurnitureFitOverlayView,
        downTime: Long,
        eventTime: Long,
        action: Int,
        x: Float,
        y: Float,
    ) {
        val event = MotionEvent.obtain(downTime, eventTime, action, x, y, 0)
        overlay.handleExternalTouchEvent(event)
        event.recycle()
    }
}
