package com.furnit.android

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.view.InputDevice
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

    @Test
    fun testFrameAlignedSegmentedFurnitureCanBePinchedLarger() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var initialOpaquePixels = 0
        var enlargedOpaquePixels = 0

        instrumentation.runOnMainSync {
            val overlay = measuredOverlay()
            val maskBitmap = segmentedFrameMask()
            overlay.setMaskAndDetections(
                mask = maskBitmap,
                dets = emptyList(),
                frameAlignedOverlay = true,
                sourceWidth = maskBitmap.width,
                sourceHeight = maskBitmap.height,
            )
            initialOpaquePixels = renderOpaquePixelCount(overlay)

            val downTime = System.currentTimeMillis()
            dispatchTouch(overlay, downTime, downTime, MotionEvent.ACTION_DOWN, 450f, 500f)
            dispatchMultiTouch(
                overlay,
                downTime,
                downTime + 16,
                MotionEvent.ACTION_POINTER_DOWN or (1 shl MotionEvent.ACTION_POINTER_INDEX_SHIFT),
                450f,
                500f,
                550f,
                500f,
            )
            dispatchMultiTouch(
                overlay,
                downTime,
                downTime + 32,
                MotionEvent.ACTION_MOVE,
                440f,
                500f,
                560f,
                500f,
            )
            dispatchMultiTouch(
                overlay,
                downTime,
                downTime + 48,
                MotionEvent.ACTION_MOVE,
                425f,
                500f,
                575f,
                500f,
            )
            dispatchMultiTouch(
                overlay,
                downTime,
                downTime + 64,
                MotionEvent.ACTION_POINTER_UP or (1 shl MotionEvent.ACTION_POINTER_INDEX_SHIFT),
                425f,
                500f,
                575f,
                500f,
            )
            dispatchTouch(overlay, downTime, downTime + 80, MotionEvent.ACTION_UP, 425f, 500f)

            enlargedOpaquePixels = renderOpaquePixelCount(overlay)
            maskBitmap.recycle()
        }

        assertTrue(
            "Pinching a frame-aligned segmented cutout should enlarge its rendered area",
            enlargedOpaquePixels > initialOpaquePixels,
        )
    }

    @Test
    fun testFrameAlignedSegmentedFurnitureCanBeDragged() {
        val instrumentation = InstrumentationRegistry.getInstrumentation()
        var before = Rect()
        var after = Rect()

        instrumentation.runOnMainSync {
            val overlay = measuredOverlay()
            val maskBitmap = segmentedFrameMask()
            overlay.setMaskAndDetections(
                mask = maskBitmap,
                dets = emptyList(),
                frameAlignedOverlay = true,
                sourceWidth = maskBitmap.width,
                sourceHeight = maskBitmap.height,
            )
            before = renderOpaqueBounds(overlay)

            val downTime = System.currentTimeMillis()
            dispatchTouch(overlay, downTime, downTime, MotionEvent.ACTION_DOWN, 500f, 500f)
            dispatchTouch(overlay, downTime, downTime + 16, MotionEvent.ACTION_MOVE, 600f, 550f)
            dispatchTouch(overlay, downTime, downTime + 32, MotionEvent.ACTION_UP, 600f, 550f)

            after = renderOpaqueBounds(overlay)
            maskBitmap.recycle()
        }

        assertTrue("Dragging should move the cutout horizontally", after.centerX() > before.centerX())
        assertTrue("Dragging should move the cutout vertically", after.centerY() > before.centerY())
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

    private fun dispatchMultiTouch(
        overlay: FurnitureFitOverlayView,
        downTime: Long,
        eventTime: Long,
        action: Int,
        firstX: Float,
        firstY: Float,
        secondX: Float,
        secondY: Float,
    ) {
        val properties = arrayOf(
            MotionEvent.PointerProperties().apply {
                id = 0
                toolType = MotionEvent.TOOL_TYPE_FINGER
            },
            MotionEvent.PointerProperties().apply {
                id = 1
                toolType = MotionEvent.TOOL_TYPE_FINGER
            },
        )
        val coordinates = arrayOf(
            MotionEvent.PointerCoords().apply {
                x = firstX
                y = firstY
                pressure = 1f
                size = 1f
            },
            MotionEvent.PointerCoords().apply {
                x = secondX
                y = secondY
                pressure = 1f
                size = 1f
            },
        )
        val event = MotionEvent.obtain(
            downTime,
            eventTime,
            action,
            2,
            properties,
            coordinates,
            0,
            0,
            1f,
            1f,
            0,
            0,
            InputDevice.SOURCE_TOUCHSCREEN,
            0,
        )
        overlay.handleExternalTouchEvent(event)
        event.recycle()
    }

    private fun segmentedFrameMask(): Bitmap {
        return Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.TRANSPARENT)
            Canvas(this).drawRect(
                160f,
                120f,
                480f,
                360f,
                Paint().apply { color = Color.WHITE },
            )
        }
    }

    private fun renderOpaquePixelCount(overlay: FurnitureFitOverlayView): Int {
        val rendered = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
        overlay.draw(Canvas(rendered))
        val pixels = IntArray(rendered.width * rendered.height)
        rendered.getPixels(pixels, 0, rendered.width, 0, 0, rendered.width, rendered.height)
        rendered.recycle()
        return pixels.count { Color.alpha(it) > 10 }
    }

    private fun renderOpaqueBounds(overlay: FurnitureFitOverlayView): Rect {
        val rendered = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
        overlay.draw(Canvas(rendered))
        val pixels = IntArray(rendered.width * rendered.height)
        rendered.getPixels(pixels, 0, rendered.width, 0, 0, rendered.width, rendered.height)
        var minX = rendered.width
        var minY = rendered.height
        var maxX = -1
        var maxY = -1
        for (y in 0 until rendered.height) {
            for (x in 0 until rendered.width) {
                if (Color.alpha(pixels[y * rendered.width + x]) <= 10) continue
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
        rendered.recycle()
        return if (maxX >= minX && maxY >= minY) {
            Rect(minX, minY, maxX + 1, maxY + 1)
        } else {
            Rect()
        }
    }
}
