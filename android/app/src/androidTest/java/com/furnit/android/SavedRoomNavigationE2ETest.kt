package com.furnit.android

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.widget.EditText
import androidx.test.core.app.ActivityScenario
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.filters.LargeTest
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.uiautomator.By
import androidx.test.uiautomator.UiDevice
import androidx.test.uiautomator.UiObject2
import androidx.test.uiautomator.Until
import org.junit.After
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * Full regression for the single-photo failure mode: real photo -> AI preview -> save -> Home ->
 * reopen saved layered room -> look, dolly and D-pad while checking rendered frames for gray holes.
 */
@LargeTest
@RunWith(AndroidJUnit4::class)
class SavedRoomNavigationE2ETest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()
    private val targetContext = instrumentation.targetContext
    private val device = UiDevice.getInstance(instrumentation)
    private val roomName = "E2E Saved Room 0715"
    private val visualFailures = mutableListOf<String>()
    private lateinit var fixtureFile: File

    @Before
    fun setUp() {
        removeSavedTestRoom()
        fixtureFile = File(targetContext.cacheDir, "saved_room_navigation.jpg")
        createSyntheticRoomFixture(fixtureFile)
    }

    @After
    fun tearDown() {
        fixtureFile.delete()
        removeSavedTestRoom()
    }

    @Test
    fun createSaveReopenAndNavigateWithoutGrayHoles() {
        val intent = Intent(targetContext, SinglePhotoRoomActivity::class.java).apply {
            putExtra(SinglePhotoRoomActivity.EXTRA_UI_TEST_IMAGE_PATH, fixtureFile.absolutePath)
        }
        ActivityScenario.launch<SinglePhotoRoomActivity>(intent).use {
            val aiRoom = waitForResource("ai_room_option", 30_000)
            aiRoom.click()

            val previewViewport = waitForResource("saved_room_viewport", 120_000)
            waitForStableFrame()
            assertFrameHasNoGrayRendererHoles(captureScreen(), previewViewport.visibleBounds, "preview")

            waitForDescription(targetContext.getString(R.string.common_save), 20_000).click()
            val nameInput = device.wait(Until.findObject(By.clazz(EditText::class.java)), 10_000)
            assertNotNull("Room-name input did not open", nameInput)
            nameInput.text = roomName
            waitForResource("room_name_save_button", 10_000).click()

            waitForText(roomName, 240_000).click()
            val savedViewport = waitForResource("saved_room_viewport", 40_000)
            waitForStableFrame()

            val baseline = captureScreen()
            assertFrameHasNoGrayRendererHoles(baseline, savedViewport.visibleBounds, "saved baseline")

            val bounds = savedViewport.visibleBounds
            // This fixture locks the viewer in portrait. Exercise vertical D-pad motion before
            // any pinch so aspect-fill's initially zero pitch margin cannot hide a regression.
            // Check the horizontal pair too; the same path owns the landscape zero-yaw case.
            var afterDpad = clickDpadAndCapture(
                "camera_dpad_up",
                baseline,
                bounds,
                "portrait D-pad up",
            )
            afterDpad = clickDpadAndCapture(
                "camera_dpad_down",
                afterDpad,
                bounds,
                "portrait D-pad down",
            )
            afterDpad = clickDpadAndCapture(
                "camera_dpad_left",
                afterDpad,
                bounds,
                "portrait D-pad left",
            )
            afterDpad = clickDpadAndCapture(
                "camera_dpad_right",
                afterDpad,
                bounds,
                "portrait D-pad right",
            )

            // Keep the existing forward-dolly and one-finger-look regression after the direct
            // D-pad checks; moving forward creates additional source-image overscan.
            savedViewport.pinchOpen(0.55f, 600)
            waitForStableFrame()
            val afterPinch = captureScreen()
            assertFrameChanged(afterDpad, afterPinch, bounds, "pinch dolly")
            assertFrameHasNoGrayRendererHoles(afterPinch, bounds, "after pinch")

            device.swipe(
                bounds.centerX() - bounds.width() / 5,
                bounds.centerY(),
                bounds.centerX() + bounds.width() / 5,
                bounds.centerY(),
                20,
            )
            waitForStableFrame()
            val afterLook = captureScreen()
            assertFrameChanged(afterPinch, afterLook, bounds, "one-finger look")
            assertFrameHasNoGrayRendererHoles(afterLook, bounds, "after one-finger look")

            assertTrue(visualFailures.joinToString(separator = "\n"), visualFailures.isEmpty())
        }
    }

    private fun clickDpadAndCapture(
        description: String,
        before: Bitmap,
        bounds: Rect,
        label: String,
    ): Bitmap {
        waitForDescription(description, 10_000).click()
        waitForStableFrame()
        return captureScreen().also { after ->
            assertFrameChanged(before, after, bounds, label)
            assertFrameHasNoGrayRendererHoles(after, bounds, label)
        }
    }

    private fun waitForText(text: String, timeoutMs: Long): UiObject2 {
        val selector = By.text(text)
        device.wait(Until.hasObject(selector), timeoutMs)
        val matches = device.findObjects(selector)
        assertTrue("Timed out waiting for text '$text'", matches.isNotEmpty())
        return matches.first()
    }

    private fun createSyntheticRoomFixture(destination: File) {
        val bitmap = Bitmap.createBitmap(960, 1280, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bitmap)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        canvas.drawColor(Color.rgb(230, 232, 235))
        paint.color = Color.rgb(194, 184, 168)
        canvas.drawRect(0f, 905f, 960f, 1280f, paint)

        paint.color = Color.rgb(46, 64, 87)
        canvas.drawRect(145f, 255f, 815f, 905f, paint)
        paint.color = Color.rgb(184, 199, 212)
        canvas.drawRect(145f, 255f, 815f, 375f, paint)
        paint.color = Color.rgb(26, 38, 54)
        paint.strokeWidth = 12f
        for (x in 160..800 step 54) {
            canvas.drawLine(x.toFloat(), 255f, x.toFloat(), 905f, paint)
        }

        paint.color = Color.rgb(82, 61, 43)
        canvas.drawRect(610f, 785f, 850f, 827f, paint)
        canvas.drawRect(635f, 827f, 661f, 1087f, paint)
        canvas.drawRect(802f, 827f, 828f, 1087f, paint)
        paint.color = Color.rgb(41, 71, 56)
        canvas.drawOval(665f, 705f, 775f, 801f, paint)
        paint.color = Color.rgb(31, 97, 59)
        paint.strokeWidth = 16f
        canvas.drawLine(720f, 785f, 720f, 735f, paint)

        destination.outputStream().use { output ->
            check(bitmap.compress(Bitmap.CompressFormat.JPEG, 92, output))
        }
        bitmap.recycle()
    }

    private fun waitForDescription(description: String, timeoutMs: Long): UiObject2 {
        val selector = By.desc(description)
        val result = device.wait(Until.findObject(selector), timeoutMs)
        assertNotNull("Timed out waiting for '$description'", result)
        return result
    }

    private fun waitForResource(resourceName: String, timeoutMs: Long): UiObject2 {
        val selector = By.res(targetContext.packageName, resourceName)
        val result = device.wait(Until.findObject(selector), timeoutMs)
        assertNotNull("Timed out waiting for resource '$resourceName'", result)
        return result
    }

    private fun captureScreen(): Bitmap = instrumentation.uiAutomation.takeScreenshot()

    private fun waitForStableFrame() {
        device.waitForIdle(1_000)
        Thread.sleep(650)
    }

    private fun assertFrameHasNoGrayRendererHoles(bitmap: Bitmap, rawBounds: Rect, label: String) {
        val bounds = insetViewport(rawBounds, bitmap)
        var sampled = 0
        var rendererGray = 0
        for (y in bounds.top until bounds.bottom step 4) {
            for (x in bounds.left until bounds.right step 4) {
                val color = bitmap.getPixel(x, y)
                val red = Color.red(color)
                val green = Color.green(color)
                val blue = Color.blue(color)
                sampled++
                if (kotlin.math.abs(red - 128) <= 3 &&
                    kotlin.math.abs(green - 128) <= 3 &&
                    kotlin.math.abs(blue - 128) <= 3
                ) rendererGray++
            }
        }
        val ratio = rendererGray.toDouble() / sampled.coerceAtLeast(1)
        if (ratio >= 0.025) {
            visualFailures += "$label contains ${(ratio * 100).toInt()}% renderer-gray holes"
        }
    }

    private fun assertFrameChanged(before: Bitmap, after: Bitmap, rawBounds: Rect, action: String) {
        val bounds = insetViewport(rawBounds, before)
        var channelDelta = 0L
        var channels = 0L
        for (y in bounds.top until bounds.bottom step 8) {
            for (x in bounds.left until bounds.right step 8) {
                val first = before.getPixel(x, y)
                val second = after.getPixel(x, y)
                channelDelta += kotlin.math.abs(Color.red(first) - Color.red(second))
                channelDelta += kotlin.math.abs(Color.green(first) - Color.green(second))
                channelDelta += kotlin.math.abs(Color.blue(first) - Color.blue(second))
                channels += 3
            }
        }
        val meanDelta = channelDelta.toDouble() / channels.coerceAtLeast(1)
        if (meanDelta <= 0.7) {
            visualFailures += "$action did not move the rendered camera (mean delta=$meanDelta)"
        }
    }

    private fun insetViewport(rawBounds: Rect, bitmap: Bitmap): Rect {
        val horizontalInset = rawBounds.width() / 10
        val verticalInset = rawBounds.height() / 7
        return Rect(
            (rawBounds.left + horizontalInset).coerceIn(0, bitmap.width - 1),
            (rawBounds.top + verticalInset).coerceIn(0, bitmap.height - 1),
            (rawBounds.right - horizontalInset).coerceIn(1, bitmap.width),
            (rawBounds.bottom - verticalInset).coerceIn(1, bitmap.height),
        )
    }

    private fun removeSavedTestRoom() {
        File(targetContext.filesDir, "rooms").listFiles()?.forEach { folder ->
            val metadata = File(folder, "metadata.txt")
            if (metadata.isFile && metadata.readText().lineSequence().any { it == "name=$roomName" }) {
                folder.deleteRecursively()
            }
        }
    }
}
