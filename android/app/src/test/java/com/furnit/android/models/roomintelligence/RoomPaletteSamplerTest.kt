package com.furnit.android.models.roomintelligence

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class RoomPaletteSamplerTest {
    @Test
    fun `mean ignores alpha below threshold and unpremultiplies channels`() {
        val ignoredTransparentRed = argb(alpha = 15, red = 15, green = 0, blue = 0)
        val halfAlphaPremultipliedOrange = argb(alpha = 128, red = 128, green = 64, blue = 0)

        val mean = BitmapStraightSrgbExtractor.meanPremultipliedArgb(
            pixels = intArrayOf(ignoredTransparentRed, halfAlphaPremultipliedOrange),
            width = 2,
            height = 1,
        )!!

        assertEquals(1f, mean.red, 0.01f)
        assertEquals(0.5f, mean.green, 0.01f)
        assertEquals(0f, mean.blue, 0.01f)
    }

    @Test
    fun `fully transparent image has no fake color`() {
        val mean = BitmapStraightSrgbExtractor.meanPremultipliedArgb(
            pixels = IntArray(4),
            width = 2,
            height = 2,
        )

        assertNull(mean)
    }

    @Test
    fun `regional fallback maps top middle and bottom deterministically`() {
        val white = argb(255, 255, 255, 255)
        val gray = argb(255, 128, 128, 128)
        val brown = argb(255, 128, 64, 32)
        val pixels = intArrayOf(
            white, white,
            gray, gray,
            brown, brown,
        )

        val palette = RoomPaletteSampler.regionalPalette(pixels, width = 2, height = 3)

        assertEquals(1f, palette.ceiling!!.dominantColors.single().red, 0.001f)
        assertEquals(128f / 255f, palette.walls!!.dominantColors.single().red, 0.001f)
        val floor = palette.floor!!
        assertEquals(128f / 255f, floor.dominantColors.single().red, 0.001f)
        assertEquals(MaterialHint.WOOD, floor.materialHint)
    }

    private fun argb(alpha: Int, red: Int, green: Int, blue: Int): Int =
        (alpha shl 24) or (red shl 16) or (green shl 8) or blue
}
