package com.furnit.android.services

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class RTMDetLiteRtTensorLayoutTest {
    @Test
    fun argbPackingMatchesSwiftBgrImageContract() {
        val source = intArrayOf(
            0xFF112233.toInt(),
            0x8044AAFE.toInt(),
        )
        val destination = FloatArray(source.size * 3)

        RTMDetLiteRtTensorLayout.packArgbToNhwcBgr(source, destination)

        assertArrayEquals(
            floatArrayOf(
                0x33.toFloat(), 0x22.toFloat(), 0x11.toFloat(),
                0xFE.toFloat(), 0xAA.toFloat(), 0x44.toFloat(),
            ),
            destination,
            0f,
        )
    }

    @Test
    fun nhwcToNchwPreservesEverySpatialChannelValue() {
        // NHWC [1,2,3,2]: each pixel stores channel 0 followed by channel 1.
        val source = floatArrayOf(
            0f, 100f,
            1f, 101f,
            2f, 102f,
            3f, 103f,
            4f, 104f,
            5f, 105f,
        )
        val destination = FloatArray(source.size)

        RTMDetLiteRtTensorLayout.nhwcToNchw(
            source = source,
            destination = destination,
            height = 2,
            width = 3,
            channels = 2,
        )

        assertArrayEquals(
            floatArrayOf(
                0f, 1f, 2f, 3f, 4f, 5f,
                100f, 101f, 102f, 103f, 104f, 105f,
            ),
            destination,
            0f,
        )
    }
}
