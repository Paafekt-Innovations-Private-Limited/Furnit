package com.furnit.android

import java.nio.ByteBuffer
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CameraFramePixelPackingTest {
    @Test
    fun `packs CameraX RGBA bytes into Android ARGB colors with row padding`() {
        val rgba = byteArrayOf(
            0x11, 0x22, 0x33, 0x44,
            0x55, 0x66, 0x77, 0x7f,
            0, 0, 0, 0,
            0x01, 0x02, 0x03, 0x7e,
            0x04, 0x05, 0x06, 0x7d,
        )
        val output = IntArray(4)

        assertTrue(
            copyRgba8888ToArgbPixels(
                source = ByteBuffer.wrap(rgba),
                width = 2,
                height = 2,
                rowStride = 12,
                pixelStride = 4,
                destination = output,
            ),
        )
        assertArrayEquals(
            intArrayOf(0x44112233, 0x7f556677, 0x7e010203, 0x7d040506),
            output,
        )
    }

    @Test
    fun `rejects a truncated CameraX RGBA plane`() {
        assertFalse(
            copyRgba8888ToArgbPixels(
                source = ByteBuffer.wrap(ByteArray(7)),
                width = 2,
                height = 1,
                rowStride = 8,
                pixelStride = 4,
                destination = IntArray(2),
            ),
        )
    }
}
