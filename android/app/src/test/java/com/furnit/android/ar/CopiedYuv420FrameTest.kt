package com.furnit.android.ar

import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertNull
import org.junit.Test

class CopiedYuv420FrameTest {
    @Test
    fun `packs independent row and pixel strides as NV21`() {
        val frame = CopiedYuv420Frame(
            width = 4,
            height = 2,
            y = byteArrayOf(1, 2, 3, 4, 99, 99, 5, 6, 7, 8),
            yRowStride = 6,
            yPixelStride = 1,
            u = byteArrayOf(10, 99, 11),
            uRowStride = 4,
            uPixelStride = 2,
            v = byteArrayOf(20, 99, 21),
            vRowStride = 3,
            vPixelStride = 2,
        )

        assertArrayEquals(
            byteArrayOf(1, 2, 3, 4, 5, 6, 7, 8, 20, 10, 21, 11),
            frame.toNv21(),
        )
    }

    @Test
    fun `supports a padded luma plane with non-unit pixel stride`() {
        val frame = CopiedYuv420Frame(
            width = 2,
            height = 2,
            y = byteArrayOf(1, 99, 2, 99, 3, 99, 4),
            yRowStride = 4,
            yPixelStride = 2,
            u = byteArrayOf(10),
            uRowStride = 1,
            uPixelStride = 1,
            v = byteArrayOf(20),
            vRowStride = 1,
            vPixelStride = 1,
        )

        assertArrayEquals(byteArrayOf(1, 2, 3, 4, 20, 10), frame.toNv21())
    }

    @Test
    fun `rejects truncated planes instead of reading outside the copy`() {
        val frame = CopiedYuv420Frame(
            width = 4,
            height = 2,
            y = byteArrayOf(1, 2, 3),
            yRowStride = 4,
            yPixelStride = 1,
            u = byteArrayOf(10, 11),
            uRowStride = 2,
            uPixelStride = 1,
            v = byteArrayOf(20, 21),
            vRowStride = 2,
            vPixelStride = 1,
        )

        assertNull(frame.toNv21())
    }

    @Test
    fun `converts neutral video-range YUV directly to opaque ARGB without JPEG loss`() {
        val black = CopiedYuv420Frame(
            width = 2,
            height = 2,
            y = byteArrayOf(16, 16, 16, 16),
            yRowStride = 2,
            yPixelStride = 1,
            u = byteArrayOf(128.toByte()),
            uRowStride = 1,
            uPixelStride = 1,
            v = byteArrayOf(128.toByte()),
            vRowStride = 1,
            vPixelStride = 1,
        )
        val white = black.copy(y = byteArrayOf(235.toByte(), 235.toByte(), 235.toByte(), 235.toByte()))

        assertArrayEquals(IntArray(4) { 0xFF000000.toInt() }, black.toArgbPixels())
        assertArrayEquals(IntArray(4) { 0xFFFFFFFF.toInt() }, white.toArgbPixels())
    }
}
