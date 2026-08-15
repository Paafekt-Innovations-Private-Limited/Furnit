package com.furnit.android.services

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class DepthPhotoGeometryTest {
    @Test
    fun depthGeometryPreservesPixelAspectAndCutsDiscontinuities() {
        val width = 9
        val height = 9
        val depth = FloatArray(width * height) { index ->
            if (index % width < 4) 1f else 2f
        }

        val geometry = GlbGenerator().depthPhotoGeometry(
            dimensions = GlbGenerator.RoomDimensions(width = 8f, height = 4f, depth = 6f),
            depthMap = depth,
            depthWidth = width,
            depthHeight = height,
            focalXPixels = 8f,
            focalYPixels = 8f,
        )

        val zs = (2 until geometry.positions.size step 3).map { geometry.positions[it] }
        assertEquals(1f, zs.maxOrNull()!! - zs.minOrNull()!!, 0.0001f)
        // Pixel (0,0) at depth 1 unprojects to (-0.5,+0.5,-1) for c=(4,4), f=8.
        assertEquals(-0.5f, geometry.positions[0], 0.0001f)
        assertEquals(0.5f, geometry.positions[1], 0.0001f)
        assertEquals(-1f, geometry.positions[2], 0.0001f)
        assertTrue(geometry.cameraVerticalFovDegrees!! > 0f)
        // The default helper still supports discontinuity rejection for non-photo callers.
        assertEquals(12, geometry.indices.size)
    }

    @Test
    fun continuousLandscapeGeometryReprojectsToEverySampledSourcePixel() {
        assertContinuousGeometryReprojects(width = 13, height = 9, focalX = 10f, focalY = 11f)
    }

    @Test
    fun continuousPortraitGeometryReprojectsToEverySampledSourcePixel() {
        assertContinuousGeometryReprojects(width = 9, height = 13, focalX = 11f, focalY = 10f)
    }

    private fun assertContinuousGeometryReprojects(
        width: Int,
        height: Int,
        focalX: Float,
        focalY: Float,
    ) {
        val depth = FloatArray(width * height) { index ->
            1.5f + (index % width) * 0.03f + (index / width) * 0.01f
        }
        val geometry = GlbGenerator().depthPhotoGeometry(
            dimensions = GlbGenerator.RoomDimensions(),
            depthMap = depth,
            depthWidth = width,
            depthHeight = height,
            focalXPixels = focalX,
            focalYPixels = focalY,
            depthDiscontinuityMeters = Float.POSITIVE_INFINITY,
        )

        val centerX = (width - 1) * 0.5f
        val centerY = (height - 1) * 0.5f
        val vertexCount = geometry.positions.size / 3
        for (vertex in 0 until vertexCount) {
            val depthMeters = -geometry.positions[vertex * 3 + 2]
            val projectedX = geometry.positions[vertex * 3] * focalX / depthMeters + centerX
            val projectedY = centerY - geometry.positions[vertex * 3 + 1] * focalY / depthMeters
            val sourceX = geometry.uvs[vertex * 2] * (width - 1)
            val sourceY = geometry.uvs[vertex * 2 + 1] * (height - 1)
            assertEquals(sourceX, projectedX, 0.0001f)
            assertEquals(sourceY, projectedY, 0.0001f)
        }

        val sampledColumns = ((width - 1) / 4) + 1
        val sampledRows = ((height - 1) / 4) + 1
        assertEquals((sampledColumns - 1) * (sampledRows - 1) * 6, geometry.indices.size)
        assertTrue(geometry.indices.maxOrNull()!! < vertexCount)
    }
}
