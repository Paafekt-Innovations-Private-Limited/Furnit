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
        val primarySurfaceZs = zs.dropLast(4)
        assertEquals(1f, primarySurfaceZs.maxOrNull()!! - primarySurfaceZs.minOrNull()!!, 0.0001f)
        assertTrue(zs.takeLast(4).all { it < primarySurfaceZs.minOrNull()!! })
        // Pixel (0,0) at depth 1 unprojects to (-0.5,+0.5,-1) for c=(4,4), f=8.
        assertEquals(-0.5f, geometry.positions[0], 0.0001f)
        assertEquals(0.5f, geometry.positions[1], 0.0001f)
        assertEquals(-1f, geometry.positions[2], 0.0001f)
        assertTrue(geometry.cameraVerticalFovDegrees!! > 0f)
        // Discontinuous foreground/background quads remain disconnected; six final indices are
        // the calibrated far-photo backing layer that fills those openings.
        assertTrue(geometry.indices.size < 30)
    }
}
