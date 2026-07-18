package com.furnit.android.ar

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class FurnitureFitArMetricsTest {

    @Test
    fun estimatedPhysicalWidthMeters_usesPinholeDistance() {
        val widthMeters = FurnitureFitArMetrics.estimatedPhysicalWidthMeters(
            bboxWidthPixels = 600f,
            distanceMeters = 2f,
            focalLengthPixels = 1200f,
        )

        assertEquals(1f, widthMeters!!, 0.0001f)
    }

    @Test
    fun rawCameraExtent_horizontalMappingSelectsXFocalAxis() {
        val extent = FurnitureFitArMetrics.rawCameraExtentFromMappedEndpoints(
            startRawX = 100f,
            startRawY = 300f,
            endRawX = 500f,
            endRawY = 300f,
        )

        assertEquals(400f, extent!!.pixels, 0.0001f)
        assertEquals(FurnitureFitArMetrics.RawCameraAxis.X, extent.focalAxis)
    }

    @Test
    fun rawCameraExtent_quarterTurnSelectsYFocalAxis() {
        val extent = FurnitureFitArMetrics.rawCameraExtentFromMappedEndpoints(
            startRawX = 500f,
            startRawY = 100f,
            endRawX = 500f,
            endRawY = 700f,
        )

        assertEquals(600f, extent!!.pixels, 0.0001f)
        assertEquals(FurnitureFitArMetrics.RawCameraAxis.Y, extent.focalAxis)
    }

    @Test
    fun rawCameraExtent_rejectsInvalidMappedCoordinates() {
        assertNull(
            FurnitureFitArMetrics.rawCameraExtentFromMappedEndpoints(
                startRawX = Float.NaN,
                startRawY = 0f,
                endRawX = 10f,
                endRawY = 0f,
            ),
        )
    }

    @Test
    fun scaleFocalLengthPixels_scalesAlongSelectedImageAxis() {
        val scaledFocalLength = FurnitureFitArMetrics.scaleFocalLengthPixels(
            focalLengthPixels = 1000f,
            intrinsicAxisPixels = 2000,
            imageAxisPixels = 1000,
        )

        assertEquals(500f, scaledFocalLength, 0.0001f)
    }
}
