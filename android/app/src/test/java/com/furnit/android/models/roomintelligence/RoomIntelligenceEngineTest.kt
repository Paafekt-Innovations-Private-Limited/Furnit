package com.furnit.android.models.roomintelligence

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class RoomIntelligenceEngineTest {
    @Test
    fun `exact dimension boundaries fit`() {
        val result = evaluate(
            room = RoomDimensions(2f, 1f, 1.4f),
            furniture = FurnitureDimensions(2f, 1f),
        )

        assertEquals(RoomIntelligenceStatus.FITS_BY_DIMENSIONS, result.status)
        val dimensionFit = result.dimensionFit!!
        assertTrue(dimensionFit.fits)
        assertFalse(dimensionFit.rotatedFootprint)
    }

    @Test
    fun `footprint rotation is evaluated as one orientation`() {
        val result = evaluate(
            room = RoomDimensions(1.3f, 2.5f, 1.4f),
            furniture = FurnitureDimensions(1.4f, 2f),
        )

        assertEquals(RoomIntelligenceStatus.FITS_BY_DIMENSIONS, result.status)
        assertTrue(result.dimensionFit!!.rotatedFootprint)
    }

    @Test
    fun `independent axis matches cannot create an impossible fit`() {
        val result = evaluate(
            room = RoomDimensions(1.3f, 2.5f, 1.1f),
            furniture = FurnitureDimensions(1.4f, 2f),
        )

        assertEquals(RoomIntelligenceStatus.DOES_NOT_FIT, result.status)
    }

    @Test
    fun `height must fit`() {
        val result = evaluate(
            room = RoomDimensions(4f, 2f, 4f),
            furniture = FurnitureDimensions(1f, 2.01f),
        )

        assertEquals(RoomIntelligenceStatus.DOES_NOT_FIT, result.status)
    }

    @Test
    fun `invalid room remains measuring and has no fit claim`() {
        val result = evaluate(
            room = RoomDimensions(Float.NaN, 2f, 4f),
            furniture = FurnitureDimensions(1f, 1f),
        )

        assertEquals(RoomIntelligenceStatus.MEASURING, result.status)
        assertNull(result.dimensionFit)
    }

    @Test
    fun `missing furniture dimensions is style only`() {
        val result = RoomIntelligenceEngine.evaluate(
            RoomIntelligenceInput(
                roomDimensions = RoomDimensions(4f, 3f, 5f),
                furnitureDimensions = null,
                roomPalette = SurfacePalette(
                    walls = SurfaceColors(listOf(StraightSrgbColor(0.7f, 0.7f, 0.7f))),
                ),
                furnitureAestheticProfile = FurnitureAestheticProfile(
                    StraightSrgbColor(0.2f, 0.2f, 0.2f),
                    styleTags = listOf("modern"),
                ),
            ),
        )

        assertEquals(RoomIntelligenceStatus.STYLE_ONLY, result.status)
    }

    @Test
    fun `derived depth clamps at both boundaries`() {
        assertEquals(
            0.25f,
            FurnitureDimensions.fromMeasuredWidthAndHeight(0.1f, 1f)!!.depthMeters,
            0f,
        )
        assertEquals(
            0.72f,
            FurnitureDimensions.fromMeasuredWidthAndHeight(1f, 1f)!!.depthMeters,
            0.0001f,
        )
        assertEquals(
            1.4f,
            FurnitureDimensions.fromMeasuredWidthAndHeight(3f, 1f)!!.depthMeters,
            0f,
        )
        assertNull(FurnitureDimensions.fromMeasuredWidthAndHeight(Float.POSITIVE_INFINITY, 1f))
    }

    @Test
    fun `result preserves metric and derived provenance without a location claim`() {
        val result = evaluate(
            room = RoomDimensions(4f, 3f, 5f),
            furniture = FurnitureDimensions(1f, 1f),
        )

        val furnitureDimensions = result.furnitureDimensions!!
        assertEquals(DimensionSource.ARCORE_METRIC, furnitureDimensions.widthSource)
        assertEquals(DimensionSource.ARCORE_METRIC, furnitureDimensions.heightSource)
        assertEquals(DimensionSource.DERIVED_FROM_WIDTH_RATIO, furnitureDimensions.depthSource)
        assertEquals(FitBasis.ROOM_ENVELOPE_ONLY, result.fitBasis)
        assertFalse(result.placementLocationEvaluated)
    }

    private fun evaluate(
        room: RoomDimensions,
        furniture: FurnitureDimensions,
    ): RoomIntelligenceResult = RoomIntelligenceEngine.evaluate(
        RoomIntelligenceInput(
            roomDimensions = room,
            furnitureDimensions = furniture,
        ),
    )
}
