package com.furnit.android.roomreconstruction

import org.junit.Assert.assertEquals
import org.junit.Test

class RoomMathParityTest {
    @Test
    fun medianAveragesEvenSamplesLikeIosMeasurementHelpers() {
        assertEquals(2.5f, RoomMath.median(listOf(4f, 1f, 3f, 2f)), 0f)
    }

    @Test
    fun upperMedianMatchesIosRoomHeightJunctionSelection() {
        assertEquals(3f, RoomMath.upperMedian(listOf(4f, 1f, 3f, 2f)), 0f)
    }

    @Test
    fun percentileRoundsToNearestOrderStatisticLikeIos() {
        val samples = (0..10).map(Int::toFloat)
        assertEquals(8f, RoomMath.percentile(samples, 0.75) ?: Float.NaN, 0f)
    }
}
