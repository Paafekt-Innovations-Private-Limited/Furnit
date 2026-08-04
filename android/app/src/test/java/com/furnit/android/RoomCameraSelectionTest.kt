package com.furnit.android

import org.junit.Assert.assertEquals
import org.junit.Test

class RoomCameraSelectionTest {
    @Test
    fun lowResolutionAuxiliaryCameraDoesNotBeatCaptureQualityCamera() {
        val selected = selectWidestCaptureQualityCamera(
            candidates = listOf(
                RoomCameraCandidate("macro", horizontalFovDegrees = 125.0, maxJpegPixels = 2_000_000),
                RoomCameraCandidate("ultra-wide", horizontalFovDegrees = 112.0, maxJpegPixels = 12_000_000),
                RoomCameraCandidate("main", horizontalFovDegrees = 74.0, maxJpegPixels = 48_000_000),
            ),
            minimumJpegPixels = 5_000_000,
        )

        assertEquals("ultra-wide", selected?.cameraId)
    }

    @Test
    fun widestQualifiedCameraWinsBeforeResolutionTieBreak() {
        val selected = selectWidestCaptureQualityCamera(
            candidates = listOf(
                RoomCameraCandidate("ultra-wide", horizontalFovDegrees = 112.0, maxJpegPixels = 8_000_000),
                RoomCameraCandidate("main", horizontalFovDegrees = 74.0, maxJpegPixels = 50_000_000),
            ),
            minimumJpegPixels = 5_000_000,
        )

        assertEquals("ultra-wide", selected?.cameraId)
    }

    @Test
    fun widestCameraStillWinsWhenNoCameraReportsEnoughPixels() {
        val selected = selectWidestCaptureQualityCamera(
            candidates = listOf(
                RoomCameraCandidate("wide", horizontalFovDegrees = 104.0, maxJpegPixels = 4_000_000),
                RoomCameraCandidate("main", horizontalFovDegrees = 72.0, maxJpegPixels = 3_000_000),
            ),
            minimumJpegPixels = 5_000_000,
        )

        assertEquals("wide", selected?.cameraId)
    }
}
