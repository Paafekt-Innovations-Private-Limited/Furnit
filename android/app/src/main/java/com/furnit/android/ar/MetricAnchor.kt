package com.furnit.android.ar

import java.io.Serializable

data class MetricAnchor(
    val pixelX: Int,
    val pixelY: Int,
    val depthMeters: Float,
    val confidence: Float,
    val originalImageWidth: Int,
    val originalImageHeight: Int,
) : Serializable
