package com.furnit.android.roomreconstruction

data class ImageLetterboxLayout(
    val canvasSide: Int,
    val sourceWidth: Int,
    val sourceHeight: Int,
    val contentWidth: Int,
    val contentHeight: Int,
    val offsetX: Int,
    val offsetY: Int,
) {
    val uniformScale: Float
        get() = contentWidth.toFloat() / maxOf(sourceWidth, 1)

    val focalScaleToSource: Float
        get() = 1f / maxOf(uniformScale, 1e-6f)

    companion object {
        fun layout(sourceWidth: Int, sourceHeight: Int, canvasSide: Int): ImageLetterboxLayout {
            val scale = minOf(
                canvasSide.toFloat() / maxOf(sourceWidth, 1),
                canvasSide.toFloat() / maxOf(sourceHeight, 1),
            )
            val contentWidth = maxOf(1, kotlin.math.round(sourceWidth * scale).toInt())
            val contentHeight = maxOf(1, kotlin.math.round(sourceHeight * scale).toInt())
            val offsetX = maxOf(0, (canvasSide - contentWidth) / 2)
            val offsetY = maxOf(0, (canvasSide - contentHeight) / 2)
            return ImageLetterboxLayout(
                canvasSide = canvasSide,
                sourceWidth = sourceWidth,
                sourceHeight = sourceHeight,
                contentWidth = contentWidth,
                contentHeight = contentHeight,
                offsetX = offsetX,
                offsetY = offsetY,
            )
        }
    }
}
