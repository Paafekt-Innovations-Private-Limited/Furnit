import CoreGraphics
import Foundation

/// Aspect-preserving letterbox into a square canvas (uniform scale, centered, padded).
struct ImageLetterboxLayout: Sendable {
    let canvasSide: Int
    let sourceWidth: Int
    let sourceHeight: Int
    let contentWidth: Int
    let contentHeight: Int
    let offsetX: Int
    let offsetY: Int

    var uniformScale: Float {
        Float(contentWidth) / Float(max(sourceWidth, 1))
    }

    /// Multiply model-space focal length by this to get focal length in source pixels (fx = fy).
    var focalScaleToSource: Float {
        1.0 / max(uniformScale, 1e-6)
    }

    static func layout(sourceWidth: Int, sourceHeight: Int, canvasSide: Int) -> ImageLetterboxLayout {
        let scale = min(
            Float(canvasSide) / Float(max(sourceWidth, 1)),
            Float(canvasSide) / Float(max(sourceHeight, 1))
        )
        let contentWidth = max(1, Int((Float(sourceWidth) * scale).rounded()))
        let contentHeight = max(1, Int((Float(sourceHeight) * scale).rounded()))
        let offsetX = max(0, (canvasSide - contentWidth) / 2)
        let offsetY = max(0, (canvasSide - contentHeight) / 2)
        return ImageLetterboxLayout(
            canvasSide: canvasSide,
            sourceWidth: sourceWidth,
            sourceHeight: sourceHeight,
            contentWidth: contentWidth,
            contentHeight: contentHeight,
            offsetX: offsetX,
            offsetY: offsetY
        )
    }

    func cropValuesFromCanvas(_ values: [Float], canvasWidth: Int, canvasHeight: Int) -> [Float]? {
        guard canvasWidth > 0,
              canvasHeight > 0,
              values.count >= canvasWidth * canvasHeight,
              offsetX + contentWidth <= canvasWidth,
              offsetY + contentHeight <= canvasHeight else {
            return nil
        }

        var cropped = [Float]()
        cropped.reserveCapacity(contentWidth * contentHeight)
        for row in offsetY..<(offsetY + contentHeight) {
            let start = row * canvasWidth + offsetX
            cropped.append(contentsOf: values[start..<(start + contentWidth)])
        }
        return cropped
    }
}

enum ImageLetterboxRenderer {
    /// Draw into a CGContext that has already been flipped (translate + scaleY -1).
    static func drawLetterboxedFlipped(
        cgImage: CGImage,
        into context: CGContext,
        layout: ImageLetterboxLayout
    ) {
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: layout.canvasSide, height: layout.canvasSide))
        let drawY = layout.canvasSide - layout.offsetY - layout.contentHeight
        context.draw(
            cgImage,
            in: CGRect(
                x: layout.offsetX,
                y: drawY,
                width: layout.contentWidth,
                height: layout.contentHeight
            )
        )
    }
}
