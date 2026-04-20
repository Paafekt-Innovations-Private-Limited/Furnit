import CoreGraphics

enum FurnitureFitGeometry {
    /// Ultralytics-style `scale_boxes` mapping from model space back to the
    /// original image space. Supports both letterboxed-square inputs and the
    /// current stretch-to-square camera path.
    static func scaleBoxesFromModel(
        box: CGRect,
        modelShape: CGSize,
        imageShape: CGSize,
        usesLetterbox: Bool,
        inputVerticallyFlipped: Bool
    ) -> CGRect {
        let edgeOffset: CGFloat = 0.0
        let x1: CGFloat
        let y1: CGFloat
        let x2: CGFloat
        let y2: CGFloat

        if usesLetterbox {
            let gain = min(modelShape.width / imageShape.width, modelShape.height / imageShape.height)
            let padX = (modelShape.width - imageShape.width * gain) / 2.0
            let padY = (modelShape.height - imageShape.height * gain) / 2.0
            x1 = (box.minX - edgeOffset - padX) / gain
            y1 = (box.minY - edgeOffset - padY) / gain
            x2 = (box.maxX + edgeOffset - padX) / gain
            y2 = (box.maxY + edgeOffset - padY) / gain
        } else {
            let gainX = imageShape.width / modelShape.width
            let gainY = imageShape.height / modelShape.height
            x1 = (box.minX - edgeOffset) * gainX
            y1 = (box.minY - edgeOffset) * gainY
            x2 = (box.maxX + edgeOffset) * gainX
            y2 = (box.maxY + edgeOffset) * gainY
        }

        let flippedY1: CGFloat
        let flippedY2: CGFloat
        if inputVerticallyFlipped {
            flippedY1 = imageShape.height - y2
            flippedY2 = imageShape.height - y1
        } else {
            flippedY1 = y1
            flippedY2 = y2
        }

        let clippedX1 = max(0, x1)
        let clippedY1 = max(0, flippedY1)
        let clippedX2 = min(imageShape.width, x2)
        let clippedY2 = min(imageShape.height, flippedY2)
        let mappedRect = CGRect(
            x: clippedX1,
            y: clippedY1,
            width: max(1, clippedX2 - clippedX1),
            height: max(1, clippedY2 - clippedY1)
        )
        return clipRectToImageBounds(
            rect: mappedRect,
            imageWidth: imageShape.width,
            imageHeight: imageShape.height
        )
    }

    static func clipRectToImageBounds(
        rect: CGRect,
        imageWidth: CGFloat,
        imageHeight: CGFloat
    ) -> CGRect {
        let clippedMinX = max(0, min(rect.minX, imageWidth))
        let clippedMinY = max(0, min(rect.minY, imageHeight))
        let clippedMaxX = max(clippedMinX, min(rect.maxX, imageWidth))
        let clippedMaxY = max(clippedMinY, min(rect.maxY, imageHeight))
        return CGRect(
            x: clippedMinX,
            y: clippedMinY,
            width: max(1, clippedMaxX - clippedMinX),
            height: max(1, clippedMaxY - clippedMinY)
        )
    }

    static func clipCompositedImageToBounds(
        _ image: CGImage,
        imageWidth: Int,
        imageHeight: Int
    ) -> CGImage {
        let clippedWidth = max(1, min(image.width, imageWidth))
        let clippedHeight = max(1, min(image.height, imageHeight))
        let clipRect = CGRect(x: 0, y: 0, width: clippedWidth, height: clippedHeight)
        return image.cropping(to: clipRect) ?? image
    }
}
