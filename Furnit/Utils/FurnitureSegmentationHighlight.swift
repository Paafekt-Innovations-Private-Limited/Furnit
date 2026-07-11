import CoreGraphics

/// Production segmentation overlay — Paafekt gold fill from mask alpha (no raw cutout pixels).
enum FurnitureSegmentationHighlight {
    private static let accentR: UInt8 = 201
    private static let accentG: UInt8 = 162
    private static let accentB: UInt8 = 75
    private static let fillAlpha: UInt8 = 108

    static func goldMaskImage(from cgImage: CGImage, alphaThreshold: UInt8 = 16) -> CGImage? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var buffer = [UInt8](repeating: 0, count: height * bytesPerRow)
        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var index = 0
        while index < buffer.count {
            let sourceAlpha = buffer[index + 3]
            if sourceAlpha >= alphaThreshold {
                let premulScale = Float(fillAlpha) / 255.0
                buffer[index] = UInt8(min(255, Int(Float(accentR) * premulScale)))
                buffer[index + 1] = UInt8(min(255, Int(Float(accentG) * premulScale)))
                buffer[index + 2] = UInt8(min(255, Int(Float(accentB) * premulScale)))
                buffer[index + 3] = fillAlpha
            } else {
                buffer[index] = 0
                buffer[index + 1] = 0
                buffer[index + 2] = 0
                buffer[index + 3] = 0
            }
            index += bytesPerPixel
        }

        return context.makeImage()
    }
}
