import CoreGraphics
import simd

/// Mean color of the furniture cutout (premultiplied RGBA from ``compositeCpuBilinearProtoMaskCutout``).
enum FurnitureSegmentationMeanColor {
    /// Un-premultiplies and averages straight sRGB channels in 0…1 for pixels with alpha ≥ `alphaThreshold`.
    static func meanStraightSRGB(cgImage: CGImage, alphaThreshold: UInt8 = 16) -> SIMD3<Float>? {
        let width = cgImage.width
        let height = cgImage.height
        guard width > 0, height > 0 else { return nil }

        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        let bufferSize = height * bytesPerRow
        var buffer = [UInt8](repeating: 0, count: bufferSize)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(
            data: &buffer,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { return nil }

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        var sumR: Double = 0
        var sumG: Double = 0
        var sumB: Double = 0
        var count: Int = 0
        var rowStart = 0
        for _ in 0..<height {
            var x = 0
            var index = rowStart
            while x < width {
                let redPremul = buffer[index]
                let greenPremul = buffer[index + 1]
                let bluePremul = buffer[index + 2]
                let alpha = buffer[index + 3]
                index += 4
                x += 1
                if alpha < alphaThreshold { continue }
                let alphaF = Double(alpha) / 255.0
                guard alphaF > 1.0 / 255.0 else { continue }
                let rStraight = min(1.0, Double(redPremul) / alphaF / 255.0)
                let gStraight = min(1.0, Double(greenPremul) / alphaF / 255.0)
                let bStraight = min(1.0, Double(bluePremul) / alphaF / 255.0)
                sumR += rStraight
                sumG += gStraight
                sumB += bStraight
                count += 1
            }
            rowStart += bytesPerRow
        }

        guard count > 0 else { return nil }
        let inv = 1.0 / Double(count)
        return SIMD3<Float>(
            Float(sumR * inv),
            Float(sumG * inv),
            Float(sumB * inv)
        )
    }
}
