import CoreVideo
import Foundation

/// Cheap pre-inference gate so Fit does not burn the detector on covered-lens / fully black frames.
enum FurnitureFitFrameUsability {
    /// Mean Rec.601 luma (0…255). Covered lens frames are typically well below this.
    static let maxMeanLuminance: Double = 12
    static let defaultSampleStep = 16

    /// Returns true when a sparse sample of the frame is essentially black.
    static func isFullyDark(_ pixelBuffer: CVPixelBuffer, sampleStep: Int = defaultSampleStep) -> Bool {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        guard width > 0, height > 0, sampleStep > 0 else { return false }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
        switch format {
        case kCVPixelFormatType_32BGRA, kCVPixelFormatType_32RGBA:
            return isFullyDarkBGRA(
                pixelBuffer,
                width: width,
                height: height,
                sampleStep: sampleStep,
                isBGRA: format == kCVPixelFormatType_32BGRA
            )
        case kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
             kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange:
            return isFullyDarkLumaPlane(pixelBuffer, width: width, height: height, sampleStep: sampleStep)
        default:
            // Unknown format: do not block inference.
            return false
        }
    }

    private static func isFullyDarkBGRA(
        _ pixelBuffer: CVPixelBuffer,
        width: Int,
        height: Int,
        sampleStep: Int,
        isBGRA: Bool
    ) -> Bool {
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return false }
        let rowBytes = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var sum = 0.0
        var count = 0
        var y = 0
        while y < height {
            let row = ptr.advanced(by: y * rowBytes)
            var x = 0
            while x < width {
                let i = x * 4
                let c0 = Double(row[i])
                let c1 = Double(row[i + 1])
                let c2 = Double(row[i + 2])
                let r = isBGRA ? c2 : c0
                let g = c1
                let b = isBGRA ? c0 : c2
                sum += 0.299 * r + 0.587 * g + 0.114 * b
                count += 1
                x += sampleStep
            }
            y += sampleStep
        }
        guard count > 0 else { return false }
        return (sum / Double(count)) <= maxMeanLuminance
    }

    private static func isFullyDarkLumaPlane(
        _ pixelBuffer: CVPixelBuffer,
        width: Int,
        height: Int,
        sampleStep: Int
    ) -> Bool {
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else { return false }
        let rowBytes = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let ptr = base.assumingMemoryBound(to: UInt8.self)
        var sum = 0
        var count = 0
        var y = 0
        while y < height {
            let row = ptr.advanced(by: y * rowBytes)
            var x = 0
            while x < width {
                sum += Int(row[x])
                count += 1
                x += sampleStep
            }
            y += sampleStep
        }
        guard count > 0 else { return false }
        return (Double(sum) / Double(count)) <= maxMeanLuminance
    }
}
