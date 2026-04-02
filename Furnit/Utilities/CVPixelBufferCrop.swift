import Foundation
import CoreVideo
import CoreImage

/// Crops a pixel buffer to an integral rectangle (top-left origin, same as typical camera buffers + CIImage).
enum CVPixelBufferCrop {

    private static let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    static func crop(buffer: CVPixelBuffer, to rect: CGRect) -> CVPixelBuffer? {
        let fullW = CVPixelBufferGetWidth(buffer)
        let fullH = CVPixelBufferGetHeight(buffer)

        let ix = max(0, Int(floor(rect.origin.x)))
        let iy = max(0, Int(floor(rect.origin.y)))
        let iw = min(fullW - ix, Int(ceil(rect.width)))
        let ih = min(fullH - iy, Int(ceil(rect.height)))

        guard iw >= 32, ih >= 32 else { return nil }

        let cropCGRect = CGRect(x: ix, y: iy, width: iw, height: ih)
        let ciImage = CIImage(cvPixelBuffer: buffer)
        let cropped = ciImage.cropped(to: cropCGRect)
            .transformed(by: CGAffineTransform(translationX: -CGFloat(ix), y: -CGFloat(iy)))

        let attrs: [String: Any] = [
            kCVPixelBufferCGImageCompatibilityKey as String: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey as String: true
        ]
        var out: CVPixelBuffer?
        let status = CVPixelBufferCreate(kCFAllocatorDefault, iw, ih, kCVPixelFormatType_32BGRA, attrs as CFDictionary, &out)
        guard status == kCVReturnSuccess, let outBuf = out else { return nil }

        Self.ciContext.render(cropped, to: outBuf)
        return outBuf
    }
}
