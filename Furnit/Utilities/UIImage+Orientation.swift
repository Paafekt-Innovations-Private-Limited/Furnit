import UIKit

extension UIImage {
    /// Re-render the image with `.up` orientation using the modern renderer API
    /// (lower transient memory than the legacy `UIGraphicsBeginImageContextWithOptions`).
    func fixedOrientation() -> UIImage {
        guard imageOrientation != .up else { return self }

        let format = UIGraphicsImageRendererFormat()
        format.scale = scale
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        return renderer.image { _ in
            draw(in: CGRect(origin: .zero, size: size))
        }
    }
}
