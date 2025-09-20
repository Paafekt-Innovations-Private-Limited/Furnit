import Foundation
import Vision
import UIKit

@MainActor
class QRCodeDetectionService: ObservableObject {
    // Published properties for UI updates
    @Published var isDetecting = false
    @Published var errorMessage: String?

    // Detection result
    struct QRDetectionResult {
        let hasQRCode: Bool
        let extractedURL: URL?
        let qrCodeString: String?
    }

    // Detect QR codes in the provided image
    func detectQRCode(in image: UIImage) async -> QRDetectionResult {
        await MainActor.run {
            isDetecting = true
            errorMessage = nil
        }

        defer {
            Task { @MainActor in
                isDetecting = false
            }
        }

        guard let cgImage = image.cgImage else {
            await MainActor.run {
                errorMessage = "Failed to convert UIImage to CGImage"
            }
            print("❌ QR Detection: Failed to convert UIImage to CGImage")
            return QRDetectionResult(hasQRCode: false, extractedURL: nil, qrCodeString: nil)
        }

        return await withCheckedContinuation { continuation in
            // Create Vision request for barcode detection
            let request = VNDetectBarcodesRequest { [weak self] request, error in
                Task { @MainActor in
                    if let error = error {
                        self?.errorMessage = "QR detection failed: \(error.localizedDescription)"
                        print("❌ QR Detection error: \(error.localizedDescription)")
                        continuation.resume(returning: QRDetectionResult(hasQRCode: false, extractedURL: nil, qrCodeString: nil))
                        return
                    }

                    guard let observations = request.results as? [VNBarcodeObservation] else {
                        print("📱 QR Detection: No QR codes found in image")
                        continuation.resume(returning: QRDetectionResult(hasQRCode: false, extractedURL: nil, qrCodeString: nil))
                        return
                    }

                    // Find QR codes
                    for observation in observations {
                        if observation.symbology == .qr,
                           let qrCodeString = observation.payloadStringValue {

                            print("🔍 QR Code detected: \(qrCodeString)")

                            // Try to extract URL from QR code
                            if let url = self?.extractURL(from: qrCodeString) {
                                print("✅ Valid URL extracted from QR code: \(url)")
                                continuation.resume(returning: QRDetectionResult(
                                    hasQRCode: true,
                                    extractedURL: url,
                                    qrCodeString: qrCodeString
                                ))
                                return
                            } else {
                                print("⚠️ QR code found but no valid URL extracted")
                                continuation.resume(returning: QRDetectionResult(
                                    hasQRCode: true,
                                    extractedURL: nil,
                                    qrCodeString: qrCodeString
                                ))
                                return
                            }
                        }
                    }

                    print("📱 QR Detection: No QR codes found in image")
                    continuation.resume(returning: QRDetectionResult(hasQRCode: false, extractedURL: nil, qrCodeString: nil))
                }
            }

            // Set up request to look specifically for QR codes
            request.symbologies = [.qr]

            // Perform the Vision request
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])

            DispatchQueue.global(qos: .userInitiated).async {
                do {
                    try handler.perform([request])
                } catch {
                    Task { @MainActor in
                        self.errorMessage = "QR detection failed: \(error.localizedDescription)"
                        print("❌ QR Detection error: \(error.localizedDescription)")
                    }
                    continuation.resume(returning: QRDetectionResult(hasQRCode: false, extractedURL: nil, qrCodeString: nil))
                }
            }
        }
    }

    // Extract URL from QR code string, handling various formats
    private func extractURL(from qrString: String) -> URL? {
        // Direct URL
        if let url = URL(string: qrString), url.scheme != nil {
            return url
        }

        // Add https:// if missing but looks like a URL
        if qrString.contains(".") && !qrString.hasPrefix("http") {
            if let url = URL(string: "https://\(qrString)") {
                return url
            }
        }

        // Check if it's a data URL or other special format
        if qrString.hasPrefix("data:") || qrString.hasPrefix("file:") {
            return URL(string: qrString)
        }

        print("⚠️ Could not extract valid URL from QR code: \(qrString)")
        return nil
    }

    // Validate if URL points to a 3D asset using multiple validation methods
    func isValid3DAssetURL(_ url: URL) async -> Bool {
        let supportedExtensions = ["usdz", "glb", "gltf", "obj", "dae"]
        let pathExtension = url.pathExtension.lowercased()

        print("🔍 URL validation: \(url.absoluteString)")
        print("   Extension: \(pathExtension)")

        // First check: URL extension validation
        if supportedExtensions.contains(pathExtension) && !pathExtension.isEmpty {
            print("   ✅ Valid 3D asset extension found")
            return true
        }

        // Second check: HTTP HEAD request to check Content-Type
        print("   🌐 No extension found, checking HTTP headers...")
        let isValidByHeaders = await checkContentTypeHeaders(url)
        if isValidByHeaders {
            print("   ✅ Valid 3D asset MIME type found in headers")
            return true
        }

        print("   ❌ No valid 3D asset indicators found")
        return false
    }

    // Check Content-Type headers via HTTP HEAD request
    private func checkContentTypeHeaders(_ url: URL) async -> Bool {
        do {
            // Create HEAD request to get headers without downloading content
            var request = URLRequest(url: url)
            request.httpMethod = "HEAD"
            request.timeoutInterval = 10.0

            let (_, response) = try await URLSession.shared.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                print("   ⚠️ Invalid HTTP response")
                return false
            }

            guard httpResponse.statusCode == 200 else {
                print("   ⚠️ HTTP error: \(httpResponse.statusCode)")
                return false
            }

            // Check Content-Type header for 3D asset MIME types
            let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type")?.lowercased() ?? ""
            print("   Content-Type: \(contentType)")

            let supported3DMimeTypes = [
                "model/vnd.usdz+zip",           // USDZ files
                "model/usd+zip",                // Alternative USDZ MIME type
                "model/gltf-binary",            // GLB files
                "model/gltf+json",              // GLTF files
                "model/obj",                    // OBJ files
                "application/octet-stream",     // Generic binary (could be GLB/USDZ)
                "application/zip"               // ZIP files (could be USDZ)
            ]

            for mimeType in supported3DMimeTypes {
                if contentType.contains(mimeType) {
                    print("   🎯 Found supported MIME type: \(mimeType)")
                    return true
                }
            }

            // Check Content-Disposition header for filename hints
            if let contentDisposition = httpResponse.value(forHTTPHeaderField: "Content-Disposition") {
                print("   Content-Disposition: \(contentDisposition)")

                // Extract filename from Content-Disposition header
                let filenamePattern = #"filename[^;=\n]*=((['"]).*?\2|[^;\n]*)"#
                if let regex = try? NSRegularExpression(pattern: filenamePattern, options: []),
                   let match = regex.firstMatch(in: contentDisposition, range: NSRange(contentDisposition.startIndex..., in: contentDisposition)),
                   let filenameRange = Range(match.range(at: 1), in: contentDisposition) {

                    let filename = String(contentDisposition[filenameRange]).trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
                    let fileExtension = (filename as NSString).pathExtension.lowercased()
                    print("   Filename from header: \(filename), Extension: \(fileExtension)")

                    if ["usdz", "glb", "gltf", "obj", "dae"].contains(fileExtension) {
                        print("   🎯 Found supported extension in filename: \(fileExtension)")
                        return true
                    }
                }
            }

            return false

        } catch {
            print("   ⚠️ HTTP HEAD request failed: \(error.localizedDescription)")
            return false
        }
    }

    // Reset detection state
    func reset() {
        isDetecting = false
        errorMessage = nil
    }
}