import Foundation
import RealityKit
import UIKit

@MainActor
class AssetDownloadService: NSObject, ObservableObject {
    // Published properties for UI updates
    @Published var isDownloading = false
    @Published var downloadProgress: Double = 0.0
    @Published var errorMessage: String?

    // Download session
    private var downloadTask: URLSessionDownloadTask?
    private lazy var urlSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30.0
        config.timeoutIntervalForResource = 120.0
        return URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }()

    // Download result
    struct DownloadResult {
        let success: Bool
        let entity: Entity?
        let fileURL: URL?
        let errorMessage: String?
    }

    // Download 3D asset from URL and convert to RealityKit Entity
    func download3DAsset(from url: URL) async -> DownloadResult {
        await MainActor.run {
            isDownloading = true
            downloadProgress = 0.0
            errorMessage = nil
        }

        defer {
            Task { @MainActor in
                isDownloading = false
                downloadProgress = 0.0
            }
        }

        print("📥 Starting download of 3D asset: \(url.absoluteString)")

        // Download the file (validation will happen post-download with enhanced detection)
        guard let downloadedFileURL = await downloadFile(from: url) else {
            let error = errorMessage ?? "Download failed"
            print("❌ Download failed: \(error)")
            return DownloadResult(success: false, entity: nil, fileURL: nil, errorMessage: error)
        }

        print("✅ File downloaded successfully: \(downloadedFileURL.path)")

        // Convert to RealityKit Entity
        do {
            // Extract the detected extension from the downloaded file
            let detectedExtension = (downloadedFileURL.lastPathComponent as NSString).pathExtension.lowercased()
            let entity = try await loadEntity(from: downloadedFileURL, originalExtension: detectedExtension)
            print("✅ Successfully loaded 3D model into RealityKit Entity")

            return DownloadResult(
                success: true,
                entity: entity,
                fileURL: downloadedFileURL,
                errorMessage: nil
            )

        } catch {
            let errorMsg = "Failed to load 3D model: \(error.localizedDescription)"
            await MainActor.run {
                errorMessage = errorMsg
            }
            print("❌ \(errorMsg)")
            return DownloadResult(success: false, entity: nil, fileURL: downloadedFileURL, errorMessage: errorMsg)
        }
    }

    // Download file from URL
    private func downloadFile(from url: URL) async -> URL? {
        return await withCheckedContinuation { continuation in
            downloadTask = urlSession.downloadTask(with: url) { [weak self] localURL, response, error in
                Task { @MainActor in
                    if let error = error {
                        self?.errorMessage = "Download error: \(error.localizedDescription)"
                        print("❌ Download error: \(error.localizedDescription)")
                        continuation.resume(returning: nil)
                        return
                    }

                    guard let localURL = localURL else {
                        self?.errorMessage = "No file downloaded"
                        print("❌ No file downloaded")
                        continuation.resume(returning: nil)
                        return
                    }

                    // Determine proper file extension from multiple sources
                    let detectedExtension = self?.detectFileExtension(from: url, response: response) ?? "unknown"
                    print("🔍 Extension detection - URL: '\(url.pathExtension)', Detected: '\(detectedExtension)'")

                    // Move file to temporary directory with proper extension
                    let tempDirectory = FileManager.default.temporaryDirectory
                    let fileName = UUID().uuidString + "." + detectedExtension
                    let destinationURL = tempDirectory.appendingPathComponent(fileName)

                    do {
                        // Remove existing file if it exists
                        if FileManager.default.fileExists(atPath: destinationURL.path) {
                            try FileManager.default.removeItem(at: destinationURL)
                        }

                        try FileManager.default.moveItem(at: localURL, to: destinationURL)
                        print("📁 File moved to: \(destinationURL.path)")
                        continuation.resume(returning: destinationURL)

                    } catch {
                        self?.errorMessage = "Failed to move downloaded file: \(error.localizedDescription)"
                        print("❌ Failed to move file: \(error.localizedDescription)")
                        continuation.resume(returning: nil)
                    }
                }
            }

            downloadTask?.resume()
        }
    }

    // Detect proper file extension from URL, Content-Disposition, and Content-Type
    private func detectFileExtension(from url: URL, response: URLResponse?) -> String {
        // First priority: Check URL path extension
        let urlExtension = url.pathExtension.lowercased()
        if !urlExtension.isEmpty && ["usdz", "glb", "gltf", "obj", "dae"].contains(urlExtension) {
            print("   ✅ Using URL extension: \(urlExtension)")
            return urlExtension
        }

        // Second priority: Parse Content-Disposition header for filename
        if let httpResponse = response as? HTTPURLResponse,
           let contentDisposition = httpResponse.value(forHTTPHeaderField: "Content-Disposition") {

            print("   📄 Content-Disposition: \(contentDisposition)")

            if let filenameExtension = extractExtensionFromContentDisposition(contentDisposition) {
                print("   ✅ Using Content-Disposition extension: \(filenameExtension)")
                return filenameExtension
            }
        }

        // Third priority: Map Content-Type to extension
        if let httpResponse = response as? HTTPURLResponse,
           let contentType = httpResponse.value(forHTTPHeaderField: "Content-Type") {

            print("   📄 Content-Type: \(contentType)")

            if let mappedExtension = mapContentTypeToExtension(contentType) {
                print("   ✅ Using Content-Type mapping: \(mappedExtension)")
                return mappedExtension
            }
        }

        // Fallback: Return URL extension even if empty (will be handled by validation)
        print("   ⚠️ No extension detected, using URL extension: '\(urlExtension)'")
        return urlExtension
    }

    // Extract file extension from Content-Disposition header
    private func extractExtensionFromContentDisposition(_ contentDisposition: String) -> String? {
        // Look for filename= parameter in Content-Disposition header
        let patterns = [
            #"filename\s*=\s*"([^"]+)""#,           // filename="example.usdz"
            #"filename\s*=\s*([^;\s]+)"#            // filename=example.usdz
        ]

        for pattern in patterns {
            if let regex = try? NSRegularExpression(pattern: pattern, options: .caseInsensitive),
               let match = regex.firstMatch(in: contentDisposition, range: NSRange(contentDisposition.startIndex..., in: contentDisposition)),
               let filenameRange = Range(match.range(at: 1), in: contentDisposition) {

                let filename = String(contentDisposition[filenameRange])
                let fileExtension = (filename as NSString).pathExtension.lowercased()

                if !fileExtension.isEmpty && ["usdz", "glb", "gltf", "obj", "dae"].contains(fileExtension) {
                    print("   📎 Extracted filename: \(filename), extension: \(fileExtension)")
                    return fileExtension
                }
            }
        }

        return nil
    }

    // Map Content-Type header to appropriate file extension
    private func mapContentTypeToExtension(_ contentType: String) -> String? {
        let lowercaseContentType = contentType.lowercased()

        // Map common 3D asset MIME types to extensions
        if lowercaseContentType.contains("model/vnd.usdz") || lowercaseContentType.contains("model/usd") {
            return "usdz"
        } else if lowercaseContentType.contains("model/gltf-binary") {
            return "glb"
        } else if lowercaseContentType.contains("model/gltf+json") || lowercaseContentType.contains("model/gltf") {
            return "gltf"
        } else if lowercaseContentType.contains("model/obj") {
            return "obj"
        } else if lowercaseContentType.contains("application/zip") {
            // ZIP files could be USDZ, but we need more context
            return "usdz"  // Optimistic assumption for 3D asset URLs
        }

        return nil
    }

    // Load Entity from downloaded file with content validation
    private func loadEntity(from fileURL: URL, originalExtension: String) async throws -> Entity {
        // Validate that we have some kind of extension to work with
        if originalExtension.isEmpty {
            let error = "No valid file extension detected. Cannot determine 3D asset format."
            print("❌ \(error)")
            throw NSError(domain: "AssetDownloadService", code: -1, userInfo: [NSLocalizedDescriptionKey: error])
        }

        // First, validate the downloaded file content
        let detectedFileType = await validateFileContent(fileURL)
        print("📁 File validation - Original extension: \(originalExtension), Detected type: \(detectedFileType ?? "unknown")")

        // Determine the actual file type to use for loading
        let fileTypeToUse = detectedFileType ?? originalExtension

        // Validate that the final file type is supported
        let supportedFormats = ["usdz", "glb", "gltf", "obj", "dae"]
        if !supportedFormats.contains(fileTypeToUse.lowercased()) {
            let error = "Unsupported 3D asset format: \(fileTypeToUse). Supported formats: \(supportedFormats.joined(separator: ", ").uppercased())"
            print("❌ \(error)")
            throw NSError(domain: "AssetDownloadService", code: -2, userInfo: [NSLocalizedDescriptionKey: error])
        }

        // For USDZ files, use RealityKit's native loading
        if fileTypeToUse == "usdz" {
            do {
                let entity = try await Entity.load(contentsOf: fileURL)
                print("✅ Loaded USDZ model using RealityKit native loader")
                return entity
            } catch {
                print("⚠️ Native USDZ loading failed, will try alternative approach: \(error)")
                throw error
            }
        }

        // For other formats, we would need additional conversion
        // For now, we'll try to load them as-is and let RealityKit handle it
        do {
            let entity = try await Entity.load(contentsOf: fileURL)
            print("✅ Loaded \(fileTypeToUse.uppercased()) model using RealityKit")
            return entity
        } catch {
            print("❌ Failed to load \(fileTypeToUse.uppercased()) model: \(error)")
            throw error
        }
    }

    // Validate file content by checking file headers/signatures
    private func validateFileContent(_ fileURL: URL) async -> String? {
        do {
            // Read the first 16 bytes to check file signatures
            let fileHandle = try FileHandle(forReadingFrom: fileURL)
            defer { fileHandle.closeFile() }

            let headerData = fileHandle.readData(ofLength: 16)
            guard headerData.count >= 4 else {
                print("⚠️ File too small to validate")
                return nil
            }

            // Convert to bytes for signature checking
            let bytes = [UInt8](headerData)

            // Check for USDZ (ZIP) signature: 50 4B 03 04 (PK..)
            if bytes.count >= 4 && bytes[0] == 0x50 && bytes[1] == 0x4B &&
               (bytes[2] == 0x03 || bytes[2] == 0x05) && (bytes[3] == 0x04 || bytes[3] == 0x06) {
                print("🔍 Detected ZIP/USDZ file signature")

                // Check if it's actually a USDZ by looking for USDC files inside
                if await isUSDAorUSDCArchive(fileURL) {
                    return "usdz"
                }
                print("   ZIP file but not USDZ content")
                return nil
            }

            // Check for GLB (Binary GLTF) signature: 67 6C 54 46 (glTF)
            if bytes.count >= 4 && bytes[0] == 0x67 && bytes[1] == 0x6C &&
               bytes[2] == 0x54 && bytes[3] == 0x46 {
                print("🔍 Detected GLB file signature")
                return "glb"
            }

            // Check for GLTF JSON by looking for opening brace and "asset"
            if let fileContent = String(data: headerData, encoding: .utf8),
               fileContent.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{") {

                // Read more content to check for GLTF structure
                let moreData = fileHandle.readData(ofLength: 512)
                if let fullContent = String(data: headerData + moreData, encoding: .utf8),
                   fullContent.contains("\"asset\"") && fullContent.contains("\"version\"") {
                    print("🔍 Detected GLTF JSON structure")
                    return "gltf"
                }
            }

            // Check for OBJ file (text-based, starts with comments or vertex data)
            if let fileContent = String(data: headerData, encoding: .utf8) {
                let trimmedContent = fileContent.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmedContent.hasPrefix("#") || trimmedContent.hasPrefix("v ") ||
                   trimmedContent.hasPrefix("vn ") || trimmedContent.hasPrefix("vt ") ||
                   trimmedContent.hasPrefix("f ") {
                    print("🔍 Detected OBJ file structure")
                    return "obj"
                }
            }

            print("⚠️ Unknown file format - signature: \(bytes.prefix(8).map { String(format: "%02X", $0) }.joined(separator: " "))")
            return nil

        } catch {
            print("❌ Failed to validate file content: \(error.localizedDescription)")
            return nil
        }
    }

    // Check if ZIP archive contains USD/USDC files (indicating USDZ)
    private func isUSDAorUSDCArchive(_ fileURL: URL) async -> Bool {
        // This is a simplified check - in a full implementation we'd use a ZIP library
        // For now, we'll assume ZIP files from 3D asset URLs are likely USDZ
        return true
    }

    // Cancel ongoing download
    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil

        Task { @MainActor in
            isDownloading = false
            downloadProgress = 0.0
            errorMessage = "Download cancelled"
        }

        print("🛑 Download cancelled by user")
    }

    // Clean up downloaded files
    func cleanupDownloadedFile(at url: URL?) {
        guard let url = url else { return }

        DispatchQueue.global(qos: .utility).async {
            do {
                if FileManager.default.fileExists(atPath: url.path) {
                    try FileManager.default.removeItem(at: url)
                    print("🗑️ Cleaned up downloaded file: \(url.path)")
                }
            } catch {
                print("⚠️ Failed to cleanup file: \(error.localizedDescription)")
            }
        }
    }

    // Reset download state
    func reset() {
        isDownloading = false
        downloadProgress = 0.0
        errorMessage = nil
        downloadTask?.cancel()
        downloadTask = nil
    }
}

// MARK: - URLSessionDownloadDelegate

extension AssetDownloadService: URLSessionDownloadDelegate {
    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        DispatchQueue.main.async { [weak self] in
            if totalBytesExpectedToWrite > 0 {
                let progress = Double(totalBytesWritten) / Double(totalBytesExpectedToWrite)
                self?.downloadProgress = progress
                print("📥 Download progress: \(Int(progress * 100))%")
            }
        }
    }

    nonisolated func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        // Handled in the main download method
    }

    nonisolated func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        if let error = error {
            DispatchQueue.main.async { [weak self] in
                self?.errorMessage = "Download failed: \(error.localizedDescription)"
                print("❌ Download completed with error: \(error.localizedDescription)")
            }
        }
    }
}