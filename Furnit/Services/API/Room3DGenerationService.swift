import Foundation
import UIKit

/// Service for handling 3D room generation via remote API
/// Manages upload, status polling, and download with progress tracking
@MainActor
class Room3DGenerationService: NSObject, ObservableObject {

    // MARK: - Published Properties

    /// Current upload progress (0.0 to 1.0)
    @Published var uploadProgress: Float = 0.0

    /// Current download progress (0.0 to 1.0)
    @Published var downloadProgress: Float = 0.0

    /// Current generation status
    @Published var status: GenerationStatus = .idle

    /// Human-readable status message
    @Published var statusMessage: String = "Ready"

    // MARK: - Configuration

    /// Base URL for the API
    private let baseURL = "https://cf45ae3674750.notebooks.jarvislabs.net/proxy/8000"

    /// Polling interval for status checks (in seconds)
    private let pollingInterval: TimeInterval = 2.5

    /// Maximum time to wait for processing (in seconds)
    private let processingTimeout: TimeInterval = 600 // 10 minutes

    // MARK: - Private Properties

    /// URLSession for upload operations with progress tracking
    private lazy var uploadSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 300
        return URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }()

    /// URLSession for download operations with progress tracking
    private lazy var downloadSession: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 600 // 10 min for large files
        return URLSession(configuration: config, delegate: self, delegateQueue: .main)
    }()

    /// Current upload task for cancellation
    private var currentUploadTask: URLSessionUploadTask?

    /// Current download task for cancellation
    private var currentDownloadTask: URLSessionDownloadTask?

    /// Current generation task for cancellation
    private var currentGenerationTask: Task<URL, Error>?

    /// Continuation for download completion
    private var downloadContinuation: CheckedContinuation<URL, Error>?

    /// Directory for saving models
    private var modelsDirectory: URL {
        let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)
    }

    // MARK: - Main Generation Flow

    /// Generate a 3D room model from an image
    /// - Parameter image: The source image to process
    /// - Returns: URL to the downloaded PLY file
    func generateRoom(from image: UIImage) async throws -> URL {
        logDebug("🚀 [Room3DGenerationService] Starting room generation")

        // Reset state
        resetState()

        // Create a task that can be cancelled
        let task = Task<URL, Error> {
            // Step 1: Upload image
            let jobId = try await uploadImage(image)
            logDebug("✅ [Room3DGenerationService] Upload complete. Job ID: \(jobId)")

            // Step 2: Poll for completion
            try await pollUntilComplete(jobId: jobId)
            logDebug("✅ [Room3DGenerationService] Processing complete")

            // Step 3: Download the PLY file
            let fileURL = try await downloadPLY(jobId: jobId)
            logDebug("✅ [Room3DGenerationService] Download complete: \(fileURL.path)")

            // Update status
            await MainActor.run {
                self.status = .completed(fileURL: fileURL)
                self.statusMessage = "Complete!"
            }

            return fileURL
        }

        currentGenerationTask = task
        return try await task.value
    }

    /// Cancel the current generation operation
    func cancelGeneration() {
        logDebug("❌ [Room3DGenerationService] Cancelling generation")

        currentGenerationTask?.cancel()
        currentUploadTask?.cancel()
        currentDownloadTask?.cancel()

        status = .failed("Cancelled")
        statusMessage = "Cancelled"
    }

    // MARK: - Upload

    /// Upload an image to the generate endpoint
    /// - Parameter image: The image to upload
    /// - Returns: The job ID from the server
    private func uploadImage(_ image: UIImage) async throws -> String {
        logDebug("📤 [Room3DGenerationService] Preparing image upload")

        status = .uploading
        statusMessage = "Uploading image..."
        uploadProgress = 0.0

        // Convert image to JPEG data with reasonable compression
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw GenerationError.invalidImage
        }

        logDebug("📤 [Room3DGenerationService] Image size: \(imageData.count / 1024) KB")

        // Create the request
        guard let url = URL(string: "\(baseURL)/api/v1/generate") else {
            throw GenerationError.uploadFailed(underlying: nil)
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"

        // Create multipart form data
        let boundary = UUID().uuidString
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        // Build the body
        var body = Data()

        // Add the file field
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"image.jpeg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        logDebug("📤 [Room3DGenerationService] Total upload size: \(body.count / 1024) KB")

        // Perform the upload with progress tracking
        return try await withCheckedThrowingContinuation { continuation in
            let task = uploadSession.uploadTask(with: request, from: body) { data, response, error in
                if let error = error {
                    logDebug("❌ [Room3DGenerationService] Upload error: \(error)")
                    continuation.resume(throwing: GenerationError.uploadFailed(underlying: error))
                    return
                }

                guard let httpResponse = response as? HTTPURLResponse else {
                    continuation.resume(throwing: GenerationError.uploadFailed(underlying: nil))
                    return
                }

                logDebug("📤 [Room3DGenerationService] Upload response status: \(httpResponse.statusCode)")

                // Accept all 2xx success status codes (200 OK, 201 Created, 202 Accepted, etc.)
                guard (200...299).contains(httpResponse.statusCode), let data = data else {
                    logDebug("❌ [Room3DGenerationService] Upload failed with status: \(httpResponse.statusCode)")
                    continuation.resume(throwing: GenerationError.uploadFailed(underlying: nil))
                    return
                }

                do {
                    let generateResponse = try JSONDecoder().decode(GenerateResponse.self, from: data)
                    logDebug("✅ [Room3DGenerationService] Got job ID: \(generateResponse.jobId)")
                    continuation.resume(returning: generateResponse.jobId)
                } catch {
                    logDebug("❌ [Room3DGenerationService] Failed to decode response: \(error)")
                    continuation.resume(throwing: GenerationError.uploadFailed(underlying: error))
                }
            }

            currentUploadTask = task
            task.resume()
        }
    }

    // MARK: - Status Polling

    /// Poll the status endpoint until job completes or fails
    /// - Parameter jobId: The job ID to check
    private func pollUntilComplete(jobId: String) async throws {
        logDebug("🔄 [Room3DGenerationService] Starting status polling for job: \(jobId)")

        status = .processing
        statusMessage = "Processing..."

        let startTime = Date()

        while true {
            // Check for cancellation
            try Task.checkCancellation()

            // Check for timeout
            if Date().timeIntervalSince(startTime) > processingTimeout {
                throw GenerationError.timeout
            }

            // Check status
            let statusResponse = try await checkStatus(jobId: jobId)

            if statusResponse.isCompleted {
                logDebug("✅ [Room3DGenerationService] Job completed!")
                return
            }

            if statusResponse.isFailed {
                let errorMessage = statusResponse.errorMessage ?? "Unknown error"
                logDebug("❌ [Room3DGenerationService] Job failed: \(errorMessage)")
                throw GenerationError.serverError(errorMessage)
            }

            // Update status message based on server status
            statusMessage = "Processing: \(statusResponse.status)..."
            logDebug("🔄 [Room3DGenerationService] Status: \(statusResponse.status)")

            // Wait before next poll
            try await Task.sleep(nanoseconds: UInt64(pollingInterval * 1_000_000_000))
        }
    }

    /// Check the status of a job
    /// - Parameter jobId: The job ID to check
    /// - Returns: The status response
    private func checkStatus(jobId: String) async throws -> StatusResponse {
        guard let url = URL(string: "\(baseURL)/api/v1/status/\(jobId)") else {
            throw GenerationError.statusCheckFailed
        }

        do {
            let (data, response) = try await URLSession.shared.data(from: url)

            guard let httpResponse = response as? HTTPURLResponse,
                  httpResponse.statusCode == 200 else {
                throw GenerationError.statusCheckFailed
            }

            return try JSONDecoder().decode(StatusResponse.self, from: data)
        } catch {
            if error is GenerationError {
                throw error
            }
            throw GenerationError.statusCheckFailed
        }
    }

    // MARK: - Download

    /// Download the PLY file for a completed job
    /// - Parameter jobId: The job ID to download
    /// - Returns: URL to the saved PLY file
    private func downloadPLY(jobId: String) async throws -> URL {
        logDebug("📥 [Room3DGenerationService] Starting PLY download for job: \(jobId)")

        status = .downloading
        statusMessage = "Downloading 3D model..."
        downloadProgress = 0.0

        guard let url = URL(string: "\(baseURL)/api/v1/download/\(jobId)") else {
            throw GenerationError.downloadFailed(underlying: nil)
        }

        // Use download task for large file streaming
        return try await withCheckedThrowingContinuation { continuation in
            self.downloadContinuation = continuation

            let task = downloadSession.downloadTask(with: url)
            currentDownloadTask = task
            task.resume()
        }
    }

    // MARK: - File Management

    /// Synchronous file save - MUST be called from nonisolated delegate context before it returns
    /// iOS deletes the temp file as soon as didFinishDownloadingTo returns, so we must copy immediately
    /// FileManager operations are thread-safe
    /// - Parameter tempURL: Temporary URL of the downloaded file
    /// - Returns: Result with saved file URL or error
    nonisolated private func saveDownloadedFileSync(from tempURL: URL) -> Result<URL, Error> {
        logDebug("💾 [Room3DGenerationService] Saving downloaded file (sync)")
        logDebug("💾 [Room3DGenerationService] Temp file path: \(tempURL.path)")

        do {
            // Get documents directory (can't use computed property in nonisolated context)
            let documentsDirectory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let modelsDir = documentsDirectory.appendingPathComponent("SavedRooms", isDirectory: true)

            // Ensure SavedRooms directory exists
            if !FileManager.default.fileExists(atPath: modelsDir.path) {
                try FileManager.default.createDirectory(at: modelsDir, withIntermediateDirectories: true)
                logDebug("📁 [Room3DGenerationService] Created SavedRooms directory")
            }

            // Generate filename with timestamp
            let dateFormatter = DateFormatter()
            dateFormatter.dateFormat = "yyyyMMdd_HHmmss"
            let timestamp = dateFormatter.string(from: Date())
            let fileName = "Room_\(timestamp).ply"

            let destinationURL = modelsDir.appendingPathComponent(fileName)
            logDebug("💾 [Room3DGenerationService] Destination: \(destinationURL.path)")

            // Copy file immediately - temp file exists now but will be deleted after delegate returns
            try FileManager.default.copyItem(at: tempURL, to: destinationURL)

            // Verify file was saved
            let attributes = try FileManager.default.attributesOfItem(atPath: destinationURL.path)
            let fileSize = attributes[.size] as? UInt64 ?? 0
            logDebug("✅ [Room3DGenerationService] File saved: \(fileName) (\(fileSize / 1024 / 1024) MB)")

            return .success(destinationURL)
        } catch {
            logDebug("❌ [Room3DGenerationService] Sync save failed: \(error)")
            return .failure(GenerationError.storageFailed(underlying: error))
        }
    }

    // MARK: - State Management

    /// Reset service state for a new generation
    private func resetState() {
        uploadProgress = 0.0
        downloadProgress = 0.0
        status = .idle
        statusMessage = "Ready"
        currentUploadTask = nil
        currentDownloadTask = nil
        currentGenerationTask = nil
    }
}

// MARK: - URLSessionTaskDelegate

extension Room3DGenerationService: URLSessionTaskDelegate {

    /// Track upload progress
    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didSendBodyData bytesSent: Int64,
        totalBytesSent: Int64,
        totalBytesExpectedToSend: Int64
    ) {
        let progress = Float(totalBytesSent) / Float(totalBytesExpectedToSend)

        Task { @MainActor in
            self.uploadProgress = progress
            self.statusMessage = "Uploading... \(Int(progress * 100))%"
            logDebug("📤 [Room3DGenerationService] Upload progress: \(Int(progress * 100))%")
        }
    }
}

// MARK: - URLSessionDownloadDelegate

extension Room3DGenerationService: URLSessionDownloadDelegate {

    /// Handle download completion
    /// CRITICAL: Must copy file SYNCHRONOUSLY before this method returns
    /// iOS deletes the temp file immediately after this delegate returns
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        logDebug("📥 [Room3DGenerationService] Download finished to temp location")

        // CRITICAL: Copy file SYNCHRONOUSLY before delegate returns
        // iOS deletes the temp file as soon as this method returns
        let result = saveDownloadedFileSync(from: location)

        // Now safe to switch to MainActor for continuation resumption
        Task { @MainActor in
            switch result {
            case .success(let savedURL):
                self.downloadContinuation?.resume(returning: savedURL)
            case .failure(let error):
                logDebug("❌ [Room3DGenerationService] Failed to save file: \(error)")
                self.downloadContinuation?.resume(throwing: error)
            }
            self.downloadContinuation = nil
        }
    }

    /// Track download progress
    nonisolated func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        // Handle unknown content length
        let progress: Float
        if totalBytesExpectedToWrite > 0 {
            progress = Float(totalBytesWritten) / Float(totalBytesExpectedToWrite)
        } else {
            // If content length unknown, show MB downloaded
            progress = 0.5 // Indeterminate
        }

        let mbDownloaded = Float(totalBytesWritten) / 1_048_576

        Task { @MainActor in
            self.downloadProgress = progress
            if totalBytesExpectedToWrite > 0 {
                self.statusMessage = "Downloading... \(Int(progress * 100))%"
            } else {
                self.statusMessage = String(format: "Downloading... %.1f MB", mbDownloaded)
            }
            logDebug("📥 [Room3DGenerationService] Download progress: \(String(format: "%.1f MB", mbDownloaded))")
        }
    }

    /// Handle download errors
    nonisolated func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error = error {
            logDebug("❌ [Room3DGenerationService] Task error: \(error)")

            Task { @MainActor in
                // Only handle if we have a pending continuation
                if let continuation = self.downloadContinuation {
                    continuation.resume(throwing: GenerationError.downloadFailed(underlying: error))
                    self.downloadContinuation = nil
                }
            }
        }
    }
}
