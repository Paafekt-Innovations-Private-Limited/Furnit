import Foundation

// MARK: - API Response Models

/// Response from POST /api/v1/generate endpoint
struct GenerateResponse: Codable {
    let jobId: String
    let status: String
    let message: String

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
        case message
    }
}

/// Response from GET /api/v1/status endpoint
struct StatusResponse: Codable {
    let jobId: String
    let status: String
    let createdAt: String
    let updatedAt: String
    let errorMessage: String?

    enum CodingKeys: String, CodingKey {
        case jobId = "job_id"
        case status
        case createdAt = "created_at"
        case updatedAt = "updated_at"
        case errorMessage = "error_message"
    }

    /// Check if job is still in progress
    var isInProgress: Bool {
        return status == "pending" || status == "processing"
    }

    /// Check if job completed successfully
    var isCompleted: Bool {
        return status == "completed"
    }

    /// Check if job failed
    var isFailed: Bool {
        return status == "failed"
    }
}

// MARK: - Generation Status

/// Tracks the current phase of the 3D generation process
enum GenerationStatus: Equatable {
    case idle
    case uploading
    case processing
    case downloading
    case completed(fileURL: URL)
    case failed(String)

    /// Human-readable description of current status
    var description: String {
        switch self {
        case .idle:
            return "Ready"
        case .uploading:
            return "Uploading image..."
        case .processing:
            return "Generating 3D model..."
        case .downloading:
            return "Downloading model..."
        case .completed:
            return "Complete"
        case .failed(let error):
            return "Failed: \(error)"
        }
    }

    /// Icon name for status display
    var iconName: String {
        switch self {
        case .idle:
            return "photo.badge.plus"
        case .uploading:
            return "arrow.up.circle"
        case .processing:
            return "gearshape.2"
        case .downloading:
            return "arrow.down.circle"
        case .completed:
            return "checkmark.circle.fill"
        case .failed:
            return "exclamationmark.triangle.fill"
        }
    }

    /// Whether this status shows a progress bar
    var showsProgress: Bool {
        switch self {
        case .uploading, .downloading:
            return true
        default:
            return false
        }
    }

    /// Whether this status shows an indeterminate spinner
    var showsSpinner: Bool {
        switch self {
        case .processing:
            return true
        default:
            return false
        }
    }
}

// MARK: - Generation Errors

/// Errors that can occur during the 3D generation process
enum GenerationError: LocalizedError {
    case invalidImage
    case uploadFailed(underlying: Error?)
    case networkUnavailable
    case serverError(String)
    case statusCheckFailed
    case downloadFailed(underlying: Error?)
    case storageFailed(underlying: Error?)
    case cancelled
    case timeout

    var errorDescription: String? {
        switch self {
        case .invalidImage:
            return "Could not process the selected image"
        case .uploadFailed(let error):
            if let error = error {
                return "Upload failed: \(error.localizedDescription)"
            }
            return "Failed to upload image"
        case .networkUnavailable:
            return "No internet connection"
        case .serverError(let message):
            return "Server error: \(message)"
        case .statusCheckFailed:
            return "Failed to check processing status"
        case .downloadFailed(let error):
            if let error = error {
                return "Download failed: \(error.localizedDescription)"
            }
            return "Failed to download 3D model"
        case .storageFailed(let error):
            if let error = error {
                return "Storage failed: \(error.localizedDescription)"
            }
            return "Failed to save 3D model"
        case .cancelled:
            return "Operation was cancelled"
        case .timeout:
            return "Operation timed out"
        }
    }

    var recoverySuggestion: String? {
        switch self {
        case .invalidImage:
            return "Try selecting a different image"
        case .uploadFailed, .downloadFailed:
            return "Check your internet connection and try again"
        case .networkUnavailable:
            return "Connect to the internet and try again"
        case .serverError:
            return "Please try again later"
        case .statusCheckFailed:
            return "Please try again"
        case .storageFailed:
            return "Free up some storage space and try again"
        case .cancelled:
            return nil
        case .timeout:
            return "The operation took too long. Please try again"
        }
    }

    /// Whether this error can be retried
    var isRetryable: Bool {
        switch self {
        case .invalidImage, .cancelled:
            return false
        default:
            return true
        }
    }
}

// MARK: - Model File Type

/// Supported 3D model file types
enum ModelFileType: String, Codable {
    case usdz
    case ply

    /// File extension including the dot
    var fileExtension: String {
        return ".\(rawValue)"
    }

    /// Display name for the file type
    var displayName: String {
        switch self {
        case .usdz:
            return "3D Model"
        case .ply:
            return "3D Room"
        }
    }

    /// Icon name for display
    var iconName: String {
        switch self {
        case .usdz:
            return "cube.fill"
        case .ply:
            return "circle.grid.3x3.fill"
        }
    }

    /// Icon color for display
    var iconColorName: String {
        switch self {
        case .usdz:
            return "green"
        case .ply:
            return "purple"
        }
    }
}
