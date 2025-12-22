import Foundation
import SwiftUI

// AR processing states for enhanced user experience
// Provides detailed status updates and progress tracking during 3D model generation
enum ARProcessingState: Equatable {
    case idle
    case pointing
    case capturing
    case detectingQR
    case downloadingAsset(progress: Double)
    case uploading
    case processing(progress: Double)
    case baking
    case downloading
    case ready
    case error(String)
    
    // Display message for UI
    var displayMessage: String {
        switch self {
        case .idle:
            return ""
        case .pointing:
            return "Point at furniture objects"
        case .capturing:
            return "Capturing image..."
        case .detectingQR:
            return "Scanning for QR codes..."
        case .downloadingAsset(let progress):
            if progress > 0.8 {
                return "Almost ready..."
            } else if progress > 0.5 {
                return "Downloading 3D model..."
            } else {
                return "Connecting to server..."
            }
        case .uploading:
            return "Uploading to server..."
        case .processing(let progress):
            if progress > 0.8 {
                return "Baking final textures..."
            } else if progress > 0.6 {
                return "Generating mesh geometry..."
            } else if progress > 0.3 {
                return "Processing depth information..."
            } else {
                return "Analyzing furniture structure..."
            }
        case .baking:
            return "Finalizing 3D model..."
        case .downloading:
            return "Downloading 3D model..."
        case .ready:
            return "Tap to place furniture"
        case .error(let message):
            return message
        }
    }
    
    // Secondary message for additional context
    var secondaryMessage: String? {
        switch self {
        case .pointing:
            return "Aim camera at chairs, tables, or sofas"
        case .capturing:
            return "Hold steady while capturing"
        case .detectingQR:
            return "Looking for QR codes with 3D asset links"
        case .downloadingAsset(let progress):
            return "Progress: \(Int(progress * 100))%"
        case .uploading:
            return "Sending image for processing"
        case .processing(let progress):
            return "Progress: \(Int(progress * 100))%"
        case .downloading:
            return "Preparing for placement"
        case .ready:
            return "Touch screen where you want to place"
        default:
            return nil
        }
    }
    
    // Whether to show progress indicator
    var showsProgress: Bool {
        switch self {
        case .capturing, .detectingQR, .downloadingAsset, .uploading, .processing, .baking, .downloading:
            return true
        default:
            return false
        }
    }
    
    // Progress value (0.0 to 1.0)
    var progressValue: Double {
        switch self {
        case .capturing:
            return 0.1
        case .detectingQR:
            return 0.15
        case .downloadingAsset(let progress):
            return 0.2 + (progress * 0.7) // Map to 20%-90% range for QR asset download
        case .uploading:
            return 0.2
        case .processing(let progress):
            return 0.2 + (progress * 0.7) // Map to 20%-90% range for backend processing
        case .baking:
            return 0.9
        case .downloading:
            return 0.95
        case .ready:
            return 1.0
        default:
            return 0.0
        }
    }
    
    // Whether the state represents an error condition
    var isError: Bool {
        if case .error = self {
            return true
        }
        return false
    }
    
    // Whether AR processing is currently active
    var isProcessing: Bool {
        switch self {
        case .capturing, .detectingQR, .downloadingAsset, .uploading, .processing, .baking, .downloading:
            return true
        default:
            return false
        }
    }
    
    // Whether the system is ready for object placement
    var isReadyToPlace: Bool {
        return self == .ready
    }
    
    // Color scheme for UI elements based on state
    var statusColor: Color {
        switch self {
        case .idle, .pointing:
            return .blue
        case .capturing, .detectingQR, .downloadingAsset, .uploading, .processing, .baking, .downloading:
            return .orange
        case .ready:
            return .green
        case .error:
            return .red
        }
    }
    
    // Icon name for the current state
    var iconName: String {
        switch self {
        case .idle, .pointing:
            return "viewfinder"
        case .capturing:
            return "camera.fill"
        case .detectingQR:
            return "qrcode.viewfinder"
        case .downloadingAsset:
            return "square.and.arrow.down"
        case .uploading:
            return "icloud.and.arrow.up"
        case .processing, .baking:
            return "gear"
        case .downloading:
            return "icloud.and.arrow.down"
        case .ready:
            return "hand.point.up.brailledot.fill"
        case .error:
            return "exclamationmark.triangle"
        }
    }
    
    // Animation style for progress indicator
    var progressAnimation: Animation? {
        switch self {
        case .capturing, .detectingQR, .uploading, .downloading:
            return .easeInOut(duration: 0.5).repeatForever(autoreverses: true)
        case .downloadingAsset, .processing, .baking:
            return .linear(duration: 1.0).repeatForever(autoreverses: false)
        default:
            return .easeInOut(duration: 0.3)
        }
    }
}

// Observable object to manage AR processing state
@MainActor
class ARProcessingStateManager: ObservableObject {
    // Current processing state
    @Published var currentState: ARProcessingState = .idle
    
    // Processing start time for duration tracking
    private var processingStartTime: Date?
    
    // Update the current processing state
    func updateState(_ newState: ARProcessingState) {
        let previousState = currentState
        currentState = newState
        
        // Log state changes for debugging
        logDebug("🔄 AR State: \(previousState) → \(newState)")
        
        // Track processing duration
        if newState.isProcessing && !previousState.isProcessing {
            processingStartTime = Date()
            logDebug("⏱️ AR processing started")
        } else if !newState.isProcessing && previousState.isProcessing {
            if let startTime = processingStartTime {
                let duration = Date().timeIntervalSince(startTime)
                logDebug("⏱️ AR processing completed in \(String(format: "%.1f", duration))s")
            }
            processingStartTime = nil
        }
    }
    
    // Reset to idle state
    func reset() {
        updateState(.idle)
        processingStartTime = nil
    }
    
    // Handle error states
    func setError(_ message: String) {
        updateState(.error(message))
    }
    
    // Update processing progress
    func updateProgress(_ progress: Double) {
        updateState(.processing(progress: progress))
    }
    
    // Get elapsed processing time
    var elapsedProcessingTime: TimeInterval? {
        guard let startTime = processingStartTime else { return nil }
        return Date().timeIntervalSince(startTime)
    }
    
    // Get formatted elapsed time string
    var formattedElapsedTime: String {
        guard let elapsed = elapsedProcessingTime else { return "" }
        return String(format: "%.0fs", elapsed)
    }
}

// Extension for convenient state transitions
extension ARProcessingStateManager {
    // Start AR session
    func startARSession() {
        updateState(.pointing)
    }
    
    // Begin image capture
    func beginCapture() {
        updateState(.capturing)
    }

    // Begin QR code detection
    func beginQRDetection() {
        updateState(.detectingQR)
    }

    // Begin asset download from QR code URL
    func beginAssetDownload(progress: Double = 0.0) {
        updateState(.downloadingAsset(progress: progress))
    }

    // Update asset download progress
    func updateAssetDownloadProgress(_ progress: Double) {
        updateState(.downloadingAsset(progress: progress))
    }

    // Begin upload process
    func beginUpload() {
        updateState(.uploading)
    }
    
    // Update processing with progress
    func updateProcessing(progress: Double) {
        updateState(.processing(progress: progress))
    }
    
    // Enter baking phase
    func beginBaking() {
        updateState(.baking)
    }
    
    // Begin model download
    func beginDownload() {
        updateState(.downloading)
    }
    
    // Ready for placement
    func readyForPlacement() {
        updateState(.ready)
    }
    
    // Handle API errors with user-friendly messages
    func handleAPIError(_ error: Error) {
        let userFriendlyMessage: String
        
        if let apiError = error as? Stable3DAPIClient.APIError {
            switch apiError {
            case .invalidURL:
                userFriendlyMessage = "Configuration error. Please try again."
            case .invalidImageData:
                userFriendlyMessage = "Invalid image. Please capture again."
            case .requestFailed(let message):
                userFriendlyMessage = "Server error: \(message)"
            case .jobNotFound:
                userFriendlyMessage = "Processing job lost. Please try again."
            case .serverError(let code):
                userFriendlyMessage = "Server error (\(code)). Please try again."
            case .networkError:
                userFriendlyMessage = "Network error. Check connection."
            case .decodingFailed:
                userFriendlyMessage = "Invalid server response. Try again."
            }
        } else {
            userFriendlyMessage = "Processing failed. Please try again."
        }
        
        setError(userFriendlyMessage)
    }
}
