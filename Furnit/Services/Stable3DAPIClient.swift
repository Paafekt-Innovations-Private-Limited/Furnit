import Foundation
import UIKit

// API client for Stable Fast 3D CUDA API
// Handles image upload, job status polling, and model download
@MainActor
class Stable3DAPIClient: ObservableObject {
    // API base URL for the backend service
    private let baseURL = "https://adf2923417480.notebooks.jarvislabs.net/proxy/8000"
    
    // URLSession for API requests with proper configuration
    private let urlSession: URLSession
    
    // API response models
    struct GenerateResponse: Codable {
        let jobId: String
        let message: String
        let estimatedTime: Int?
        
        enum CodingKeys: String, CodingKey {
            case jobId = "job_id"
            case message
            case estimatedTime = "estimated_time"
        }
    }
    
    struct StatusResponse: Codable {
        let status: String
        let progress: Double?
        let message: String?
        let estimatedTime: Int?
        
        enum CodingKeys: String, CodingKey {
            case status
            case progress
            case message
            case estimatedTime = "estimated_time"
        }
        
        // Custom initializer to handle various response formats
        init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            
            // Status is required
            status = try container.decode(String.self, forKey: .status)
            
            // All other fields are optional with defaults
            progress = try container.decodeIfPresent(Double.self, forKey: .progress)
            message = try container.decodeIfPresent(String.self, forKey: .message)
            estimatedTime = try container.decodeIfPresent(Int.self, forKey: .estimatedTime)
        }
        
        // Manual initializer for fallback parsing
        init(status: String, progress: Double? = nil, message: String? = nil, estimatedTime: Int? = nil) {
            self.status = status
            self.progress = progress
            self.message = message
            self.estimatedTime = estimatedTime
        }
    }
    
    // Helper struct for manual parsing
    private struct ManualStatusResponse {
        let status: String
        let progress: Double?
        let message: String?
        let estimatedTime: Int?
    }
    
    struct HealthResponse: Codable {
        let status: String
        let version: String
        let cuda: Bool?
    }
    
    // Job processing states
    enum JobStatus {
        case pending
        case processing
        case completed
        case failed
        case unknown
        
        init(from statusString: String) {
            switch statusString.lowercased() {
            case "pending", "queued":
                self = .pending
            case "processing", "running", "generating":
                self = .processing
            case "completed", "finished", "done":
                self = .completed
            case "failed", "error":
                self = .failed
            default:
                self = .unknown
            }
        }
    }
    
    // API errors
    enum APIError: Error, LocalizedError {
        case invalidURL
        case invalidImageData
        case requestFailed(String)
        case decodingFailed(String)
        case jobNotFound
        case serverError(Int)
        case networkError(Error)
        
        var errorDescription: String? {
            switch self {
            case .invalidURL:
                return "Invalid API URL"
            case .invalidImageData:
                return "Invalid image data for upload"
            case .requestFailed(let message):
                return "Request failed: \(message)"
            case .decodingFailed(let message):
                return "Failed to decode response: \(message)"
            case .jobNotFound:
                return "Job not found on server"
            case .serverError(let code):
                return "Server error: HTTP \(code)"
            case .networkError(let error):
                return "Network error: \(error.localizedDescription)"
            }
        }
    }
    
    init() {
        // Configure URLSession with timeout for long-running requests
        let configuration = URLSessionConfiguration.default
        configuration.timeoutIntervalForRequest = 30.0
        configuration.timeoutIntervalForResource = 300.0 // 5 minutes for downloads
        self.urlSession = URLSession(configuration: configuration)
        
        print("🌐 Stable3DAPIClient initialized with base URL: \(baseURL)")
    }
    
    // Check API health and availability
    func checkHealth() async throws -> HealthResponse {
        guard let url = URL(string: "\(baseURL)/health") else {
            throw APIError.invalidURL
        }
        
        print("🏥 Checking API health at: \(url)")
        
        do {
            let (data, response) = try await urlSession.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.requestFailed("Invalid response type")
            }
            
            guard httpResponse.statusCode == 200 else {
                throw APIError.serverError(httpResponse.statusCode)
            }
            
            let healthResponse = try JSONDecoder().decode(HealthResponse.self, from: data)
            print("✅ API health check successful: \(healthResponse)")
            return healthResponse
            
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }
    
    // Submit image for 3D generation
    func generateModel(from image: UIImage) async throws -> GenerateResponse {
        guard let url = URL(string: "\(baseURL)/generate") else {
            throw APIError.invalidURL
        }
        
        // Convert UIImage to JPEG data with reasonable compression
        guard let imageData = image.jpegData(compressionQuality: 0.8) else {
            throw APIError.invalidImageData
        }
        
        print("📤 Submitting image for 3D generation")
        print("   Image size: \(image.size)")
        print("   Data size: \(imageData.count) bytes")
        
        // Create multipart form data request
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        // Build multipart body
        var body = Data()
        
        // Add image file with correct field name expected by API
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"file\"; filename=\"furniture.jpg\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: image/jpeg\r\n\r\n".data(using: .utf8)!)
        body.append(imageData)
        body.append("\r\n".data(using: .utf8)!)
        
        // Close multipart body
        body.append("--\(boundary)--\r\n".data(using: .utf8)!)
        
        request.httpBody = body
        
        do {
            let (data, response) = try await urlSession.data(for: request)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.requestFailed("Invalid response type")
            }
            
            if httpResponse.statusCode == 200 || httpResponse.statusCode == 202 {
                let generateResponse = try JSONDecoder().decode(GenerateResponse.self, from: data)
                print("✅ 3D generation job submitted successfully")
                print("   Job ID: \(generateResponse.jobId)")
                print("   Message: \(generateResponse.message)")
                return generateResponse
                
            } else {
                // Enhanced error handling for different HTTP status codes
                let errorMessage = try parseErrorResponse(data: data, statusCode: httpResponse.statusCode)
                
                switch httpResponse.statusCode {
                case 422:
                    throw APIError.requestFailed("Invalid image data: \(errorMessage)")
                case 400:
                    throw APIError.requestFailed("Bad request: \(errorMessage)")
                default:
                    throw APIError.serverError(httpResponse.statusCode)
                }
            }
            
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }
    
    // Check status of generation job
    func checkJobStatus(jobId: String) async throws -> StatusResponse {
        guard let url = URL(string: "\(baseURL)/status/\(jobId)") else {
            throw APIError.invalidURL
        }
        
        print("🔍 Checking status for job: \(jobId)")
        
        do {
            let (data, response) = try await urlSession.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.requestFailed("Invalid response type")
            }
            
            switch httpResponse.statusCode {
            case 200:
                // Log raw response for debugging
                if let responseString = String(data: data, encoding: .utf8) {
                    print("🔍 Raw status response: \(responseString)")
                }
                
                do {
                    let statusResponse = try JSONDecoder().decode(StatusResponse.self, from: data)
                    print("📊 Job status: \(statusResponse.status)")
                    if let progress = statusResponse.progress {
                        print("   Progress: \(Int(progress * 100))%")
                    }
                    return statusResponse
                } catch {
                    print("⚠️ Failed to decode status response: \(error)")
                    print("📄 Response data: \(String(data: data, encoding: .utf8) ?? "Unable to decode")")
                    
                    // Try to parse manually as fallback
                    if let jsonObject = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                        print("🔍 Parsed JSON object: \(jsonObject)")
                        
                        // Try to extract status manually
                        if let status = jsonObject["status"] as? String {
                            print("📊 Manually extracted status: \(status)")
                            
                            // Create manual StatusResponse
                            let manualResponse = ManualStatusResponse(
                                status: status,
                                progress: jsonObject["progress"] as? Double,
                                message: jsonObject["message"] as? String,
                                estimatedTime: jsonObject["estimated_time"] as? Int
                            )
                            
                            // Convert to our expected format
                            return StatusResponse(
                                status: manualResponse.status,
                                progress: manualResponse.progress,
                                message: manualResponse.message,
                                estimatedTime: manualResponse.estimatedTime
                            )
                        }
                    }
                    
                    throw APIError.decodingFailed("Status response: \(error.localizedDescription)")
                }
                
            case 404:
                throw APIError.jobNotFound
                
            default:
                throw APIError.serverError(httpResponse.statusCode)
            }
            
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }
    
    // Download completed 3D model
    func downloadModel(jobId: String) async throws -> Data {
        guard let url = URL(string: "\(baseURL)/download/\(jobId)") else {
            throw APIError.invalidURL
        }
        
        print("⬇️ Downloading 3D model for job: \(jobId)")
        
        do {
            let (data, response) = try await urlSession.data(from: url)
            
            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.requestFailed("Invalid response type")
            }
            
            switch httpResponse.statusCode {
            case 200:
                print("✅ 3D model downloaded successfully")
                print("   File size: \(data.count) bytes")
                
                // Verify we got USDZ data by checking file signature
                if data.count >= 8 {
                    let magicBytes = data.prefix(8)
                    let magicString = String(data: magicBytes, encoding: .ascii) ?? ""
                    print("   File format signature: \(magicString.prefix(4))")
                    
                    // USDZ files start with "PK" (ZIP archive) since USDZ is a ZIP container
                    if magicBytes.starts(with: [0x50, 0x4B]) {
                        print("   ✅ USDZ file format confirmed (ZIP container)")
                    } else {
                        print("   ⚠️ Unexpected file format - expected USDZ")
                    }
                }
                
                return data
                
            case 404:
                throw APIError.jobNotFound
                
            default:
                throw APIError.serverError(httpResponse.statusCode)
            }
            
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }
    
    // Poll job status with incremental delays (2s, 5s, 8s pattern)
    func pollJobStatus(jobId: String, onStatusUpdate: @escaping (StatusResponse) -> Void) async throws -> StatusResponse {
        let delays: [TimeInterval] = [2.0, 5.0, 8.0] // Incremental polling delays
        var delayIndex = 0
        
        print("🔄 Starting incremental polling for job: \(jobId)")
        
        while true {
            do {
                let status = try await checkJobStatus(jobId: jobId)
                onStatusUpdate(status)
                
                let jobStatus = JobStatus(from: status.status)
                
                switch jobStatus {
                case .completed:
                    print("✅ Job completed successfully")
                    return status
                    
                case .failed:
                    let errorMessage = status.message ?? "Job failed with unknown error"
                    throw APIError.requestFailed(errorMessage)
                    
                case .pending, .processing, .unknown:
                    // Continue polling with incremental delay
                    let currentDelay = delays[min(delayIndex, delays.count - 1)]
                    print("⏳ Job still \(status.status), waiting \(currentDelay)s before next check")
                    
                    try await Task.sleep(nanoseconds: UInt64(currentDelay * 1_000_000_000))
                    delayIndex += 1
                    continue
                }
                
            } catch {
                print("⚠️ Error during polling: \(error)")
                throw error
            }
        }
    }
    
    // Complete workflow: generate, poll, and download
    func generateComplete3DModel(from image: UIImage, onStatusUpdate: @escaping (String, Double?) -> Void) async throws -> Data {
        // Step 1: Submit image for generation
        onStatusUpdate("Uploading image to server...", nil)
        let generateResponse = try await generateModel(from: image)
        
        // Step 2: Poll until completion
        onStatusUpdate("Processing 3D model...", 0.0)
        let _ = try await pollJobStatus(jobId: generateResponse.jobId) { status in
            let progress = status.progress ?? 0.0
            let statusMessage = self.getStatusMessage(for: status.status, progress: progress)
            onStatusUpdate(statusMessage, progress)
        }
        
        // Step 3: Download the completed model
        onStatusUpdate("Downloading 3D model...", 0.95)
        let modelData = try await downloadModel(jobId: generateResponse.jobId)
        
        onStatusUpdate("3D model ready!", 1.0)
        return modelData
    }
    
    // Parse error response from API
    private func parseErrorResponse(data: Data, statusCode: Int) throws -> String {
        // Try to parse structured error response (FastAPI format)
        if let errorData = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            // Handle FastAPI validation error format
            if let detail = errorData["detail"] as? [[String: Any]] {
                let messages = detail.compactMap { error in
                    if let msg = error["msg"] as? String,
                       let loc = error["loc"] as? [String] {
                        return "\(loc.joined(separator: ".")): \(msg)"
                    }
                    return error["msg"] as? String
                }.joined(separator: ", ")
                return messages.isEmpty ? "Validation failed" : messages
            }
            
            // Handle simple error format
            if let message = errorData["message"] as? String {
                return message
            }
            
            // Handle detail string format
            if let detail = errorData["detail"] as? String {
                return detail
            }
        }
        
        // Fallback to raw response if possible
        if let responseString = String(data: data, encoding: .utf8) {
            return responseString
        }
        
        return "HTTP \(statusCode) error"
    }
    
    // Generate user-friendly status messages
    private func getStatusMessage(for status: String, progress: Double) -> String {
        switch status.lowercased() {
        case "pending", "queued":
            return "Waiting in queue..."
        case "processing", "running":
            if progress > 0.8 {
                return "Baking final textures..."
            } else if progress > 0.6 {
                return "Generating mesh geometry..."
            } else if progress > 0.3 {
                return "Processing depth information..."
            } else {
                return "Analyzing furniture structure..."
            }
        case "generating":
            return "Creating 3D model..."
        case "finished", "completed", "done":
            return "3D model generation complete!"
        default:
            return "Processing 3D model..."
        }
    }
}