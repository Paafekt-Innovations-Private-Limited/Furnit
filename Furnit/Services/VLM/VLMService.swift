// VLMService.swift
// Service for calling Vision-Language Models (Claude, GPT-4V, Gemini)
// VLM acts as "interior designer brain" - comments on geometry, doesn't compute it

import Foundation
import UIKit

// MARK: - VLM Provider

/// Supported VLM providers
public enum VLMProvider: String, Codable, CaseIterable {
    case claude = "claude"
    case openai = "openai"
    case gemini = "gemini"

    public var displayName: String {
        switch self {
        case .claude: return "Claude (Anthropic)"
        case .openai: return "GPT-4 Vision (OpenAI)"
        case .gemini: return "Gemini Pro Vision (Google)"
        }
    }

    public var defaultModel: String {
        switch self {
        case .claude: return "claude-3-5-sonnet-20241022"
        case .openai: return "gpt-4o"
        case .gemini: return "gemini-1.5-pro"
        }
    }
}

// MARK: - VLM Service

/// Service for getting design suggestions from VLMs
public class VLMService {

    // MARK: - Configuration

    public struct Config {
        public var provider: VLMProvider = .claude
        public var apiKey: String?
        public var baseURL: String?
        public var maxTokens: Int = 1024
        public var temperature: Float = 0.7
        public var timeout: TimeInterval = 30

        public init(provider: VLMProvider = .claude, apiKey: String? = nil) {
            self.provider = provider
            self.apiKey = apiKey
        }
    }

    public var config: Config

    // MARK: - Initialization

    public init(config: Config = Config()) {
        self.config = config
    }

    // MARK: - Main API

    /// Get design suggestions from VLM
    /// - Parameters:
    ///   - request: The design request with room context and candidates
    ///   - image: Optional annotated room image
    /// - Returns: VLM response with recommendations
    public func getDesignSuggestions(
        request: VLMDesignRequest,
        image: UIImage? = nil
    ) async throws -> VLMDesignResponse {
        guard let apiKey = config.apiKey, !apiKey.isEmpty else {
            throw VLMError.missingAPIKey
        }

        // Build request with image if provided
        var requestWithImage = request
        if let image = image,
           let imageData = image.jpegData(compressionQuality: 0.8) {
            requestWithImage.annotatedImageBase64 = imageData.base64EncodedString()
        }

        switch config.provider {
        case .claude:
            return try await callClaude(request: requestWithImage)
        case .openai:
            return try await callOpenAI(request: requestWithImage)
        case .gemini:
            return try await callGemini(request: requestWithImage)
        }
    }

    /// Quick suggestion without full analysis
    public func quickSuggest(
        furniture: FurnitureItem,
        room: RoomContext,
        bestCandidate: PlacementCandidate
    ) async throws -> String {
        let request = VLMDesignRequest(
            roomContext: room,
            furniture: furniture,
            candidates: [bestCandidate],
            questions: ["Is this a good placement? Brief answer."]
        )

        let response = try await getDesignSuggestions(request: request)
        return response.recommendation
    }

    // MARK: - Claude API

    private func callClaude(request: VLMDesignRequest) async throws -> VLMDesignResponse {
        let url = URL(string: config.baseURL ?? "https://api.anthropic.com/v1/messages")!

        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue(config.apiKey, forHTTPHeaderField: "x-api-key")
        httpRequest.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
        httpRequest.timeoutInterval = config.timeout

        // Build message content
        var content: [[String: Any]] = []

        // Add image if present
        if let imageBase64 = request.annotatedImageBase64 {
            content.append([
                "type": "image",
                "source": [
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": imageBase64
                ]
            ])
        }

        // Add text prompt
        content.append([
            "type": "text",
            "text": request.generatePrompt()
        ])

        let body: [String: Any] = [
            "model": config.provider.defaultModel,
            "max_tokens": config.maxTokens,
            "messages": [
                ["role": "user", "content": content]
            ]
        ]

        httpRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: httpRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw VLMError.invalidResponse
        }

        if httpResponse.statusCode != 200 {
            let errorText = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw VLMError.apiError(code: httpResponse.statusCode, message: errorText)
        }

        return try parseClaudeResponse(data)
    }

    private func parseClaudeResponse(_ data: Data) throws -> VLMDesignResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let content = json["content"] as? [[String: Any]],
              let firstContent = content.first,
              let text = firstContent["text"] as? String else {
            throw VLMError.parseError("Could not parse Claude response")
        }

        return parseVLMText(text)
    }

    // MARK: - OpenAI API

    private func callOpenAI(request: VLMDesignRequest) async throws -> VLMDesignResponse {
        let url = URL(string: config.baseURL ?? "https://api.openai.com/v1/chat/completions")!

        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.setValue("Bearer \(config.apiKey ?? "")", forHTTPHeaderField: "Authorization")
        httpRequest.timeoutInterval = config.timeout

        // Build message content
        var content: [[String: Any]] = []

        // Add image if present
        if let imageBase64 = request.annotatedImageBase64 {
            content.append([
                "type": "image_url",
                "image_url": [
                    "url": "data:image/jpeg;base64,\(imageBase64)"
                ]
            ])
        }

        // Add text prompt
        content.append([
            "type": "text",
            "text": request.generatePrompt()
        ])

        let body: [String: Any] = [
            "model": config.provider.defaultModel,
            "max_tokens": config.maxTokens,
            "temperature": config.temperature,
            "messages": [
                ["role": "user", "content": content]
            ]
        ]

        httpRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: httpRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw VLMError.invalidResponse
        }

        if httpResponse.statusCode != 200 {
            let errorText = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw VLMError.apiError(code: httpResponse.statusCode, message: errorText)
        }

        return try parseOpenAIResponse(data)
    }

    private func parseOpenAIResponse(_ data: Data) throws -> VLMDesignResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = json["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let text = message["content"] as? String else {
            throw VLMError.parseError("Could not parse OpenAI response")
        }

        return parseVLMText(text)
    }

    // MARK: - Gemini API

    private func callGemini(request: VLMDesignRequest) async throws -> VLMDesignResponse {
        let baseURL = config.baseURL ?? "https://generativelanguage.googleapis.com/v1beta/models"
        let model = config.provider.defaultModel
        let url = URL(string: "\(baseURL)/\(model):generateContent?key=\(config.apiKey ?? "")")!

        var httpRequest = URLRequest(url: url)
        httpRequest.httpMethod = "POST"
        httpRequest.setValue("application/json", forHTTPHeaderField: "Content-Type")
        httpRequest.timeoutInterval = config.timeout

        // Build parts
        var parts: [[String: Any]] = []

        // Add image if present
        if let imageBase64 = request.annotatedImageBase64 {
            parts.append([
                "inline_data": [
                    "mime_type": "image/jpeg",
                    "data": imageBase64
                ]
            ])
        }

        // Add text prompt
        parts.append([
            "text": request.generatePrompt()
        ])

        let body: [String: Any] = [
            "contents": [
                ["parts": parts]
            ],
            "generationConfig": [
                "maxOutputTokens": config.maxTokens,
                "temperature": config.temperature
            ]
        ]

        httpRequest.httpBody = try JSONSerialization.data(withJSONObject: body)

        let (data, response) = try await URLSession.shared.data(for: httpRequest)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw VLMError.invalidResponse
        }

        if httpResponse.statusCode != 200 {
            let errorText = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw VLMError.apiError(code: httpResponse.statusCode, message: errorText)
        }

        return try parseGeminiResponse(data)
    }

    private func parseGeminiResponse(_ data: Data) throws -> VLMDesignResponse {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let candidates = json["candidates"] as? [[String: Any]],
              let firstCandidate = candidates.first,
              let content = firstCandidate["content"] as? [String: Any],
              let parts = content["parts"] as? [[String: Any]],
              let firstPart = parts.first,
              let text = firstPart["text"] as? String else {
            throw VLMError.parseError("Could not parse Gemini response")
        }

        return parseVLMText(text)
    }

    // MARK: - Response Parsing

    private func parseVLMText(_ text: String) -> VLMDesignResponse {
        var response = VLMDesignResponse()
        response.rawResponse = text
        response.recommendation = text

        // Try to extract structured information
        let lowerText = text.lowercased()

        // Detect recommended option
        if lowerText.contains("option 1") || lowerText.contains("first option") || lowerText.contains("first placement") {
            response.recommendedCandidateIndex = 0
        } else if lowerText.contains("option 2") || lowerText.contains("second option") {
            response.recommendedCandidateIndex = 1
        } else if lowerText.contains("option 3") || lowerText.contains("third option") {
            response.recommendedCandidateIndex = 2
        }

        // Extract adjustments
        response.adjustments = extractAdjustments(from: text)

        // Extract style suggestions
        response.styleSuggestions = extractStyleSuggestions(from: text)

        // Extract concerns
        response.concerns = extractConcerns(from: text)

        // Estimate confidence based on language
        response.confidence = estimateConfidence(from: text)

        return response
    }

    private func extractAdjustments(from text: String) -> [PlacementAdjustment] {
        var adjustments: [PlacementAdjustment] = []

        // Look for rotation suggestions
        let rotationPatterns = [
            ("rotate", "90", Float.pi / 2),
            ("rotate", "45", Float.pi / 4),
            ("rotate", "180", Float.pi),
            ("turn", "90", Float.pi / 2),
            ("angle", "45", Float.pi / 4)
        ]

        let lowerText = text.lowercased()
        for (verb, degrees, radians) in rotationPatterns {
            if lowerText.contains(verb) && lowerText.contains(degrees) {
                adjustments.append(PlacementAdjustment(
                    type: .rotate,
                    amount: radians,
                    reason: "VLM suggested \(degrees)° rotation"
                ))
                break
            }
        }

        // Look for shift suggestions
        let shiftPatterns = [
            ("shift", "left", PlacementAdjustment.AdjustmentType.shiftX, Float(-0.2)),
            ("shift", "right", .shiftX, Float(0.2)),
            ("move", "left", .shiftX, Float(-0.2)),
            ("move", "right", .shiftX, Float(0.2)),
            ("move", "forward", .shiftZ, Float(0.2)),
            ("move", "back", .shiftZ, Float(-0.2)),
            ("shift", "20cm", .shiftX, Float(0.2)),
            ("shift", "40cm", .shiftX, Float(0.4))
        ]

        for (verb, direction, type, amount) in shiftPatterns {
            if lowerText.contains(verb) && lowerText.contains(direction) {
                adjustments.append(PlacementAdjustment(
                    type: type,
                    amount: amount,
                    reason: "VLM suggested shifting \(direction)"
                ))
                break
            }
        }

        return adjustments
    }

    private func extractStyleSuggestions(from text: String) -> [String] {
        var suggestions: [String] = []

        // Color suggestions
        let colors = ["lighter", "darker", "neutral", "warm", "cool", "white", "gray", "beige", "navy", "forest green"]
        for color in colors {
            if text.lowercased().contains(color) {
                suggestions.append("Consider \(color) tones")
                break
            }
        }

        // Material suggestions
        let materials = ["fabric", "leather", "wood", "metal", "glass", "natural", "matte", "glossy"]
        for material in materials {
            if text.lowercased().contains(material) {
                suggestions.append("Consider \(material) finish")
                break
            }
        }

        return suggestions
    }

    private func extractConcerns(from text: String) -> [String] {
        var concerns: [String] = []

        let concernPhrases = [
            ("cramped", "Space may feel cramped"),
            ("tight", "Clearance is tight"),
            ("blocked", "May block traffic flow"),
            ("dark", "Area may be too dark"),
            ("crowded", "Room may feel crowded"),
            ("close to", "Close proximity to other items"),
            ("clearance", "Limited clearance around furniture")
        ]

        let lowerText = text.lowercased()
        for (phrase, concern) in concernPhrases {
            if lowerText.contains(phrase) {
                concerns.append(concern)
            }
        }

        return Array(Set(concerns))  // Remove duplicates
    }

    private func estimateConfidence(from text: String) -> Float {
        let lowerText = text.lowercased()

        // High confidence indicators
        let highConfidence = ["definitely", "clearly", "perfect", "ideal", "excellent", "best choice"]
        for phrase in highConfidence {
            if lowerText.contains(phrase) {
                return 0.9
            }
        }

        // Medium confidence indicators
        let mediumConfidence = ["good", "works well", "suitable", "appropriate", "reasonable"]
        for phrase in mediumConfidence {
            if lowerText.contains(phrase) {
                return 0.7
            }
        }

        // Low confidence indicators
        let lowConfidence = ["might", "could", "perhaps", "consider", "uncertain", "depends"]
        for phrase in lowConfidence {
            if lowerText.contains(phrase) {
                return 0.5
            }
        }

        return 0.6  // Default medium confidence
    }
}

// MARK: - VLM Errors

public enum VLMError: LocalizedError {
    case missingAPIKey
    case invalidResponse
    case apiError(code: Int, message: String)
    case parseError(String)
    case timeout
    case networkError(Error)

    public var errorDescription: String? {
        switch self {
        case .missingAPIKey:
            return "API key not configured"
        case .invalidResponse:
            return "Invalid response from VLM"
        case .apiError(let code, let message):
            return "API error (\(code)): \(message)"
        case .parseError(let detail):
            return "Could not parse response: \(detail)"
        case .timeout:
            return "Request timed out"
        case .networkError(let error):
            return "Network error: \(error.localizedDescription)"
        }
    }
}

// MARK: - VLM Manager (Singleton)

/// Shared manager for VLM operations
public class VLMManager {
    public static let shared = VLMManager()

    public let spatialReasoning = SpatialReasoningService()
    public var vlmService: VLMService

    private init() {
        self.vlmService = VLMService()
    }

    /// Configure the VLM service with API credentials
    public func configure(provider: VLMProvider, apiKey: String? = nil) {
        vlmService = VLMService(config: VLMService.Config(provider: provider, apiKey: apiKey))
    }

    /// Complete placement analysis with geometry + VLM
    /// - Parameters:
    ///   - furniture: Furniture to place
    ///   - room: Room context
    ///   - image: Optional room image for VLM
    ///   - includeVLM: Whether to call VLM for suggestions
    /// - Returns: Complete placement result
    public func analyzePlacement(
        furniture: FurnitureItem,
        room: RoomContext,
        image: UIImage? = nil,
        includeVLM: Bool = true
    ) async -> PlacementResult {
        let startTime = Date()

        // Step 1: Generate candidates using spatial reasoning
        let candidates = spatialReasoning.generateCandidates(furniture: furniture, room: room)
        let solverTime = Date().timeIntervalSince(startTime) * 1000

        let fits = !candidates.filter { $0.isValid }.isEmpty

        var vlmResponse: VLMDesignResponse?
        var vlmTime: Double?

        // Step 2: Get VLM suggestions (if enabled and API key available)
        if includeVLM && vlmService.config.apiKey != nil {
            let vlmStartTime = Date()
            do {
                let request = VLMDesignRequest(
                    roomContext: room,
                    furniture: furniture,
                    candidates: Array(candidates.prefix(3))
                )
                vlmResponse = try await vlmService.getDesignSuggestions(request: request, image: image)
            } catch {
                print("VLM error: \(error.localizedDescription)")
            }
            vlmTime = Date().timeIntervalSince(vlmStartTime) * 1000
        }

        return PlacementResult(
            fits: fits,
            candidates: candidates,
            vlmResponse: vlmResponse,
            solverTimeMs: solverTime,
            vlmTimeMs: vlmTime
        )
    }
}
