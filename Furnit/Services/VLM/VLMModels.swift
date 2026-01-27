// VLMModels.swift
// Data models for VLM (Vision-Language Model) spatial reasoning integration
// VLM provides semantic reasoning on top of geometry computed by YOLO/SHARP pipeline

import Foundation
import simd
import CoreGraphics

// MARK: - Room Context

/// Complete room context for spatial reasoning
public struct RoomContext: Codable {
    /// Room dimensions in meters
    public var dimensions: RoomDimensions

    /// Floor polygon vertices (normalized 0-1 or meters)
    public var floorPolygon: [CGPoint]

    /// Detected walls with their properties
    public var walls: [WallInfo]

    /// Detected obstacles (existing furniture, fixtures)
    public var obstacles: [ObstacleInfo]

    /// Door and window zones (clearance required)
    public var openings: [OpeningInfo]

    /// Walkway corridors that should remain clear
    public var walkways: [WalkwayInfo]

    /// Camera pose when image was captured
    public var cameraPose: CameraPose?

    /// Room style classification (optional)
    public var styleHint: String?

    public init(
        dimensions: RoomDimensions = RoomDimensions(),
        floorPolygon: [CGPoint] = [],
        walls: [WallInfo] = [],
        obstacles: [ObstacleInfo] = [],
        openings: [OpeningInfo] = [],
        walkways: [WalkwayInfo] = [],
        cameraPose: CameraPose? = nil,
        styleHint: String? = nil
    ) {
        self.dimensions = dimensions
        self.floorPolygon = floorPolygon
        self.walls = walls
        self.obstacles = obstacles
        self.openings = openings
        self.walkways = walkways
        self.cameraPose = cameraPose
        self.styleHint = styleHint
    }
}

/// Room dimensions in meters
public struct RoomDimensions: Codable {
    public var width: Float = 0
    public var depth: Float = 0
    public var height: Float = 0

    public var floorArea: Float { width * depth }
    public var volume: Float { width * depth * height }

    public init(width: Float = 0, depth: Float = 0, height: Float = 0) {
        self.width = width
        self.depth = depth
        self.height = height
    }
}

/// Wall information
public struct WallInfo: Codable {
    public var id: String
    public var startPoint: CGPoint  // Floor-level start
    public var endPoint: CGPoint    // Floor-level end
    public var height: Float
    public var normal: SIMD3<Float>? // Outward-facing normal
    public var material: String?     // "painted", "brick", "window_wall"

    public init(id: String, startPoint: CGPoint, endPoint: CGPoint, height: Float = 2.5, normal: SIMD3<Float>? = nil, material: String? = nil) {
        self.id = id
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.height = height
        self.normal = normal
        self.material = material
    }
}

/// Obstacle (existing furniture or fixture)
public struct ObstacleInfo: Codable {
    public var id: String
    public var className: String        // "sofa", "table", "lamp"
    public var footprint: [CGPoint]     // 2D polygon on floor
    public var boundingBox: CGRect      // Axis-aligned bounding box
    public var height: Float
    public var isMovable: Bool

    public init(id: String, className: String, footprint: [CGPoint] = [], boundingBox: CGRect = .zero, height: Float = 1.0, isMovable: Bool = true) {
        self.id = id
        self.className = className
        self.footprint = footprint
        self.boundingBox = boundingBox
        self.height = height
        self.isMovable = isMovable
    }
}

/// Door/window opening
public struct OpeningInfo: Codable {
    public enum OpeningType: String, Codable {
        case door
        case window
        case archway
    }

    public var id: String
    public var type: OpeningType
    public var position: CGPoint        // Center position on floor
    public var width: Float
    public var height: Float
    public var swingClearance: Float    // Required clearance for door swing
    public var wallId: String?          // Which wall this is on

    public init(id: String, type: OpeningType, position: CGPoint = .zero, width: Float = 0.9, height: Float = 2.1, swingClearance: Float = 0.9, wallId: String? = nil) {
        self.id = id
        self.type = type
        self.position = position
        self.width = width
        self.height = height
        self.swingClearance = swingClearance
        self.wallId = wallId
    }
}

/// Walkway corridor definition
public struct WalkwayInfo: Codable {
    public var id: String
    public var startPoint: CGPoint
    public var endPoint: CGPoint
    public var minWidth: Float          // Minimum required width (typically 0.6-0.9m)
    public var priority: Int            // Higher = more important to keep clear

    public init(id: String, startPoint: CGPoint = .zero, endPoint: CGPoint = .zero, minWidth: Float = 0.6, priority: Int = 1) {
        self.id = id
        self.startPoint = startPoint
        self.endPoint = endPoint
        self.minWidth = minWidth
        self.priority = priority
    }
}

/// Camera pose for AR context
public struct CameraPose: Codable {
    public var position: SIMD3<Float>
    public var forward: SIMD3<Float>
    public var up: SIMD3<Float>
    public var fovDegrees: Float

    public init(position: SIMD3<Float> = .zero, forward: SIMD3<Float> = SIMD3(0, 0, -1), up: SIMD3<Float> = SIMD3(0, 1, 0), fovDegrees: Float = 60) {
        self.position = position
        self.forward = forward
        self.up = up
        self.fovDegrees = fovDegrees
    }
}

// MARK: - Furniture

/// Furniture to be placed
public struct FurnitureItem: Codable {
    public var id: String
    public var name: String
    public var category: String         // "seating", "table", "storage", "bed"

    /// Physical dimensions in meters
    public var width: Float
    public var depth: Float
    public var height: Float

    /// Required clearance around furniture (front, back, left, right)
    public var clearance: FurnitureClearance

    /// Preferred placement constraints
    public var placementHints: PlacementHints

    /// Style attributes for VLM
    public var style: FurnitureStyle?

    public var footprintArea: Float { width * depth }

    public init(
        id: String,
        name: String,
        category: String = "furniture",
        width: Float = 1.0,
        depth: Float = 1.0,
        height: Float = 1.0,
        clearance: FurnitureClearance = FurnitureClearance(),
        placementHints: PlacementHints = PlacementHints(),
        style: FurnitureStyle? = nil
    ) {
        self.id = id
        self.name = name
        self.category = category
        self.width = width
        self.depth = depth
        self.height = height
        self.clearance = clearance
        self.placementHints = placementHints
        self.style = style
    }
}

/// Clearance requirements around furniture
public struct FurnitureClearance: Codable {
    public var front: Float = 0.6   // Space needed in front (e.g., for seating)
    public var back: Float = 0.1    // Space behind
    public var left: Float = 0.3    // Space on left
    public var right: Float = 0.3   // Space on right

    public init(front: Float = 0.6, back: Float = 0.1, left: Float = 0.3, right: Float = 0.3) {
        self.front = front
        self.back = back
        self.left = left
        self.right = right
    }

    /// Create clearance envelope polygon for a given furniture footprint and rotation
    public func envelope(forWidth width: Float, depth: Float, atRotation yaw: Float) -> [CGPoint] {
        let totalWidth = left + width + right
        let totalDepth = back + depth + front

        // Create envelope centered at origin, then would be transformed by caller
        let halfW = totalWidth / 2
        let halfD = totalDepth / 2

        return [
            CGPoint(x: CGFloat(-halfW), y: CGFloat(-halfD)),
            CGPoint(x: CGFloat(halfW), y: CGFloat(-halfD)),
            CGPoint(x: CGFloat(halfW), y: CGFloat(halfD)),
            CGPoint(x: CGFloat(-halfW), y: CGFloat(halfD))
        ]
    }
}

/// Hints for furniture placement
public struct PlacementHints: Codable {
    public var preferAgainstWall: Bool = false      // Sofa, bed headboard
    public var preferCorner: Bool = false           // Corner desk, lamp
    public var preferCentered: Bool = false         // Dining table, coffee table
    public var preferNearWindow: Bool = false       // Reading chair, desk
    public var avoidNearDoor: Bool = true           // Don't block doors
    public var faceTowardCenter: Bool = false       // Seating should face room center

    public init(
        preferAgainstWall: Bool = false,
        preferCorner: Bool = false,
        preferCentered: Bool = false,
        preferNearWindow: Bool = false,
        avoidNearDoor: Bool = true,
        faceTowardCenter: Bool = false
    ) {
        self.preferAgainstWall = preferAgainstWall
        self.preferCorner = preferCorner
        self.preferCentered = preferCentered
        self.preferNearWindow = preferNearWindow
        self.avoidNearDoor = avoidNearDoor
        self.faceTowardCenter = faceTowardCenter
    }
}

/// Style attributes for VLM reasoning
public struct FurnitureStyle: Codable {
    public var primaryColor: String?        // "navy blue", "walnut", "white"
    public var material: String?            // "fabric", "leather", "wood", "metal"
    public var styleCategory: String?       // "modern", "traditional", "scandinavian"
    public var finish: String?              // "matte", "glossy", "natural"

    public init(primaryColor: String? = nil, material: String? = nil, styleCategory: String? = nil, finish: String? = nil) {
        self.primaryColor = primaryColor
        self.material = material
        self.styleCategory = styleCategory
        self.finish = finish
    }
}

// MARK: - Placement Candidates

/// A candidate placement pose
public struct PlacementCandidate: Codable, Identifiable {
    public var id: String

    /// Position in room coordinates (meters)
    public var x: Float
    public var y: Float     // Height off floor (usually 0)
    public var z: Float

    /// Rotation in radians (yaw only for floor placement)
    public var yaw: Float

    /// Computed scores
    public var scores: PlacementScores

    /// Constraint violations
    public var violations: [ConstraintViolation]

    /// Whether this placement is physically valid
    public var isValid: Bool { violations.filter { $0.severity == .blocking }.isEmpty }

    /// Combined score (higher = better)
    public var compositeScore: Float { scores.composite }

    public init(
        id: String = UUID().uuidString,
        x: Float = 0,
        y: Float = 0,
        z: Float = 0,
        yaw: Float = 0,
        scores: PlacementScores = PlacementScores(),
        violations: [ConstraintViolation] = []
    ) {
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.scores = scores
        self.violations = violations
    }
}

/// Scores for a placement candidate
public struct PlacementScores: Codable {
    public var fit: Float = 0               // Does furniture fit without collision (0-1)
    public var clearance: Float = 0         // How much clearance around furniture (0-1)
    public var walkway: Float = 0           // Walkway accessibility score (0-1)
    public var wallAlignment: Float = 0     // Alignment with walls (0-1)
    public var cameraVisibility: Float = 0  // Visible from current camera (0-1)
    public var styleMatch: Float = 0        // Style compatibility (0-1, from VLM or heuristic)

    /// Weights for composite score
    public static let defaultWeights = PlacementWeights()

    /// Compute weighted composite score
    public var composite: Float {
        let w = Self.defaultWeights
        return w.fit * fit +
               w.clearance * clearance +
               w.walkway * walkway +
               w.wallAlignment * wallAlignment +
               w.cameraVisibility * cameraVisibility +
               w.styleMatch * styleMatch
    }

    public init(fit: Float = 0, clearance: Float = 0, walkway: Float = 0, wallAlignment: Float = 0, cameraVisibility: Float = 0, styleMatch: Float = 0) {
        self.fit = fit
        self.clearance = clearance
        self.walkway = walkway
        self.wallAlignment = wallAlignment
        self.cameraVisibility = cameraVisibility
        self.styleMatch = styleMatch
    }
}

/// Weights for composite scoring
public struct PlacementWeights: Codable {
    public var fit: Float = 0.30
    public var clearance: Float = 0.20
    public var walkway: Float = 0.20
    public var wallAlignment: Float = 0.15
    public var cameraVisibility: Float = 0.05
    public var styleMatch: Float = 0.10

    public init() {}
}

/// Constraint violation
public struct ConstraintViolation: Codable {
    public enum ViolationType: String, Codable {
        case wallCollision
        case obstacleCollision
        case insufficientClearance
        case walkwayBlocked
        case doorSwingBlocked
        case outOfBounds
        case tooNearDoor
    }

    public enum Severity: String, Codable {
        case blocking   // Cannot place here
        case warning    // Can place but not ideal
        case info       // Minor issue
    }

    public var type: ViolationType
    public var severity: Severity
    public var message: String
    public var relatedObjectId: String?
    public var overlapAmount: Float?        // How much overlap in meters

    public init(type: ViolationType, severity: Severity, message: String, relatedObjectId: String? = nil, overlapAmount: Float? = nil) {
        self.type = type
        self.severity = severity
        self.message = message
        self.relatedObjectId = relatedObjectId
        self.overlapAmount = overlapAmount
    }
}

// MARK: - VLM Request/Response

/// Request to VLM for design suggestions
public struct VLMDesignRequest: Codable {
    /// Room context with measurements
    public var roomContext: RoomContext

    /// Furniture being placed
    public var furniture: FurnitureItem

    /// Top candidate placements from solver
    public var candidates: [PlacementCandidate]

    /// Base64-encoded room image with overlay (optional)
    public var annotatedImageBase64: String?

    /// Specific questions for VLM
    public var questions: [String]

    public init(
        roomContext: RoomContext,
        furniture: FurnitureItem,
        candidates: [PlacementCandidate],
        annotatedImageBase64: String? = nil,
        questions: [String] = []
    ) {
        self.roomContext = roomContext
        self.furniture = furniture
        self.candidates = candidates
        self.annotatedImageBase64 = annotatedImageBase64
        self.questions = questions
    }

    /// Generate prompt for VLM
    public func generatePrompt() -> String {
        var prompt = """
        You are an interior design expert analyzing furniture placement.

        ## Room Context
        - Dimensions: \(String(format: "%.1f", roomContext.dimensions.width))m × \(String(format: "%.1f", roomContext.dimensions.depth))m × \(String(format: "%.1f", roomContext.dimensions.height))m
        - Floor Area: \(String(format: "%.1f", roomContext.dimensions.floorArea)) m²
        """

        if let style = roomContext.styleHint {
            prompt += "\n- Style: \(style)"
        }

        prompt += """


        ## Furniture to Place
        - Name: \(furniture.name)
        - Category: \(furniture.category)
        - Dimensions: \(String(format: "%.2f", furniture.width))m × \(String(format: "%.2f", furniture.depth))m × \(String(format: "%.2f", furniture.height))m
        """

        if let style = furniture.style {
            if let color = style.primaryColor { prompt += "\n- Color: \(color)" }
            if let material = style.material { prompt += "\n- Material: \(material)" }
            if let styleCategory = style.styleCategory { prompt += "\n- Style: \(styleCategory)" }
        }

        prompt += "\n\n## Candidate Placements (ranked by solver)\n"

        for (i, candidate) in candidates.prefix(3).enumerated() {
            prompt += """

            ### Option \(i + 1)
            - Position: (\(String(format: "%.2f", candidate.x))m, \(String(format: "%.2f", candidate.z))m)
            - Rotation: \(String(format: "%.0f", candidate.yaw * 180 / .pi))°
            - Fit Score: \(String(format: "%.2f", candidate.scores.fit))
            - Clearance Score: \(String(format: "%.2f", candidate.scores.clearance))
            """

            if !candidate.violations.isEmpty {
                let warnings = candidate.violations.filter { $0.severity != .blocking }
                if !warnings.isEmpty {
                    prompt += "\n- Notes: \(warnings.map { $0.message }.joined(separator: "; "))"
                }
            }
        }

        prompt += """


        ## Your Task
        Based on the room measurements and candidate placements:
        1. Recommend which placement feels best aesthetically and why
        2. Suggest any small adjustments (rotation, shift) to improve the placement
        3. Provide color/material suggestions that would complement the room
        4. Note any potential issues with the placement that the measurements might not capture

        Keep your response concise and actionable.
        """

        if !questions.isEmpty {
            prompt += "\n\n## Specific Questions\n"
            for question in questions {
                prompt += "- \(question)\n"
            }
        }

        return prompt
    }
}

/// Response from VLM
public struct VLMDesignResponse: Codable {
    /// Recommended candidate index (0-based)
    public var recommendedCandidateIndex: Int?

    /// Explanation for recommendation
    public var recommendation: String

    /// Suggested adjustments
    public var adjustments: [PlacementAdjustment]

    /// Style/color suggestions
    public var styleSuggestions: [String]

    /// Potential issues noted
    public var concerns: [String]

    /// Confidence in recommendation (0-1)
    public var confidence: Float

    /// Raw response text from VLM
    public var rawResponse: String?

    public init(
        recommendedCandidateIndex: Int? = nil,
        recommendation: String = "",
        adjustments: [PlacementAdjustment] = [],
        styleSuggestions: [String] = [],
        concerns: [String] = [],
        confidence: Float = 0,
        rawResponse: String? = nil
    ) {
        self.recommendedCandidateIndex = recommendedCandidateIndex
        self.recommendation = recommendation
        self.adjustments = adjustments
        self.styleSuggestions = styleSuggestions
        self.concerns = concerns
        self.confidence = confidence
        self.rawResponse = rawResponse
    }
}

/// Suggested adjustment from VLM
public struct PlacementAdjustment: Codable {
    public enum AdjustmentType: String, Codable {
        case rotate
        case shiftX
        case shiftZ
        case alignWithWall
        case alignWithFurniture
    }

    public var type: AdjustmentType
    public var amount: Float            // Meters or radians
    public var reason: String

    public init(type: AdjustmentType, amount: Float, reason: String) {
        self.type = type
        self.amount = amount
        self.reason = reason
    }
}

// MARK: - Solver Result

/// Complete result from spatial reasoning + VLM
public struct PlacementResult {
    /// Whether furniture fits in the room
    public var fits: Bool

    /// All candidate placements with scores
    public var candidates: [PlacementCandidate]

    /// Best candidate (highest composite score)
    public var bestCandidate: PlacementCandidate? {
        candidates.filter { $0.isValid }.max { $0.compositeScore < $1.compositeScore }
    }

    /// VLM recommendation (if requested)
    public var vlmResponse: VLMDesignResponse?

    /// Processing time
    public var solverTimeMs: Double
    public var vlmTimeMs: Double?

    public init(
        fits: Bool,
        candidates: [PlacementCandidate],
        vlmResponse: VLMDesignResponse? = nil,
        solverTimeMs: Double = 0,
        vlmTimeMs: Double? = nil
    ) {
        self.fits = fits
        self.candidates = candidates
        self.vlmResponse = vlmResponse
        self.solverTimeMs = solverTimeMs
        self.vlmTimeMs = vlmTimeMs
    }
}

// Note: SIMD3 already conforms to Codable in Swift 5.9+
