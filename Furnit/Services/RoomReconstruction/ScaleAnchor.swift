import CoreGraphics
import UIKit
import simd

typealias RoomDepthMap = [[Float]]

struct ScaleObservation: Sendable {
    let source: String
    let tier: AnchorTier
    /// Multiply raw Depth Anything values by this scale to get meters.
    let depthScale: Double
    /// Detection or quality confidence in 0...1.
    let detConf: Double
    /// Estimated final room height if this observation scale is used.
    let impliedRoomHeight: Double
    let debug: String
}

typealias ScaleCandidate = ScaleObservation

enum AnchorTier: Int, Sendable {
    case architectural = 1
    case codeFixture = 2
    case cameraHeight = 3
    case furniture = 4
}

struct ScaleReference: Sendable {
    let meanMeters: Double
    let coefficientOfVariation: Double
    let tier: AnchorTier
}

enum ScaleReferences {
    static let all: [String: ScaleReference] = [
        "tile": ScaleReference(meanMeters: 0.0, coefficientOfVariation: 0.03, tier: .architectural),
        "door_width": ScaleReference(meanMeters: 0.83, coefficientOfVariation: 0.06, tier: .architectural),
        "toilet_seat": ScaleReference(meanMeters: 0.40, coefficientOfVariation: 0.12, tier: .codeFixture),
        "counter": ScaleReference(meanMeters: 0.90, coefficientOfVariation: 0.10, tier: .codeFixture),
        "outlet_switch": ScaleReference(meanMeters: 1.20, coefficientOfVariation: 0.10, tier: .codeFixture),
        "sink_rim": ScaleReference(meanMeters: 0.85, coefficientOfVariation: 0.14, tier: .codeFixture),
        // Furniture is intentionally listed for exclusion and debug only; it must not emit scale observations.
        "chair": ScaleReference(meanMeters: 0.95, coefficientOfVariation: 0.38, tier: .furniture),
        "couch": ScaleReference(meanMeters: 0.80, coefficientOfVariation: 0.40, tier: .furniture),
        "bed": ScaleReference(meanMeters: 0.55, coefficientOfVariation: 0.45, tier: .furniture),
        "plant": ScaleReference(meanMeters: 0.90, coefficientOfVariation: 0.55, tier: .furniture),
    ]
}

protocol ScaleAnchor {
    /// Returns 0...n candidates. Failures are represented by an empty result.
    func candidates(ctx: SceneContext) -> [ScaleCandidate]
}

struct ScaleObjectBox {
    let classIdx: Int
    let confidence: Float
    let rect: CGRect
}

struct SceneContext {
    let rawDepth: RoomDepthMap
    let focalPx: Double
    let levelingRotation: simd_float3x3
    let furnitureBoxes: [CGRect]
    let objectBoxes: [ScaleObjectBox]
    let workingImage: UIImage
    let imageSize: CGSize
    let rawCameraHeightMeters: Double?
    let fallbackDepthScale: Double
    let impliedRoomHeightForScale: (Double) -> Double?
}
