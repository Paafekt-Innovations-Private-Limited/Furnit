import Foundation
import simd

public struct FurnitureProfile: Equatable, Sendable {
    public let primaryColor: SIMD3<Float>
    public let accentColor: SIMD3<Float>?
    public let styleTags: [String]

    public init(primaryColor: SIMD3<Float>, accentColor: SIMD3<Float>?, styleTags: [String]) {
        self.primaryColor = primaryColor
        self.accentColor = accentColor
        self.styleTags = styleTags
    }
}

public enum HarmonyType: String, Sendable {
    case analogous
    case complementary
    case triadic
    case splitComplementary
    case neutral
    case clash
}

public struct AestheticScore: Equatable, Sendable {
    public let harmonyScore: Float
    public let harmonyType: HarmonyType
    public let contrastScore: Float
    public let styleCompatibilityScore: Float
    public let recommendations: [String]

    public init(
        harmonyScore: Float,
        harmonyType: HarmonyType,
        contrastScore: Float,
        styleCompatibilityScore: Float,
        recommendations: [String]
    ) {
        self.harmonyScore = harmonyScore
        self.harmonyType = harmonyType
        self.contrastScore = contrastScore
        self.styleCompatibilityScore = styleCompatibilityScore
        self.recommendations = recommendations
    }
}

private struct LCHColor: Sendable {
    let L: Float
    let C: Float
    let H: Float
}

private struct WeightedRoomColor: Sendable {
    let color: SIMD3<Float>
    let weight: Float
}

private func srgbToLCH(_ rgb: SIMD3<Float>) -> LCHColor {
    func linearize(_ component: Float) -> Float {
        let clamped = min(max(component, 0), 1)
        return clamped <= 0.04045 ? clamped / 12.92 : pow((clamped + 0.055) / 1.055, 2.4)
    }

    let r = linearize(rgb.x)
    let g = linearize(rgb.y)
    let b = linearize(rgb.z)
    let x = r * 0.4124 + g * 0.3576 + b * 0.1805
    let y = r * 0.2126 + g * 0.7152 + b * 0.0722
    let z = r * 0.0193 + g * 0.1192 + b * 0.9505

    func f(_ value: Float) -> Float {
        value > 0.008856 ? pow(value, 1.0 / 3.0) : (7.787 * value + 16.0 / 116.0)
    }

    let fx = f(x / 0.95047)
    let fy = f(y / 1.00000)
    let fz = f(z / 1.08883)
    let l = 116 * fy - 16
    let a = 500 * (fx - fy)
    let bLab = 200 * (fy - fz)
    let c = sqrt(a * a + bLab * bLab)
    var h = atan2(bLab, a) * (180.0 / Float.pi)
    if h < 0 { h += 360 }
    return LCHColor(L: l, C: c, H: h)
}

private func hueDifference(_ lhs: Float, _ rhs: Float) -> Float {
    let diff = abs(lhs - rhs)
    return diff > 180 ? 360 - diff : diff
}

private func clamp01(_ value: Float) -> Float {
    min(max(value, 0), 1)
}

private func bellCurveScore(value: Float, target: Float, tolerance: Float) -> Float {
    guard tolerance > 0 else { return value == target ? 1 : 0 }
    let normalized = (value - target) / tolerance
    return Float(Foundation.exp(Double(-0.5 * normalized * normalized)))
}

public struct AestheticAdvisor {
    private static let compatibleStyles: [String: Set<String>] = [
        "modern": ["modern", "minimalist", "contemporary", "industrial"],
        "rustic": ["rustic", "farmhouse", "traditional", "eclectic"],
        "scandinavian": ["scandinavian", "minimalist", "modern", "nordic"],
        "industrial": ["industrial", "modern", "eclectic"],
        "traditional": ["traditional", "rustic", "classic"]
    ]

    private let palette: SurfacePalette
    private let roomStyleTags: [String]

    public init(palette: SurfacePalette, roomStyleTags: [String] = []) {
        self.palette = palette
        self.roomStyleTags = roomStyleTags
    }

    public func evaluate(furniture: FurnitureProfile) -> AestheticScore {
        let harmony = harmonyScore(for: furniture.primaryColor)
        let contrast = contrastScore(for: furniture.primaryColor)
        let style = styleScore(for: furniture.styleTags)
        let recommendations = buildRecommendations(harmony: harmony, contrast: contrast, style: style)
        return AestheticScore(
            harmonyScore: harmony.score,
            harmonyType: harmony.type,
            contrastScore: contrast,
            styleCompatibilityScore: style,
            recommendations: recommendations
        )
    }

    private func harmonyScore(for furnitureColor: SIMD3<Float>) -> (score: Float, type: HarmonyType) {
        let roomColors = weightedRoomColors()
        guard !roomColors.isEmpty else { return (0.58, .neutral) }

        let furnitureLCH = srgbToLCH(furnitureColor)
        if furnitureLCH.C < 12 {
            let bestNeutral = roomColors
                .map { room -> Float in
                    let roomLCH = srgbToLCH(room.color)
                    let lightnessAlignment = 1 - min(abs(roomLCH.L - furnitureLCH.L) / 40, 1)
                    return clamp01((0.62 + lightnessAlignment * 0.24) * room.weight)
                }
                .max() ?? 0.72
            return (bestNeutral, .neutral)
        }

        var bestScore: Float = 0.18
        var bestType: HarmonyType = .clash
        for room in roomColors {
            let roomLCH = srgbToLCH(room.color)
            let hueDelta = hueDifference(furnitureLCH.H, roomLCH.H)

            let analogous = bellCurveScore(value: hueDelta, target: 18, tolerance: 18)
            let complementary = bellCurveScore(value: hueDelta, target: 180, tolerance: 22)
            let triadic = bellCurveScore(value: hueDelta, target: 120, tolerance: 16)
            let splitComplementary = bellCurveScore(value: hueDelta, target: 150, tolerance: 18)
            let neutral = clamp01(1 - max(furnitureLCH.C, roomLCH.C) / 70) * 0.92

            let rawCandidates: [(HarmonyType, Float)] = [
                (.analogous, analogous),
                (.complementary, complementary),
                (.triadic, triadic),
                (.splitComplementary, splitComplementary),
                (.neutral, neutral)
            ]
            let bestRaw = rawCandidates.max(by: { $0.1 < $1.1 }) ?? (.clash, 0.18)
            let chromaBalance = 1 - min(abs(furnitureLCH.C - roomLCH.C) / 55, 1)
            let lightnessBalance = 1 - min(abs(furnitureLCH.L - roomLCH.L) / 45, 1)
            let weightedScore = clamp01(
                (bestRaw.1 * 0.62 + chromaBalance * 0.20 + lightnessBalance * 0.18) * room.weight
            )

            if weightedScore > bestScore {
                bestScore = weightedScore
                bestType = bestRaw.0
            }
        }

        if bestScore < 0.34 {
            return (bestScore, .clash)
        }
        return (bestScore, bestType)
    }

    private func contrastScore(for furnitureColor: SIMD3<Float>) -> Float {
        let furnitureLCH = srgbToLCH(furnitureColor)
        let references = [
            palette.floor?.dominantColors.first,
            palette.walls?.dominantColors.first,
            palette.ceiling?.dominantColors.first
        ].compactMap { $0 }

        guard !references.isEmpty else { return 0.5 }

        let scores = references.map { color -> Float in
            let roomLCH = srgbToLCH(color)
            let delta = abs(furnitureLCH.L - roomLCH.L)
            let perceptualSeparation = bellCurveScore(value: delta, target: 26, tolerance: 18)
            let safetyPenalty: Float = delta < 8 ? 0.25 : 0
            return clamp01(perceptualSeparation - safetyPenalty)
        }
        return scores.max() ?? 0.5
    }

    private func styleScore(for furnitureTags: [String]) -> Float {
        let normalizedRoom = roomStyleTags.map { $0.lowercased() }
        let normalizedFurniture = furnitureTags.map { $0.lowercased() }
        guard !normalizedRoom.isEmpty, !normalizedFurniture.isEmpty else { return 0.5 }
        var matches = 0
        for roomTag in normalizedRoom {
            let compatible = Self.compatibleStyles[roomTag, default: []]
            matches += normalizedFurniture.filter { compatible.contains($0) || $0 == roomTag }.count
        }
        return min(Float(matches) / Float(max(normalizedFurniture.count, normalizedRoom.count)), 1)
    }

    private func buildRecommendations(
        harmony: (score: Float, type: HarmonyType),
        contrast: Float,
        style: Float
    ) -> [String] {
        var result: [String] = []
        if harmony.type == .clash {
            result.append("roomViewer.aestheticRecoNeutralClash".localized)
        } else if harmony.type == .analogous {
            result.append("roomViewer.aestheticRecoAnalogousHarmony".localized)
        } else if harmony.type == .complementary {
            result.append("roomViewer.aestheticRecoComplementaryFocal".localized)
        }
        if contrast < 0.15 {
            result.append("roomViewer.aestheticRecoContrastLow".localized)
        } else if contrast > 0.80 {
            result.append("roomViewer.aestheticRecoContrastHigh".localized)
        }
        if style < 0.30, let first = roomStyleTags.first {
            let tagLabel = Self.localizedRoomStyleTag(first)
            result.append("roomViewer.aestheticRecoStyleMismatch".localized(tagLabel))
        }
        if result.isEmpty {
            result.append("roomViewer.aestheticRecoBroadlyCompatible".localized)
        }
        return result
    }

    private static func localizedRoomStyleTag(_ raw: String) -> String {
        let slug = String(raw.lowercased().filter { $0.isLetter || $0.isNumber })
        guard !slug.isEmpty else { return raw.capitalized }
        let key = "roomViewer.styleTag." + slug
        let translated = key.localized
        return translated == key ? raw.capitalized : translated
    }

    private func weightedRoomColors() -> [WeightedRoomColor] {
        var colors: [WeightedRoomColor] = []
        for color in palette.floor?.dominantColors ?? [] {
            colors.append(WeightedRoomColor(color: color, weight: 1.00))
        }
        for color in palette.walls?.dominantColors ?? [] {
            colors.append(WeightedRoomColor(color: color, weight: 1.12))
        }
        for color in palette.ceiling?.dominantColors ?? [] {
            colors.append(WeightedRoomColor(color: color, weight: 0.72))
        }

        guard !colors.isEmpty else { return [] }
        let maxWeight = colors.map(\.weight).max() ?? 1
        return colors.map { WeightedRoomColor(color: $0.color, weight: clamp01($0.weight / maxWeight)) }
    }
}
