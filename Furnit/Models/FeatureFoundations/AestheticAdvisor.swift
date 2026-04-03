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

private func srgbToLCH(_ rgb: SIMD3<Float>) -> (L: Float, C: Float, H: Float) {
    func linearize(_ component: Float) -> Float {
        component <= 0.04045 ? component / 12.92 : pow((component + 0.055) / 1.055, 2.4)
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
    return (l, c, h)
}

private func hueDifference(_ lhs: Float, _ rhs: Float) -> Float {
    let diff = abs(lhs - rhs)
    return diff > 180 ? 360 - diff : diff
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
        var roomColors: [SIMD3<Float>] = []
        roomColors.append(contentsOf: palette.floor?.dominantColors ?? [])
        roomColors.append(contentsOf: palette.walls?.dominantColors ?? [])
        guard !roomColors.isEmpty else { return (0.6, .neutral) }

        let (_, furnitureChroma, furnitureHue) = srgbToLCH(furnitureColor)
        if furnitureChroma < 15 { return (0.75, .neutral) }

        var best: (Float, HarmonyType) = (0.2, .clash)
        for roomColor in roomColors {
            let (_, roomChroma, roomHue) = srgbToLCH(roomColor)
            if roomChroma < 15 {
                if best.0 < 0.75 { best = (0.75, .neutral) }
                continue
            }
            let diff = hueDifference(furnitureHue, roomHue)
            let candidate: (Float, HarmonyType)
            switch diff {
            case 0..<30: candidate = (0.85, .analogous)
            case 150..<210: candidate = (0.80, .complementary)
            case 110..<130, 230..<250: candidate = (0.72, .triadic)
            case 30..<60: candidate = (0.60, .splitComplementary)
            default: candidate = (0.30, .clash)
            }
            if candidate.0 > best.0 { best = candidate }
        }
        return best
    }

    private func contrastScore(for furnitureColor: SIMD3<Float>) -> Float {
        let (furnitureLightness, _, _) = srgbToLCH(furnitureColor)
        let roomReference = palette.floor?.dominantColors.first ?? palette.walls?.dominantColors.first ?? SIMD3<Float>(repeating: 0.5)
        let (roomLightness, _, _) = srgbToLCH(roomReference)
        return min(abs(furnitureLightness - roomLightness) / 100, 1)
    }

    private func styleScore(for furnitureTags: [String]) -> Float {
        guard !roomStyleTags.isEmpty, !furnitureTags.isEmpty else { return 0.5 }
        var matches = 0
        for roomTag in roomStyleTags {
            let compatible = Self.compatibleStyles[roomTag, default: []]
            matches += furnitureTags.filter { compatible.contains($0) }.count
        }
        return min(Float(matches) / Float(max(furnitureTags.count, roomStyleTags.count)), 1)
    }

    private func buildRecommendations(
        harmony: (score: Float, type: HarmonyType),
        contrast: Float,
        style: Float
    ) -> [String] {
        var result: [String] = []
        if harmony.type == .clash {
            result.append("Consider a more neutral color to reduce palette clash.")
        }
        if contrast < 0.15 {
            result.append("Increase contrast against the floor for better visual separation.")
        }
        if style < 0.30, let first = roomStyleTags.first {
            result.append("A style closer to \(first) will read as more coherent in this room.")
        }
        if result.isEmpty {
            result.append("Current furniture profile is broadly compatible with the room palette.")
        }
        return result
    }
}
