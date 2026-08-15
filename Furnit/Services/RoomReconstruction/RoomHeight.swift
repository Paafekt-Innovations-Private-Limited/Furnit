import simd

struct LeveledDepthPointGrid {
    let width: Int
    let height: Int
    let viewDirectionHorizontal: SIMD2<Float>?
    private let points: [SIMD3<Float>]
    private let valid: [Bool]

    init(
        depth: [Float],
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rotation: simd_float3x3
    ) {
        self.width = width
        self.height = height
        let pixelCount = max(0, width * height)
        var points = [SIMD3<Float>](repeating: .zero, count: pixelCount)
        var valid = [Bool](repeating: false, count: pixelCount)

        if width > 1, height > 1, depth.count == pixelCount, fx > 1, fy > 1 {
            for y in 0..<height {
                let cameraYUnit = (Float(y) - cy) / fy
                for x in 0..<width {
                    let index = y * width + x
                    let d = depth[index]
                    guard d.isFinite, d > 0 else { continue }
                    let cameraPoint = SIMD3<Float>(
                        (Float(x) - cx) * d / fx,
                        cameraYUnit * d,
                        d
                    )
                    points[index] = rotation * cameraPoint
                    valid[index] = true
                }
            }
        }

        let view = rotation * SIMD3<Float>(0, 0, 1)
        let horizontal = SIMD2<Float>(view.x, view.z)
        let horizontalLength = simd_length(horizontal)
        self.viewDirectionHorizontal = horizontalLength > 1e-4 ? horizontal / horizontalLength : nil
        self.points = points
        self.valid = valid
    }

    func point(x: Int, y: Int, scale: Float = 1) -> SIMD3<Float>? {
        guard x >= 0, x < width, y >= 0, y < height else { return nil }
        let index = y * width + x
        guard index < valid.count, valid[index] else { return nil }
        let point = points[index]
        return abs(scale - 1) > 1e-6 ? point * scale : point
    }
}

struct RoomHeightResult {
    let height: Float
    let confidence: Float
    let approximate: Bool
    let vFloor: Float?
    let vCeil: Float?
    let vHorizon: Float?
    let normalSign: Float
    let debug: String
    let selfCheckDebug: String
}

struct RoomWidthResult {
    let width: Float
    let confidence: Float
    let approximate: Bool
    let debug: String
}

enum RoomHeight {
    static func roomHeightSingleView(
        depth: [Float],
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        pitch: Float,
        rotation: simd_float3x3,
        cameraHeight: Float = 1.60
    ) -> RoomHeightResult {
        let pointGrid = LeveledDepthPointGrid(
            depth: depth,
            width: width,
            height: height,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            rotation: rotation
        )
        return roomHeightSingleView(
            pointGrid: pointGrid,
            fy: fy,
            cy: cy,
            pitch: pitch,
            cameraHeight: cameraHeight
        )
    }

    static func roomHeightSingleView(
        pointGrid: LeveledDepthPointGrid,
        fy: Float,
        cy: Float,
        pitch: Float,
        cameraHeight: Float = 1.60
    ) -> RoomHeightResult {
        let variants: [(pitchSign: Float, normalSign: Float)] = [
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]

        var best: RoomHeightResult?
        let junctionsByNormalSign = junctionRows(pointGrid: pointGrid)
        for variant in variants {
            let junctions = variant.normalSign > 0
                ? junctionsByNormalSign.positive
                : junctionsByNormalSign.negative
            let result = estimateOnce(
                junctions: junctions,
                fy: fy,
                cy: cy,
                pitch: pitch * variant.pitchSign,
                cameraHeight: cameraHeight,
                normalSign: variant.normalSign,
                variantLabel: "pitchSign=\(Int(variant.pitchSign)) normalSign=\(Int(variant.normalSign))"
            )
            if result.selfCheckDebug.contains("ok=true"), result.confidence >= 0.5 {
                return result
            }
            if best == nil || result.confidence > best!.confidence {
                best = result
            }
        }
        return best ?? RoomHeightResult(
            height: 0,
            confidence: 0.2,
            approximate: true,
            vFloor: nil,
            vCeil: nil,
            vHorizon: nil,
            normalSign: 1,
            debug: "height_unavailable",
            selfCheckDebug: "ok=false no_variants"
        )
    }

    private struct JunctionRows {
        let floor: [Float]
        let ceiling: [Float]
    }

    private struct SignedJunctionRows {
        let positive: JunctionRows
        let negative: JunctionRows
    }

    /// A negative normal sign only swaps floor and ceiling; wall classification is unchanged.
    /// Classify each sampled column once, then derive all four directional junction searches.
    private static func junctionRows(pointGrid: LeveledDepthPointGrid) -> SignedJunctionRows {
        let width = pointGrid.width
        let height = pointGrid.height
        var positiveFloorRows: [Float] = []
        var positiveCeilingRows: [Float] = []
        var negativeFloorRows: [Float] = []
        var negativeCeilingRows: [Float] = []
        let xStart = max(1, width * 2 / 5)
        let xEnd = min(width - 2, width * 3 / 5)
        let xStep = max(1, (xEnd - xStart) / 180)

        func transitionRow(
            surfaces: [Surface?],
            target: Surface,
            descending: Bool
        ) -> Float? {
            var lastTarget = -1
            let rows: AnySequence<Int> = descending
                ? AnySequence(stride(from: height - 2, through: 1, by: -1))
                : AnySequence(stride(from: 1, to: height - 1, by: 1))
            for y in rows {
                guard let surface = surfaces[y] else { continue }
                if surface == target {
                    lastTarget = y
                } else if surface == .wall, lastTarget >= 0 {
                    return Float(lastTarget)
                }
            }
            return lastTarget >= 0 ? Float(lastTarget) : nil
        }

        var x = xStart
        while x <= xEnd {
            var surfaces = [Surface?](repeating: nil, count: height)
            if height > 2 {
                for y in 1..<(height - 1) {
                    surfaces[y] = worldNormal(pointGrid, x, y, 1).map(classify)
                }
            }
            if let row = transitionRow(surfaces: surfaces, target: .floor, descending: true) {
                positiveFloorRows.append(row)
            }
            if let row = transitionRow(surfaces: surfaces, target: .ceiling, descending: false) {
                positiveCeilingRows.append(row)
            }
            if let row = transitionRow(surfaces: surfaces, target: .ceiling, descending: true) {
                negativeFloorRows.append(row)
            }
            if let row = transitionRow(surfaces: surfaces, target: .floor, descending: false) {
                negativeCeilingRows.append(row)
            }
            x += xStep
        }
        return SignedJunctionRows(
            positive: JunctionRows(floor: positiveFloorRows, ceiling: positiveCeilingRows),
            negative: JunctionRows(floor: negativeFloorRows, ceiling: negativeCeilingRows)
        )
    }

    private static func estimateOnce(
        junctions: JunctionRows,
        fy: Float,
        cy: Float,
        pitch: Float,
        cameraHeight: Float,
        normalSign: Float,
        variantLabel: String
    ) -> RoomHeightResult {
        let clampedCameraHeight = min(1.75, max(1.55, cameraHeight))
        let horizonRow = cy + fy * tan(pitch)

        let floorRows = junctions.floor
        let ceilingRows = junctions.ceiling

        guard floorRows.count >= 10, ceilingRows.count >= 10 else {
            return RoomHeightResult(
                height: 0,
                confidence: 0.2,
                approximate: true,
                vFloor: nil,
                vCeil: nil,
                vHorizon: horizonRow,
                normalSign: normalSign,
                debug: "junctions_missing f=\(floorRows.count) c=\(ceilingRows.count) \(variantLabel)",
                selfCheckDebug: "vCeil=nan vHorizon=\(horizonRow) vFloor=nan ok=false \(variantLabel)"
            )
        }

        let floorRow = median(floorRows)
        let ceilingRow = median(ceilingRows)
        let selfCheckOK = (floorRow > ceilingRow) && (ceilingRow < horizonRow) && (horizonRow < floorRow)
        let selfCheck = String(
            format: "vCeil=%.1f vHorizon=%.1f vFloor=%.1f ok=%@ %@",
            ceilingRow,
            horizonRow,
            floorRow,
            selfCheckOK ? "true" : "false",
            variantLabel
        )

        let alpha = atan((floorRow - horizonRow) / fy)
        let beta = atan((horizonRow - ceilingRow) / fy)
        guard alpha > 0.02, beta > 0 else {
            return RoomHeightResult(
                height: 0,
                confidence: 0.2,
                approximate: true,
                vFloor: floorRow,
                vCeil: ceilingRow,
                vHorizon: horizonRow,
                normalSign: normalSign,
                debug: String(
                    format: "bad_angles a=%.3f b=%.3f vH=%.1f vF=%.1f vC=%.1f %@",
                    alpha,
                    beta,
                    horizonRow,
                    floorRow,
                    ceilingRow,
                    variantLabel
                ),
                selfCheckDebug: selfCheck
            )
        }

        var roomHeight = clampedCameraHeight * (tan(alpha) + tan(beta)) / tan(alpha)
        let floorStd = stdev(floorRows)
        let ceilingStd = stdev(ceilingRows)
        let rowsAgree = floorStd < 40 && ceilingStd < 40
        var confidence: Float = (selfCheckOK && rowsAgree) ? 0.85 : (selfCheckOK ? 0.5 : 0.3)
        if !(2.0...3.6).contains(roomHeight) {
            confidence = min(confidence, 0.3)
        }
        roomHeight = min(3.6, max(2.0, roomHeight))

        let debug = String(
            format: "H=%.3f hc=%.2f a=%.1fdeg b=%.1fdeg vH=%d vF=%d vC=%d fN=%d cN=%d fSd=%d cSd=%d %@",
            roomHeight,
            clampedCameraHeight,
            alpha * 57.2958,
            beta * 57.2958,
            Int(horizonRow),
            Int(floorRow),
            Int(ceilingRow),
            floorRows.count,
            ceilingRows.count,
            Int(floorStd),
            Int(ceilingStd),
            variantLabel
        )
        return RoomHeightResult(
            height: roomHeight,
            confidence: confidence,
            approximate: confidence < 0.7,
            vFloor: floorRow,
            vCeil: ceilingRow,
            vHorizon: horizonRow,
            normalSign: normalSign,
            debug: debug,
            selfCheckDebug: selfCheck
        )
    }

    private static func median(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        let sorted = values.sorted()
        return sorted[sorted.count / 2]
    }

    private static func stdev(_ values: [Float]) -> Float {
        guard values.count > 1 else { return 999 }
        let mean = values.reduce(0, +) / Float(values.count)
        return sqrt(values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Float(values.count))
    }

    private static func cameraPoint(
        _ x: Int,
        _ y: Int,
        _ depth: Float,
        _ fx: Float,
        _ fy: Float,
        _ cx: Float,
        _ cy: Float
    ) -> SIMD3<Float> {
        SIMD3<Float>((Float(x) - cx) * depth / fx, (Float(y) - cy) * depth / fy, depth)
    }

    private static func worldNormal(
        _ pointGrid: LeveledDepthPointGrid,
        _ x: Int,
        _ y: Int,
        _ normalSign: Float
    ) -> SIMD3<Float>? {
        let width = pointGrid.width
        let height = pointGrid.height
        guard x > 0, x < width - 1, y > 0, y < height - 1 else { return nil }
        guard let left = pointGrid.point(x: x - 1, y: y),
              let right = pointGrid.point(x: x + 1, y: y),
              let up = pointGrid.point(x: x, y: y - 1),
              let down = pointGrid.point(x: x, y: y + 1) else {
            return nil
        }

        let cross = simd_cross(right - left, down - up)
        let length = simd_length(cross)
        guard length > 1e-6 else { return nil }
        let leveled = (cross / length) * normalSign
        // Existing leveling frame has positive Y down; flip to Y-up for classification.
        return simd_normalize(SIMD3<Float>(leveled.x, -leveled.y, leveled.z))
    }

    private enum Surface {
        case floor
        case ceiling
        case wall
    }

    private static func classify(_ normal: SIMD3<Float>) -> Surface {
        if normal.y > 0.7 { return .floor }
        if normal.y < -0.7 { return .ceiling }
        return .wall
    }

    private static func floorWallRow(
        _ pointGrid: LeveledDepthPointGrid,
        col x: Int,
        _ normalSign: Float
    ) -> Float? {
        var lastFloor = -1
        for y in stride(from: pointGrid.height - 2, through: 1, by: -1) {
            guard let normal = worldNormal(pointGrid, x, y, normalSign) else { continue }
            let surface = classify(normal)
            if surface == .floor {
                lastFloor = y
            } else if surface == .wall, lastFloor >= 0 {
                return Float(lastFloor)
            }
        }
        return lastFloor >= 0 ? Float(lastFloor) : nil
    }

    private static func ceilingWallRow(
        _ pointGrid: LeveledDepthPointGrid,
        col x: Int,
        _ normalSign: Float
    ) -> Float? {
        var lastCeiling = -1
        for y in stride(from: 1, to: pointGrid.height - 1, by: 1) {
            guard let normal = worldNormal(pointGrid, x, y, normalSign) else { continue }
            let surface = classify(normal)
            if surface == .ceiling {
                lastCeiling = y
            } else if surface == .wall, lastCeiling >= 0 {
                return Float(lastCeiling)
            }
        }
        return lastCeiling >= 0 ? Float(lastCeiling) : nil
    }

    static func roomWidthSingleView(
        depth: [Float],
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        rotation: simd_float3x3,
        vFloor: Float?,
        vHorizon: Float?,
        vCeil: Float?,
        normalSign: Float,
        cameraHeight: Float = 1.60
    ) -> RoomWidthResult {
        let pointGrid = LeveledDepthPointGrid(
            depth: depth,
            width: width,
            height: height,
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            rotation: rotation
        )
        return roomWidthSingleView(
            pointGrid: pointGrid,
            vFloor: vFloor,
            vHorizon: vHorizon,
            vCeil: vCeil,
            normalSign: normalSign,
            cameraHeight: cameraHeight
        )
    }

    static func roomWidthSingleView(
        pointGrid: LeveledDepthPointGrid,
        vFloor: Float?,
        vHorizon: Float?,
        vCeil: Float?,
        normalSign: Float,
        cameraHeight: Float = 1.60
    ) -> RoomWidthResult {
        guard let vFloor,
              let vHorizon,
              let vCeil,
              vFloor > vHorizon,
              vFloor > vCeil else {
            return RoomWidthResult(width: 0, confidence: 0.2, approximate: true, debug: "junctions_missing")
        }

        let backWall = backWallPixelWidth(
            pointGrid: pointGrid,
            vFloor: vFloor,
            vCeil: vCeil,
            normalSign: normalSign
        )
        let denominator = vFloor - vHorizon
        guard backWall.span > 0, denominator > 1 else {
            return RoomWidthResult(
                width: 0,
                confidence: 0.2,
                approximate: true,
                debug: "backwall_missing count=\(backWall.count) denom=\(Int(denominator))"
            )
        }

        let clampedCameraHeight = min(1.75, max(1.55, cameraHeight))
        var roomWidth = backWall.span * clampedCameraHeight / denominator
        var confidence: Float = backWall.count > 2_000 ? 0.8 : 0.5
        if !(1.0...12.0).contains(roomWidth) {
            confidence = 0.3
        }
        roomWidth = min(12.0, max(1.0, roomWidth))
        return RoomWidthResult(
            width: roomWidth,
            confidence: confidence,
            approximate: confidence < 0.7,
            debug: String(
                format: "W=%.3f span=%dpx xL=%d xR=%d count=%d denom=%d",
                roomWidth,
                Int(backWall.span),
                backWall.xLeft,
                backWall.xRight,
                backWall.count,
                Int(denominator)
            )
        )
    }

    private static func backWallPixelWidth(
        pointGrid: LeveledDepthPointGrid,
        vFloor: Float,
        vCeil: Float,
        normalSign: Float
    ) -> (span: Float, count: Int, xLeft: Int, xRight: Int) {
        guard let viewDirection = pointGrid.viewDirectionHorizontal else { return (0, 0, 0, 0) }

        func isBackWall(_ normal: SIMD3<Float>) -> Bool {
            let horizontal = SIMD2<Float>(normal.x, normal.z)
            let length = simd_length(horizontal)
            if length < 0.3 { return false }
            return abs(simd_dot(horizontal / length, viewDirection)) > 0.7
        }

        let yTop = Int(vCeil + (vFloor - vCeil) * 0.35)
        let yBottom = Int(vCeil + (vFloor - vCeil) * 0.65)
        let minY = max(1, min(pointGrid.height - 2, yTop))
        let maxY = max(1, min(pointGrid.height - 2, yBottom))
        guard minY <= maxY else { return (0, 0, 0, 0) }

        var xCounts = [Int](repeating: 0, count: pointGrid.width)
        var backWallCount = 0
        for y in minY...maxY {
            for x in 1..<(pointGrid.width - 1) {
                guard let normal = worldNormal(pointGrid, x, y, normalSign) else { continue }
                if isBackWall(normal) {
                    xCounts[x] += 1
                    backWallCount += 1
                }
            }
        }
        guard backWallCount > 200 else { return (0, backWallCount, 0, 0) }

        func xValue(atSortedIndex targetIndex: Int) -> Int {
            var cumulativeCount = 0
            for x in 0..<xCounts.count {
                cumulativeCount += xCounts[x]
                if targetIndex < cumulativeCount { return x }
            }
            return max(0, xCounts.count - 1)
        }

        let leftIndex = min(backWallCount - 1, max(0, Int(Double(backWallCount - 1) * 0.025)))
        let rightIndex = min(backWallCount - 1, max(0, Int(Double(backWallCount - 1) * 0.975)))
        let xLeft = xValue(atSortedIndex: leftIndex)
        let xRight = xValue(atSortedIndex: rightIndex)
        return (Float(xRight - xLeft), backWallCount, xLeft, xRight)
    }
}
