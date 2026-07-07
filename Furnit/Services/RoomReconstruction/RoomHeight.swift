import simd

struct RoomHeightResult {
    let height: Float
    let confidence: Float
    let approximate: Bool
    let debug: String
    let selfCheckDebug: String
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
        let variants: [(pitchSign: Float, normalSign: Float)] = [
            (1, 1),
            (-1, 1),
            (1, -1),
            (-1, -1),
        ]

        var best: RoomHeightResult?
        for variant in variants {
            let result = estimateOnce(
                depth: depth,
                width: width,
                height: height,
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                pitch: pitch * variant.pitchSign,
                rotation: rotation,
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
            debug: "height_unavailable",
            selfCheckDebug: "ok=false no_variants"
        )
    }

    private static func estimateOnce(
        depth: [Float],
        width: Int,
        height: Int,
        fx: Float,
        fy: Float,
        cx: Float,
        cy: Float,
        pitch: Float,
        rotation: simd_float3x3,
        cameraHeight: Float,
        normalSign: Float,
        variantLabel: String
    ) -> RoomHeightResult {
        let clampedCameraHeight = min(1.75, max(1.55, cameraHeight))
        let horizonRow = cy + fy * tan(pitch)

        var floorRows: [Float] = []
        var ceilingRows: [Float] = []
        let xStart = max(1, width * 2 / 5)
        let xEnd = min(width - 2, width * 3 / 5)
        let xStep = max(1, (xEnd - xStart) / 180)
        var x = xStart
        while x <= xEnd {
            if let row = floorWallRow(depth, width, height, col: x, fx, fy, cx, cy, rotation, normalSign) {
                floorRows.append(row)
            }
            if let row = ceilingWallRow(depth, width, height, col: x, fx, fy, cx, cy, rotation, normalSign) {
                ceilingRows.append(row)
            }
            x += xStep
        }

        guard floorRows.count >= 10, ceilingRows.count >= 10 else {
            return RoomHeightResult(
                height: 0,
                confidence: 0.2,
                approximate: true,
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
        _ depth: [Float],
        _ width: Int,
        _ height: Int,
        _ x: Int,
        _ y: Int,
        _ fx: Float,
        _ fy: Float,
        _ cx: Float,
        _ cy: Float,
        _ rotation: simd_float3x3,
        _ normalSign: Float
    ) -> SIMD3<Float>? {
        guard x > 0, x < width - 1, y > 0, y < height - 1 else { return nil }
        let centerDepth = depth[y * width + x]
        if !centerDepth.isFinite || centerDepth <= 0 { return nil }
        let leftDepth = depth[y * width + x - 1]
        let rightDepth = depth[y * width + x + 1]
        let upDepth = depth[(y - 1) * width + x]
        let downDepth = depth[(y + 1) * width + x]
        if leftDepth <= 0 || rightDepth <= 0 || upDepth <= 0 || downDepth <= 0 { return nil }

        let left = cameraPoint(x - 1, y, leftDepth, fx, fy, cx, cy)
        let right = cameraPoint(x + 1, y, rightDepth, fx, fy, cx, cy)
        let up = cameraPoint(x, y - 1, upDepth, fx, fy, cx, cy)
        let down = cameraPoint(x, y + 1, downDepth, fx, fy, cx, cy)
        let cameraNormal = simd_normalize(simd_cross(right - left, down - up)) * normalSign
        let leveled = rotation * cameraNormal
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
        _ depth: [Float],
        _ width: Int,
        _ height: Int,
        col x: Int,
        _ fx: Float,
        _ fy: Float,
        _ cx: Float,
        _ cy: Float,
        _ rotation: simd_float3x3,
        _ normalSign: Float
    ) -> Float? {
        var lastFloor = -1
        for y in stride(from: height - 2, through: 1, by: -1) {
            guard let normal = worldNormal(depth, width, height, x, y, fx, fy, cx, cy, rotation, normalSign) else { continue }
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
        _ depth: [Float],
        _ width: Int,
        _ height: Int,
        col x: Int,
        _ fx: Float,
        _ fy: Float,
        _ cx: Float,
        _ cy: Float,
        _ rotation: simd_float3x3,
        _ normalSign: Float
    ) -> Float? {
        var lastCeiling = -1
        for y in stride(from: 1, to: height - 1, by: 1) {
            guard let normal = worldNormal(depth, width, height, x, y, fx, fy, cx, cy, rotation, normalSign) else { continue }
            let surface = classify(normal)
            if surface == .ceiling {
                lastCeiling = y
            } else if surface == .wall, lastCeiling >= 0 {
                return Float(lastCeiling)
            }
        }
        return lastCeiling >= 0 ? Float(lastCeiling) : nil
    }
}
